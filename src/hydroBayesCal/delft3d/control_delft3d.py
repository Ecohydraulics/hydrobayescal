"""
Delft3D-FLOW binding for HydroBayesCal.

This module couples HydroBayesCal to the structured-grid **Delft3D-FLOW**
engine (Deltares, Delft3D 4 suite). It mirrors the TELEMAC
(:mod:`hydroBayesCal.telemac.control_telemac`) and OpenFOAM
(:mod:`hydroBayesCal.openfoam.control_openfoam`) bindings: the Python
attribute names are shared across solvers, while the *string and file
conventions* are Delft3D-specific:

* ``<case>.mdf`` -- master definition FLOW file (the control file); the engine
  is launched through ``config_d_hydro.xml`` and the ``run_dflow2d3d.sh``
  launcher (which wraps the ``d_hydro`` executable).
* Bed roughness via Chezy / Manning / White-Colebrook (``Roumet``, ``Ccofu``,
  ``Ccofv`` keywords in the ``.mdf``); eddy viscosity/diffusivity via
  ``Vicouv`` / ``Dicouv``.
* ``trim-<case>`` -- map (field) output; ``trih-<case>`` -- history
  (monitoring-point) output. The binding enforces ``FlNcdf = #maphis#`` in the
  ``.mdf`` so that the map output is additionally written as NetCDF
  (``trim-<case>.nc``), which is read with :mod:`netCDF4` instead of the
  binary NEFIS ``.dat``/``.def`` pair.
* ``tri-diag.<case>`` -- diagnostic file; a run is only accepted when it
  contains no ``*** ERROR`` lines (Deltares convention).

The runtime environment follows the native Linux installation described at
`hydro-informatics.com <https://hydro-informatics.com/get-started/delft3d.html>`_:
an installation prefix (default ``$HOME/opt/delft3d-flow``) whose ``env.sh``
must be sourced before ``run_dflow2d3d.sh`` (serial) or
``run_dflow2d3d_parallel.sh`` (Intel-MPI parallel) is called from the case
directory. See the :doc:`usage-delft3d <usage-delft3d>` page for the workflow.
"""

import os
import re
import glob
import json
import csv
import shutil
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import spatial

from hydroBayesCal.hysim import HydroSimulations
from hydroBayesCal.function_pool import logger

#: Marker so callers / tests can detect that the binding is available.
DELFT3D_BINDING_IMPLEMENTED = True

#: Default user-space installation prefix of the native Delft3D-FLOW build
#: (see the installer script ``install-delft3d-flow-native.sh``).
DEFAULT_ENV_SCRIPT = "~/opt/delft3d-flow/env.sh"

#: MDF value of the ``Roumet`` keyword per bed-roughness formulation.
ROUGHNESS_METHODS = {
    "chezy": "C",
    "manning": "M",
    "whitecolebrook": "Z",
    "white-colebrook": "Z",
    "nikuradse": "Z",
}


class Delft3DController:
    """
    Low-level handle on a single Delft3D-FLOW case directory.

    Edits the ``<case>.mdf`` master definition file, launches the solver
    through the ``run_dflow2d3d.sh`` / ``run_dflow2d3d_parallel.sh`` launchers
    (after sourcing the installation's ``env.sh``), verifies the diagnostic
    file, and reads the NetCDF map output (``trim-<case>.nc``).
    """

    def __init__(
        self,
        case_dir: str,
        env_script: str = DEFAULT_ENV_SCRIPT,
        d_hydro_config: str = "config_d_hydro.xml",
        launcher: str = "run_dflow2d3d.sh",
        parallel_launcher: str = "run_dflow2d3d_parallel.sh",
    ):
        self.case_dir = os.path.normpath(case_dir)
        self.env_script = os.path.expanduser(env_script)
        self.d_hydro_config = d_hydro_config
        self.launcher = launcher
        self.parallel_launcher = parallel_launcher

    # Turns a relative path into an absolute one inside the case directory.
    def _case_path(self, relpath: str) -> str:
        if os.path.isabs(relpath):
            return relpath
        return os.path.join(self.case_dir, relpath)

    @property
    def mdf_file(self) -> str:
        """Absolute path to the case ``.mdf`` file.

        Resolved from the ``<mdfFile>`` entry of ``config_d_hydro.xml``;
        falls back to the single ``*.mdf`` file in the case directory.
        """
        config_path = self._case_path(self.d_hydro_config)
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                match = re.search(r"<mdfFile>\s*(.*?)\s*</mdfFile>", f.read())
            if match:
                return self._case_path(match.group(1))

        candidates = sorted(glob.glob(os.path.join(self.case_dir, "*.mdf")))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f"Cannot resolve the .mdf file of case {self.case_dir}: no <mdfFile> entry in "
            f"{self.d_hydro_config} and {len(candidates)} *.mdf candidates found."
        )

    @property
    def case_name(self) -> str:
        """Runid of the case (``<case>.mdf`` without extension), used in the
        ``trim-<case>`` / ``trih-<case>`` / ``tri-diag.<case>`` output names."""
        return os.path.splitext(os.path.basename(self.mdf_file))[0]

    @staticmethod
    def _format_mdf_value(value: Any) -> str:
        """Format a value in MDF style: numbers in scientific notation,
        strings wrapped in ``#`` markers (Delft3D string convention)."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("#"):
                return stripped
            try:
                return f"{float(stripped):.7e}"
            except ValueError:
                return f"#{stripped}#"
        return f"{float(value):.7e}"

    def update_mdf_parameter(self, keyword: str, value: Any) -> None:
        """Set ``keyword = value`` in the case ``.mdf`` file.

        The keyword is matched case-insensitively at the start of a line
        (MDF convention). If the keyword is absent it is appended, which
        Delft3D-FLOW accepts because keyword order in the MDF is free.
        """
        path = self.mdf_file
        formatted = self._format_mdf_value(value)

        with open(path, "r") as f:
            lines = f.readlines()

        pat = re.compile(rf"^(\s*{re.escape(keyword)}\s*=\s*).*$", re.IGNORECASE)
        replaced = False
        new_lines = []
        for line in lines:
            m = pat.match(line)
            if m and not replaced:
                new_lines.append(f"{m.group(1)}{formatted}\n")
                replaced = True
            else:
                new_lines.append(line)

        if not replaced:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append(f"{keyword:<6} = {formatted}\n")
            logger.info(f"Keyword '{keyword}' not found in {os.path.basename(path)} -- appended.")

        with open(path, "w") as f:
            f.writelines(new_lines)

    def set_roughness(self, value: float, formulation: str = "Manning") -> None:
        """Write a uniform bed roughness to the ``.mdf``.

        Sets ``Roumet`` to the requested formulation (``C`` Chezy, ``M``
        Manning, ``Z`` White-Colebrook) and the uniform roughness
        coefficients ``Ccofu`` / ``Ccofv`` in both grid directions.
        """
        method = ROUGHNESS_METHODS.get(formulation.replace(" ", "").lower())
        if method is None:
            raise ValueError(
                f"Unknown roughness formulation '{formulation}'. "
                f"Choose one of: Chezy, Manning, WhiteColebrook."
            )
        self.update_mdf_parameter("Roumet", f"#{method}#")
        self.update_mdf_parameter("Ccofu", float(value))
        self.update_mdf_parameter("Ccofv", float(value))

    def ensure_netcdf_output(self) -> None:
        """Force NetCDF map and history output (``FlNcdf = #maphis#``) so the
        results can be read without the binary NEFIS libraries."""
        self.update_mdf_parameter("FlNcdf", "#maphis#")

    # Runs a command and streams output line by line to stdout and optionally to a log file.
    # Raises RuntimeError if the process exits with a non-zero return code.
    def _run_streamed(self, cmd: Iterable[str], log_path: Optional[str] = None) -> None:
        env = os.environ.copy()
        env.pop("DISPLAY", None)

        log_fh = open(log_path, "w") if log_path else None
        try:
            process = subprocess.Popen(
                list(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=self.case_dir,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line.rstrip())
                if log_fh:
                    log_fh.write(line)

            process.wait()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with return code {process.returncode}: {' '.join(cmd)}"
                )
        finally:
            if log_fh:
                log_fh.close()

    def run_simulation(self, nprocs: int = 1) -> None:
        """Launch Delft3D-FLOW in the case directory.

        Sources the installation's ``env.sh`` (Intel oneAPI runtime,
        ``DELFT3D_HOME``, ``PATH`` and ``LD_LIBRARY_PATH``), then calls
        ``run_dflow2d3d.sh <config>`` or, for ``nprocs > 1``,
        ``run_dflow2d3d_parallel.sh <nprocs> <config>``. The launcher reads
        ``config_d_hydro.xml`` from the current working directory.
        """
        if not os.path.isfile(self.env_script):
            raise FileNotFoundError(
                f"Delft3D-FLOW environment script not found: {self.env_script}. "
                "Point env_script to the <prefix>/env.sh of your Delft3D-FLOW installation "
                "(default $HOME/opt/delft3d-flow/env.sh)."
            )

        if int(nprocs) > 1:
            run_cmd = f'"{self.parallel_launcher}" {int(nprocs)} "{self.d_hydro_config}"'
        else:
            run_cmd = f'"{self.launcher}" "{self.d_hydro_config}"'

        bash_cmd = f'source "{self.env_script}" && cd "{self.case_dir}" && {run_cmd}'
        run_log = os.path.join(self.case_dir, "log.run")

        print("Running Delft3D-FLOW simulation...")
        self._run_streamed(["bash", "-c", bash_cmd], log_path=run_log)
        self.check_run()
        print(f"Simulation finished. Log saved to {run_log}")

    def check_run(self) -> None:
        """Verify the run following the Delft3D 4 smoke-test conventions:
        the diagnostic file ``tri-diag.<case>`` must contain no ``*** ERROR``
        lines and the map output must exist."""
        case = self.case_name

        diag_path = self._case_path(f"tri-diag.{case}")
        if os.path.isfile(diag_path):
            with open(diag_path, "r", errors="replace") as f:
                errors = [line.strip() for line in f if "*** ERROR" in line]
            if errors:
                raise RuntimeError(
                    f"Delft3D-FLOW reported errors in {diag_path}:\n" + "\n".join(errors[:10])
                )
        else:
            logger.warning(f"Diagnostic file {diag_path} not found -- cannot verify the run.")

        has_map = os.path.isfile(self._case_path(f"trim-{case}.nc")) or os.path.isfile(
            self._case_path(f"trim-{case}.dat")
        )
        if not has_map:
            raise RuntimeError(
                f"Delft3D-FLOW finished but wrote no map output (trim-{case}.*) in {self.case_dir}."
            )

    @staticmethod
    def _to_float_array(var_data) -> np.ndarray:
        """Convert a (possibly masked) NetCDF array to float with NaN fill."""
        arr = np.ma.filled(np.ma.masked_invalid(var_data), np.nan)
        return np.asarray(arr, dtype=float)

    def extract_map_results(
        self, n_avg_timesteps: int = 1
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Read the NetCDF map output ``trim-<case>.nc``.

        Averages the last ``n_avg_timesteps`` map time steps and returns the
        horizontal coordinates of the active water-level (zeta) points together
        with a dictionary of fields using the standard HydroBayesCal quantity
        names:

        * ``WATER_LEVEL`` -- free-surface elevation ``S1``,
        * ``WATER_DEPTH`` -- ``S1 + DPS`` (bed depth ``DPS`` is positive down),
        * ``U_x`` / ``U_y`` -- east/north velocity components, obtained from the
          grid-aligned ``U1`` / ``V1`` values rotated by the local grid
          orientation ``ALFAS`` (for 3D runs, ``U1``/``V1`` are averaged over
          the layers). The staggered u/v face values are taken at the zeta-point
          index, i.e. within half a grid cell of the exact cell centre.
        * ``U_MAG`` -- velocity magnitude from ``U_x`` / ``U_y``.

        Inactive and permanently dry cells (``KCS != 1``) are removed.
        """
        try:
            import netCDF4
        except ImportError as err:
            raise ImportError(
                "Reading Delft3D-FLOW map output requires the netCDF4 package. "
                "Install it with: pip install hydroBayesCal[delft3d]"
            ) from err

        nc_path = self._case_path(f"trim-{self.case_name}.nc")
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(
                f"NetCDF map file not found: {nc_path}. The binding sets 'FlNcdf = #maphis#' in "
                "the .mdf before each run; verify the Delft3D-FLOW build supports NetCDF output."
            )

        with netCDF4.Dataset(nc_path, "r") as ds:
            def read(name):
                return self._to_float_array(ds.variables[name][:]) if name in ds.variables else None

            xz = read("XZ")
            yz = read("YZ")
            if xz is None or yz is None:
                raise KeyError(f"XZ/YZ zeta-point coordinates not found in {nc_path}.")

            s1_all = read("S1")
            if s1_all is None:
                raise KeyError(f"Water level 'S1' not found in {nc_path}.")

            n_times = s1_all.shape[0]
            n = min(max(1, int(n_avg_timesteps)), n_times)
            if n < n_avg_timesteps:
                logger.warning(
                    f"Requested n_avg_timesteps={n_avg_timesteps} but only {n_times} map "
                    f"time steps available. Averaging over {n}."
                )
            logger.info(f"Found {n_times} map time steps, averaging last {n}.")

            def time_mean(arr):
                # Mean over the trailing time steps; then collapse a remaining
                # layer axis (3D runs) so a horizontal field is left.
                out = np.nanmean(arr[-n:], axis=0)
                while out.ndim > xz.ndim:
                    out = np.nanmean(out, axis=0)
                return out

            s1 = time_mean(s1_all)

            dps = read("DPS0")
            if dps is None:
                dps = read("DPS")
                if dps is not None and dps.ndim > xz.ndim:
                    dps = time_mean(dps)
            water_depth = s1 + dps if dps is not None else None

            u1_all, v1_all = read("U1"), read("V1")
            alfas = read("ALFAS")
            u_east = v_north = None
            if u1_all is not None and v1_all is not None:
                u_grid = time_mean(u1_all)
                v_grid = time_mean(v1_all)
                if u_grid.shape != s1.shape or v_grid.shape != s1.shape:
                    logger.warning(
                        f"U1/V1 map shapes {u_grid.shape}/{v_grid.shape} do not match the "
                        f"zeta-point grid {s1.shape} -- skipping velocity extraction."
                    )
                elif alfas is not None:
                    alfas_rad = np.deg2rad(alfas)
                    u_east = u_grid * np.cos(alfas_rad) - v_grid * np.sin(alfas_rad)
                    v_north = u_grid * np.sin(alfas_rad) + v_grid * np.cos(alfas_rad)
                else:
                    logger.warning("ALFAS grid orientation not found -- returning grid-aligned velocities.")
                    u_east, v_north = u_grid, v_grid

            kcs = read("KCS")

        # Keep only active internal cells with valid coordinates.
        mask = np.isfinite(xz.ravel()) & np.isfinite(yz.ravel()) & np.isfinite(s1.ravel())
        if kcs is not None:
            mask &= kcs.ravel() == 1

        coords = np.column_stack([xz.ravel()[mask], yz.ravel()[mask]])

        results: Dict[str, np.ndarray] = {"WATER_LEVEL": s1.ravel()[mask]}
        if water_depth is not None:
            results["WATER_DEPTH"] = water_depth.ravel()[mask]
        if u_east is not None and v_north is not None:
            ux = u_east.ravel()[mask]
            uy = v_north.ravel()[mask]
            results["U_x"] = ux
            results["U_y"] = uy
            results["U_MAG"] = np.sqrt(ux**2 + uy**2)

        logger.info(
            f"Extracted {coords.shape[0]} active zeta points with quantities: {sorted(results)}"
        )
        return coords, results


# =============================================================================
# Delft3DModel - BAL-compatible wrapper around Delft3DController
# =============================================================================


class Delft3DModel(HydroSimulations):
    """
    BAL-compatible Delft3D-FLOW model wrapper.

    Provides the interface expected by ``bal_delft3d.py`` (mirroring
    ``OpenFOAMModel``): for each experimental-design run the Delft3D-FLOW case
    template is copied, the calibration parameters are written into the
    ``<case>.mdf``, the solver is launched through ``run_dflow2d3d.sh`` and
    the calibration quantities are extracted at the measurement points from
    the NetCDF map output.

    Parameters
    ----------
    case_template_dir : str
        Delft3D-FLOW case template that is copied for each run. Must contain
        the ``<case>.mdf`` master definition file, ``config_d_hydro.xml``, and
        all attribute files (grid ``.grd``/``.enc``, bathymetry ``.dep``,
        boundary conditions ``.bnd``/``.bct`` ...).
    env_script : str
        ``env.sh`` of the Delft3D-FLOW installation prefix, default
        ``"~/opt/delft3d-flow/env.sh"`` (native build convention). Sourced in
        every solver shell.
    d_hydro_config : str
        Runtime configuration read by the launcher, default
        ``"config_d_hydro.xml"``; its ``<mdfFile>`` entry names the ``.mdf``.
    n_processors : int
        Number of MPI processes; ``1`` uses ``run_dflow2d3d.sh``, larger
        values use ``run_dflow2d3d_parallel.sh``.
    roughness_formulation : str
        Bed-roughness law written to ``Roumet`` when a ``"roughness"``
        calibration parameter is updated: ``"Chezy"``, ``"Manning"`` or
        ``"WhiteColebrook"``.
    n_avg_timesteps : int
        Number of final map time steps to average when extracting outputs.
    **kwargs
        Common :class:`~hydroBayesCal.hysim.HydroSimulations` parameters
        (``model_dir``, ``res_dir``, ``calibration_pts_file_path``,
        ``calibration_parameters``, ``param_values``,
        ``calibration_quantities``, ``init_runs``, ``max_runs`` ...).

    Notes
    -----
    Calibration parameter names are dispatched as follows:

    * ``"roughness"`` (or ``"Ccof"``) -- uniform bed roughness: sets
      ``Roumet`` (per ``roughness_formulation``) plus ``Ccofu`` and ``Ccofv``.
    * any other name -- treated as a literal MDF keyword and written with
      ``update_mdf_parameter`` (e.g. ``"Vicouv"``, ``"Dicouv"``).

    ``calibration_quantities`` / ``extraction_quantities`` use the standard
    HydroBayesCal names ``"WATER_LEVEL"``, ``"WATER_DEPTH"``, ``"U_x"``,
    ``"U_y"``, ``"U_MAG"``.
    """

    def __init__(
        self,
        case_template_dir,
        env_script=DEFAULT_ENV_SCRIPT,
        d_hydro_config="config_d_hydro.xml",
        n_processors=1,
        roughness_formulation="Manning",
        n_avg_timesteps=1,
        control_file="",
        model_dir="",
        res_dir="",
        calibration_pts_file_path="",
        n_cpus=1,
        init_runs=5,
        calibration_parameters=None,
        param_values=None,
        extraction_quantities=None,
        calibration_quantities=None,
        dict_output_name="extraction-data",
        user_param_values=False,
        max_runs=50,
        complete_bal_mode=True,
        only_bal_mode=False,
        delete_complex_outputs=False,
        validation=False,
        multitask_selection="variables",
        *args,
        **kwargs,
    ):
        # Delft3D-specific directory defaults, resolved before the base
        # constructor runs so the standard result layout is built on them.
        case_template_dir = os.path.abspath(case_template_dir)
        model_dir = os.path.abspath(model_dir) if model_dir else os.path.dirname(case_template_dir)
        res_dir = os.path.abspath(res_dir) if res_dir else os.path.join(model_dir, "results")

        # Fall back to a minimal single-parameter Manning-roughness setup so a
        # bare Delft3D-FLOW case is still usable without an explicit config.
        calibration_parameters = calibration_parameters or ["roughness"]
        param_values = param_values or [[0.02, 0.04]]
        extraction_quantities = extraction_quantities or [
            "WATER_LEVEL",
            "WATER_DEPTH",
            "U_x",
            "U_y",
            "U_MAG",
        ]
        calibration_quantities = calibration_quantities or ["WATER_DEPTH"]

        # Resolve the master definition file from the template so the base
        # class records the actual Delft3D control file (<case>.mdf).
        template_controller = Delft3DController(
            case_template_dir, env_script=env_script, d_hydro_config=d_hydro_config
        )
        if not control_file:
            control_file = os.path.basename(template_controller.mdf_file)

        # The base class owns the common state: parameters, observations and
        # variances (from the calibration CSV), and the standard result folder
        # layout (asr_dir, calibration-data/<quantities>, restart_data, plots,
        # surrogate-gpe). This keeps the Delft3D binding aligned with Telemac.
        os.makedirs(model_dir, exist_ok=True)
        super().__init__(
            control_file=control_file,
            model_dir=model_dir,
            res_dir=res_dir,
            calibration_pts_file_path=calibration_pts_file_path,
            n_cpus=n_cpus,
            init_runs=init_runs,
            calibration_parameters=calibration_parameters,
            param_values=param_values,
            calibration_quantities=calibration_quantities,
            extraction_quantities=extraction_quantities,
            dict_output_name=dict_output_name,
            user_param_values=user_param_values,
            max_runs=max_runs,
            complete_bal_mode=complete_bal_mode,
            only_bal_mode=only_bal_mode,
            delete_complex_outputs=delete_complex_outputs,
            validation=validation,
            multitask_selection=multitask_selection,
        )

        # Delft3D-specific attributes
        self.case_template_dir = case_template_dir
        self.env_script = env_script
        self.d_hydro_config = d_hydro_config
        self.n_processors = int(n_processors)
        self.roughness_formulation = roughness_formulation
        self.n_avg_timesteps = max(1, int(n_avg_timesteps))
        # Alias consumed by the GP layer (see bal_delft3d.py).
        self.parameter_ranges = self.param_values

        # The base class sets these only when a calibration file is present;
        # provide robust fallbacks so the BAL driver never sees None.
        if self.num_calibration_quantities is None:
            self.num_calibration_quantities = len(self.calibration_quantities)
        if self.nloc is None:
            self.nloc = 0

        # XY coordinates of the calibration points (Delft3D-FLOW map output is
        # extracted on the horizontal curvilinear grid).
        self._load_control_points()

        # Results storage
        self.model_evaluations = None

        logger.info(
            f"Delft3DModel initialized: control file '{self.control_file}', "
            f"{self.ndim} parameter(s), {self.nloc} locations"
        )

    def _load_control_points(self):
        """Read the XY coordinates of the calibration points.

        Observations, variances and ``nloc`` are set by the base class from the
        calibration CSV (``<quantity>_DATA`` / ``<quantity>_ERROR`` columns).
        The map-output extraction additionally needs the point coordinates,
        which are read here from the same dataframe the base class stored.
        """
        df = self.calibration_pts_df
        if df is None:
            self.control_points = np.array([])
            return

        x_col = next((c for c in df.columns if c.lower() == "x"), None)
        y_col = next((c for c in df.columns if c.lower() == "y"), None)

        if x_col and y_col:
            self.control_points = df[[x_col, y_col]].values
        else:
            self.control_points = np.array([])

    def update_model_controls(self, controller, params):
        """Write one set of calibration parameter values into the ``.mdf``.

        ``"roughness"``/``"Ccof"`` set the uniform bed roughness (``Roumet``
        plus ``Ccofu``/``Ccofv``); any other parameter name is written as a
        literal MDF keyword.
        """
        for pname, pval in zip(self.calibration_parameters, params):
            if pname.replace(" ", "").lower() in ("roughness", "ccof"):
                print(
                    f"Updating uniform bed roughness ({self.roughness_formulation}) "
                    f"Ccofu = Ccofv = {float(pval)} ..."
                )
                controller.set_roughness(float(pval), self.roughness_formulation)
            else:
                print(f"Updating MDF keyword '{pname}' = {float(pval)} ...")
                controller.update_mdf_parameter(pname, float(pval))

    def run_multiple_simulations(
        self,
        collocation_points,
        complete_bal_mode=True,
        validation=False,
        bal_iteration=None,
        bal_new_set_parameters=None,
        start_index=0,
    ):
        """Run multiple Delft3D-FLOW simulations - BAL interface.

        ``start_index`` resumes a staged initial design: the cumulative design is passed
        together with the number of rows already simulated, so growing the design runs
        only the new block while the run numbering and the accumulated outputs stay
        continuous. See :mod:`~hydroBayesCal.surrogate.initial_design`.
        """
        start_index = int(start_index or 0)
        design = None
        if bal_new_set_parameters is not None:
            params_to_run = np.atleast_2d(bal_new_set_parameters)
            start_idx = collocation_points.shape[0]
        else:
            design = np.atleast_2d(collocation_points)
            params_to_run = design[start_index:]
            start_idx = start_index
        # Outputs of the blocks already simulated, so a staged design accumulates
        # instead of replacing what the earlier blocks produced.
        previous_evaluations = self.model_evaluations if start_index else None

        all_results = []
        all_detailed_results = []  # for comprehensive CSV: one row per run x control point

        for i, params in enumerate(params_to_run):
            run_idx = start_idx + i
            case_name = f"run_{run_idx:04d}"
            case_dir = os.path.join(self.model_dir, case_name)

            logger.info(f"\n{'='*60}")
            logger.info(f"Simulation {run_idx + 1}: {case_name}")
            logger.info(f"Parameters: {dict(zip(self.calibration_parameters, params))}")

            # Copy template
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir)
            shutil.copytree(self.case_template_dir, case_dir)

            # Create controller for this run
            d3d = Delft3DController(
                case_dir,
                env_script=self.env_script,
                d_hydro_config=self.d_hydro_config,
            )

            try:
                # Guarantee readable (NetCDF) map output and update parameters
                d3d.ensure_netcdf_output()
                self.update_model_controls(d3d, params)

                # Run simulation (checks tri-diag.<case> for *** ERROR lines)
                d3d.run_simulation(nprocs=self.n_processors)

                # Extract fields from the map output (averaged over n_avg_timesteps)
                coords, fields = d3d.extract_map_results(n_avg_timesteps=self.n_avg_timesteps)

                if self.nloc == 0:
                    # No measurements file yet - save raw fields to disk so they can
                    # be re-extracted at control points once measurements.csv is ready.
                    raw_dir = os.path.join(self.restart_data_folder, "raw_fields")
                    os.makedirs(raw_dir, exist_ok=True)
                    np.save(os.path.join(raw_dir, f"coords_{run_idx:04d}.npy"), coords)
                    for qty, values in fields.items():
                        np.save(os.path.join(raw_dir, f"{qty}_{run_idx:04d}.npy"), values)
                    logger.info(f"No measurements file - raw fields saved to {raw_dir} for run {run_idx}.")

                    # Save collocation points so they can be reloaded later
                    current_cp = (design[: start_idx + i + 1] if design is not None
                                  else params_to_run[: i + 1])
                    np.save(os.path.join(self.restart_data_folder, "collocation_points.npy"), current_cp)

                else:
                    # Measurements file exists - interpolate to control points as normal
                    results = self._extract_at_control_points(coords, fields)

                    run_results = []
                    for loc_idx in range(self.nloc):
                        for qty in self.calibration_quantities:
                            if qty in results:
                                run_results.append(float(results[qty][loc_idx]))
                            else:
                                run_results.append(np.nan)

                    all_results.append(run_results)

                    # Store detailed results: one row per control point
                    for pt_idx, cp in enumerate(self.control_points):
                        row = {
                            "run_idx": run_idx,
                            **{p: float(v) for p, v in zip(self.calibration_parameters, params)},
                            "x": float(cp[0]),
                            "y": float(cp[1]),
                        }
                        for qty in self.extraction_quantities:
                            row[qty] = (
                                float(results[qty][pt_idx]) if qty in results else np.nan
                            )
                        all_detailed_results.append(row)

                    # Save per-run JSON inside the run folder
                    self._save_run_results(case_dir, run_idx, params, results)

                    # Update model_evaluations
                    current_results = np.array(all_results)
                    if bal_new_set_parameters is None:
                        self.model_evaluations = (
                            np.vstack([previous_evaluations, current_results])
                            if previous_evaluations is not None else current_results)
                        current_cp = design[: start_idx + i + 1]
                    else:
                        self.model_evaluations = (
                            np.vstack([self.model_evaluations, current_results])
                            if self.model_evaluations is not None
                            else current_results
                        )
                        current_cp = np.vstack([collocation_points, params_to_run[: i + 1]])

                    # Save cumulative results after every run
                    self._save_all_results(current_cp, all_detailed_results)

                # Delete the run folder immediately to free disk space
                if self.delete_complex_outputs:
                    self._cleanup_run(case_dir)

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                logger.error(f"Simulation {case_name} failed: {e}")
                logger.error(tb)
                print(f"\n{'='*60}", flush=True)
                print(f"EXCEPTION IN RUN {case_name}: {e}", flush=True)
                print(tb, flush=True)
                print(f"{'='*60}\n", flush=True)
                if self.nloc > 0:
                    all_results.append([np.nan] * (self.nloc * len(self.calibration_quantities)))
                    for pt_idx, cp in enumerate(self.control_points):
                        row = {
                            "run_idx": run_idx,
                            **{p: float(v) for p, v in zip(self.calibration_parameters, params)},
                            "x": float(cp[0]),
                            "y": float(cp[1]),
                        }
                        for qty in self.extraction_quantities:
                            row[qty] = np.nan
                        all_detailed_results.append(row)

        # model_evaluations is already up to date from the per-run updates above.
        # Final save to ensure all results are on disk after the full batch.
        # Skip if nloc=0 (no measurements file yet) - raw fields were saved per-run instead.
        if self.nloc > 0:
            if bal_new_set_parameters is not None:
                all_cp = np.vstack([collocation_points, params_to_run])
            else:
                all_cp = design
            self._save_all_results(all_cp, all_detailed_results)

    def _extract_at_control_points(self, coords, fields):
        """Extract the map-output quantities at the calibration points.

        Nearest-neighbour lookup on the active zeta points of the curvilinear
        grid (2D horizontal query via a cKDTree).
        """
        if len(self.control_points) == 0:
            return {qty: np.array([]) for qty in self.extraction_quantities}

        tree = spatial.cKDTree(coords)
        _, indices = tree.query(self.control_points)

        results = {}
        for qty in self.extraction_quantities:
            if qty in fields:
                results[qty] = fields[qty][indices]
            else:
                logger.warning(
                    f"Extraction quantity '{qty}' not available in the Delft3D map output "
                    f"(available: {sorted(fields)})."
                )
                results[qty] = np.full(len(self.control_points), np.nan)
        return results

    def _save_run_results(self, case_dir, run_idx, params, results):
        """Save individual run results."""
        output = {
            "run_index": int(run_idx),
            "parameters": {k: float(v) for k, v in zip(self.calibration_parameters, params)},
            "results": {k: np.asarray(v).tolist() for k, v in results.items()},
        }
        with open(os.path.join(case_dir, f"{self.dict_output_name}.json"), "w") as f:
            json.dump(output, f, indent=2)

    def _cleanup_run(self, case_dir):
        """Delete the entire run folder to free disk space.

        All results have already been saved to:
          - model_evaluations.npy  (GP training data)
          - collocation_points.npy
          - results-detailed-*.csv / .npy
          - initial-model-outputs.json
        so the raw Delft3D-FLOW output (NEFIS/NetCDF trim and trih files) is no
        longer needed.
        """
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
            logger.info(f"Deleted run folder to free disk space: {case_dir}")

    def output_processing(self, output_data_path=None, **kwargs):
        """Load existing results for BAL restart."""
        if output_data_path and os.path.isfile(output_data_path):
            with open(output_data_path, "r") as f:
                data = json.load(f)
            self.restart_collocation_points = np.array(data["collocation_points"])
            self.model_evaluations = np.array(data["model_evaluations"])
            return self.model_evaluations
        raise FileNotFoundError(f"Output data not found: {output_data_path}")
