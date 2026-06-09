import os
import re
import glob
import json
import sys
import shutil
import subprocess
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import spatial


class OpenFOAMController:
    def __init__(self, case_dir: str):
        self.case_dir = os.path.normpath(case_dir)

    # Turns a relative path into an absolute one inside the case directory.
    # Strips any leading case or template folder name to avoid double-nesting.
    def _case_path(self, relpath: str) -> str:
        if os.path.isabs(relpath):
            return relpath

        rp = relpath.replace("\\", "/").lstrip("/")
        if rp.startswith("./"):
            rp = rp[2:]

        template_folder = os.path.basename(os.path.dirname(self.case_dir))
        case_folder = os.path.basename(self.case_dir)

        for prefix in (template_folder, case_folder):
            if rp == prefix:
                rp = ""
            elif rp.startswith(prefix + "/"):
                rp = rp[len(prefix) + 1 :]

        return os.path.join(self.case_dir, rp)

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
                raise RuntimeError(f"Command failed with return code {process.returncode}: {' '.join(cmd)}")
        finally:
            if log_fh:
                log_fh.close()

    # Runs command silently, raises CalledProcessError on failure.
    def _run_checked(self, cmd: Iterable[str]) -> None:
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        subprocess.run(list(cmd), check=True, env=env, cwd=self.case_dir)

    # Removes all processorN directories from the case folder before decomposePar.
    def _clean_processor_dirs(self) -> None:
        for p in glob.glob(os.path.join(self.case_dir, "processor[0-9]*")):
            if os.path.isdir(p):
                shutil.rmtree(p)

    # Checks that system/decomposeParDict exists and returns its path.
    def _ensure_decompose_dict(self) -> str:
        path = self._case_path("system/decomposeParDict")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing system/decomposeParDict in case: {self.case_dir}\n"
                "Parallel runs need this file so decomposePar can create processor directories."
            )
        return path

    # Updates numberOfSubdomains in decomposeParDict to match the requested processor count.
    def _set_number_of_subdomains(self, nprocs: int) -> None:
        path = self._ensure_decompose_dict()

        with open(path, "r") as f:
            lines = f.readlines()

        pat = re.compile(r"^(\s*numberOfSubdomains\s+)\d+(\s*;\s*)$")
        replaced = False
        new_lines = []

        for line in lines:
            m = pat.match(line)
            if m:
                new_lines.append(f"{m.group(1)}{int(nprocs)}{m.group(2)}\n")
                replaced = True
            else:
                new_lines.append(line)

        if not replaced:
            insert_at = len(new_lines)
            for i, line in enumerate(new_lines):
                if line.strip() == "}":
                    insert_at = i
                    break
            new_lines.insert(insert_at, f"\nnumberOfSubdomains {int(nprocs)};\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

    # Cleans processor directories, sets the subdomain count, and runs decomposePar.
    def decompose_parallel_case(self, nprocs: int) -> None:
        self._clean_processor_dirs()
        self._set_number_of_subdomains(nprocs)
        self._run_checked(["decomposePar", "-case", self.case_dir, "-force"])

    # Reassembles the decomposed case into a single directory via reconstructPar.
    def reconstruct_parallel_case(self) -> None:
        self._run_checked(["reconstructPar", "-case", self.case_dir])

    # Modifies type, Ks, or value for a given patch in an OpenFOAM field file.
    def update_boundary_condition(self, file: str, patch: str, field_type: str, bc_type: str, value: Any) -> None:
        path = self._case_path(file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        inside_boundary_field = False
        inside_patch = False
        patch_found = False
        brace_level = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not inside_boundary_field and stripped.startswith("boundaryField"):
                inside_boundary_field = True
                new_lines.append(line)
                continue

            if inside_boundary_field:
                brace_level += line.count("{") - line.count("}")

                if not inside_patch:
                    if re.match(rf"^\s*{re.escape(patch)}\s*\{{\s*$", line):
                        inside_patch = True
                        patch_found = True
                        new_lines.append(line)
                        continue
                    if re.match(rf"^\s*{re.escape(patch)}\s*$", line):
                        if i + 1 < len(lines) and lines[i + 1].strip() == "{":
                            inside_patch = True
                            patch_found = True
                            new_lines.append(line)
                            continue

                if inside_patch:
                    if re.match(r"\s*type\s+", line):
                        new_lines.append(f"        type {bc_type};\n")
                        continue

                    if re.match(r"\s*Ks\s+", line):
                        if value is None:
                            raise ValueError(f"Missing 'value' for patch '{patch}' in {file}")
                        new_lines.append(f"        Ks uniform {float(value):.5f};\n")
                        continue

                    if re.match(r"\s*value\s+", line):
                        new_lines.append("        value uniform 0;\n")
                        continue

                    if "}" in line:
                        inside_patch = False
                        new_lines.append(line)
                        continue

            new_lines.append(line)

        if not patch_found:
            raise ValueError(f"Patch '{patch}' not found in {file}")

        with open(path, "w") as f:
            f.writelines(new_lines)

    # Updates a scalar key inside a named subdictionary in an OpenFOAM dict file.
    def update_dictionary_entry(self, file: str, subdict: str, key: str, value: float) -> None:
        path = self._case_path(file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        inside_subdict = False
        pending_subdict = False
        brace_level = 0
        key_updated = False

        subdict_inline = re.compile(rf"^\s*{re.escape(subdict)}\s*\{{\s*$")
        subdict_nameonly = re.compile(rf"^\s*{re.escape(subdict)}\s*$")
        key_re = re.compile(rf"^\s*{re.escape(key)}\s+")

        for line in lines:
            stripped = line.strip()

            if not inside_subdict and not pending_subdict:
                if subdict_inline.match(line):
                    inside_subdict = True
                    brace_level = 1
                    new_lines.append(line)
                    continue
                if subdict_nameonly.match(line):
                    pending_subdict = True
                    new_lines.append(line)
                    continue

            if pending_subdict:
                if stripped == "{":
                    inside_subdict = True
                    pending_subdict = False
                    brace_level = 1
                    new_lines.append(line)
                    continue
                pending_subdict = False

            if inside_subdict:
                brace_level += line.count("{") - line.count("}")

                if key_re.match(line):
                    new_lines.append(f"        {key} {float(value):.5f};\n")
                    key_updated = True
                    continue

                new_lines.append(line)

                if brace_level <= 0:
                    inside_subdict = False
                continue

            new_lines.append(line)

        if not key_updated:
            raise ValueError(f"Key '{key}' not found in {file}::{subdict}")

        with open(path, "w") as f:
            f.writelines(new_lines)

    # Dispatches all calibration parameter updates: Cmu goes to turbulenceProperties,
    # everything else is written as a boundary condition; alpha.water fields are skipped.
    def update_model_controls(self, params: Dict[str, Dict[str, Any]]) -> None:
        skip_fields = {"alpha.water", "alpha.water.orig"}

        for patch, param in params.items():
            file = param["file"]

            if patch == "Cmu":
                value = float(param["value"])
                print("Updating model coefficient 'Cmu' in constant/turbulenceProperties...")
                self.update_dictionary_entry(
                    file="constant/turbulenceProperties",
                    subdict="kEpsilonCoeffs",
                    key="Cmu",
                    value=value,
                )
                continue

            if patch == "ks":
                value = float(param["value"])
                print(f"Updating Ks={value} on patch 'bottom' in 0/nut...")
                self.update_boundary_condition(
                    file=param["file"],
                    patch=param["patch"],
                    field_type=param["field_type"],
                    bc_type=param["bc_type"],
                    value=value,
                )
                continue

            field_type = param.get("field_type", "")
            bc_type = param["bc_type"]
            value = param.get("value", None)

            field_name = os.path.basename(file)
            if field_name in skip_fields:
                print(f"Skipping field '{field_name}' (managed by setFields).")
                continue

            print(f"Updating boundary condition for patch '{patch}' in {file}...")
            self.update_boundary_condition(file, patch, field_type, bc_type, value)

    # Runs the solver; if nprocs > 1 decomposes first and reconstructs after.
    def run_simulation(self, nprocs: int = 8, solver: str = "interFoam") -> None:
        print("Running OpenFOAM simulation...")
        run_log = os.path.join(self.case_dir, "log.run")

        if int(nprocs) > 1:
            self.decompose_parallel_case(int(nprocs))
            mpi_cmd = (
                f"source $HOME/OpenFOAM/OpenFOAM-v2412/etc/bashrc && "
                f"mpirun -np {int(nprocs)} {solver} -parallel"
            )
            self._run_streamed(
                ["bash", "-c", mpi_cmd],
                log_path=run_log,
            )
            self.reconstruct_parallel_case()
        else:
            self._run_streamed([solver, "-case", self.case_dir], log_path=run_log)

        print(f"Simulation finished. Log saved to {run_log}")

    # Converts only the latest time directory to VTK format using foamToVTK.
    def convert_to_vtk(self) -> None:
        print("Converting to VTK...")
        cmd = (
            f"source $HOME/OpenFOAM/OpenFOAM-v2412/etc/bashrc && "
            f"foamToVTK -case {self.case_dir}"
        )
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        result = subprocess.run(["bash", "-c", cmd], env=env, cwd=self.case_dir,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"foamToVTK stderr: {result.stderr}", flush=True)
            print(f"foamToVTK stdout: {result.stdout}", flush=True)
            raise RuntimeError(f"foamToVTK failed with exit code {result.returncode}")
        print("VTK conversion completed.")

    def extract_fields_from_vtk(self, alpha_threshold: float = 0.5, n_avg_timesteps: int = 1) -> Tuple[Any, Any, Any]:
        """
        Extract velocity (U) and turbulent kinetic energy (k) fields from VTK output,
        averaged over the last n_avg_timesteps timesteps, filtered to water phase only.

        k is read directly from the OpenFOAM k field (k-epsilon RANS turbulent kinetic energy).

        If n_avg_timesteps=1 (default), only the last timestep is used (original behaviour).
        If n_avg_timesteps=N, the last N VTK files are averaged, giving a time-averaged result
        over N * writeInterval seconds.

        Args:
            alpha_threshold: Only include points where alpha.water > threshold (default 0.5)

        Returns:
            Tuple of (coordinates, U_mean, k_mean) for water-phase points only.
            k is None if not found in the VTK files.
        """
        vtk_dir = os.path.join(self.case_dir, "VTK")
        if not os.path.isdir(vtk_dir):
            raise FileNotFoundError(f"VTK directory does not exist: {vtk_dir}")

        # Collect and sort all available VTK files by timestep index.
        # OpenFOAM v2412 foamToVTK creates: VTK/<casename>_<stepIndex>/internal.vtu
        # Sort by the numeric suffix of the parent folder (step index, not simulation time).
        vtk_files_sorted = sorted(
            glob.glob(os.path.join(vtk_dir, "**", "internal.vtu"), recursive=True),
            key=lambda p: int(os.path.basename(os.path.dirname(p)).rsplit("_", 1)[-1])
            if os.path.basename(os.path.dirname(p)).rsplit("_", 1)[-1].isdigit() else 0
        )
        if not vtk_files_sorted:
            raise FileNotFoundError(f"No internal.vtu files found in: {vtk_dir}")
        print(f"Found {len(vtk_files_sorted)} VTK timesteps, averaging last {n_avg_timesteps}", flush=True)

        # Select last n_avg_timesteps files
        n = min(n_avg_timesteps, len(vtk_files_sorted))
        if n < n_avg_timesteps:
            logger.warning(
                f"Requested n_avg_timesteps={n_avg_timesteps} but only {n} VTK files available. "
                f"Averaging over {n} timesteps."
            )
        selected_files = vtk_files_sorted[-n:]
        logger.info(f"Averaging over {n} VTK timesteps: "
                    f"{[os.path.basename(os.path.dirname(f)) for f in selected_files]}")

        # Read first file to get coords and water mask (assumed fixed across timesteps)
        mesh0 = pv.read(selected_files[0])
        if "U" not in mesh0.point_data:
            raise KeyError("Velocity field 'U' not found in VTK point_data.")

        coords = mesh0.points
        has_k = "k" in mesh0.point_data

        if "alpha.water" in mesh0.point_data:
            alpha = mesh0.point_data["alpha.water"]
            water_mask = alpha > alpha_threshold
            coords = coords[water_mask]
            logger.info(f"Water phase: {water_mask.sum()}/{len(alpha)} points (alpha > {alpha_threshold})")
        else:
            logger.warning("alpha.water not found in VTK - using all points")
            water_mask = None

        # Accumulate U and k across selected timesteps
        U_sum = np.zeros_like(mesh0.point_data["U"][water_mask] if water_mask is not None else mesh0.point_data["U"])
        k_sum = np.zeros(U_sum.shape[0]) if has_k else None

        for vtk_file in selected_files:
            mesh = pv.read(vtk_file)
            U_t = mesh.point_data["U"]
            if water_mask is not None:
                U_t = U_t[water_mask]
            U_sum += U_t

            if has_k and "k" in mesh.point_data:
                k_t = mesh.point_data["k"]
                if water_mask is not None:
                    k_t = k_t[water_mask]
                k_sum += k_t

        U_mean = U_sum / n
        k_mean = k_sum / n if has_k else None

        if k_mean is None:
            logger.warning("k field not found in VTK - TKE will be NaN")
        else:
            logger.info(f"k (averaged over {n} steps): Mean={k_mean.mean():.4e}, Max={k_mean.max():.4e}")

        return coords, U_mean, k_mean
# =============================================================================
# OpenFOAMModel - BAL-compatible wrapper around OpenFOAMController
# =============================================================================

from hydroBayesCal.hysim import HydroSimulations
from hydroBayesCal.function_pool import logger


class OpenFOAMModel(HydroSimulations):
    """
    BAL-compatible wrapper around OpenFOAMController.

    Provides the interface expected by bal_openfoam.py while using
    your existing OpenFOAMController for the actual OpenFOAM operations.
    """

    def __init__(
        self,
        case_template_dir,
        solver_name="interFoam",
        n_processors=8,
        results_filename_base="results_interfoam",
        alpha_water_name="alpha.water",
        water_surface_alpha=0.5,
        reference_z=0.0,
        control_file="system/controlDict",
        model_dir="",
        res_dir="",
        calibration_pts_file_path="",
        n_cpus=8,
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
        n_avg_timesteps=1,
        *args,
        **kwargs
    ):
        # OpenFOAM-specific directory defaults, resolved before the base
        # constructor runs so the standard result layout is built on them.
        case_template_dir = os.path.abspath(case_template_dir)
        model_dir = os.path.abspath(model_dir) if model_dir else os.path.dirname(case_template_dir)
        res_dir = os.path.abspath(res_dir) if res_dir else os.path.join(model_dir, "results")

        # Fall back to a minimal single-parameter k-epsilon Cmu setup so a bare
        # OpenFOAM case is still usable without an explicit calibration config.
        calibration_parameters = calibration_parameters or ["Cmu"]
        param_values = param_values or [[0.06, 0.12]]
        extraction_quantities = extraction_quantities or ["U_x", "U_y", "U_z"]
        calibration_quantities = calibration_quantities or ["U_x", "U_y", "U_z"]

        # The base class owns the common state: parameters, observations and
        # variances (from the calibration CSV), and the standard result folder
        # layout (asr_dir, calibration-data/<quantities>, restart_data, plots,
        # surrogate-gpe). This keeps the OpenFOAM binding aligned with Telemac.
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

        # OpenFOAM-specific attributes
        self.case_template_dir = case_template_dir
        self.solver_name = solver_name
        self.n_processors = n_processors
        self.results_filename_base = results_filename_base
        self.alpha_water_name = alpha_water_name
        self.water_surface_alpha = water_surface_alpha
        self.reference_z = reference_z
        self.n_avg_timesteps = max(1, int(n_avg_timesteps))  # minimum 1
        # Alias consumed by the GP layer (see bal_openfoam.py).
        self.parameter_ranges = self.param_values

        # The base class sets these only when a calibration file is present;
        # provide robust fallbacks so the BAL driver never sees None.
        if self.num_calibration_quantities is None:
            self.num_calibration_quantities = len(self.calibration_quantities)
        if self.nloc is None:
            self.nloc = 0

        # XYZ coordinates of the calibration points. The base class stores the
        # dataframe and the observations/variances; the OpenFOAM field
        # extraction additionally needs the point coordinates.
        self._load_control_points()

        # Check that k is written to VTK output
        self._check_k_in_controldict()

        # Results storage
        self.model_evaluations = None

        logger.info(f"OpenFOAMModel initialized: {self.ndim} parameter(s), {self.nloc} locations")

    def _check_k_in_controldict(self):
        """Check that the k field is written in the case template controlDict.

        OpenFOAM does not always write k to VTK by default. If k is missing,
        TKE and velocity fluctuations will be NaN in all outputs.
        """
        controldict_path = os.path.join(self.case_template_dir, "system", "controlDict")
        if not os.path.isfile(controldict_path):
            logger.warning(
                f"controlDict not found at {controldict_path} -- "
                "cannot verify that 'k' is written to VTK. "
                "TKE and fluctuations will be NaN if 'k' is absent from VTK output."
            )
            return

        with open(controldict_path, "r") as f:
            content = f.read()

        # k is written if it appears in writeFields or if writeFormat includes it,
        # or if there is no writeFields restriction (i.e. all fields are written)
        has_write_fields = "writeFields" in content
        k_explicitly_listed = bool(re.search(r'\bk\b', content))

        if has_write_fields and not k_explicitly_listed:
            logger.warning(
                "controlDict contains 'writeFields' but 'k' does not appear to be listed. "
                "TKE and velocity fluctuations will be NaN. "
                "Add 'k' to writeFields in system/controlDict to fix this."
            )
        else:
            logger.info("controlDict check passed: 'k' field appears to be available for VTK output.")

    def _load_control_points(self):
        """Read the XYZ coordinates of the calibration points.

        Observations, variances and ``nloc`` are set by the base class from the
        calibration CSV (``<quantity>_DATA`` / ``<quantity>_ERROR`` columns).
        The OpenFOAM field extraction additionally needs the point coordinates,
        which are read here from the same dataframe the base class stored.
        """
        df = self.calibration_pts_df
        if df is None:
            self.control_points = np.array([])
            return

        x_col = next((c for c in df.columns if c.lower() == 'x'), None)
        y_col = next((c for c in df.columns if c.lower() == 'y'), None)
        z_col = next((c for c in df.columns if c.lower() == 'z'), None)

        if x_col and y_col and z_col:
            self.control_points = df[[x_col, y_col, z_col]].values
        else:
            self.control_points = np.array([])

    def run_multiple_simulations(
        self,
        collocation_points,
        complete_bal_mode=True,
        validation=False,
        bal_iteration=None,
        bal_new_set_parameters=None
    ):
        """Run multiple simulations - BAL interface."""

        if bal_new_set_parameters is not None:
            params_to_run = np.atleast_2d(bal_new_set_parameters)
            start_idx = collocation_points.shape[0]
        else:
            params_to_run = np.atleast_2d(collocation_points)
            start_idx = 0

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

            # Note: nut internalField is already uniform 0 in the case template.
            # No reset needed.

            # Create controller for this run
            foam = OpenFOAMController(case_dir)

            # Build params dict for update_model_controls
            update_params = {}
            for pname, pval in zip(self.calibration_parameters, params):
                if pname.lower() == "cmu":
                    update_params["Cmu"] = {
                        "file": "constant/turbulenceProperties",
                        "value": float(pval),
                    }
                elif pname.lower() == "ks":
                    update_params["ks"] = {
                        "file": "0/nut",
                        "patch": "bottom",
                        "field_type": "scalar",
                        "bc_type": "nutkRoughWallFunction",
                        "value": float(pval),
                    }
                else:
                    raise NotImplementedError(f"No dispatch defined for calibration parameter '{pname}'. "
                                              f"Add a branch in run_multiple_simulations.")

            try:
                # Update parameters
                foam.update_model_controls(update_params)

                # Run simulation
                foam.run_simulation(nprocs=self.n_processors, solver=self.solver_name)

                # Convert to VTK
                foam.convert_to_vtk()

                # Extract fields from VTK (averaged over n_avg_timesteps)
                coords, U, k = foam.extract_fields_from_vtk(n_avg_timesteps=self.n_avg_timesteps)

                # Sanity check: verify fields differ from previous run.
                # Identical fields across runs indicate a VTK extraction bug (e.g. missing -latestTime).
                if run_idx > start_idx:
                    prev_U_path = os.path.join(self.restart_data_folder, "raw_fields", f"U_{run_idx - 1:04d}.npy")
                    if os.path.isfile(prev_U_path):
                        prev_U = np.load(prev_U_path)
                        if prev_U.shape == U.shape and np.allclose(prev_U, U, rtol=1e-6):
                            raise RuntimeError(
                                f"SANITY CHECK FAILED: Run {run_idx} produced identical U fields to run {run_idx - 1}. "
                                f"This almost certainly means foamToVTK is reading the wrong time directory. "
                                f"Check that -latestTime is being applied and that the simulation actually ran."
                            )
                        else:
                            logger.info(f"Sanity check passed: run {run_idx} fields differ from run {run_idx - 1}.")

                if self.nloc == 0:
                    # No measurements file yet  save raw fields to disk so they can
                    # be re-extracted at control points once measurements.csv is ready.
                    # This allows the initial runs to complete and free disk space
                    # without needing measurement coordinates.
                    raw_dir = os.path.join(self.restart_data_folder, "raw_fields")
                    os.makedirs(raw_dir, exist_ok=True)
                    np.save(os.path.join(raw_dir, f"coords_{run_idx:04d}.npy"), coords)
                    np.save(os.path.join(raw_dir, f"U_{run_idx:04d}.npy"), U)
                    if k is not None:
                        np.save(os.path.join(raw_dir, f"k_{run_idx:04d}.npy"), k)
                    logger.info(f"No measurements file  raw fields saved to {raw_dir} for run {run_idx}.")

                    # Save collocation points so they can be reloaded later
                    current_cp = params_to_run[:i + 1]
                    np.save(os.path.join(self.restart_data_folder, "collocation_points.npy"), current_cp)

                else:
                    # Measurements file exists  interpolate to control points as normal
                    results = self._extract_at_control_points(coords, U, k)

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
                            "x": float(cp[0]), "y": float(cp[1]), "z": float(cp[2]),
                            "U_x": float(results["U_x"][pt_idx]),
                            "U_y": float(results["U_y"][pt_idx]),
                            "U_z": float(results["U_z"][pt_idx]),
                            "U_magnitude": float(results["U_magnitude"][pt_idx]),
                            "u_fluct": float(results["u_fluct"][pt_idx]),
                            "v_fluct": float(results["v_fluct"][pt_idx]),
                            "w_fluct": float(results["w_fluct"][pt_idx]),
                            "TKE": float(results["TKE"][pt_idx]),
                        }
                        all_detailed_results.append(row)

                    # Save per-run JSON inside the run folder
                    self._save_run_results(case_dir, run_idx, params, results)

                    # Update model_evaluations
                    current_results = np.array(all_results)
                    if bal_new_set_parameters is None:
                        self.model_evaluations = current_results
                        current_cp = params_to_run[:i + 1]
                    else:
                        self.model_evaluations = np.vstack([
                            self.model_evaluations, current_results
                        ]) if self.model_evaluations is not None else current_results
                        current_cp = np.vstack([collocation_points, params_to_run[:i + 1]])

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
                            "x": float(cp[0]), "y": float(cp[1]), "z": float(cp[2]),
                            "U_x": np.nan, "U_y": np.nan, "U_z": np.nan,
                            "U_magnitude": np.nan,
                            "u_fluct": np.nan, "v_fluct": np.nan, "w_fluct": np.nan,
                            "TKE": np.nan,
                        }
                        all_detailed_results.append(row)

        # model_evaluations is already up to date from the per-run updates above.
        # Final save to ensure all results are on disk after the full batch.
        # Skip if nloc=0 (no measurements file yet)  raw fields were saved per-run instead.
        if self.nloc > 0:
            if bal_new_set_parameters is not None:
                all_cp = np.vstack([collocation_points, params_to_run])
            else:
                all_cp = params_to_run
            self._save_all_results(all_cp, all_detailed_results)

    def _extract_at_control_points(self, coords, U, k=None):
        """Extract velocity and TKE at control points.

        TKE is read directly from the OpenFOAM k field (RANS turbulent kinetic energy).

        Args:
            coords: Array of VTK point coordinates (water phase only)
            U: Velocity array at VTK points, shape (n_points, 3)
            k: RANS turbulent kinetic energy array from OpenFOAM k field, or None

        Returns:
            dict with U_x, U_y, U_z, U_magnitude, u_fluct, v_fluct, w_fluct, TKE at control points
        """
        if len(self.control_points) == 0:
            return {
                "U_x": np.array([]), "U_y": np.array([]), "U_z": np.array([]),
                "U_magnitude": np.array([]),
                "u_fluct": np.array([]), "v_fluct": np.array([]), "w_fluct": np.array([]),
                "TKE": np.array([]),
            }

        tree = spatial.cKDTree(coords)
        _, indices = tree.query(self.control_points)

        U_cp = U[indices]

        # TKE read directly from OpenFOAM k field
        # Isotropic fluctuation components estimated as sqrt(2k/3) for reporting only
        if k is not None:
            k_cp = k[indices]
            fluct = np.sqrt(np.maximum(2.0 / 3.0 * k_cp, 0.0))
        else:
            k_cp = np.full(len(self.control_points), np.nan)
            fluct = np.full(len(self.control_points), np.nan)

        return {
            "U_x": U_cp[:, 0],
            "U_y": U_cp[:, 1],
            "U_z": U_cp[:, 2],
            "U_magnitude": np.linalg.norm(U_cp, axis=1),
            "u_fluct": fluct,
            "v_fluct": fluct,
            "w_fluct": fluct,
            "TKE": k_cp,        # k from OpenFOAM k-epsilon field
        }

    def _save_run_results(self, case_dir, run_idx, params, results):
        """Save individual run results."""
        output = {
            "run_index": int(run_idx),
            "parameters": {k: float(v) for k, v in zip(self.calibration_parameters, params)},
            "results": {k: v.tolist() for k, v in results.items()}
        }
        with open(os.path.join(case_dir, f"{self.dict_output_name}.json"), 'w') as f:
            json.dump(output, f, indent=2)

    def _save_all_results(self, collocation_points, detailed_results=None):
        """Save all results to calibration folder.

        Saves:
        - collocation_points.npy: calibration parameter values tested, shape (n_runs, n_params)
        - model_evaluations.npy: flat model outputs, shape (n_runs, nloc * n_quantities)
        - initial-model-outputs.json: same data as JSON
        - collocation-points-{quantities}.csv: calibration parameter values as CSV
        - results-detailed-{quantities}.csv: comprehensive table with one row per
          run x control point, including coordinates, U, fluctuations and TKE
        - results-detailed-{quantities}.npy: same data as structured numpy array
        """
        np.save(os.path.join(self.calibration_folder, "collocation_points.npy"), collocation_points)
        np.save(os.path.join(self.calibration_folder, "model_evaluations.npy"), self.model_evaluations)

        output = {
            "collocation_points": collocation_points.tolist(),
            "model_evaluations": self.model_evaluations.tolist(),
            "calibration_parameters": self.calibration_parameters,
            "calibration_quantities": self.calibration_quantities,
            "n_runs": int(collocation_points.shape[0])
        }
        with open(os.path.join(self.calibration_folder, "initial-model-outputs.json"), 'w') as f:
            json.dump(output, f, indent=2)

        # Also save to restart_data_folder so only_bal_mode can find it
        with open(os.path.join(self.restart_data_folder, "initial-model-outputs.json"), 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved initial-model-outputs.json to restart_data folder for BAL restart.")

        # Collocation points CSV (calibration parameter values)
        quantities_str = '_'.join(self.calibration_quantities)
        csv_path = os.path.join(self.calibration_folder, f"collocation-points-{quantities_str}.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.calibration_parameters)
            writer.writerows(collocation_points.tolist())
        logger.info(f"Saved collocation points CSV to {csv_path}")

        # Comprehensive results CSV and npy (run x control point rows)
        if detailed_results:
            detailed_csv_path = os.path.join(
                self.calibration_folder, f"results-detailed-{quantities_str}.csv"
            )
            fieldnames = list(detailed_results[0].keys())
            # Append if file exists, write fresh if not
            file_exists = os.path.isfile(detailed_csv_path)
            with open(detailed_csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(detailed_results)
            logger.info(f"Saved detailed results CSV to {detailed_csv_path}")

            # Also save as npy structured array
            dtype = [(k, 'f8') for k in fieldnames]
            arr = np.array([tuple(r[k] for k in fieldnames) for r in detailed_results], dtype=dtype)
            npy_path = os.path.join(
                self.calibration_folder, f"results-detailed-{quantities_str}.npy"
            )
            np.save(npy_path, arr)
            logger.info(f"Saved detailed results npy to {npy_path}")

    def save_calibration_data(self, it, collocation_points, bayesian_dict):
        """Write per-iteration CSV files to ``calibration-data/<quantities>/``.

        Called once per BAL iteration from ``bal_openfoam.py`` after
        ``estimate_bme()``. Produces three files per iteration::

            collocation_points_N{n_tp}.csv   parameter values tested so far
            model_results_N{n_tp}.csv        simulation outputs (model_evaluations)
            bayesian_scores.csv              BME, RE, IE, ELPD for all iterations

        ``bayesian_scores.csv`` is appended on each call (one row per iteration).
        The posterior is saved as a separate ``.npy`` file because it is a
        variable-length array (rejection sampling keeps only accepted samples).
        """
        n_tp = int(collocation_points.shape[0])
        folder = self.calibration_folder

        # 1. Collocation points CSV
        cp_path = os.path.join(folder, f"collocation_points_N{n_tp:03d}.csv")
        cp_df = pd.DataFrame(collocation_points, columns=self.calibration_parameters)
        cp_df.index.name = "run_idx"
        cp_df.to_csv(cp_path)

        # 2. Model evaluations CSV
        quantities_str = '_'.join(self.calibration_quantities)
        col_names = [
            f"{qty}_z{i}"
            for i in range(self.nloc)
            for qty in self.calibration_quantities
        ]
        if self.model_evaluations is not None:
            me_path = os.path.join(folder, f"model_results_N{n_tp:03d}.csv")
            me_df = pd.DataFrame(self.model_evaluations, columns=col_names)
            me_df.index.name = "run_idx"
            me_df.to_csv(me_path)

        # 3. Posterior npy (variable size - one file per iteration)
        posterior = bayesian_dict['posterior'][it]
        if posterior is not None and len(posterior) > 0:
            post_path = os.path.join(folder, f"posterior_N{n_tp:03d}.npy")
            np.save(post_path, posterior)

        # 4. Bayesian scores CSV (one growing file, appended each call)
        scores_path = os.path.join(folder, "bayesian_scores.csv")
        scores_row = {
            "iteration": it,
            "N_tp": n_tp,
            "BME":  bayesian_dict['BME'][it],
            "RE":   bayesian_dict['RE'][it],
            "IE":   bayesian_dict['IE'][it],
            "ELPD": bayesian_dict['ELPD'][it],
            "post_size": int(bayesian_dict['post_size'][it]),
        }
        scores_df = pd.DataFrame([scores_row])
        write_header = not os.path.isfile(scores_path)
        scores_df.to_csv(scores_path, mode='a', header=write_header, index=False)

        logger.info(
            f"Saved calibration-data for iteration {it} "
            f"(N_tp={n_tp}, BME={bayesian_dict['BME'][it]:.4e}, "
            f"RE={bayesian_dict['RE'][it]:.4f})"
        )

    def _cleanup_run(self, case_dir):
        """Delete the entire run folder to free disk space.

        All results have already been saved to:
          - model_evaluations.npy  (GP training data)
          - collocation_points.npy
          - results-detailed-*.csv / .npy
          - initial-model-outputs.json
        so the raw OpenFOAM output is no longer needed.
        """
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
            logger.info(f"Deleted run folder to free disk space: {case_dir}")

    def output_processing(self, output_data_path=None, **kwargs):
        """Load existing results for BAL restart."""
        if output_data_path and os.path.isfile(output_data_path):
            with open(output_data_path, 'r') as f:
                data = json.load(f)
            self.restart_collocation_points = np.array(data["collocation_points"])
            self.model_evaluations = np.array(data["model_evaluations"])
            return self.model_evaluations
        raise FileNotFoundError(f"Output data not found: {output_data_path}") 
