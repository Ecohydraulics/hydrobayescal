import os
import re
import glob
import json
import shutil
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy import spatial


#: k-epsilon coefficients that can be calibrated: config name (lowercase) -> name in
#: kEpsilonCoeffs. The coefficient must already be in the template's kEpsilonCoeffs.
KEPSILON_COEFFS = {
    "cmu": "Cmu",
    "c1": "C1",
    "c2": "C2",
    "c3": "C3",
    "sigmak": "sigmak",
    "sigmaeps": "sigmaEps",
}

#: Bed wall functions for ks (in order of preference). atmNutkWallFunction uses z0 and
#: also works if the first cell is smaller than the roughness; nutkRoughWallFunction
#: uses Ks and needs the first cell to be larger than Ks.
ATM_ROUGH_WALL = "atmNutkWallFunction"
KS_ROUGH_WALL = "nutkRoughWallFunction"

#: Fully rough flow: roughness length z0 = ks / 30 (Nikuradse).
KS_TO_Z0 = 1.0 / 30.0

#: Turbulence fields whose atm wall functions carry their own z0 entry and must be
#: kept consistent with the z0 written to nut.
ATM_Z0_FIELDS = ("nut", "epsilon", "omega")


def _is_number(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _set_patch_entry(path: str, patch: str, key: str, new_value: str) -> bool:
    """Replace the line ``key ...;`` of ``patch`` in boundaryField (line based, also
    works for full fields in time directories). Returns True if a line was replaced.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    key_re = re.compile(rf"^(\s*){re.escape(key)}\s+[^;]*;")
    patch_re = re.compile(rf"^\s*{re.escape(patch)}\s*(\{{\s*)?$")
    in_bf = in_patch = False
    depth = 0
    replaced = False
    for i, line in enumerate(lines):
        if not in_bf:
            in_bf = line.strip().startswith("boundaryField")
            continue
        if not in_patch:
            if patch_re.match(line):
                in_patch = True
                depth = line.count("{") - line.count("}")
            continue
        depth += line.count("{") - line.count("}")
        m = key_re.match(line)
        if m:
            lines[i] = f"{m.group(1)}{key:<16}{new_value};\n"
            replaced = True
        if depth <= 0 and "}" in line:
            break

    if replaced:
        with open(path, "w") as f:
            f.writelines(lines)
    return replaced


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

    # Turbulence dictionary of the case: constant/momentumTransport (Foundation 8+)
    # or constant/turbulenceProperties (ESI, also the default).
    def _turbulence_dictionary(self) -> str:
        for candidate in ("constant/momentumTransport",
                          "constant/turbulenceProperties"):
            if os.path.isfile(self._case_path(candidate)):
                return candidate
        return "constant/turbulenceProperties"

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
        ks_written = False
        cs_present = False  # True once we've seen (and preserved) an existing Cs line
        skip_multiline_value = False  # True while consuming a nonuniform list after 'value'
        # Ks/Cs are only written for nutkRoughWallFunction
        is_rough_wall = bc_type == "nutkRoughWallFunction"

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip the rest of a multi-line 'value nonuniform' list (already
            # replaced by 'value uniform 0;'), until ';' or ');'
            if skip_multiline_value:
                if stripped in (";", ");"):
                    skip_multiline_value = False
                continue

            if not inside_boundary_field and stripped.startswith("boundaryField"):
                inside_boundary_field = True
                new_lines.append(line)
                continue

            if inside_boundary_field:
                if not inside_patch:
                    if re.match(rf"^\s*{re.escape(patch)}\s*\{{\s*$", line):
                        inside_patch = True
                        patch_found = True
                        ks_written = False
                        cs_present = False
                        new_lines.append(line)
                        continue
                    if re.match(rf"^\s*{re.escape(patch)}\s*$", line):
                        if i + 1 < len(lines) and lines[i + 1].strip() == "{":
                            inside_patch = True
                            patch_found = True
                            ks_written = False
                            cs_present = False
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
                        ks_written = True
                        continue

                    if is_rough_wall and re.match(r"\s*Cs\s+", line):
                        # keep the Cs value of the template
                        cs_present = True
                        new_lines.append(line)
                        continue

                    if re.match(r"\s*value\s+", line):
                        # Insert Ks/Cs before value line if not yet written
                        if is_rough_wall and not ks_written and value is not None:
                            new_lines.append(f"        Ks uniform {float(value):.5f};\n")
                            ks_written = True
                            if not cs_present:
                                new_lines.append("        Cs uniform 0.5;\n")
                                cs_present = True
                        new_lines.append("        value uniform 0;\n")
                        # multi-line nonuniform list: skip its body
                        if not stripped.endswith(";"):
                            skip_multiline_value = True
                        continue

                    if "}" in line:
                        # Last resort: insert before closing brace
                        if is_rough_wall and not ks_written and value is not None:
                            new_lines.append(f"        Ks uniform {float(value):.5f};\n")
                            ks_written = True
                            if not cs_present:
                                new_lines.append("        Cs uniform 0.5;\n")
                                cs_present = True
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

    # Writes the roughness length z0 into the z0 entry of `patch` for every atm wall
    # function field (nut, epsilon, omega) in 0/ and in every other time directory
    # (a hot-start case reads its boundary conditions from {startTime}/, not 0/).
    def update_roughness_length(self, patch: str, z0: float, ks: Optional[float] = None) -> None:
        label = f"z0={z0:.6g}" + (f" (ks={ks:.6g})" if ks is not None else "")
        time_dirs = ["0"] + [
            e for e in sorted(os.listdir(self.case_dir))
            if e != "0" and os.path.isdir(os.path.join(self.case_dir, e)) and _is_number(e)
        ]
        for tdir in time_dirs:
            for field in ATM_Z0_FIELDS:
                path = os.path.join(self.case_dir, tdir, field)
                if not os.path.isfile(path):
                    continue
                if _set_patch_entry(path, patch, "z0", f"constant {z0:.6g}"):
                    print(f"Updating {label} on patch '{patch}' in {tdir}/{field}")
                elif field == "nut" and tdir == "0":
                    # nut must carry the z0 entry; anything else means the case
                    # template is not set up the way the binding expects.
                    raise ValueError(f"No 'z0' entry on patch '{patch}' in 0/nut")

    # Dispatches all calibration parameter updates: k-epsilon coefficients go to
    # turbulenceProperties, everything else is written as a boundary condition;
    # alpha.water fields are skipped.
    def update_model_controls(self, params: Dict[str, Dict[str, Any]]) -> None:
        skip_fields = {"alpha.water", "alpha.water.orig"}

        for patch, param in params.items():
            file = param["file"]

            if patch in KEPSILON_COEFFS.values():
                value = float(param["value"])
                # Use the given turbulence dictionary if it exists, otherwise the one
                # of the case (turbulenceProperties or momentumTransport)
                turb_file = param.get("file") or self._turbulence_dictionary()
                if not os.path.isfile(self._case_path(turb_file)):
                    turb_file = self._turbulence_dictionary()
                print(f"Updating model coefficient '{patch}' in {turb_file}...")
                self.update_dictionary_entry(
                    file=turb_file,
                    subdict="kEpsilonCoeffs",
                    key=patch,
                    value=value,
                )
                continue

            if patch == "ks":
                value = float(param["value"])
                if param["bc_type"] == ATM_ROUGH_WALL:
                    # z0 bed: write z0 = ks/30 to nut, epsilon and omega
                    self.update_roughness_length(param["patch"], value * KS_TO_Z0, ks=value)
                    continue
                print(f"Updating Ks={value} on patch '{param['patch']}' in 0/nut...")
                self.update_boundary_condition(
                    file=param["file"],
                    patch=param["patch"],
                    field_type=param["field_type"],
                    bc_type=param["bc_type"],
                    value=value,
                )
                # Also update Ks in the other time directories: a hot start reads the
                # boundary conditions from {startTime}/nut, not 0/nut
                for entry in sorted(os.listdir(self.case_dir)):
                    if entry == "0":
                        continue
                    try:
                        float(entry)   # skip non-numeric dirs (constant, system, ...)
                    except ValueError:
                        continue
                    nut_path = os.path.join(self.case_dir, entry, "nut")
                    if os.path.isfile(nut_path):
                        print(f"  Also updating Ks={value} in {entry}/nut (hot-start time dir)...")
                        # Line replacement on the bed patch only (other walls keep
                        # their own Ks)
                        if _set_patch_entry(nut_path, param["patch"], "Ks", f"uniform {value:.5f}"):
                            print(f"    Ks updated to {value:.5f} in {entry}/nut.")
                        else:
                            print(f"    WARNING: Ks line not found in {entry}/nut - file unchanged.")
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
            # OpenFOAM environment must be loaded before. Set HBC_MPI_LAUNCHER if the
            # launcher is not mpirun (e.g. mpiexec)
            launcher = os.environ.get("HBC_MPI_LAUNCHER", "mpirun")
            mpi_cmd = f"{launcher} -np {int(nprocs)} {solver} -parallel"
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
        # Remove any pre-existing VTK folder (e.g. copied from case template)
        # to ensure foamToVTK only converts this run's own time directories.
        vtk_dir_pre = os.path.join(self.case_dir, "VTK")
        if os.path.isdir(vtk_dir_pre):
            shutil.rmtree(vtk_dir_pre)
            print(f"Removed pre-existing VTK folder: {vtk_dir_pre}", flush=True)
        print("Converting to VTK...")
        cmd = f"foamToVTK -case {self.case_dir}"
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        result = subprocess.run(["bash", "-c", cmd], env=env, cwd=self.case_dir,
                                capture_output=True, text=True)
        if result.returncode != 0:
            # Not fatal: extraction reads the case directly, VTK is only for ParaView
            print(f"foamToVTK stderr: {result.stderr}", flush=True)
            print(f"foamToVTK stdout: {result.stdout}", flush=True)
            logger.warning(
                f"foamToVTK failed with exit code {result.returncode}; continuing, "
                f"since field extraction reads the case directory itself."
            )
            return
        print("VTK conversion completed.")

    def extract_fields_from_vtk(self, alpha_threshold: float = 0.5, n_avg_timesteps: int = 1) -> Tuple[Any, Any, Any, Any]:
        """
        Extract velocity (U), turbulent kinetic energy (k) and water volume
        fraction (alpha.water) from the finished run, averaged over the last
        ``n_avg_timesteps`` write times, restricted to the water phase.

        Reads the case directory with pyvista's OpenFOAM reader (no foamToVTK needed,
        works for ESI and Foundation, ASCII and binary).

        k is the total TKE: modelled k plus the resolved part 0.5 * trace(UPrime2Mean)
        if UPrime2Mean is written by fieldAverage.

        Args:
            alpha_threshold: keep only points with alpha.water > this (default 0.5).
                A single-phase (rigid-lid) case carries no alpha.water; every cell
                is then water and all points are kept.
            n_avg_timesteps: number of trailing write times to average.

        Returns:
            (coordinates, U_mean, k_mean, alpha_mean) for water-phase points; k_mean
            is None when the case carries no k field, alpha_mean is None when it
            carries no alpha.water (single-phase case).
        """
        foam_files = sorted(glob.glob(os.path.join(self.case_dir, "*.foam")))
        foam_file = foam_files[0] if foam_files else os.path.join(self.case_dir, "case.foam")
        if not os.path.isfile(foam_file):
            # pyvista needs a .foam file
            open(foam_file, "a").close()

        reader = pv.OpenFOAMReader(foam_file)
        reader.enable_all_cell_arrays()
        try:
            reader.enable_all_point_arrays()
        except AttributeError:
            pass

        times = [float(x) for x in reader.time_values]
        # skip t=0 (initial condition) unless it is the only time
        positive = [x for x in times if x > 0.0]
        times = positive or times
        if not times:
            raise RuntimeError(
                f"No time steps found in {self.case_dir}. Did the solver run?"
            )
        times.sort()

        n = min(int(n_avg_timesteps), len(times))
        if n < int(n_avg_timesteps):
            logger.warning(
                f"Requested n_avg_timesteps={n_avg_timesteps} but only {n} write "
                f"time(s) available. Averaging over {n}."
            )
        selected = times[-n:]
        logger.info(f"Averaging over {n} write time(s): {selected}")

        coords = None
        water_mask = None
        has_k = False
        has_alpha = False
        U_sum = None
        k_sum = None
        alpha_sum = None

        for time_value in selected:
            reader.set_active_time_value(time_value)
            mesh = reader.read()["internalMesh"]
            # OpenFOAM writes cell-centred data; the calibration samples at points.
            mesh = mesh.cell_data_to_point_data()

            if "U" not in mesh.point_data:
                raise KeyError(
                    f"Velocity field 'U' not found at t={time_value} in {self.case_dir}."
                )

            if coords is None:
                coords = mesh.points
                has_k = "k" in mesh.point_data
                has_alpha = "alpha.water" in mesh.point_data
                if has_alpha:
                    alpha = mesh.point_data["alpha.water"]
                    water_mask = alpha > alpha_threshold
                    coords = coords[water_mask]
                    logger.info(
                        f"Water phase: {int(water_mask.sum())}/{len(alpha)} points "
                        f"(alpha > {alpha_threshold})"
                    )
                else:
                    # single-phase case: all cells are water
                    logger.info("no alpha.water field (single-phase case) - using all points")
                # Accumulate U, k and alpha.water across the selected write times
                U_sum = np.zeros_like(
                    mesh.point_data["U"][water_mask] if water_mask is not None
                    else mesh.point_data["U"]
                )
                k_sum = np.zeros(U_sum.shape[0]) if has_k else None
                alpha_sum = np.zeros(U_sum.shape[0]) if has_alpha else None

            U_t = mesh.point_data["U"]
            if water_mask is not None:
                U_t = U_t[water_mask]
            U_sum += U_t

            if has_k and "k" in mesh.point_data:
                k_t = mesh.point_data["k"]
                if water_mask is not None:
                    k_t = k_t[water_mask]
                k_sum += k_t

            if has_alpha and "alpha.water" in mesh.point_data:
                alpha_t = mesh.point_data["alpha.water"]
                if water_mask is not None:
                    alpha_t = alpha_t[water_mask]
                alpha_sum += alpha_t

        U_mean = U_sum / n
        k_mean = k_sum / n if has_k else None
        alpha_mean = alpha_sum / n if has_alpha else None

        if k_mean is None:
            logger.warning("k field not found - TKE will be NaN")
        else:
            logger.info(
                f"k_mod (averaged over {n} step(s)): Mean={k_mean.mean():.4e}, Max={k_mean.max():.4e}"
            )
            # Add the resolved TKE 0.5 * trace(UPrime2Mean) (URANS, e.g. vortex shedding),
            # since the measured TKE contains both parts
            if "UPrime2Mean" in mesh.point_data:
                R = mesh.point_data["UPrime2Mean"]
                if water_mask is not None:
                    R = R[water_mask]
                k_res = _resolved_tke(R)
                logger.info(
                    f"k_res (0.5*tr(UPrime2Mean)): Mean={k_res.mean():.4e}, Max={k_res.max():.4e}"
                )
                k_mean = k_mean + k_res
            else:
                logger.warning(
                    "UPrime2Mean not found (fieldAverage with prime2Mean on U missing?) - "
                    "TKE is the modelled k only, without the resolved URANS part."
                )

        return coords, U_mean, k_mean, alpha_mean


def _resolved_tke(R):
    """Resolved TKE, 0.5 * trace, from a UPrime2Mean array (n_points x 6 or 9)."""
    R = np.asarray(R)
    if R.shape[1] == 9:
        diag = R[:, [0, 4, 8]]
    else:
        # VTK order (xx, yy, zz, xy, yz, xz); if this gives negative normal stresses
        # use the OpenFOAM order (xx, xy, xz, yy, yz, zz)
        diag = R[:, [0, 1, 2]]
        if (diag < -1e-6 * np.abs(R).max()).any():
            diag = R[:, [0, 3, 5]]
    return 0.5 * np.maximum(diag, 0.0).sum(axis=1)


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

    #: Quantities returned by _extract_at_control_points (valid calibration and
    #: extraction quantities). U_magnitude is the old name of U_MAG, ALPHA_WATER is
    #: the interFoam water volume fraction at the point.
    EXTRACTABLE_QUANTITIES = (
        "U_x", "U_y", "U_z",
        "U_MAG", "U_magnitude",
        "u_fluct", "v_fluct", "w_fluct",
        "TKE",
        "ALPHA_WATER",
    )

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
        gpe_error=0.0,
        measurement_error=0.10,
        model_structural_error=0.0,
        write_vtk=False,
        *args,
        **kwargs
    ):
        # Default folders
        case_template_dir = os.path.abspath(case_template_dir)
        model_dir = os.path.abspath(model_dir) if model_dir else os.path.dirname(case_template_dir)
        res_dir = os.path.abspath(res_dir) if res_dir else os.path.join(model_dir, "results")

        # Defaults if no calibration setup is given
        calibration_parameters = calibration_parameters or ["Cmu"]
        param_values = param_values or [[0.06, 0.12]]
        extraction_quantities = extraction_quantities or ["U_x", "U_y", "U_z"]
        calibration_quantities = calibration_quantities or ["U_x", "U_y", "U_z"]

        # Base class: parameters, observations, variances and result folders (as Telemac)
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
            # error terms, used by the base class for the variances
            gpe_error=gpe_error,
            measurement_error=measurement_error,
            model_structural_error=model_structural_error,
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
        # Write VTK for every run (off by default, only needed for ParaView)
        self.write_vtk = bool(write_vtk)
        # Alias consumed by the GP layer (see bal_openfoam.py).
        self.parameter_ranges = self.param_values

        # Check calibration quantities and parameters before any simulation
        self._validate_calibration_quantities()
        self._validate_calibration_parameters()

        # Fallbacks if there is no calibration file
        if self.num_calibration_quantities is None:
            self.num_calibration_quantities = len(self.calibration_quantities)
        if self.nloc is None:
            self.nloc = 0

        # XYZ coordinates of the calibration points
        self._load_control_points()

        # Check that k is written to VTK output
        self._check_k_in_controldict()

        # Find the bed patch for ks in 0/nut
        self.ks_patch, self.ks_bc_type = self._detect_ks_patch()
        if self.ks_patch:
            how = (f"z0 = ks/30 written to {', '.join(ATM_Z0_FIELDS)}" if self.ks_bc_type == ATM_ROUGH_WALL
                   else "Ks written to nut")
            logger.info(f"Auto-detected ks roughness patch: '{self.ks_patch}' ({self.ks_bc_type}; {how})")
        else:
            logger.warning(
                f"Could not auto-detect a {ATM_ROUGH_WALL} or {KS_ROUGH_WALL} bed patch in the case "
                "template's 0/nut; a 'ks' calibration parameter will fail if used."
            )

        # Results storage
        self.model_evaluations = None
        # Per-point history in the TELEMAC initial-model-outputs.json layout:
        # {point_id: [{quantity: value, ...} per run]}. See _record_point_outputs.
        self.point_outputs = {}

        logger.info(f"OpenFOAMModel initialized: {self.ndim} parameter(s), {self.nloc} locations")

    def _validate_calibration_quantities(self):
        """Check every calibration target against what the extraction can produce.

        Raises
        ------
        ValueError
            If ``calibration_quantities`` names a field absent from
            :attr:`EXTRACTABLE_QUANTITIES`, listing the offending names and the
            valid ones.
        """
        unknown = [
            q for q in (self.calibration_quantities or [])
            if q not in self.EXTRACTABLE_QUANTITIES
        ]
        if unknown:
            raise ValueError(
                f"Unknown calibration_quantities for the OpenFOAM binding: {unknown}. "
                f"Valid field names are: {', '.join(self.EXTRACTABLE_QUANTITIES)}."
            )

    def _validate_calibration_parameters(self):
        """Check every calibration parameter against what the dispatch can route.

        Raises
        ------
        ValueError
            If ``calibration_parameters`` names a parameter that is neither ``ks``
            nor a :data:`KEPSILON_COEFFS` entry, listing the offending names and
            the valid ones.
        """
        valid = tuple(KEPSILON_COEFFS.values()) + ("ks",)
        unknown = [
            p for p in (self.calibration_parameters or [])
            if p.lower() not in KEPSILON_COEFFS and p.lower() != "ks"
        ]
        if unknown:
            raise ValueError(
                f"Unknown calibration_parameters for the OpenFOAM binding: {unknown}. "
                f"Valid parameter names are: {', '.join(valid)}. Names are matched "
                f"case-insensitively. To calibrate something else, add a branch to the "
                f"dispatch in OpenFOAMModel.run_multiple_simulations."
            )

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

    def _detect_ks_patch(self):
        """Find the bed patch in ``0/nut`` that the ``ks`` parameter writes to.

        An atmNutkWallFunction patch is used first, otherwise nutkRoughWallFunction.
        If there are several, the one named bottom/base/bed is used.

        Returns
        -------
        (str, str) or (None, None)
            The patch name and its wall function type.
        """
        nut_path = os.path.join(self.case_template_dir, "0", "nut")
        if not os.path.isfile(nut_path):
            return None, None

        with open(nut_path, "r") as f:
            content = f.read()

        for bc_type in (ATM_ROUGH_WALL, KS_ROUGH_WALL):
            patches = re.findall(rf"(\w+)\s*\{{\s*[^{{}}]*?type\s+{bc_type}\s*;", content, re.DOTALL)
            if len(patches) == 1:
                return patches[0], bc_type
            if len(patches) > 1:
                beds = [p for p in patches if p.lower() in ("bottom", "base", "bed")]
                if len(beds) == 1:
                    return beds[0], bc_type
                raise ValueError(
                    f"Several {bc_type} patches in {nut_path} ({', '.join(patches)}) and no single "
                    "bed patch (bottom/base/bed) among them: cannot tell which one 'ks' belongs to."
                )
        return None, None

    def _load_control_points(self):
        """Read the XYZ coordinates of the calibration points (observations and
        variances are read by the base class)."""
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
        bal_new_set_parameters=None,
        start_index=0,
    ):
        """Run multiple simulations (BAL interface).

        start_index: number of runs already done (staged initial design); only the
        remaining runs are simulated.
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
        # keep outputs of runs already done (staged design)
        previous_evaluations = self.model_evaluations if start_index else None
        # per-point history: keep only the runs before this batch
        for point_id in self.point_outputs:
            self.point_outputs[point_id] = self.point_outputs[point_id][:start_idx]

        all_results = []
        all_detailed_results = []  # for comprehensive CSV: one row per run x control point
        # _save_all_results appends to the csv, so only new rows are passed
        n_detailed_saved = 0

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
                if pname.lower() in KEPSILON_COEFFS:
                    update_params[KEPSILON_COEFFS[pname.lower()]] = {
                        "file": "constant/turbulenceProperties",
                        "value": float(pval),
                    }
                elif pname.lower() == "ks":
                    update_params["ks"] = {
                        "file": "0/nut",
                        "patch": self.ks_patch,
                        "field_type": "scalar",
                        "bc_type": self.ks_bc_type,
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

                # Convert to VTK (only if write_vtk, not needed for the extraction)
                if getattr(self, "write_vtk", False):
                    foam.convert_to_vtk()

                # Extract fields, averaged over the last n_avg_timesteps write times
                coords, U, k, alpha = foam.extract_fields_from_vtk(n_avg_timesteps=self.n_avg_timesteps)

                # Sanity check: verify fields differ from previous run.
                # Identical fields across runs indicate an extraction bug.
                if run_idx > start_idx:
                    prev_U_path = os.path.join(self.restart_data_folder, "raw_fields", f"U_{run_idx - 1:04d}.npy")
                    if os.path.isfile(prev_U_path):
                        prev_U = np.load(prev_U_path)
                        if prev_U.shape == U.shape and np.allclose(prev_U, U, rtol=1e-6):
                            raise RuntimeError(
                                f"SANITY CHECK FAILED: Run {run_idx} produced identical U fields to run {run_idx - 1}. "
                                f"Check that the simulation actually ran and that the parameter was written to the case."
                            )
                        else:
                            logger.info(f"Sanity check passed: run {run_idx} fields differ from run {run_idx - 1}.")

                if self.nloc == 0:
                    # No measurements file yet: save raw fields so they can be
                    # extracted at the control points later
                    raw_dir = os.path.join(self.restart_data_folder, "raw_fields")
                    os.makedirs(raw_dir, exist_ok=True)
                    np.save(os.path.join(raw_dir, f"coords_{run_idx:04d}.npy"), coords)
                    np.save(os.path.join(raw_dir, f"U_{run_idx:04d}.npy"), U)
                    if k is not None:
                        np.save(os.path.join(raw_dir, f"k_{run_idx:04d}.npy"), k)
                    if alpha is not None:
                        np.save(os.path.join(raw_dir, f"alpha_{run_idx:04d}.npy"), alpha)
                    logger.info(f"No measurements file - raw fields saved to {raw_dir} for run {run_idx}.")

                    # Save collocation points so they can be reloaded later
                    current_cp = (design[:start_idx + i + 1] if design is not None
                                  else params_to_run[:i + 1])
                    np.save(os.path.join(self.restart_data_folder, "collocation_points.npy"), current_cp)

                else:
                    # Measurements file exists: extract at control points
                    results = self._extract_at_control_points(coords, U, k, alpha)

                    # quantities are checked in __init__, a missing key is an error
                    run_results = []
                    for loc_idx in range(self.nloc):
                        for qty in self.calibration_quantities:
                            run_results.append(float(results[qty][loc_idx]))

                    all_results.append(run_results)
                    self._record_point_outputs(results)

                    # Store detailed results: one row per control point. Every
                    # extractable quantity is written here regardless of what is
                    # being calibrated, so calibration_quantities can be re-chosen
                    # later from this table without re-running OpenFOAM (see
                    # output_processing / --only_bal_mode).
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
                            "ALPHA_WATER": float(results["ALPHA_WATER"][pt_idx]),
                        }
                        all_detailed_results.append(row)

                    # Save per-run JSON inside the run folder
                    self._save_run_results(case_dir, run_idx, params, results)

                    # Update model_evaluations
                    current_results = np.array(all_results)
                    if bal_new_set_parameters is None:
                        self.model_evaluations = (
                            np.vstack([previous_evaluations, current_results])
                            if previous_evaluations is not None else current_results)
                        current_cp = design[:start_idx + i + 1]
                    else:
                        self.model_evaluations = np.vstack([
                            self.model_evaluations, current_results
                        ]) if self.model_evaluations is not None else current_results
                        current_cp = np.vstack([collocation_points, params_to_run[:i + 1]])

                    # Save after every run (only new rows, the csv is appended)
                    self._save_all_results(current_cp, all_detailed_results[n_detailed_saved:])
                    n_detailed_saved = len(all_detailed_results)
                    self._save_model_results_calibration_csv()

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
                    self._record_point_outputs(None)
                    for pt_idx, cp in enumerate(self.control_points):
                        row = {
                            "run_idx": run_idx,
                            **{p: float(v) for p, v in zip(self.calibration_parameters, params)},
                            "x": float(cp[0]), "y": float(cp[1]), "z": float(cp[2]),
                            "U_x": np.nan, "U_y": np.nan, "U_z": np.nan,
                            "U_magnitude": np.nan,
                            "u_fluct": np.nan, "v_fluct": np.nan, "w_fluct": np.nan,
                            "TKE": np.nan,
                            "ALPHA_WATER": np.nan,
                        }
                        all_detailed_results.append(row)

        # Final save (not if nloc = 0, then raw fields were saved per run)
        if self.nloc > 0:
            if bal_new_set_parameters is not None:
                all_cp = np.vstack([collocation_points, params_to_run])
            else:
                all_cp = design
            self._save_all_results(all_cp, all_detailed_results[n_detailed_saved:])
            self._save_model_results_calibration_csv()

    def _save_model_results_calibration_csv(self):
        """model-results-calibration-*.csv as in the TELEMAC binding: one row per run,
        one column per (point, quantity). Rewritten completely at every call.
        """
        if self.model_evaluations is None:
            return
        quantities_str = '_'.join(self.calibration_quantities)
        column_headers = [
            f"{i}_{qty}"
            for i in range(1, self.nloc + 1)
            for qty in self.calibration_quantities
        ]
        csv_path = os.path.join(
            self.calibration_folder, f"model-results-calibration-{quantities_str}.csv"
        )
        np.savetxt(csv_path, self.model_evaluations, delimiter=',', fmt='%.8f',
                   header=','.join(column_headers))
        logger.info(f"Saved model results calibration CSV to {csv_path}")

    def _point_ids(self):
        """Calibration point IDs: the first column of the calibration CSV, as TELEMAC keys them."""
        df = self.calibration_pts_df
        return [] if df is None else [str(k) for k in df.iloc[:, 0]]

    def _point_output_quantities(self):
        """Quantities stored in initial-model-outputs.json: extraction and calibration
        quantities, so the calibration quantities can be changed in a restart.
        """
        quantities = [q for q in (self.extraction_quantities or [])
                      if q in self.EXTRACTABLE_QUANTITIES]
        return quantities + [q for q in self.calibration_quantities if q not in quantities]

    def _record_point_outputs(self, results):
        """Append one run to ``point_outputs``; ``results=None`` records a failed run as NaN."""
        quantities = self._point_output_quantities()
        for pt_idx, point_id in enumerate(self._point_ids()):
            self.point_outputs.setdefault(point_id, []).append({
                q: float(results[q][pt_idx]) if results is not None else float("nan")
                for q in quantities
            })

    def _save_all_results(self, collocation_points, detailed_results=None):
        """Base-class bookkeeping, then initial-model-outputs.json in the TELEMAC layout."""
        super()._save_all_results(collocation_points, detailed_results)
        self._save_initial_model_outputs()

    def _save_initial_model_outputs(self):
        """Write initial-model-outputs.json as in the TELEMAC binding:
        {point_id: [{quantity: value, ...}, ...]}, one entry per run.
        """
        for folder in (self.calibration_folder, self.restart_data_folder):
            with open(os.path.join(folder, "initial-model-outputs.json"), "w") as f:
                json.dump(self.point_outputs, f, indent=4)

    def _extract_at_control_points(self, coords, U, k=None, alpha=None):
        """Extract velocity, TKE and water fraction at control points.

        TKE is the total (modelled k + resolved) TKE from ``extract_fields_from_vtk``.

        Args:
            coords: Array of VTK point coordinates (water phase only)
            U: Velocity array at VTK points, shape (n_points, 3)
            k: total turbulent kinetic energy array (modelled + resolved), or None
            alpha: interFoam alpha.water array at VTK points, or None (single-phase)

        Returns:
            dict keyed by :attr:`EXTRACTABLE_QUANTITIES`: U_x, U_y, U_z, U_MAG
            (with U_magnitude retained as a legacy alias of the same values),
            u_fluct, v_fluct, w_fluct, TKE and ALPHA_WATER at the control points
        """
        if len(self.control_points) == 0:
            # empty arrays for all quantities
            return {q: np.array([]) for q in self.EXTRACTABLE_QUANTITIES}

        tree = spatial.cKDTree(coords)
        _, indices = tree.query(self.control_points)

        U_cp = U[indices]

        # TKE = modelled k + resolved 0.5*tr(UPrime2Mean), see extract_fields_from_vtk
        # Isotropic fluctuation components estimated as sqrt(2k/3) for reporting only
        if k is not None:
            k_cp = k[indices]
            fluct = np.sqrt(np.maximum(2.0 / 3.0 * k_cp, 0.0))
        else:
            k_cp = np.full(len(self.control_points), np.nan)
            fluct = np.full(len(self.control_points), np.nan)

        if alpha is not None:
            alpha_cp = alpha[indices]
        else:
            alpha_cp = np.full(len(self.control_points), np.nan)

        u_mag = np.linalg.norm(U_cp, axis=1)

        return {
            "U_x": U_cp[:, 0],
            "U_y": U_cp[:, 1],
            "U_z": U_cp[:, 2],
            "U_MAG": u_mag,        # the documented name, shared with the Delft3D binding
            "U_magnitude": u_mag,  # legacy alias, keeps older result files readable
            "u_fluct": fluct,
            "v_fluct": fluct,
            "w_fluct": fluct,
            "TKE": k_cp,        # modelled k + resolved URANS TKE
            "ALPHA_WATER": alpha_cp,  # interFoam water volume fraction at the point
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

    def output_processing(self, output_data_path=None, run_range_filtering=None, **kwargs):
        """Load stored results for a BAL restart.

        Reads the TELEMAC layout written by _save_initial_model_outputs. The flat layout
        and the re-selection from detailed-model-outputs.csv are done by the base class.
        run_range_filtering = (first, last), 1-based, as in the TELEMAC binding.
        """
        data = None
        if output_data_path and os.path.isfile(output_data_path):
            with open(output_data_path, 'r') as f:
                data = json.load(f)
        if data is None or "model_evaluations" in data:
            return self._load_stored_outputs(output_data_path, run_range_filtering=run_range_filtering)

        point_ids = self._point_ids()
        missing = [p for p in point_ids if p not in data]
        if missing:
            raise KeyError(f"Calibration points {missing} not found in {output_data_path}.")

        n_runs = len(data[point_ids[0]])
        first, last = run_range_filtering or (1, n_runs)
        self.point_outputs = {p: data[p][first - 1:last] for p in point_ids}

        try:
            rows = [
                [float(self.point_outputs[p][r][q])
                 for p in point_ids for q in self.calibration_quantities]
                for r in range(len(self.point_outputs[point_ids[0]]))
            ]
        except KeyError as e:
            raise KeyError(
                f"Calibration quantity {e} is not stored in {output_data_path}; it can only "
                f"be re-selected if it was extracted when the runs were made."
            ) from None
        self.model_evaluations = np.array(rows)
        return self.model_evaluations
