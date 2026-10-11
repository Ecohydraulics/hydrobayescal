"""Tests for choosing the OpenFOAM calibration quantities after the runs.

``run_multiple_simulations`` writes every quantity in
``OpenFOAMModel.EXTRACTABLE_QUANTITIES`` to ``results-detailed-*.csv`` for every
run x control point, and ``HydroSimulations._save_all_results`` keeps the rows of
all runs in ``restart_data/detailed-model-outputs.csv``. ``output_processing`` then
rebuilds the surrogate training matrix for whatever ``calibration_quantities`` is
configured now, so ``--only_bal_mode True --calibration_quantities "..."`` needs
no new OpenFOAM runs.

Exercised through the methods directly with a stub carrying only the attributes
they read, so the tests need no solver stack and no trained surrogate.
"""
import csv
import json

import numpy as np
import pytest

from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel

DETAILED_COLS = [
    "run_idx", "Cmu", "x", "y", "z",
    "U_x", "U_y", "U_z", "U_magnitude",
    "u_fluct", "v_fluct", "w_fluct", "TKE", "ALPHA_WATER",
]


def _stub(tmp_path, calibration_quantities, nloc=2):
    restart = tmp_path / "restart_data"
    calib = tmp_path / "calibration-data" / "_".join(calibration_quantities)
    restart.mkdir(parents=True, exist_ok=True)
    calib.mkdir(parents=True, exist_ok=True)
    return type("Stub", (), {
        "restart_data_folder": str(restart),
        "calibration_folder": str(calib),
        "calibration_quantities": list(calibration_quantities),
        "calibration_parameters": ["Cmu"],
        "nloc": nloc,
        "EXTRACTABLE_QUANTITIES": OpenFOAMModel.EXTRACTABLE_QUANTITIES,
        "RESELECTION_DETAILED_FILE": OpenFOAMModel.RESELECTION_DETAILED_FILE,
        "model_evaluations": None,
        "restart_collocation_points": None,
        # output_processing delegates to these two methods of the shared base class, so the stub has to carry them: a bare stub raised AttributeError on the rebuild path.
        "_load_stored_outputs": OpenFOAMModel._load_stored_outputs,
        "_rebuild_model_evaluations_from_detailed": OpenFOAMModel._rebuild_model_evaluations_from_detailed,
    })()


def _write_detailed(folder, runs):
    """runs: list of (Cmu, [per-point {quantity: value}]). Writes the mirror file."""
    path = f"{folder}/{OpenFOAMModel.RESELECTION_DETAILED_FILE}"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAILED_COLS)
        w.writeheader()
        for run_idx, (cmu, points) in enumerate(runs):
            for pt_idx, values in enumerate(points):
                row = {c: 0.0 for c in DETAILED_COLS}
                row["run_idx"] = run_idx
                row["Cmu"] = cmu
                row["x"] = float(pt_idx)
                row.update(values)
                w.writerow(row)
    return path


TWO_RUNS = [
    (0.07, [{"U_x": 1.0, "TKE": 0.10}, {"U_x": 2.0, "TKE": 0.20}]),
    (0.09, [{"U_x": 1.5, "TKE": 0.15}, {"U_x": 2.5, "TKE": 0.25}]),
]


def test_rebuild_two_quantities(tmp_path):
    stub = _stub(tmp_path, ["U_x", "TKE"])
    _write_detailed(stub.restart_data_folder, TWO_RUNS)

    me, cp = OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)

    # For each control point, both quantities in order: [Ux_p0, TKE_p0, Ux_p1, TKE_p1]
    assert me.tolist() == [[1.0, 0.10, 2.0, 0.20], [1.5, 0.15, 2.5, 0.25]]
    assert cp.tolist() == [[0.07], [0.09]]


def test_rebuild_single_quantity_subset(tmp_path):
    stub = _stub(tmp_path, ["TKE"])
    _write_detailed(stub.restart_data_folder, TWO_RUNS)

    me, _ = OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)

    assert me.tolist() == [[0.10, 0.20], [0.15, 0.25]]


def test_rebuild_rejects_a_quantity_absent_from_the_table(tmp_path):
    stub = _stub(tmp_path, ["WATER_DEPTH"])
    _write_detailed(stub.restart_data_folder, TWO_RUNS)

    with pytest.raises(ValueError, match="no column"):
        OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)


def test_rebuild_detects_control_point_count_mismatch(tmp_path):
    stub = _stub(tmp_path, ["U_x"], nloc=3)
    _write_detailed(stub.restart_data_folder, TWO_RUNS)  # only 2 points per run

    with pytest.raises(ValueError, match="expected nloc=3"):
        OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)


def test_rebuild_missing_table_raises_filenotfound(tmp_path):
    stub = _stub(tmp_path, ["U_x"])

    with pytest.raises(FileNotFoundError):
        OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)


def test_output_processing_fast_path_when_quantities_match(tmp_path):
    stub = _stub(tmp_path, ["U_x"])
    json_path = f"{stub.restart_data_folder}/initial-model-outputs.json"
    with open(json_path, "w") as f:
        json.dump({
            "collocation_points": [[0.07], [0.09]],
            "model_evaluations": [[1.0, 2.0], [1.5, 2.5]],
            "calibration_quantities": ["U_x"],
        }, f)

    me = OpenFOAMModel.output_processing(stub, output_data_path=json_path)

    # Returned verbatim from the JSON, detailed table not consulted.
    assert me.tolist() == [[1.0, 2.0], [1.5, 2.5]]
    assert stub.restart_collocation_points.tolist() == [[0.07], [0.09]]


def test_output_processing_trusts_a_json_without_a_recorded_quantity_list(tmp_path):
    """Backward compatibility: a pre-existing JSON that never recorded its
    quantities is used as-is rather than forcing a rebuild it cannot satisfy."""
    stub = _stub(tmp_path, ["U_x", "U_y"])
    json_path = f"{stub.restart_data_folder}/initial-model-outputs.json"
    with open(json_path, "w") as f:
        json.dump({
            "collocation_points": [[0.07]],
            "model_evaluations": [[1.0, 2.0]],
        }, f)

    me = OpenFOAMModel.output_processing(stub, output_data_path=json_path)

    assert me.tolist() == [[1.0, 2.0]]


def test_output_processing_rebuilds_when_quantities_differ(tmp_path):
    stub = _stub(tmp_path, ["TKE"])          # configured now
    _write_detailed(stub.restart_data_folder, TWO_RUNS)
    json_path = f"{stub.restart_data_folder}/initial-model-outputs.json"
    with open(json_path, "w") as f:
        json.dump({
            "collocation_points": [[0.07], [0.09]],
            "model_evaluations": [[1.0, 2.0], [1.5, 2.5]],
            "calibration_quantities": ["U_x"],   # what the runs were saved as
        }, f)

    me = OpenFOAMModel.output_processing(stub, output_data_path=json_path)

    assert me.tolist() == [[0.10, 0.20], [0.15, 0.25]]
    assert stub.restart_collocation_points.tolist() == [[0.07], [0.09]]


def test_output_processing_applies_run_range_on_the_rebuild_path(tmp_path):
    stub = _stub(tmp_path, ["U_x"])
    _write_detailed(stub.restart_data_folder, TWO_RUNS + [
        (0.11, [{"U_x": 9.0}, {"U_x": 9.9}]),
    ])

    me = OpenFOAMModel.output_processing(
        stub, output_data_path=None, run_range_filtering=(1, 2)
    )

    assert me.tolist() == [[1.0, 2.0], [1.5, 2.5]]   # third run trimmed off
    assert stub.restart_collocation_points.tolist() == [[0.07], [0.09]]


def test_output_processing_raises_when_nothing_is_on_disk(tmp_path):
    stub = _stub(tmp_path, ["U_x"])

    with pytest.raises(FileNotFoundError):
        OpenFOAMModel.output_processing(stub, output_data_path=None)


def _detailed_rows(run_idx, cmu, points):
    """Rows as ``run_multiple_simulations`` builds them, every extractable quantity per point."""
    rows = []
    for pt_idx, values in enumerate(points):
        row = {c: 0.0 for c in DETAILED_COLS}
        row.update(run_idx=run_idx, Cmu=cmu, x=float(pt_idx))
        row.update(values)
        rows.append(row)
    return rows


def _simulate(stub, runs, first_run_idx=0):
    """Store runs the way the binding does: all rows of the block after every run, and once more after the loop."""
    from hydroBayesCal.hysim import HydroSimulations

    n_q = len(stub.calibration_quantities)
    block = []
    for offset, (cmu, points) in enumerate(runs):
        block.extend(_detailed_rows(first_run_idx + offset, cmu, points))
        n_runs = first_run_idx + offset + 1
        stub.model_evaluations = np.zeros((n_runs, stub.nloc * n_q))
        HydroSimulations._save_all_results(stub, np.zeros((n_runs, 1)), list(block))
    HydroSimulations._save_all_results(stub, np.zeros((n_runs, 1)), list(block))


def test_reselection_reads_what_the_binding_really_writes(tmp_path):
    """The rebuild must work on a table produced by the saving path, not only on a hand-written one.

    The binding saves the rows of the whole block after every run. While those saves were appended, every run was stored several times and the rebuild rejected the table for its row count, so re-selection failed on any real result folder although each of its parts passed its own test.
    """
    stub = _stub(tmp_path, ["U_x"])
    _simulate(stub, TWO_RUNS)

    stub.calibration_quantities = ["TKE"]
    me = OpenFOAMModel.output_processing(
        stub, output_data_path=f"{stub.restart_data_folder}/initial-model-outputs.json")

    assert me.tolist() == [[0.10, 0.20], [0.15, 0.25]]
    assert stub.restart_collocation_points.tolist() == [[0.07], [0.09]]


def test_reselection_keeps_earlier_runs_when_bal_continues_on_new_quantities(tmp_path):
    """Runs made before a change of calibration quantities stay available after BAL continued.

    A restart on other quantities works in a new, empty calibration folder. A copy of that folder's detailed table would hold only the runs simulated since the restart, and the next re-selection would silently train the surrogate on those few runs.
    """
    stub = _stub(tmp_path, ["U_x"])
    _simulate(stub, TWO_RUNS)

    # restart on TKE and let BAL add one run, which lands in the folder of the new quantities
    stub.calibration_quantities = ["TKE"]
    new_folder = tmp_path / "calibration-data" / "TKE"
    new_folder.mkdir(parents=True)
    stub.calibration_folder = str(new_folder)
    _simulate(stub, [(0.11, [{"U_x": 3.0, "TKE": 0.30}, {"U_x": 4.0, "TKE": 0.40}])], first_run_idx=2)

    # restart once more, on both quantities
    stub.calibration_quantities = ["U_x", "TKE"]
    me = OpenFOAMModel.output_processing(
        stub, output_data_path=f"{stub.restart_data_folder}/initial-model-outputs.json")

    assert me.tolist() == [[1.0, 0.10, 2.0, 0.20], [1.5, 0.15, 2.5, 0.25], [3.0, 0.30, 4.0, 0.40]]
    assert stub.restart_collocation_points.tolist() == [[0.07], [0.09], [0.11]]


def test_rebuild_ignores_identical_repeats_from_tables_written_up_to_1_9_0(tmp_path):
    """Result folders that already exist stay usable: their tables repeat each run, row for row."""
    stub = _stub(tmp_path, ["TKE"])
    path = _write_detailed(stub.restart_data_folder, TWO_RUNS)
    with open(path) as f:
        header, *body = f.read().splitlines()
    with open(path, "w") as f:
        f.write("\n".join([header] + body[:2] + body) + "\n")  # run 0 stored twice

    me, cp = OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)

    assert me.tolist() == [[0.10, 0.20], [0.15, 0.25]]
    assert cp.tolist() == [[0.07], [0.09]]


def test_rebuild_still_rejects_repeats_that_disagree(tmp_path):
    """Only identical repeats are dropped. Two different row sets for one run are a real conflict."""
    stub = _stub(tmp_path, ["TKE"])
    path = _write_detailed(stub.restart_data_folder, TWO_RUNS + [(0.07, [{"TKE": 9.0}, {"TKE": 9.9}])])
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(text.replace("\n2,", "\n0,"))  # relabel the third run as run 0

    with pytest.raises(ValueError, match="expected nloc=2"):
        OpenFOAMModel._rebuild_model_evaluations_from_detailed(stub)


def test_delft3d_binding_can_reselect_its_calibration_quantities(tmp_path):
    """Delft3D stores every extraction quantity per run through the same base class, so it shares the re-selection."""
    from hydroBayesCal.delft3d.control_delft3d import Delft3DModel

    stub = _stub(tmp_path, ["U_x"])
    _simulate(stub, TWO_RUNS)

    stub.calibration_quantities = ["TKE"]
    me = Delft3DModel.output_processing(
        stub, output_data_path=f"{stub.restart_data_folder}/initial-model-outputs.json")

    assert me.tolist() == [[0.10, 0.20], [0.15, 0.25]]
