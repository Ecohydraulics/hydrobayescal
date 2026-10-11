"""Tests for choosing the OpenFOAM calibration quantities after the runs.

``run_multiple_simulations`` writes every quantity in
``OpenFOAMModel.EXTRACTABLE_QUANTITIES`` to ``results-detailed-*.csv`` for every
run x control point, and ``_mirror_detailed_results_for_reselection`` copies that
table to ``restart_data/detailed-model-outputs.csv``. ``output_processing`` then
rebuilds the surrogate training matrix for whatever ``calibration_quantities`` is
configured now, so ``--only_bal_mode True --calibration_quantities "..."`` needs
no new OpenFOAM runs.

Exercised through the methods directly with a stub carrying only the attributes
they read, so the tests need no solver stack and no trained surrogate.
"""
import csv
import json
import os

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


def test_mirror_copies_the_detailed_table_to_restart_data(tmp_path):
    stub = _stub(tmp_path, ["U_x", "TKE"])
    src = f"{stub.calibration_folder}/results-detailed-U_x_TKE.csv"
    with open(src, "w", newline="") as f:
        f.write("run_idx,U_x,TKE\n0,1.0,0.1\n")

    OpenFOAMModel._mirror_detailed_results_for_reselection(stub)

    dst = f"{stub.restart_data_folder}/{OpenFOAMModel.RESELECTION_DETAILED_FILE}"
    with open(dst) as f:
        assert f.read() == "run_idx,U_x,TKE\n0,1.0,0.1\n"


def test_mirror_is_a_noop_when_there_is_no_detailed_table(tmp_path):
    stub = _stub(tmp_path, ["U_x"])
    OpenFOAMModel._mirror_detailed_results_for_reselection(stub)  # must not raise
    dst = f"{stub.restart_data_folder}/{OpenFOAMModel.RESELECTION_DETAILED_FILE}"
    assert not os.path.exists(dst)
