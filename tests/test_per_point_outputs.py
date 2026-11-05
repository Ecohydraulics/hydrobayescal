"""Tests for ``initial-model-outputs.json`` in the per-point layout of the TELEMAC binding, written and read by the base class for OpenFOAM and Delft3D."""
import json

import numpy as np
import pandas as pd
import pytest

from hydroBayesCal.hysim import HydroSimulations

QUANTITIES = ("U_x", "U_MAG", "TKE")


def _stub(tmp_path, calibration_quantities):
    calibration = tmp_path / "calibration-data" / "_".join(calibration_quantities)
    restart = tmp_path / "restart_data"
    calibration.mkdir(parents=True, exist_ok=True)
    restart.mkdir(parents=True, exist_ok=True)
    methods = ("_point_ids", "_per_point_outputs", "_load_per_point_outputs", "_load_stored_outputs",
               "_rebuild_model_evaluations_from_detailed")
    return type("Stub", (), {
        "calibration_folder": str(calibration),
        "restart_data_folder": str(restart),
        "calibration_parameters": ["Cmu"],
        "calibration_quantities": list(calibration_quantities),
        "calibration_pts_df": pd.DataFrame({"Point": ["P1", "P2"], "X": [0.0, 1.0], "Y": [0.0, 0.0]}),
        "nloc": 2,
        "model_evaluations": None,
        "restart_collocation_points": None,
        "RESELECTION_DETAILED_FILE": HydroSimulations.RESELECTION_DETAILED_FILE,
        **{name: getattr(HydroSimulations, name) for name in methods},
    })()


def _value(run_idx, point, quantity):
    return 100.0 * run_idx + 10.0 * point + QUANTITIES.index(quantity)


def _save_runs(stub, cmus, failed=()):
    """Save the way the bindings do: all rows of the block after every run."""
    block = []
    for run_idx, cmu in enumerate(cmus):
        for point in range(2):
            row = {"run_idx": run_idx, "Cmu": cmu, "x": float(point), "y": 0.0, "z": 0.0}
            row.update({q: np.nan if run_idx in failed else _value(run_idx, point, q) for q in QUANTITIES})
            block.append(row)
        stub.model_evaluations = np.array(
            [[_value(r, p, q) for p in range(2) for q in stub.calibration_quantities] for r in range(run_idx + 1)])
        HydroSimulations._save_all_results(stub, np.array(cmus[:run_idx + 1]).reshape(-1, 1), list(block))


def test_the_outputs_are_stored_per_point_with_every_quantity(tmp_path):
    stub = _stub(tmp_path, ["U_x"])
    _save_runs(stub, [0.07, 0.08, 0.09], failed=(1,))

    for folder in (stub.restart_data_folder, stub.calibration_folder):
        with open(f"{folder}/initial-model-outputs.json") as handle:
            stored = json.load(handle)
        assert list(stored) == ["P1", "P2"]
        assert len(stored["P2"]) == 3
        assert stored["P2"][2] == {q: _value(2, 1, q) for q in QUANTITIES}
        assert np.isnan(stored["P1"][1]["TKE"]), "a failed run is stored as NaN"

    table = pd.read_csv(f"{stub.calibration_folder}/model-results-calibration-U_x.csv")
    assert list(table.columns) == ["1_U_x", "2_U_x"] and len(table) == 3


def test_any_stored_quantity_can_become_the_calibration_target(tmp_path):
    _save_runs(_stub(tmp_path, ["U_x"]), [0.07, 0.08, 0.09])
    restart = _stub(tmp_path, ["TKE", "U_MAG"])

    outputs = restart._load_stored_outputs(f"{restart.restart_data_folder}/initial-model-outputs.json")

    expected = [[_value(r, p, q) for p in range(2) for q in ("TKE", "U_MAG")] for r in range(3)]
    assert outputs.tolist() == expected
    assert restart.restart_collocation_points.ravel().tolist() == pytest.approx([0.07, 0.08, 0.09])


def test_a_run_range_trims_outputs_and_parameters_alike(tmp_path):
    stub = _stub(tmp_path, ["U_x"])
    _save_runs(stub, [0.07, 0.08, 0.09])

    outputs = stub._load_stored_outputs(f"{stub.restart_data_folder}/initial-model-outputs.json",
                                        run_range_filtering=(2, 3))

    assert outputs[:, 0].tolist() == [_value(1, 0, "U_x"), _value(2, 0, "U_x")]
    assert stub.restart_collocation_points.ravel().tolist() == pytest.approx([0.08, 0.09])


def test_a_quantity_that_was_not_extracted_is_named(tmp_path):
    _save_runs(_stub(tmp_path, ["U_x"]), [0.07])
    restart = _stub(tmp_path, ["WATER_DEPTH"])

    with pytest.raises(ValueError, match="WATER_DEPTH"):
        restart._load_stored_outputs(f"{restart.restart_data_folder}/initial-model-outputs.json")


def test_another_calibration_points_file_is_refused(tmp_path):
    _save_runs(_stub(tmp_path, ["U_x"]), [0.07])
    restart = _stub(tmp_path, ["U_x"])
    restart.calibration_pts_df = pd.DataFrame({"Point": ["P1", "P3"], "X": [0.0, 2.0], "Y": [0.0, 0.0]})

    with pytest.raises(ValueError, match="P3"):
        restart._load_stored_outputs(f"{restart.restart_data_folder}/initial-model-outputs.json")


def test_a_flat_file_of_an_earlier_release_still_loads(tmp_path):
    stub = _stub(tmp_path, ["U_x"])
    with open(f"{stub.restart_data_folder}/initial-model-outputs.json", "w") as handle:
        json.dump({"collocation_points": [[0.07], [0.08]], "model_evaluations": [[1.0, 2.0], [3.0, 4.0]],
                   "calibration_parameters": ["Cmu"], "calibration_quantities": ["U_x"], "n_runs": 2}, handle)

    outputs = stub._load_stored_outputs(f"{stub.restart_data_folder}/initial-model-outputs.json")

    assert outputs.tolist() == [[1.0, 2.0], [3.0, 4.0]]
