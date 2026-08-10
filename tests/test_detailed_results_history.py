"""Tests for the shared detailed-results bookkeeping on ``HydroSimulations``.

``_save_all_results`` lives on the base class so that every binding writes its
results the same way; it used to be duplicated verbatim in the OpenFOAM and
Delft3D bindings, which is how the ``.npy`` and ``.csv`` copies of the same table
drifted apart.

The ``.csv`` is appended to across BAL iterations while the ``.npy`` is rewritten
from scratch, so the ``.npy`` has to be rebuilt from the accumulated file rather
than from the rows of the current call. It is exercised through the base class
directly, with a stub carrying only the attributes the method reads, so the test
needs no solver stack.
"""
import numpy as np
import pytest

from hydroBayesCal.hysim import HydroSimulations


@pytest.fixture
def saver(tmp_path):
    """A minimal stand-in exposing exactly what _save_all_results reads."""
    calibration = tmp_path / "calibration-data"
    restart = tmp_path / "restart_data"
    calibration.mkdir()
    restart.mkdir()

    stub = type("Stub", (), {
        "calibration_folder": str(calibration),
        "restart_data_folder": str(restart),
        "calibration_parameters": ["ks"],
        "calibration_quantities": ["U_x"],
        "model_evaluations": np.zeros((1, 1)),
    })()
    return stub


def _rows(run_idx, n=2):
    """Detailed-results rows as a binding would build them, one per point."""
    return [{"run": float(run_idx), "point": float(p), "U_x": float(run_idx) + p / 10}
            for p in range(n)]


def _save(stub, run_idx, n_runs, rows):
    stub.model_evaluations = np.zeros((n_runs, 1))
    HydroSimulations._save_all_results(
        stub, np.arange(n_runs, dtype=float).reshape(n_runs, 1), rows
    )


def test_npy_keeps_the_full_history_across_calls(saver):
    """Later calls must not truncate the .npy to the rows of that call alone.

    ``run_multiple_simulations`` calls this once per batch, so a BAL run appends
    a handful of rows at a time. Rebuilding the array from only the new rows left
    the .npy holding the last batch while the .csv held everything.
    """
    _save(saver, 0, 1, _rows(0))
    _save(saver, 1, 2, _rows(1))
    _save(saver, 2, 3, _rows(2))

    arr = np.load(f"{saver.calibration_folder}/results-detailed-U_x.npy")

    assert len(arr) == 6, "expected 3 calls x 2 points of accumulated history"
    assert list(arr["run"]) == [0, 0, 1, 1, 2, 2]
    assert arr["U_x"][-1] == pytest.approx(2.1)


def test_npy_and_csv_agree(saver):
    """The two copies of the table are written from the same rows, in the same order."""
    _save(saver, 0, 1, _rows(0))
    _save(saver, 1, 2, _rows(1))

    arr = np.load(f"{saver.calibration_folder}/results-detailed-U_x.npy")
    with open(f"{saver.calibration_folder}/results-detailed-U_x.csv") as f:
        csv_lines = f.read().splitlines()

    assert csv_lines[0] == "run,point,U_x"          # header written once
    assert len(csv_lines) - 1 == len(arr)
    assert [float(line.split(",")[0]) for line in csv_lines[1:]] == list(arr["run"])


def test_restart_outputs_are_written_for_only_bal_mode(saver):
    """only_bal_mode reloads initial-model-outputs.json from restart_data."""
    _save(saver, 0, 1, _rows(0))

    import json
    with open(f"{saver.restart_data_folder}/initial-model-outputs.json") as f:
        restart = json.load(f)

    assert restart["n_runs"] == 1
    assert restart["calibration_parameters"] == ["ks"]
    assert restart["calibration_quantities"] == ["U_x"]


def test_restart_collocation_points_csv_is_written(saver):
    """``__init__`` loads this file when only_bal_mode is set, so it must exist.

    Only the TELEMAC binding ever wrote it, so ``only_bal_mode=True`` died with a
    FileNotFoundError for OpenFOAM and Delft3D before a single BAL iteration ran.
    Writing it here gives every binding the restart path.
    """
    _save(saver, 0, 3, _rows(0))

    path = f"{saver.restart_data_folder}/initial-collocation-points.csv"
    points = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)

    assert points.shape == (3, 1)
    with open(path) as f:
        assert f.readline().strip() == "ks"      # header names the parameters


def test_the_initial_design_stays_in_the_leading_rows(saver):
    """The file is rewritten each call, and the reader takes max_rows=init_runs.

    During BAL it therefore holds the accumulated set, not the initial design
    alone. That is only safe while the initial points remain the leading rows, so
    a restart with init_runs=2 still reads the design it started from.
    """
    initial = np.array([[10.0], [20.0]])
    HydroSimulations._save_all_results(saver, initial, None)

    saver.model_evaluations = np.zeros((4, 1))
    accumulated = np.vstack([initial, [[30.0], [40.0]]])
    HydroSimulations._save_all_results(saver, accumulated, None)

    path = f"{saver.restart_data_folder}/initial-collocation-points.csv"
    restart = np.loadtxt(path, delimiter=",", skiprows=1, max_rows=2, ndmin=2)

    assert restart.tolist() == [[10.0], [20.0]]
