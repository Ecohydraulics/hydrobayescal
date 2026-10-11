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


# --------------------------------------------------------------------------- #
# the call pattern the bindings actually use
# --------------------------------------------------------------------------- #
def _binding_rows(run_idx, n=2):
    """Rows as the OpenFOAM and Delft3D bindings build them.

    Keyed by ``run_idx`` and carrying every extracted quantity, whichever of them is
    a calibration target at the moment.
    """
    return [{"run_idx": run_idx, "ks": 0.01 * (run_idx + 1), "x": float(p), "y": 0.0,
             "U_x": run_idx + p / 10, "TKE": 0.1 * run_idx + p / 100}
            for p in range(n)]


def _read_csv(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_saving_the_cumulative_rows_after_every_run_does_not_repeat_them(saver):
    """One row set per run, however often the block is saved.

    ``run_multiple_simulations`` saves after every run, so that a crash loses one run
    and not the block, and once more after the loop, each time with all rows of the
    block. Appending them blindly stored run 0 of an N-run block N+1 times, which
    inflated the table and made it impossible to match rows to calibration points.
    """
    rows = []
    for run_idx in range(3):
        rows.extend(_binding_rows(run_idx))
        _save(saver, run_idx, run_idx + 1, list(rows))
    _save(saver, 2, 3, list(rows))  # the final save after the loop

    arr = np.load(f"{saver.calibration_folder}/results-detailed-U_x.npy")
    table = _read_csv(f"{saver.calibration_folder}/results-detailed-U_x.csv")

    assert list(arr["run_idx"]) == [0, 0, 1, 1, 2, 2]
    assert [float(r["run_idx"]) for r in table] == [0, 0, 1, 1, 2, 2]
    assert arr["U_x"][-1] == pytest.approx(2.1)


def test_a_rerun_replaces_the_stored_rows_of_that_run(saver):
    """A run simulated again, e.g. after a failure, must not leave its old rows behind."""
    failed = [dict(row, U_x=float("nan"), TKE=float("nan")) for row in _binding_rows(0)]
    _save(saver, 0, 1, failed)
    _save(saver, 0, 1, _binding_rows(0))

    arr = np.load(f"{saver.calibration_folder}/results-detailed-U_x.npy")

    assert list(arr["run_idx"]) == [0, 0]
    assert np.isfinite(arr["U_x"]).all()


def test_restart_table_survives_a_change_of_calibration_targets(saver, tmp_path):
    """The table in ``restart_data`` keeps the runs made under other calibration targets.

    A different target set gets its own, initially empty calibration folder, so the
    per-target table of that folder knows only the runs simulated since the switch. The
    table in ``restart_data`` is what a re-selection of the calibration targets reads,
    and it has to hold all runs, not a copy of whichever per-target table was written
    last.
    """
    _save(saver, 1, 2, _binding_rows(0) + _binding_rows(1))

    other_folder = tmp_path / "calibration-data-TKE"
    other_folder.mkdir()
    saver.calibration_folder = str(other_folder)
    saver.calibration_quantities = ["TKE"]
    _save(saver, 2, 3, _binding_rows(2))  # one BAL run under the new targets

    restart = _read_csv(f"{saver.restart_data_folder}/detailed-model-outputs.csv")
    per_target = np.load(f"{other_folder}/results-detailed-TKE.npy")

    assert [float(r["run_idx"]) for r in restart] == [0, 0, 1, 1, 2, 2]
    assert [float(r["TKE"]) for r in restart][-1] == pytest.approx(0.21)
    assert list(per_target["run_idx"]) == [2, 2]


def test_a_quantity_extracted_only_by_later_runs_gets_its_own_column(saver):
    """Columns are merged, never shifted: earlier runs read NaN for the new quantity."""
    early = [{k: v for k, v in row.items() if k != "TKE"} for row in _binding_rows(0)]
    _save(saver, 0, 1, early)
    _save(saver, 1, 2, _binding_rows(1))

    restart = _read_csv(f"{saver.restart_data_folder}/detailed-model-outputs.csv")

    assert [float(r["U_x"]) for r in restart] == pytest.approx([0.0, 0.1, 1.0, 1.1])
    assert np.isnan([float(r["TKE"]) for r in restart[:2]]).all()
    assert [float(r["TKE"]) for r in restart[2:]] == pytest.approx([0.1, 0.11])
