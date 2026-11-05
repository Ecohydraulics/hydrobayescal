"""Tests for the extraction time per calibration point (``extraction['point_time_column']``).

A small SELAFIN file is written whose WATER DEPTH equals the index of the output step (outputs every 10 s, steps 0 to 9), so the step or the window the extraction used can be read off the extracted value.
"""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tqdm")

from hydroBayesCal.telemac.config_telemac import point_time_index, point_times
from hydroBayesCal.telemac.control_telemac import TelemacModel
from hydroBayesCal.telemac.pputils.ppmodules.selafin_io_pp import ppSELAFIN

N_STEPS = 10
OUTPUT_PERIOD = 10.0


def _write_selafin(path):
    x = np.array([0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    slf = ppSELAFIN(str(path))
    slf.setPrecision("f", 4)
    slf.setTitle("hydroBayesCal test")
    slf.setVarNames(["WATER DEPTH"])
    slf.setVarUnits(["M"])
    slf.setIPARAM([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    slf.setMesh(2, 4, 3, np.array([[1, 2, 3], [1, 3, 4]]), np.array([1, 2, 3, 4]), x, y)
    slf.writeHeader()
    for step in range(N_STEPS):
        slf.writeVariables(step * OUTPUT_PERIOD, np.full((1, 4), float(step)))
    slf.close()


def _extract(tmp_path, times, **kwargs):
    model_dir, results = tmp_path / "simulation", tmp_path / "results"
    (results / "restart_data").mkdir(parents=True)
    model_dir.mkdir()
    _write_selafin(model_dir / "telemac_1.slf")

    model = TelemacModel.__new__(TelemacModel)
    model.restart_data_folder = str(results / "restart_data")
    model.asr_dir = str(results)
    model.init_runs = 5
    model.gaia_cas = None
    model.gaia_results_filename = None
    model.tm_results_filename = "telemac_1.slf"
    model.gaia_layer_average = None
    model.point_time_column = kwargs.pop("model_column", None)

    points = pd.DataFrame({"Point": ["P1", "P2"], "X": [2.0, 7.0], "Y": [3.0, 6.0], "Z": [0.0, 0.0],
                           "Sample time": times})
    model.extract_data_point("telemac_1.slf", points, "extraction", ["WATER DEPTH"], 1,
                             str(model_dir), str(results), output_extraction="nearest", **kwargs)
    with open(results / "extraction-detailed.json") as handle:
        output = json.load(handle)
    return [output[p][0]["WATER DEPTH"] for p in ("P1", "P2")]


def test_without_a_configured_column_the_extraction_is_unchanged(tmp_path):
    """A column that happens to hold dates is not read unless it is configured."""
    values = _extract(tmp_path, ["2024-05-01", "2024-06-01"], output_extraction_time="mean_last", n=3)

    assert values == pytest.approx([8.0, 8.0])


def test_mean_last_averages_the_steps_that_end_at_the_point_time(tmp_path):
    values = _extract(tmp_path, [50.0, None], output_extraction_time="mean_last", n=3,
                      point_time_column="Sample time")

    assert values == pytest.approx([4.0, 8.0]), "P1: steps 3 to 5; P2 has no time: steps 7 to 9"


def test_the_column_configured_on_the_model_is_used(tmp_path):
    values = _extract(tmp_path, [50.0, 90.0], output_extraction_time="mean_last", n=3,
                      model_column="sample TIME")

    assert values == pytest.approx([4.0, 8.0])


@pytest.mark.parametrize("mode", ["last", "index"])
def test_last_and_index_take_the_step_written_at_the_point_time(tmp_path, mode):
    values = _extract(tmp_path, ["30", ""], output_extraction_time=mode, time_index=1,
                      point_time_column="Sample time")

    assert values == pytest.approx([3.0, 9.0 if mode == "last" else 1.0])


def test_a_window_that_starts_before_the_first_output_is_refused(tmp_path):
    with pytest.raises(ValueError, match="only 2 were written"):
        _extract(tmp_path, [10.0, None], output_extraction_time="mean_last", n=3,
                 point_time_column="Sample time")


def test_a_time_without_output_is_refused_with_the_nearest_output_time(tmp_path):
    with pytest.raises(ValueError, match=r"Point P2.*t = 55.0 s.*nearest output time is 50.0 s"):
        _extract(tmp_path, [50.0, 55.0], output_extraction_time="last",
                 point_time_column="Sample time")


def test_values_that_are_not_seconds_are_refused():
    points = pd.DataFrame({"Point": ["P1", "P2", "P3"], "time": [3600, "2024-05-01", np.nan]})

    with pytest.raises(ValueError, match=r"\['P2'\]"):
        point_times(points, "time")


def test_empty_cells_are_points_without_a_time():
    points = pd.DataFrame({"Point": ["P1", "P2", "P3"], "time": [3600, "", np.nan]})

    times = point_times(points, "TIME")

    assert times[0] == 3600.0 and np.isnan(times[1:]).all()


def test_a_missing_column_is_named():
    with pytest.raises(ValueError, match="no such column"):
        point_times(pd.DataFrame({"Point": ["P1"]}), "time")


def test_single_precision_times_are_matched():
    model_times = np.arange(0, 1e6, 3600.0, dtype=np.float32)

    assert point_time_index(model_times, 7200.0, "P1") == 2
    assert point_time_index(model_times, float(model_times[200]) + 0.01, "P1") == 200


# --------------------------------------------------------------------------- #
# the standalone extraction agrees with the calibration workflow
# --------------------------------------------------------------------------- #
def test_the_standalone_extraction_takes_the_same_steps(tmp_path):
    from hydroBayesCal.extract import extract_results

    path = tmp_path / "telemac_1.slf"
    _write_selafin(path)

    df = extract_results(str(path), "WATER DEPTH", x=[2.0, 7.0], y=[3.0, 6.0], time="mean_last",
                         n_last=3, method="nearest", point_time=[50.0, np.nan])
    assert df["WATER DEPTH"].tolist() == pytest.approx([4.0, 8.0])

    df = extract_results(str(path), "WATER DEPTH", x=[2.0], y=[3.0], time="index", time_index=1,
                         method="nearest", point_time=30.0)
    assert df["WATER DEPTH"].tolist() == pytest.approx([3.0])

    with pytest.raises(ValueError, match="nearest output time"):
        extract_results(str(path), "WATER DEPTH", x=[2.0], y=[3.0], point_time=55.0)


@pytest.mark.parametrize("times", [[50.0, None], [None, None]], ids=["point-time", "global"])
def test_mean_last_over_no_step_is_refused(tmp_path, times):
    """An empty window gave NaN at a point with a time, and the mean over the whole run without one."""
    with pytest.raises(ValueError, match="n_last"):
        _extract(tmp_path, times, output_extraction_time="mean_last", n=0, point_time_column="Sample time")


def test_the_standalone_extraction_refuses_mean_last_over_no_step(tmp_path):
    from hydroBayesCal.extract import extract_results

    path = tmp_path / "telemac_1.slf"
    _write_selafin(path)
    with pytest.raises(ValueError, match="n_last"):
        extract_results(str(path), "WATER DEPTH", x=[2.0], y=[3.0], time="mean_last", n_last=0, point_time=50.0)
