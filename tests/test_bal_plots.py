"""Smoke tests for the collocation-point plot of the BAL diagnostics.

The figure is only checked for being written. What is pinned here is that the plot finds its input, because the lookup of the collocation-points file referenced names the module never imported and so failed on the first call that did not pass the points in memory.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from hydroBayesCal.visualize.bal_plots import BALPlots

RNG = np.random.default_rng(1909)
N_INITIAL, N_BAL = 8, 5


class _Plotter(BALPlots):
    def __init__(self, folder):
        self.save_folder = folder


def _case(tmp_path, n_params):
    """A calibration folder with a collocation-points file and the matching BAL dictionary."""
    names = [f"zone{i + 2}" for i in range(n_params)]
    points = RNG.uniform(0.01, 0.2, size=(N_INITIAL + N_BAL, n_params))
    variable_name = "WATER DEPTH_SCALAR VELOCITY"
    np.savetxt(tmp_path / f"collocation-points-{variable_name}.csv", points, delimiter=",",
               header=",".join(names), comments="")
    dictionary = {
        "N_tp": np.arange(N_INITIAL, N_INITIAL + N_BAL + 1, dtype=float),
        "util_func": np.array(["dkl", "dkl", "bme", "dkl", "bme"], dtype=object),
        "calibration_parameters": names,
    }
    return names, points, variable_name, dictionary


@pytest.mark.parametrize("n_params", [2, 3])
def test_collocation_points_are_read_from_the_calibration_folder(tmp_path, n_params):
    matplotlib.pyplot.rcParams.update({"text.usetex": False})
    names, _, variable_name, dictionary = _case(tmp_path, n_params)

    _Plotter(tmp_path).plot_collocation_points(
        dictionary, collocation_points_folder=str(tmp_path), variable_name=variable_name,
        param_values=[[0.01, 0.2]] * n_params, show=False)

    assert any(path.suffix == ".png" for path in tmp_path.iterdir())


def test_a_differently_named_file_is_found_when_it_is_the_only_one(tmp_path):
    """The file name follows the calibration targets, and their order in a config can differ from the one used for the run."""
    matplotlib.pyplot.rcParams.update({"text.usetex": False})
    _, _, _, dictionary = _case(tmp_path, 2)

    _Plotter(tmp_path).plot_collocation_points(
        dictionary, collocation_points_folder=str(tmp_path),
        variable_name="SCALAR VELOCITY_WATER DEPTH", show=False)

    assert any(path.suffix == ".png" for path in tmp_path.iterdir())


def test_a_missing_file_is_reported_as_such(tmp_path):
    names, points, _, dictionary = _case(tmp_path, 2)
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="collocation-points"):
        _Plotter(tmp_path).plot_collocation_points(
            dictionary, collocation_points_folder=str(empty), variable_name="WATER DEPTH",
            show=False)


def test_points_given_in_memory_need_no_file(tmp_path):
    matplotlib.pyplot.rcParams.update({"text.usetex": False})
    names, points, _, dictionary = _case(tmp_path, 2)
    out = tmp_path / "plots"
    out.mkdir()

    _Plotter(out).plot_collocation_points(dictionary, collocation_points=points,
                                          color_by="iteration", show=False)

    assert any(path.suffix == ".png" for path in out.iterdir())
