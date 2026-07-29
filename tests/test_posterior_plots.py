"""Tests for the per-iteration posterior diagnostic plots.

The figures themselves are only smoke-tested (LaTeX rendering is a system
dependency); what is pinned here is the series assembly, in particular that a
``BAL_dictionary.pkl`` written before the per-iteration keys existed still yields
the full trace by reconstructing it from the stored posteriors.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("seaborn")

from hydroBayesCal.surrogate.posterior_analysis import record_iteration
from hydroBayesCal.visualize.posterior_plots import PosteriorPlots

RNG = np.random.default_rng(4711)
NAMES = ["zone2", "zone3"]
BOUNDS = [[0.0, 1.0], [0.0, 1.0]]


def _bal_dictionary(n_iterations=4, with_keys=True, drop_first=True):
    """A minimal BAL dictionary with a posterior that sharpens over the iterations."""
    posteriors, dictionary = [], {}
    for it in range(n_iterations):
        if drop_first and it == 0:
            posteriors.append(None)  # rejection sampling accepted nothing
            continue
        spread = 0.12 / (it + 1)
        posteriors.append(np.clip(
            RNG.multivariate_normal([0.3, 0.6], np.eye(2) * spread ** 2, 1500),
            0.0, 1.0))

    dictionary["posterior"] = posteriors
    dictionary["prior"] = RNG.uniform(size=(4000, 2))
    dictionary["N_tp"] = np.arange(10, 10 + n_iterations, dtype=float)

    if with_keys:
        for it, posterior in enumerate(posteriors):
            if posterior is None:
                continue
            record_iteration(dictionary, it, posterior, prior=dictionary["prior"],
                             parameter_names=NAMES, prior_bounds=BOUNDS, log=False)
    return dictionary


def test_iteration_series_uses_stored_keys():
    dictionary = _bal_dictionary(with_keys=True)
    series = PosteriorPlots._iteration_series(dictionary, NAMES, BOUNDS)

    assert series["peak"].shape == (3, 2)          # iteration 0 has no posterior
    assert series["hdi"].shape == (3, 2, 2)
    assert np.allclose(series["n_tp"], [11.0, 12.0, 13.0])
    assert np.all(np.isfinite(series["density_percentile"]))


def test_iteration_series_rebuilds_from_a_legacy_dictionary():
    """Result files written before the per-iteration keys existed must still plot."""
    legacy = _bal_dictionary(with_keys=False)
    assert "marginal_optima" not in legacy

    series = PosteriorPlots._iteration_series(legacy, NAMES, BOUNDS)

    assert series["peak"].shape == (3, 2)
    assert np.allclose(series["peak"], [0.3, 0.6], atol=0.12)
    assert np.all(np.isfinite(series["variance_reduction"]))


def test_iteration_series_matches_between_stored_and_rebuilt():
    stored = PosteriorPlots._iteration_series(_bal_dictionary(with_keys=True), NAMES, BOUNDS)
    rebuilt = PosteriorPlots._iteration_series(_bal_dictionary(with_keys=False), NAMES, BOUNDS)
    assert stored["peak"].shape == rebuilt["peak"].shape


def test_iteration_series_raises_without_any_posterior():
    with pytest.raises(ValueError, match="No iteration"):
        PosteriorPlots._iteration_series({"posterior": [None, None]}, NAMES, BOUNDS)


class _Plotter(PosteriorPlots):
    def __init__(self, folder):
        self.save_folder = folder


def test_plots_are_written(tmp_path):
    matplotlib.pyplot.rcParams.update({"text.usetex": False})
    dictionary = _bal_dictionary()
    plotter = _Plotter(tmp_path)

    plotter.plot_parameter_optimum_convergence(dictionary, NAMES, param_values=BOUNDS)
    plotter.plot_marginal_vs_joint(dictionary, NAMES, param_values=BOUNDS)

    written = {path.name for path in tmp_path.iterdir()}
    assert "parameter_optimum_convergence.png" in written
    assert "parameter_optimum_convergence_variance_reduction.png" in written
    assert "marginal_vs_joint.png" in written
