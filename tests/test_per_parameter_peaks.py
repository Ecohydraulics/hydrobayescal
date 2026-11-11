"""The per-parameter posterior peaks (1.13.0): ties, resampling and the default decision.

Synthetic posteriors only; seconds to run.
"""
import numpy as np
import pytest

from hydroBayesCal.surrogate.posterior_analysis import (
    CALIBRATED_ESTIMATE,
    _histogram_peak,
    analyze_posterior,
    bagged_marginal_peak,
    marginal_optima,
    record_iteration,
)

BOUNDS_UNIT = [[0.0, 1.0], [0.0, 1.0]]


def test_adjacent_tied_bins_are_one_plateau():
    # two bins of 3 samples each, side by side: the peak is the mean of all six, not of
    # the lower bin (argmax took the first tied bin up to 1.12.2)
    column = np.array([0.05, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.75])
    counts, edges = np.histogram(column, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.8])
    assert counts.tolist() == [1, 3, 3, 0, 1]
    assert _histogram_peak(column, counts, edges) == pytest.approx(
        np.mean([0.12, 0.15, 0.18, 0.22, 0.25, 0.28]))


def test_of_separate_tied_bins_the_one_nearest_the_median_is_taken():
    column = np.array([0.11, 0.12, 0.13, 0.45, 0.46, 0.47, 0.48, 0.71, 0.72, 0.73])
    counts, edges = np.histogram(column, bins=[0.0, 0.2, 0.4, 0.6, 0.8])
    assert counts.tolist() == [3, 0, 4, 3]
    # one clear maximum: unchanged rule
    assert _histogram_peak(column, counts, edges) == pytest.approx(0.465)
    column = np.array([0.11, 0.12, 0.13, 0.45, 0.46, 0.71, 0.72, 0.73, 0.74])
    counts, edges = np.histogram(column, bins=[0.0, 0.2, 0.4, 0.6, 0.8])
    counts = np.array([3, 0, 2, 3])
    # bins 0 and 3 tie; the median 0.46 is nearer the upper one
    assert _histogram_peak(column, counts, edges) == pytest.approx(np.mean([0.71, 0.72, 0.73, 0.74]))


def test_the_peak_of_a_flat_topped_marginal_does_not_depend_on_where_the_bins_start():
    """A plateau from 0.30 to 0.36 with tapering flanks, the shape of the zone 3 posterior
    of the inn-KB15 trial: the bagged peak stays near the middle of the plateau for every
    sample of it, a single histogram jumps across it."""
    rng = np.random.default_rng(3)
    peaks, singles = [], []
    for _ in range(12):
        column = np.concatenate([rng.uniform(0.30, 0.36, 300), rng.normal(0.30, 0.04, 70),
                                 rng.normal(0.36, 0.03, 70)])
        peak, spread = bagged_marginal_peak(column)
        single, _ = bagged_marginal_peak(column, n_resamples=0)
        peaks.append(peak)
        singles.append(single)
        assert np.isfinite(spread) and spread > 0
    assert np.std(peaks) < np.std(singles)
    assert 0.31 < np.mean(peaks) < 0.35


def test_the_bagged_peak_is_reproducible():
    column = np.random.default_rng(1).normal(0.4, 0.05, 500)
    assert bagged_marginal_peak(column) == bagged_marginal_peak(column)


def test_a_parameter_pinned_at_its_bound_stays_pinned():
    rng = np.random.default_rng(2)
    pinned = 1.0 - np.abs(rng.normal(0.0, 0.03, 2000))
    free = rng.normal(0.5, 0.05, 2000)
    result = marginal_optima(np.column_stack([pinned, free]), prior_bounds=BOUNDS_UNIT,
                             parameter_names=["a", "b"])
    assert "pinned_at_upper_bound" in result["flags"][0]
    assert result["peak"][0] > 0.95
    assert np.all(np.isfinite(result["peak_spread"]))


def test_the_default_calibrated_set_is_the_per_parameter_peaks():
    assert CALIBRATED_ESTIMATE == "marginal"
    rng = np.random.default_rng(4)
    # independent, but the joint and the marginal maxima differ by more than the old
    # gap tolerance of 0.25 posterior standard deviations (a skewed marginal)
    posterior = np.column_stack([rng.gamma(2.0, 0.08, 6000) + 0.05,
                                 rng.normal(0.6, 0.05, 6000)])
    posterior = posterior[(posterior >= 0).all(1) & (posterior <= 1).all(1)]
    analysis = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)
    decision = analysis["decision"]
    assert decision["source"] == "marginal-peak vector"
    assert np.allclose(decision["vector"], analysis["marginal"]["peak"])
    assert [label for label, _ in decision["must_verify"]] == ["marginal_peak", "joint_map"]
    joint = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                              prior_bounds=BOUNDS_UNIT, estimate="joint")["decision"]
    assert joint["source"] != "marginal-peak vector" or np.allclose(
        joint["vector"], decision["vector"])


def test_a_ridge_posterior_by_default_reports_the_peaks_with_a_warning():
    rng = np.random.default_rng(5)
    t = rng.uniform(0.0, 1.0, 8000)
    posterior = np.column_stack([0.15 + 0.7 * t + rng.normal(0, 0.012, 8000),
                                 0.85 - 0.7 * t + rng.normal(0, 0.012, 8000)])
    decision = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)["decision"]
    assert decision["source"] == "marginal-peak vector"
    assert "Check it" in decision["message"]
    assert "joint_map" in [label for label, _ in decision["must_verify"]]


def test_an_unknown_estimate_is_refused():
    posterior = np.random.default_rng(6).normal(0.5, 0.05, (2000, 2))
    with pytest.raises(ValueError, match="estimate"):
        analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                          prior_bounds=BOUNDS_UNIT, estimate="mean")


def test_the_spread_of_the_peaks_is_recorded_per_iteration():
    posterior = np.random.default_rng(7).multivariate_normal([0.3, 0.6], np.eye(2) * 0.0025, 2000)
    bayesian_dict = {"posterior": [posterior]}
    record_iteration(bayesian_dict, 0, posterior, parameter_names=["a", "b"],
                     prior_bounds=BOUNDS_UNIT, log=False)
    spread = bayesian_dict["marginal_peak_spread"][0]
    assert spread.shape == (2,) and np.all(spread > 0)
