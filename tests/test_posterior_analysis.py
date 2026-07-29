"""Tests for :mod:`hydroBayesCal.surrogate.posterior_analysis`.

All of these run on synthetic posteriors in seconds: no solver, no surrogate and no
GP training. The load-bearing case is
``test_equifinality_inconsistent_for_anticorrelated_pair``, which pins the behaviour
that motivated the module: per-parameter marginal peaks that are individually
correct but jointly implausible.
"""
import numpy as np
import pytest

from hydroBayesCal.surrogate.posterior_analysis import (
    analyze_posterior,
    marginal_bin_count,
    assemble_candidates,
    detect_posterior_modes,
    equifinality_diagnostic,
    joint_optimum,
    marginal_optima,
    record_iteration,
    select_posterior_iteration,
    track_iteration,
    write_candidate_report,
    write_user_collocation_points,
)

RNG = np.random.default_rng(20260728)
BOUNDS_UNIT = [[0.0, 1.0], [0.0, 1.0]]


def _truncated_normal(mean, std, low, high, size):
    """Draw from a normal truncated to [low, high] by rejection."""
    out = np.empty(0)
    while out.size < size:
        draw = RNG.normal(mean, std, size=2 * size)
        out = np.concatenate([out, draw[(draw >= low) & (draw <= high)]])
    return out[:size]


# ---------------------------------------------------------------------------
# marginal optima
# ---------------------------------------------------------------------------
def test_marginal_peak_recovers_known_mode():
    true_modes = [0.30, 0.70, 0.15]
    bounds = [[0.0, 1.0]] * 3
    posterior = np.column_stack([
        _truncated_normal(mode, 0.05, 0.0, 1.0, 50000) for mode in true_modes])

    result = marginal_optima(posterior, prior_bounds=bounds,
                             parameter_names=["a", "b", "c"])

    assert np.allclose(result["peak"], true_modes, atol=0.02)
    assert result["verdict"] == "well_identified"


def test_derived_bin_count_beats_a_fixed_ten_bin_histogram():
    """Why the bin rule was changed: 10 bins quantise the optimum to a tenth of the range."""
    true_mode = 0.34
    posterior = _truncated_normal(true_mode, 0.03, 0.0, 1.0, 20000).reshape(-1, 1)

    derived_peak = marginal_optima(posterior, prior_bounds=[[0.0, 1.0]])["peak"][0]

    counts, edges = np.histogram(posterior[:, 0], bins=10)
    index = int(np.argmax(counts))
    in_bin = posterior[:, 0][(posterior[:, 0] >= edges[index])
                             & (posterior[:, 0] <= edges[index + 1])]
    fixed_ten_peak = float(np.mean(in_bin))

    assert abs(derived_peak - true_mode) < 0.02
    assert abs(derived_peak - true_mode) < abs(fixed_ten_peak - true_mode)


def test_bin_count_rule_scales_with_sample_size_and_spread():
    rng = np.random.default_rng(3)
    narrow = rng.normal(0.5, 0.01, 20000)
    broad = rng.normal(0.5, 0.20, 20000)
    small = rng.normal(0.5, 0.05, 300)

    # More samples in the same spread resolve the peak more finely.
    assert marginal_bin_count(rng.normal(0.5, 0.05, 20000)) > marginal_bin_count(small)
    # The occupancy cap holds: never more than n / samples_per_bin bins.
    assert marginal_bin_count(small) <= max(8, small.size // 25)
    # A floor always applies, even for a degenerate spread.
    assert marginal_bin_count(np.full(500, 0.3)) >= 8
    # Bin width tracks the spread, so both give a comparable number of bins.
    assert 0.2 < marginal_bin_count(narrow) / marginal_bin_count(broad) < 5.0


def test_peak_is_read_off_the_samples_not_a_smoothed_curve():
    """Against a bound, a symmetric kernel would pull the apparent peak inward."""
    rng = np.random.default_rng(5)
    pinned = np.clip(rng.normal(1.08, 0.06, 20000), 0.0, 1.0).reshape(-1, 1)

    result = marginal_optima(pinned, prior_bounds=[[0.0, 1.0]])

    assert result["peak"][0] > 0.98
    assert "pinned_at_upper_bound" in result["flags"][0]


def test_identifiability_flags():
    n = 20000
    bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    posterior = np.column_stack([
        RNG.uniform(0.0, 1.0, n),                               # unconstrained
        _truncated_normal(1.0, 0.05, 0.0, 1.0, n),              # pinned at the top
        _truncated_normal(0.5, 0.02, 0.0, 1.0, n),              # sharp and interior
    ])

    result = marginal_optima(posterior, prior_bounds=bounds,
                             parameter_names=["flat", "pinned", "sharp"])

    assert "non_identifiable" in result["flags"][0]
    assert result["variance_reduction"][0] < 0.10
    assert "pinned_at_upper_bound" in result["flags"][1]
    assert result["flags"][2] == []
    assert result["verdict"] == "partially_identified"


def test_bimodal_marginal_detected():
    n = 20000
    column = np.concatenate([_truncated_normal(0.2, 0.03, 0.0, 1.0, n // 2),
                             _truncated_normal(0.8, 0.03, 0.0, 1.0, n // 2)])
    result = marginal_optima(column.reshape(-1, 1), prior_bounds=[[0.0, 1.0]])

    assert result["n_marginal_modes"][0] == 2
    assert "multimodal_marginal" in result["flags"][0]


def test_low_posterior_sample_dominates_the_verdict():
    posterior = _truncated_normal(0.5, 0.05, 0.0, 1.0, 40).reshape(-1, 1)
    result = marginal_optima(posterior, prior_bounds=[[0.0, 1.0]])
    assert result["verdict"] == "low_posterior_sample"


def test_degenerate_column_is_flagged_not_fatal():
    posterior = np.column_stack([np.full(500, 0.4),
                                 _truncated_normal(0.5, 0.05, 0.0, 1.0, 500)])
    result = marginal_optima(posterior, prior_bounds=BOUNDS_UNIT)
    assert "degenerate_marginal" in result["flags"][0]
    assert np.isfinite(result["peak"]).all()


# ---------------------------------------------------------------------------
# equifinality: the core of the module
# ---------------------------------------------------------------------------
def _anticorrelated_ridge(n=20000):
    """Thin anti-diagonal ridge whose two marginals peak in the empty centre.

    Two equally weighted lobes at opposite ends of the ridge give each marginal a
    peak near 0.5, but the joint posterior has almost no mass there.
    """
    half = n // 2
    t1 = RNG.normal(0.25, 0.04, half)
    t2 = RNG.normal(0.75, 0.04, half)
    t = np.concatenate([t1, t2])
    noise = RNG.normal(0.0, 0.01, n)
    return np.column_stack([t + noise, 1.0 - t + noise])


def test_equifinality_inconsistent_for_anticorrelated_pair():
    posterior = _anticorrelated_ridge()
    marginal = marginal_optima(posterior, prior_bounds=BOUNDS_UNIT,
                               parameter_names=["friction", "shields"])
    joint = joint_optimum(posterior, method="knn")

    # Force the pathological combination: each marginal is symmetric about 0.5, so
    # the component-wise "optimum" sits in the middle of the empty ridge.
    marginal_peak_vector = np.array([0.5, 0.5])
    result = equifinality_diagnostic(
        posterior, marginal_peak_vector, joint_optimum_vector=joint["vector"],
        density=joint["density"], sample_log_density=joint["log_density"],
        parameter_names=marginal["parameter_names"])

    assert result["verdict"] == "inconsistent"
    assert result["density_percentile"] < 10.0
    assert result["max_abs_correlation"] > 0.6
    assert "friction" in result["message"] and "shields" in result["message"]


def test_equifinality_consistent_for_independent_parameters():
    n = 20000
    posterior = np.column_stack([
        _truncated_normal(0.3, 0.05, 0.0, 1.0, n),
        _truncated_normal(0.6, 0.05, 0.0, 1.0, n)])
    marginal = marginal_optima(posterior, prior_bounds=BOUNDS_UNIT)
    joint = joint_optimum(posterior, method="knn")
    result = equifinality_diagnostic(
        posterior, marginal["peak"], joint_optimum_vector=joint["vector"],
        density=joint["density"], sample_log_density=joint["log_density"])

    assert result["verdict"] == "consistent"
    assert result["max_abs_correlation"] < 0.3
    assert np.allclose(marginal["peak"], joint["vector"], atol=0.06)


# ---------------------------------------------------------------------------
# joint optimum
# ---------------------------------------------------------------------------
def test_joint_optimum_locates_a_unimodal_posterior():
    n = 20000
    posterior = np.column_stack([
        _truncated_normal(0.4, 0.05, 0.0, 1.0, n),
        _truncated_normal(0.7, 0.05, 0.0, 1.0, n)])

    auto = joint_optimum(posterior, method="auto")
    knn = joint_optimum(posterior, method="knn")["vector"]

    assert auto["method_used"] == "knn"
    assert np.allclose(auto["vector"], knn)
    assert np.allclose(knn, [0.4, 0.7], atol=0.05)


def test_joint_optimum_rejects_an_unknown_density_method():
    posterior = RNG.uniform(size=(200, 2))
    with pytest.raises(ValueError, match="Unknown joint density method"):
        joint_optimum(posterior, method="kde")


def test_joint_optimum_is_an_actual_posterior_sample():
    posterior = _anticorrelated_ridge(5000)
    result = joint_optimum(posterior, method="knn")
    assert np.allclose(posterior[result["index"]], result["vector"])


def test_joint_optimum_likelihood_method_requires_a_surrogate():
    posterior = RNG.uniform(size=(200, 2))
    with pytest.raises(ValueError, match="needs surrogate"):
        joint_optimum(posterior, method="likelihood")


def test_joint_optimum_likelihood_method_with_a_stub_surrogate():
    """A linear stub whose best-fitting sample can be computed by hand."""
    posterior = RNG.uniform(0.0, 1.0, size=(500, 2))
    observations = np.array([[1.0, 2.0]])
    error = np.array([0.01, 0.01])

    class _StubSurrogate:
        @staticmethod
        def predict_(input_sets):
            x = np.atleast_2d(input_sets)
            return {"output": np.column_stack([2.0 * x[:, 0], 3.0 * x[:, 1]])}

    result = joint_optimum(posterior, method="likelihood", surrogate=_StubSurrogate(),
                           observations=observations, error=error)

    residuals = np.column_stack([2.0 * posterior[:, 0] - 1.0,
                                 3.0 * posterior[:, 1] - 2.0])
    expected = int(np.argmin(np.sum(residuals ** 2 / error, axis=1)))
    assert result["index"] == expected


# ---------------------------------------------------------------------------
# posterior modes
# ---------------------------------------------------------------------------
def test_mode_detection_weights():
    n = 6000
    heavy = RNG.multivariate_normal([0.2, 0.2], np.eye(2) * 0.0016, int(0.7 * n))
    light = RNG.multivariate_normal([0.8, 0.8], np.eye(2) * 0.0016, int(0.3 * n))
    posterior = np.clip(np.vstack([heavy, light]), 0.0, 1.0)

    result = detect_posterior_modes(posterior, prior_bounds=BOUNDS_UNIT, random_state=0)

    assert result["n_modes"] == 2
    assert result["verdict"] == "multimodal_equifinality"
    assert np.allclose(np.sort(result["weights"])[::-1], [0.7, 0.3], atol=0.05)
    representatives = result["representatives"][np.argsort(result["weights"])[::-1]]
    assert np.allclose(representatives[0], [0.2, 0.2], atol=0.1)
    assert np.allclose(representatives[1], [0.8, 0.8], atol=0.1)


def test_continuous_ridge_is_one_mode_not_several():
    """A trade-off ridge is one connected family of solutions, not distinct modes.

    A Gaussian mixture needs several components to follow a curved ridge, so without
    the connectivity test this is reported as multimodal equifinality.
    """
    rng = np.random.default_rng(11)
    t = rng.uniform(0.1, 0.9, 6000)
    posterior = np.column_stack([t + rng.normal(0, 0.01, t.size),
                                 (t - 0.5) ** 2 + rng.normal(0, 0.01, t.size)])
    bounds = [[0.0, 1.0], [-0.5, 0.5]]

    result = detect_posterior_modes(posterior, prior_bounds=bounds, random_state=0)
    assert result["n_modes"] == 1
    assert result["verdict"] == "unimodal"

    # Without the connectivity test the mixture components survive as false modes.
    naive = detect_posterior_modes(posterior, prior_bounds=bounds, random_state=0,
                                   merge_connected=False, min_separation=0.0)
    assert naive["n_modes"] > 1


def test_separated_blobs_survive_the_connectivity_merge():
    """Two solutions with a density valley between them must stay separate."""
    rng = np.random.default_rng(12)
    posterior = np.vstack([
        rng.multivariate_normal([0.15, 0.15], np.eye(2) * 0.0009, 3000),
        rng.multivariate_normal([0.85, 0.85], np.eye(2) * 0.0009, 3000)])
    result = detect_posterior_modes(posterior, prior_bounds=BOUNDS_UNIT, random_state=0)

    assert result["n_modes"] == 2
    assert result["verdict"] == "multimodal_equifinality"


def test_mode_representative_is_a_posterior_sample():
    n = 4000
    posterior = np.vstack([
        RNG.multivariate_normal([0.2, 0.2], np.eye(2) * 0.0016, n // 2),
        RNG.multivariate_normal([0.8, 0.8], np.eye(2) * 0.0016, n // 2)])
    result = detect_posterior_modes(posterior, prior_bounds=BOUNDS_UNIT)
    for representative in result["representatives"]:
        assert np.isclose(np.abs(posterior - representative).sum(axis=1).min(), 0.0)


def test_unconstrained_parameter_does_not_manufacture_modes():
    """A parameter the data say nothing about must not split the mode count.

    Two genuine modes in the identified parameters, plus one uniform parameter: a
    mixture fitted on all columns happily halves the uniform spread and reports four
    modes that differ in nothing meaningful.
    """
    n = 6000
    identified = np.vstack([
        RNG.multivariate_normal([0.2, 0.3], np.eye(2) * 0.0009, n // 2),
        RNG.multivariate_normal([0.8, 0.7], np.eye(2) * 0.0009, n // 2)])
    unconstrained = RNG.uniform(0.0, 1.0, n).reshape(-1, 1)
    posterior = np.clip(np.hstack([identified, unconstrained]), 0.0, 1.0)
    bounds = [[0.0, 1.0]] * 3

    naive = detect_posterior_modes(posterior, prior_bounds=bounds, random_state=0)
    masked = detect_posterior_modes(posterior, prior_bounds=bounds, random_state=0,
                                    relevant_mask=np.array([True, True, False]))

    assert masked["n_modes"] == 2
    assert masked["n_modes"] <= naive["n_modes"]

    # analyze_posterior derives the mask from the identifiability flags itself.
    analysis = analyze_posterior(posterior=posterior, prior_bounds=bounds,
                                 parameter_names=["a", "b", "unconstrained"])
    assert analysis["modes"]["n_modes"] == 2


def test_mode_detection_skipped_for_small_samples():
    posterior = RNG.uniform(size=(30, 2))
    result = detect_posterior_modes(posterior, prior_bounds=BOUNDS_UNIT)
    assert result["n_modes"] == 1
    assert "skipped" in result["message"]


# ---------------------------------------------------------------------------
# candidates and CSV contract
# ---------------------------------------------------------------------------
def _analysis_fixture():
    n = 4000
    posterior = np.vstack([
        RNG.multivariate_normal([0.2, 0.3], np.eye(2) * 0.0016, int(0.6 * n)),
        RNG.multivariate_normal([0.8, 0.7], np.eye(2) * 0.0016, int(0.4 * n))])
    posterior = np.clip(posterior, 0.0, 1.0)
    return analyze_posterior(posterior=posterior, prior_bounds=BOUNDS_UNIT,
                             parameter_names=["zone2", "zone3"])


def test_candidates_include_all_requested_kinds():
    analysis = _analysis_fixture()
    labels = analysis["candidates"]["labels"]

    assert "marginal_peak" in labels
    assert "joint_map" in labels
    assert "posterior_mean" in labels
    assert any(label.startswith("mode_") for label in labels)
    assert analysis["candidates"]["vectors"].shape[1] == 2


def test_candidates_are_clipped_to_the_calibration_range():
    marginal = {"peak": np.array([2.0, -1.0])}
    posterior = RNG.uniform(size=(500, 2))
    candidates = assemble_candidates(
        marginal, None, None, posterior, parameter_names=["a", "b"],
        include=("marginal_peak",), prior_bounds=BOUNDS_UNIT)
    assert np.all(candidates["vectors"] >= 0.0)
    assert np.all(candidates["vectors"] <= 1.0)


def test_user_collocation_csv_contract(tmp_path):
    """The file must parse with the exact call HydroSimulations.__init__ makes."""
    analysis = _analysis_fixture()
    names = ["zone2", "zone3"]
    path = write_user_collocation_points(analysis["candidates"], names, str(tmp_path))

    loaded = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    assert loaded.shape == analysis["candidates"]["vectors"].shape
    assert np.allclose(loaded, analysis["candidates"]["vectors"], atol=1e-8)

    with open(path) as handle:
        header = handle.readline().strip().split(",")
    assert header == names


def test_user_collocation_csv_single_candidate_stays_2d(tmp_path):
    candidates = {"vectors": np.array([[0.1, 0.2]])}
    path = write_user_collocation_points(candidates, ["a", "b"], str(tmp_path))
    loaded = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    assert loaded.shape == (1, 2)


def test_user_collocation_csv_backs_up_an_existing_file(tmp_path):
    candidates = {"vectors": np.array([[0.1, 0.2]])}
    path = write_user_collocation_points(candidates, ["a", "b"], str(tmp_path))
    write_user_collocation_points({"vectors": np.array([[0.3, 0.4]])}, ["a", "b"],
                                  str(tmp_path))
    backup = np.loadtxt(path + ".bak", delimiter=",", skiprows=1, ndmin=2)
    assert np.allclose(backup, [[0.1, 0.2]])


def test_user_collocation_csv_rejects_a_width_mismatch(tmp_path):
    with pytest.raises(ValueError, match="calibration parameters"):
        write_user_collocation_points({"vectors": np.zeros((2, 3))}, ["a", "b"],
                                      str(tmp_path))


def test_candidate_report_files(tmp_path):
    analysis = _analysis_fixture()
    candidates_path, diagnostics_path = write_candidate_report(analysis, str(tmp_path))

    with open(candidates_path) as handle:
        rows = handle.read().strip().splitlines()
    assert len(rows) == 1 + len(analysis["candidates"]["labels"])

    with open(diagnostics_path) as handle:
        rows = handle.read().strip().splitlines()
    assert len(rows) == 1 + 2  # header plus one row per parameter


# ---------------------------------------------------------------------------
# per-iteration tracking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, np.zeros((0, 2)), np.array([[0.1, 0.2]]),
                                 np.column_stack([np.full(50, 0.3), np.full(50, 0.4)])])
def test_track_iteration_never_raises(bad):
    summary = track_iteration(bad, parameter_names=["a", "b"], prior_bounds=BOUNDS_UNIT)
    assert set(summary) == {"peak", "hdi", "variance_reduction", "flags", "gap",
                            "post_size"}


def test_record_iteration_populates_additive_keys():
    posterior = RNG.multivariate_normal([0.3, 0.6], np.eye(2) * 0.0025, 2000)
    bayesian_dict = {"posterior": [posterior, None]}

    record_iteration(bayesian_dict, 0, posterior, parameter_names=["a", "b"],
                     prior_bounds=BOUNDS_UNIT)

    assert bayesian_dict["marginal_optima"][0].shape == (2,)
    assert bayesian_dict["marginal_hdi"][0].shape == (2, 2)
    assert bayesian_dict["variance_reduction"][0].shape == (2,)
    assert bayesian_dict["marginal_joint_gap"][0]["verdict"] in (
        "consistent", "acceptable", "inconsistent")
    assert bayesian_dict["calibration_parameters"] == ["a", "b"]


def test_record_iteration_on_a_legacy_dictionary_without_the_keys():
    posterior = RNG.multivariate_normal([0.3, 0.6], np.eye(2) * 0.0025, 500)
    bayesian_dict = {"posterior": [posterior]}
    record_iteration(bayesian_dict, 0, posterior, prior_bounds=BOUNDS_UNIT)
    assert "marginal_optima" in bayesian_dict


# ---------------------------------------------------------------------------
# BAL dictionary handling
# ---------------------------------------------------------------------------
def test_select_posterior_iteration_skips_empty_entries():
    good = RNG.uniform(size=(100, 2))
    bayesian_dict = {"posterior": [good, None, np.zeros((0, 2))]}
    posterior, index = select_posterior_iteration(bayesian_dict, iteration=-1)
    assert index == 0
    assert posterior.shape == (100, 2)


def test_select_posterior_iteration_raises_when_all_empty():
    with pytest.raises(ValueError, match="No valid posterior"):
        select_posterior_iteration({"posterior": [None, None]})


def test_analyze_posterior_on_a_legacy_bal_dictionary():
    """Result files written before this module existed carry none of the new keys."""
    posterior = RNG.multivariate_normal([0.3, 0.6], np.eye(2) * 0.0025, 1500)
    legacy = {
        "posterior": [None, posterior],
        "prior": RNG.uniform(size=(5000, 2)),
        "BME": np.zeros(2), "RE": np.zeros(2), "N_tp": np.zeros(2),
    }
    analysis = analyze_posterior(bayesian_dict=legacy,
                                 parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)
    assert analysis["iteration"] == 1
    assert analysis["marginal"]["peak"].shape == (2,)
    assert analysis["candidates"]["vectors"].shape[1] == 2


def test_analyze_posterior_infers_names_and_bounds_from_the_dictionary():
    posterior = RNG.multivariate_normal([0.3, 0.6], np.eye(2) * 0.0025, 1500)
    bayesian_dict = {
        "posterior": [posterior],
        "prior": RNG.uniform(size=(3000, 2)),
        "calibration_parameters": ["zone2", "zone3"],
        "param_values": [[0.0, 1.0], [0.0, 1.0]],
    }
    analysis = analyze_posterior(bayesian_dict=bayesian_dict)
    assert analysis["marginal"]["parameter_names"] == ["zone2", "zone3"]
