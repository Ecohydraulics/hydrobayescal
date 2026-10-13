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
    # The failure path has to return the same keys as the success path: a consumer
    # reading the summary must not have to know whether the diagnostic worked.
    assert set(summary) == {"peak", "hdi", "variance_reduction", "flags", "gap",
                            "joint", "joint_log_density", "n_modes", "post_size"}
    assert summary["n_modes"] == 1


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
    assert bayesian_dict["joint_optimum"][0].shape == (2,)
    assert np.isfinite(bayesian_dict["joint_log_density"][0])
    assert bayesian_dict["posterior_modes"][0] >= 1
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


# ---------------------------------------------------------------------------
# global maximum of the joint posterior
# ---------------------------------------------------------------------------
class _BimodalSurrogate:
    """Emulator whose likelihood has two maxima, one deeper than the other.

    The deep one sits at (0.25, 0.75) and reproduces the observations exactly; the
    shallow one at (0.75, 0.25) does not. The accepted sample below is deliberately
    dominated by the *shallow* mode, so the highest-density sample is in the wrong basin
    and only a mode-seeded search can find the right one.
    """

    deep = np.array([0.25, 0.75])
    shallow = np.array([0.75, 0.25])

    def predict_(self, input_sets, **kwargs):
        points = np.atleast_2d(input_sets)
        near_deep = np.exp(-np.sum((points - self.deep) ** 2, axis=1) / (2 * 0.02 ** 2))
        near_shallow = 0.7 * np.exp(
            -np.sum((points - self.shallow) ** 2, axis=1) / (2 * 0.02 ** 2))
        response = near_deep + near_shallow
        output = np.column_stack([response, 2 * response, 0.5 * response])
        return {"output": output, "std": np.full_like(output, 0.01)}


def _bimodal_posterior():
    return np.vstack([
        RNG.normal(_BimodalSurrogate.deep, 0.02, (120, 2)),
        RNG.normal(_BimodalSurrogate.shallow, 0.02, (400, 2)),
    ])


def test_refinement_seeded_by_the_modes_finds_the_global_maximum():
    """Why the refinement exists: the densest sample can be in the wrong basin."""
    posterior = _bimodal_posterior()
    observations = np.array([[1.0, 2.0, 0.5]])
    error = np.full(3, 0.02 ** 2)

    analysis = analyze_posterior(
        posterior=posterior, parameter_names=["a", "b"], prior_bounds=BOUNDS_UNIT,
        surrogate=_BimodalSurrogate(), observations=observations, error=error,
        refine=True)
    joint = analysis["joint"]

    plain = posterior[int(np.argmax(joint["log_density"]))]
    assert np.allclose(plain, _BimodalSurrogate.shallow, atol=0.08)  # wrong basin
    assert joint["refined"] is True
    assert np.allclose(joint["refined_vector"], _BimodalSurrogate.deep, atol=0.01)
    # Not every start agreed, which is the honest signal that there are two maxima.
    assert joint["global_agreement"] < 1.0


def test_refinement_without_a_surrogate_falls_back_and_says_so():
    posterior = RNG.multivariate_normal([0.3, 0.6], np.eye(2) * 0.0025, 800)
    result = joint_optimum(posterior, refine=True)
    assert result["refined"] is False
    assert np.allclose(result["refined_vector"], result["vector"])


def test_refined_optimum_stays_inside_the_calibration_range():
    posterior = np.column_stack([_truncated_normal(0.02, 0.01, 0.0, 1.0, 500),
                                 _truncated_normal(0.98, 0.01, 0.0, 1.0, 500)])
    observations = np.array([[1.0, 2.0, 0.5]])
    analysis = analyze_posterior(
        posterior=posterior, parameter_names=["a", "b"], prior_bounds=BOUNDS_UNIT,
        surrogate=_BimodalSurrogate(), observations=observations,
        error=np.full(3, 0.02 ** 2), refine=True)
    vector = analysis["joint"]["refined_vector"]
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)


# ---------------------------------------------------------------------------
# marginal against joint: the decision rule
# ---------------------------------------------------------------------------
def test_independent_posterior_promotes_the_marginal_peaks():
    posterior = np.column_stack([_truncated_normal(0.30, 0.05, 0.0, 1.0, 8000),
                                 _truncated_normal(0.70, 0.05, 0.0, 1.0, 8000)])
    analysis = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)
    decision = analysis["decision"]

    assert analysis["equifinality"]["verdict"] == "consistent"
    assert decision["source"] == "marginal-peak vector"
    assert np.allclose(decision["vector"], [0.30, 0.70], atol=0.03)


def test_a_ridge_posterior_reports_the_joint_optimum_not_the_marginal_peaks():
    """The case the whole module exists for: individually optimal, jointly wrong."""
    t = RNG.uniform(0.0, 1.0, 8000)
    posterior = np.column_stack([0.15 + 0.7 * t + RNG.normal(0, 0.012, 8000),
                                 0.85 - 0.7 * t + RNG.normal(0, 0.012, 8000)])
    analysis = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)
    decision = analysis["decision"]

    assert analysis["equifinality"]["verdict"] in ("coupled", "inconsistent")
    assert decision["source"].endswith("joint posterior maximum")
    assert np.allclose(decision["vector"], analysis["joint"]["vector"])
    assert "marginal_peak" in [label for label, _ in decision["must_verify"]] or \
           decision["verdict"] == "inconsistent"


def test_a_multimodal_posterior_reports_no_single_parameter_set():
    posterior = np.vstack([RNG.normal([0.2, 0.2], 0.03, (4000, 2)),
                           RNG.normal([0.8, 0.8], 0.03, (4000, 2))])
    analysis = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)
    decision = analysis["decision"]

    assert analysis["modes"]["n_modes"] == 2
    assert decision["verdict"] == "equifinal"
    assert decision["vector"] is None
    assert len(decision["must_verify"]) == 2


def test_the_decision_reaches_the_candidate_report(tmp_path):
    posterior = np.column_stack([_truncated_normal(0.30, 0.05, 0.0, 1.0, 4000),
                                 _truncated_normal(0.70, 0.05, 0.0, 1.0, 4000)])
    analysis = analyze_posterior(posterior=posterior, parameter_names=["a", "b"],
                                 prior_bounds=BOUNDS_UNIT)
    candidates_path, _ = write_candidate_report(analysis, str(tmp_path))

    with open(candidates_path) as handle:
        rows = [line.split(",") for line in handle.read().strip().splitlines()]
    header = rows[0]
    assert "selected" in header
    column = header.index("selected")
    assert sum(row[column] == "calibrated" for row in rows[1:]) == 1


# ---------------------------------------------------------------------------
# posterior samples with repeated rows (posterior_sampling_method='bayesian_weighting')
# ---------------------------------------------------------------------------
def _resampled_posterior(sigma, n_prior=25000, seed=0):
    """A unimodal, well-identified posterior at (0.6, 0.4), drawn from a uniform prior in the two ways the package offers.

    Returns the prior, the weighted resample of it (rows repeat) and the rejection sample (rows are distinct).
    """
    rng = np.random.default_rng(seed)
    prior = rng.uniform(size=(n_prior, 2))
    log_weight = -0.5 * (((prior - np.array([0.6, 0.4])) / sigma) ** 2).sum(axis=1)
    weight = np.exp(log_weight - log_weight.max())
    resampled = prior[rng.choice(n_prior, size=n_prior, replace=True, p=weight / weight.sum())]
    accepted = prior[rng.uniform(size=n_prior) < weight]
    return prior, resampled, accepted


def test_effective_sample_size_of_distinct_and_of_repeated_rows():
    from hydroBayesCal.surrogate.posterior_analysis import effective_sample_size

    distinct = np.arange(12.0).reshape(6, 2)
    assert effective_sample_size(distinct) == pytest.approx(6.0)
    repeated = distinct[[0, 0, 0, 1]]  # one row three times, one row once
    assert effective_sample_size(repeated) == pytest.approx(4.0 ** 2 / (3.0 ** 2 + 1.0 ** 2))
    assert effective_sample_size(np.repeat(distinct[:1], 500, axis=0)) == pytest.approx(1.0)


def test_the_joint_density_of_a_resampled_posterior_is_not_capped_by_its_repeated_rows():
    """A row that occurs more often than there are neighbors has only itself as neighbors.

    All its neighbor distances were zero then, and its density was set to a cap that most rows shared: about 20,000 of 25,000 here. The joint optimum was a tie-break among them, and any point off the sample ranked near the bottom.
    """
    _, resampled, _ = _resampled_posterior(sigma=0.05)

    result = joint_optimum(resampled, method="knn")

    log_density = np.asarray(result["log_density"])
    assert np.all(np.isfinite(log_density))
    at_the_maximum = np.unique(resampled[log_density >= log_density.max() - 1e-9], axis=0)
    assert len(at_the_maximum) == 1, "one parameter set is the densest, not a tie of thousands"
    assert np.linalg.norm(np.asarray(result["vector"]) - np.array([0.6, 0.4])) < 0.03


@pytest.mark.parametrize("seed", range(4))
def test_marginal_peaks_of_a_resampled_unimodal_posterior_are_not_called_inconsistent(seed):
    """The equifinality check compares the density at the marginal peaks with the densities at the samples.

    With the capped densities, the vector of marginal peaks of this unimodal posterior ranked at the 2nd to 5th percentile, and the verdict was "inconsistent" for every seed.
    """
    prior, resampled, _ = _resampled_posterior(sigma=0.012, seed=seed)

    peaks = marginal_optima(resampled, prior=prior)["peak"]
    joint = joint_optimum(resampled, method="knn", prior=prior)["vector"]
    verdict = equifinality_diagnostic(resampled, peaks, joint_optimum_vector=joint)

    assert verdict["verdict"] in ("consistent", "acceptable")
    assert verdict["density_percentile"] > 10.0


@pytest.mark.parametrize("seed", range(4))
def test_the_marginals_of_a_resampled_unimodal_posterior_have_one_mode(seed):
    """The histogram of a marginal can resolve what its samples are worth, not what their rows count.

    Binned by the number of rows, the marginals of this posterior (25,000 rows that repeat about 170 parameter sets) broke up into 8 to 14 modes each.
    """
    prior, resampled, _ = _resampled_posterior(sigma=0.012, seed=seed)

    marginals = marginal_optima(resampled, prior=prior)

    assert max(marginals["n_marginal_modes"]) <= 2
    assert np.allclose(marginals["peak"], [0.6, 0.4], atol=0.03)


def test_distinct_samples_are_ranked_as_before():
    """For a rejection sample nothing changes: the density differs from the earlier estimate by a constant only."""
    from scipy.spatial import cKDTree

    from hydroBayesCal.surrogate.posterior_analysis import _DensityEstimator

    _, _, accepted = _resampled_posterior(sigma=0.1)
    estimator = _DensityEstimator(accepted)
    off_sample = np.array([[0.55, 0.45], [0.2, 0.9]])

    z = (accepted - accepted.mean(axis=0)) / accepted.std(axis=0)
    tree, k = cKDTree(z), estimator.k_neighbors + 1

    def earlier(points):
        distances, _ = tree.query((points - accepted.mean(axis=0)) / accepted.std(axis=0), k=k)
        return np.array([-2.0 * np.log(row[row > 0][-1]) for row in distances])

    assert len(np.unique(accepted, axis=0)) == len(accepted)
    difference = np.concatenate([estimator.log_density(accepted) - earlier(accepted),
                                 estimator.log_density(off_sample) - earlier(off_sample)])
    assert np.ptp(difference) < 1e-9


def test_the_sample_size_guards_count_what_a_resampled_posterior_is_worth():
    """25,000 rows that repeat about a hundred parameter sets are not 25,000 samples.

    The guards compared the number of rows with their thresholds and let such a posterior pass as if it were large.
    """
    from hydroBayesCal.surrogate.posterior_analysis import effective_sample_size

    prior, resampled, accepted = _resampled_posterior(sigma=0.012)
    effective = effective_sample_size(resampled)
    assert len(resampled) == 25000 and effective < 100

    marginals = marginal_optima(resampled, prior=prior)
    assert marginals["verdict"] == "low_posterior_sample"
    assert marginals["post_size"] == 25000
    assert marginals["effective_post_size"] == pytest.approx(effective)
    assert "independent" in marginals["message"]

    modes = detect_posterior_modes(resampled, prior=prior)
    assert modes["n_modes"] == 1 and "skipped" in modes["message"]

    # a rejection sample keeps its count
    assert marginal_optima(accepted, prior=prior)["effective_post_size"] == pytest.approx(len(accepted))


# ---------------------------------------------------------------------------
# refinement on an emulator that computes in single precision
# ---------------------------------------------------------------------------
class _SinglePrecisionSurrogate:
    """Three outputs that are linear in two parameters, evaluated in single precision like the gpytorch emulators.

    The parameters have the ranges of a friction coefficient and of a parameter in the hundreds, and the observations are reproduced exactly at ``truth``.
    """

    lower = np.array([0.02, 100.0])
    span = np.array([0.04, 200.0])
    truth = np.array([0.045, 230.0])

    def __init__(self, with_std=True):
        self.with_std = with_std

    def predict_(self, input_sets, **kwargs):
        unit = ((np.atleast_2d(input_sets) - self.lower) / self.span).astype(np.float32)
        output = np.column_stack([unit[:, 0] + unit[:, 1], unit[:, 0] - unit[:, 1],
                                  np.float32(0.5) * unit[:, 0] + np.float32(2.0) * unit[:, 1]])
        result = {"output": output.astype(float)}
        if self.with_std:
            # the emulator is the less certain, the farther the first parameter is from the middle of its range
            std = np.float32(0.01) + np.float32(0.2) * np.abs(unit[:, [0]] - np.float32(0.5)) * np.ones((1, 3), dtype=np.float32)
            result["std"] = std.astype(float)
        return result


def _single_precision_case(with_std=False):
    surrogate = _SinglePrecisionSurrogate(with_std)
    observations = surrogate.predict_(surrogate.truth)["output"]
    bounds = np.column_stack([surrogate.lower, surrogate.lower + surrogate.span])
    starts = surrogate.lower + np.array([[0.2, 0.8], [0.8, 0.2], [0.4, 0.4], [0.9, 0.9], [0.55, 0.7]]) * surrogate.span
    return surrogate, observations, np.full(3, 0.02 ** 2), bounds, starts


def test_refinement_moves_on_a_single_precision_emulator():
    """The optimizer has to see the slope of the likelihood.

    Its default finite-difference step of 1e-8 is below the resolution of a single-precision emulator: the difference of two likelihoods was zero or round-off, five of six starts did not move, and the unmoved starting points were reported as a refined optimum and as several local maxima. The step is now 1e-3 of the calibration range of every parameter.
    """
    from hydroBayesCal.surrogate.posterior_analysis import refine_joint_optimum

    surrogate, observations, error, bounds, starts = _single_precision_case()

    result = refine_joint_optimum(starts, surrogate, observations, error, seeds=starts, prior_bounds=bounds, n_starts=5)

    assert result["refined"] is True
    assert result["global_agreement"] == 1.0, "one maximum, reached from every start"
    assert np.all(np.abs(result["refined_vector"] - surrogate.truth) < 2e-3 * surrogate.span)
    assert np.all(np.abs(result["local_optima"] - surrogate.truth) < 5e-3 * surrogate.span)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_refinement_on_a_trained_emulator(precision):
    """The same with a trained gpytorch emulator instead of a stand-in, in the precision of earlier releases and in the present one.

    The starts lie around the optimum, like the posterior samples the refinement is seeded with. The likelihood includes the variance of the emulator here, which flattens it where the emulator is uncertain: from a start far outside the posterior, a search can end in a corner of the calibration ranges that has no training run.
    """
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining
    from hydroBayesCal.surrogate.posterior_analysis import refine_joint_optimum

    rng = np.random.default_rng(3)
    model = lambda p: np.column_stack([p[:, 0] + p[:, 1], p[:, 0] - p[:, 1], 0.5 * p[:, 0] + 2.0 * p[:, 1]])
    x = rng.uniform(size=(25, 2))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    emulator = GPyTraining(collocation_points=x, model_evaluations=model(x), likelihood=likelihood,
                           kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)),
                           training_iter=150, optimizer="adam", lr=0.07, verbose=False, precision=precision)
    torch.manual_seed(0)
    emulator.train_()
    truth = np.array([0.6, 0.4])
    starts = truth + np.array([[0.08, -0.06], [-0.07, 0.05], [0.05, 0.08], [-0.04, -0.07]])

    result = refine_joint_optimum(starts, emulator, model(truth[None, :]), np.full(3, 0.02 ** 2), seeds=starts,
                                  prior_bounds=BOUNDS_UNIT, n_starts=4)

    assert result["global_agreement"] == 1.0
    assert np.linalg.norm(result["refined_vector"] - truth) < 0.02


def test_the_refined_likelihood_is_the_one_of_the_calibration():
    """The calibration adds the emulator variance to the observation variance (``include_surrogate_error``, on by default).

    The refinement maximized the likelihood without it, although its documentation promised the likelihood of the calibration. An optimum found that way may sit where the emulator is uncertain.
    """
    from hydroBayesCal.surrogate.posterior_analysis import _surrogate_log_likelihood

    surrogate, observations, error, _, starts = _single_precision_case(with_std=True)
    prediction = surrogate.predict_(starts)

    with_emulator_error = _surrogate_log_likelihood(surrogate, starts, observations, error)
    without = _surrogate_log_likelihood(surrogate, starts, observations, error, include_surrogate_error=False)

    residual = prediction["output"] - observations
    total = error + prediction["std"] ** 2
    expected = -0.5 * np.sum(residual ** 2 / total + np.log(total / error), axis=1)
    assert with_emulator_error == pytest.approx(expected, rel=1e-9)
    assert without == pytest.approx(-0.5 * np.sum(residual ** 2 / error, axis=1), rel=1e-9)


def test_the_analysis_takes_the_likelihood_setting_from_the_stored_calibration(monkeypatch):
    from hydroBayesCal.surrogate import posterior_analysis

    surrogate, observations, error, bounds, _ = _single_precision_case(with_std=True)
    rng = np.random.default_rng(0)
    posterior = surrogate.truth + rng.normal(scale=0.02, size=(600, 2)) * surrogate.span
    seen = []
    original = posterior_analysis._surrogate_log_likelihood
    monkeypatch.setattr(posterior_analysis, "_surrogate_log_likelihood",
                        lambda *a, include_surrogate_error=True, **k:
                        (seen.append(include_surrogate_error), original(*a, include_surrogate_error=include_surrogate_error, **k))[1])

    for stored in (False, True):
        seen.clear()
        bal = {"posterior": [posterior], "param_values": bounds.tolist(), "calibration_parameters": ["a", "b"],
               "include_surrogate_error": stored}
        analyze_posterior(bayesian_dict=bal, surrogate=surrogate, observations=observations, error=error, refine=True)
        assert seen and set(seen) == {stored}
