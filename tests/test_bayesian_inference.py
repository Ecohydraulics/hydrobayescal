"""Tests for the likelihood paths of :class:`BayesianInference`.

Focus: the opt-in surrogate-uncertainty path. ``error`` carries variances while
``model_error`` carries standard deviations, the diagonal fast path must reproduce
the dense one exactly, and including the surrogate uncertainty must broaden the
posterior rather than merely shift it.
"""
import numpy as np
import pytest

pytest.importorskip("tqdm")

from hydroBayesCal.surrogate.bal_functions import BayesianInference

RNG = np.random.default_rng(99)
N_OBS = 6
MC = 200


def _setup(model_error_value=None):
    observations = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    predictions = observations + RNG.normal(0.0, 0.2, size=(MC, N_OBS))
    error = np.full(N_OBS, 0.04)  # variances
    model_error = (None if model_error_value is None
                   else np.full((MC, N_OBS), model_error_value))
    return predictions, observations, error, model_error


def test_error_is_a_variance_vector_on_the_diagonal():
    predictions, observations, error, _ = _setup()
    inference = BayesianInference(model_predictions=predictions,
                                  observations=observations, error=error)
    assert np.allclose(np.diagonal(inference.cov_mat), error)


def test_diagonal_fast_path_matches_the_dense_path():
    predictions, observations, error, model_error = _setup(0.1)

    fast = BayesianInference(model_predictions=predictions, observations=observations,
                             error=error, model_error=model_error)
    assert fast._has_diagonal_augmented_covariance()
    fast.calculate_likelihood_with_error_diagonal()

    dense = BayesianInference(model_predictions=predictions, observations=observations,
                              error=error, model_error=model_error)
    dense.calculate_likelihood_with_error()

    assert np.allclose(fast.log_likelihood, np.ravel(dense.log_likelihood), atol=1e-9)


def test_estimate_bme_dispatches_to_the_fast_path():
    predictions, observations, error, model_error = _setup(0.1)
    inference = BayesianInference(model_predictions=predictions,
                                  observations=observations, error=error,
                                  model_error=model_error, prior=RNG.uniform(size=(MC, 2)))
    inference.estimate_bme()
    assert inference.log_likelihood.ndim == 1
    assert np.isfinite(inference.RE)


def test_zero_model_error_matches_the_manual_likelihood_up_to_a_constant():
    """calculate_likelihood_manual drops the normalising constant; the shape must agree."""
    predictions, observations, error, _ = _setup()
    model_error = np.zeros((MC, N_OBS))

    manual = BayesianInference(model_predictions=predictions, observations=observations,
                               error=error)
    manual.calculate_likelihood_manual()

    with_error = BayesianInference(model_predictions=predictions,
                                   observations=observations, error=error,
                                   model_error=model_error)
    with_error.calculate_likelihood_with_error_diagonal()

    difference = np.ravel(with_error.log_likelihood) - np.ravel(manual.log_likelihood)
    assert np.allclose(difference, difference[0], atol=1e-9)


def test_surrogate_error_broadens_the_posterior():
    """The quantitative statement of the over-sharp-posterior finding."""
    predictions, observations, error, _ = _setup()
    prior = RNG.uniform(size=(MC, 2))

    sharp = BayesianInference(model_predictions=predictions, observations=observations,
                              error=error, prior=prior)
    sharp.estimate_bme()

    broad = BayesianInference(model_predictions=predictions, observations=observations,
                              error=error, model_error=np.full((MC, N_OBS), 0.5),
                              prior=prior)
    broad.estimate_bme()

    assert broad.posterior.shape[0] > sharp.posterior.shape[0]


def test_full_covariance_falls_back_to_the_dense_path():
    predictions, observations, error, model_error = _setup(0.1)
    inference = BayesianInference(model_predictions=predictions,
                                  observations=observations, error=error,
                                  model_error=model_error)
    inference.cov_mat = inference.cov_mat + 0.001  # no longer diagonal
    assert not inference._has_diagonal_augmented_covariance()


# ---------------------------------------------------------------------------
# numerical stability of the evidence
# ---------------------------------------------------------------------------
def _inference(n_obs, obs_std, model_error_value, mc=400, residual_sigma=1.0, seed=7):
    """A calibration-shaped problem at a realistic size."""
    rng = np.random.default_rng(seed)
    observations = rng.uniform(0.5, 2.5, size=(1, n_obs))
    error = np.full(n_obs, obs_std ** 2)                       # variances
    predictions = observations + rng.normal(0.0, residual_sigma * obs_std, size=(mc, n_obs))
    model_error = (None if model_error_value is None
                   else np.full((mc, n_obs), model_error_value))
    return BayesianInference(model_predictions=predictions, observations=observations,
                             error=error, model_error=model_error,
                             prior=rng.uniform(size=(mc, 3)))


def test_bme_does_not_overflow_at_a_realistic_problem_size():
    """100 calibration points x 3 targets at 2 cm precision, with surrogate error.

    Regression: the model_error paths kept the full normalising constant, so the
    log-likelihood went large and positive and mean(exp(.)) reached inf, making
    RE = ELPD - log(inf) = -inf.
    """
    inference = _inference(n_obs=300, obs_std=0.02, model_error_value=0.01)
    inference.estimate_bme()

    assert np.all(inference.log_likelihood <= 0.0)
    assert np.isfinite(inference.log_BME)
    assert np.isfinite(inference.BME)
    assert np.isfinite(inference.RE)


def test_re_is_finite_when_the_linear_bme_underflows():
    """Regression for a bug present on the DEFAULT path, without any model error.

    At 600 outputs and a poor fit, mean(exp(log_likelihood)) underflows to exactly
    0.0, which used to give RE = nan. That nan then collapsed every BAL candidate
    score to 0.0 and the training point was chosen arbitrarily.
    """
    inference = _inference(n_obs=600, obs_std=0.05, model_error_value=None,
                           residual_sigma=6.0)
    inference.estimate_bme()

    assert float(np.mean(np.exp(inference.log_likelihood))) == 0.0
    assert inference.BME == 0.0
    assert np.isfinite(inference.log_BME)
    assert np.isfinite(inference.RE)


def test_re_is_finite_when_the_linear_bme_overflows():
    """The guard itself, exercised deterministically."""
    inference = _inference(n_obs=10, obs_std=0.1, model_error_value=0.05)
    inference.calculate_likelihood_with_error_diagonal = lambda: setattr(
        inference, "log_likelihood", np.full(400, 800.0))
    inference.estimate_bme()

    assert np.isinf(inference.BME)
    assert np.isclose(inference.log_BME, 800.0)
    assert np.isfinite(inference.RE)


def test_zero_model_error_reproduces_the_manual_likelihood_exactly():
    """The reference normalisation is the whole basis of the fix.

    With the observation-covariance constant as the reference, the model-error path
    collapses onto calculate_likelihood_manual at model_error = 0, so the two
    likelihood conventions agree exactly rather than up to a constant.
    """
    predictions, observations, error, _ = _setup()
    manual = BayesianInference(model_predictions=predictions, observations=observations,
                               error=error)
    manual.calculate_likelihood_manual()

    with_error = BayesianInference(model_predictions=predictions,
                                   observations=observations, error=error,
                                   model_error=np.zeros((MC, N_OBS)))
    with_error.calculate_likelihood_with_error_diagonal()

    assert np.allclose(np.ravel(with_error.log_likelihood),
                       np.ravel(manual.log_likelihood), atol=1e-12)


def test_log_likelihood_is_non_positive_with_model_error():
    """v >= e always, so log(v/e) >= 0 and overflow is structurally impossible."""
    rng = np.random.default_rng(3)
    predictions, observations, error, _ = _setup()
    for _ in range(5):
        model_error = rng.uniform(0.0, 0.5, size=(MC, N_OBS))
        inference = BayesianInference(model_predictions=predictions,
                                      observations=observations, error=error,
                                      model_error=model_error)
        inference.calculate_likelihood_with_error_diagonal()
        assert np.all(inference.log_likelihood <= 0.0)


def test_log_bme_equals_the_naive_mean_where_that_is_representable():
    predictions, observations, error, model_error = _setup(0.1)
    inference = BayesianInference(model_predictions=predictions, observations=observations,
                                  error=error, model_error=model_error,
                                  prior=RNG.uniform(size=(MC, 2)))
    inference.estimate_bme()

    naive = float(np.mean(np.exp(inference.log_likelihood)))
    assert np.isclose(inference.log_BME, np.log(naive), rtol=1e-10)


def test_bme_is_comparable_across_the_flag():
    """Same convention on both paths, so archived and new runs are on one scale."""
    predictions, observations, error, _ = _setup()
    prior = RNG.uniform(size=(MC, 2))

    without = BayesianInference(model_predictions=predictions, observations=observations,
                                error=error, prior=prior)
    without.estimate_bme()
    with_zero = BayesianInference(model_predictions=predictions, observations=observations,
                                  error=error, model_error=np.zeros((MC, N_OBS)),
                                  prior=prior)
    with_zero.estimate_bme()

    assert np.isclose(without.log_BME, with_zero.log_BME, atol=1e-12)


def test_dense_path_survives_a_determinant_underflow():
    """np.linalg.det of a 150x150 diagonal of 0.0025 underflows to 0.0; slogdet does not."""
    n_obs, mc = 150, 20
    rng = np.random.default_rng(11)
    observations = rng.uniform(0.5, 2.5, size=(1, n_obs))
    error = np.full(n_obs, 0.05 ** 2)
    predictions = observations + rng.normal(0.0, 0.05, size=(mc, n_obs))
    model_error = np.full((mc, n_obs), 0.02)

    fast = BayesianInference(model_predictions=predictions, observations=observations,
                             error=error, model_error=model_error)
    fast.calculate_likelihood_with_error_diagonal()
    dense = BayesianInference(model_predictions=predictions, observations=observations,
                              error=error, model_error=model_error)
    dense.calculate_likelihood_with_error()

    assert np.all(np.isfinite(dense.log_likelihood))
    assert np.allclose(np.ravel(dense.log_likelihood), np.ravel(fast.log_likelihood),
                       atol=1e-8)


def test_non_positive_observation_variance_is_rejected():
    predictions, observations, error, _ = _setup()
    error = error.copy()
    error[2] = 0.0
    with pytest.raises(ValueError, match="variance"):
        BayesianInference(model_predictions=predictions, observations=observations,
                          error=error)


# --------------------------------------------------------------------------- #
# posterior sampling: weighted resampling vs rejection sampling
# --------------------------------------------------------------------------- #
def _peaked_problem(mc=4000, n_obs=30):
    """A sharply peaked likelihood - the regime where rejection sampling starves.

    One parameter scales the prediction, so the likelihood is tight around the value
    that reproduces the observations. This is the ordinary shape of a *successful*
    calibration, which is exactly when the acceptance rate collapses.
    """
    rng = np.random.default_rng(7)
    prior = rng.uniform(0.0, 1.0, size=(mc, 2))
    shape = np.linspace(0.5, 1.5, n_obs)
    predictions = shape[None, :] * (0.5 + prior[:, [0]])
    observations = (shape * (0.5 + 0.7))[None, :]
    error = np.full(n_obs, 0.01 ** 2)
    return predictions, observations, error, prior


def test_weighted_resampling_returns_a_full_size_posterior():
    """Rejection sampling starves on a peaked likelihood; weighting must not.

    The count is the whole point: a posterior of a few dozen samples cannot support
    a density estimate, however many prior samples were evaluated.
    """
    predictions, observations, error, prior = _peaked_problem()

    rejection = BayesianInference(model_predictions=predictions, observations=observations,
                                  error=error, prior=prior,
                                  sampling_method="rejection_sampling")
    rejection.estimate_bme()

    weighted = BayesianInference(model_predictions=predictions, observations=observations,
                                 error=error, prior=prior,
                                 sampling_method="bayesian_weighting")
    weighted.estimate_bme()

    assert rejection.posterior is not None
    assert weighted.posterior is not None, "bayesian_weighting produced no posterior"
    assert len(rejection.posterior) < 0.05 * len(prior), "expected a starved acceptance"
    assert len(weighted.posterior) == len(prior)
    assert weighted.posterior_output.shape == (len(prior), predictions.shape[1])


def test_the_two_samplers_agree_on_the_posterior():
    """Different mechanics, same target distribution - so the fix cannot be a
    full-size sample of the wrong thing."""
    predictions, observations, error, prior = _peaked_problem()

    rejection = BayesianInference(model_predictions=predictions, observations=observations,
                                  error=error, prior=prior,
                                  sampling_method="rejection_sampling")
    rejection.estimate_bme()
    weighted = BayesianInference(model_predictions=predictions, observations=observations,
                                 error=error, prior=prior,
                                 sampling_method="bayesian_weighting")
    weighted.estimate_bme()

    # the informed parameter: means within a small fraction of the prior width
    assert abs(weighted.posterior[:, 0].mean() - rejection.posterior[:, 0].mean()) < 0.02
    # and the log-evidence is a property of the likelihood, not of the sampler
    assert np.isclose(weighted.log_BME, rejection.log_BME, rtol=1e-9)


def test_weighted_resampling_reports_its_effective_sample_size():
    """Resampling with replacement repeats rows, so the count alone overstates the
    information. ESS must be reported, and must not exceed the drawn size."""
    predictions, observations, error, prior = _peaked_problem()
    weighted = BayesianInference(model_predictions=predictions, observations=observations,
                                 error=error, prior=prior,
                                 sampling_method="bayesian_weighting")
    weighted.estimate_bme()

    assert weighted.ess is not None
    assert 0 < weighted.ess <= len(prior)
    # post_index must address rows of the original prior
    assert weighted.post_index.min() >= 0
    assert weighted.post_index.max() < len(prior)
    assert np.allclose(weighted.posterior, prior[weighted.post_index])
