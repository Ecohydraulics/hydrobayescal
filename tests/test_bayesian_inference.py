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
