"""Tests for the covariance that Bayesian active learning draws emulator outputs from.

The BAL utility samples a multivariate normal distribution at every candidate parameter set. Its sampler refuses a covariance with a negative eigenvalue, and a covariance computed in single precision can have one where the posterior variance is orders of magnitude below the prior variance, i.e. close to a training run. The refusal is a ``ValueError`` in the active-learning loop, after the simulations of the initial design have been paid for, so both the emulator and the utility have to cope with it.
"""
import numpy as np
import pytest
from scipy import stats

from hydroBayesCal.surrogate.bal_functions import SequentialDesign, project_to_psd

#: Eigenvalues 2.1e-6 and -1.0e-7: the off-diagonal term exceeds the variances, as round-off produces it.
INDEFINITE = np.array([[1.0e-6, 1.1e-6], [1.1e-6, 1.0e-6]])
#: Eigenvalues 2.2e-6 and 0.8e-6.
DEFINITE = np.array([[2.0e-6, 0.5e-6], [0.5e-6, 1.0e-6]])


def test_the_sampler_refuses_a_covariance_that_is_indefinite_by_round_off():
    """The premise of this module: this is the error that ended a calibration."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        stats.multivariate_normal(mean=np.zeros(2), cov=INDEFINITE, allow_singular=True)


def test_projection_turns_an_indefinite_block_into_a_covariance():
    projected = project_to_psd(INDEFINITE)

    stats.multivariate_normal(mean=np.zeros(2), cov=projected, allow_singular=True)  # must not raise
    eigenvalues = np.linalg.eigvalsh(projected)
    assert eigenvalues.min() >= 0.0
    assert eigenvalues.max() == pytest.approx(2.1e-6, rel=1e-9), "the dominant direction is untouched"
    assert np.array_equal(projected, projected.T)


def test_projection_leaves_a_definite_block_as_it_is():
    assert np.array_equal(project_to_psd(DEFINITE), DEFINITE)


def test_projection_reports_how_far_the_matrix_was_from_a_covariance():
    _, ratio = project_to_psd(INDEFINITE, return_ratio=True)
    assert ratio == pytest.approx(-1.0e-7 / 2.1e-6, rel=1e-6)
    _, ratio = project_to_psd(DEFINITE, return_ratio=True)
    assert ratio == 0.0


def test_projection_handles_a_stack_of_blocks_and_changes_only_the_indefinite_ones():
    stack = np.stack([DEFINITE, INDEFINITE, DEFINITE])[None]  # [n_samples=1, n_locations=3, 2, 2]

    projected = project_to_psd(stack)

    assert projected.shape == stack.shape
    assert np.array_equal(projected[0, 0], DEFINITE)
    assert np.array_equal(projected[0, 2], DEFINITE)
    assert np.linalg.eigvalsh(projected[0, 1]).min() >= 0.0


def test_the_bal_utility_survives_a_covariance_that_is_indefinite_by_round_off():
    """Whatever emulator supplies the covariance, round-off in it must not end a calibration in its BAL loop."""
    design = type("Stub", (), {"mc_exploration": 400})()  # all that bayesian_active_learning reads from self
    y_mean = np.array([1.0, 2.0, 1.5, 2.5])
    blocks = [INDEFINITE, DEFINITE]  # one quantity x quantity block per location

    score = SequentialDesign.bayesian_active_learning(
        design, y_mean=y_mean, y_std=blocks, observations=y_mean[None, :],
        error=np.full(4, 1.0e-4), utility_function="dkl")

    assert isinstance(float(score), float)


def test_the_bal_utility_still_refuses_a_covariance_that_is_not_one_at_all():
    """Clipping round-off is not a license to accept garbage: non-finite entries stay an error."""
    design = type("Stub", (), {"mc_exploration": 50})()
    broken = np.array([[np.nan, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        SequentialDesign.bayesian_active_learning(
            design, y_mean=np.array([1.0, 2.0]), y_std=[broken], observations=np.array([[1.0, 2.0]]),
            error=np.full(2, 1.0e-4), utility_function="dkl")
