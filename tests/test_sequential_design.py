"""Tests for the selection of the next training point (``SequentialDesign.run_sequential_design``).

The emulator is replaced by an analytic stand-in, so the tests see the selection logic alone: which candidates of the prior sample are scored, how the exploitation and the exploration score are combined, and what happens to a candidate whose utility cannot be computed. A wrong selection does not raise anything. It spends a full-complexity simulation on a parameter set that no criterion chose.
"""
import numpy as np
import pytest

from hydroBayesCal.surrogate.bal_functions import SequentialDesign


class _Design:
    """What ``run_sequential_design`` reads from the experimental design."""
    explore_method = "random"
    exploit_method = "bal"
    util_func = "dkl"
    n_new_samples = 1


class _Emulator:
    """Analytic stand-in for a trained emulator with three outputs at one calibration point."""

    def __init__(self, training_points, absurd_row=None):
        self.training_points = np.asarray(training_points, dtype=float)
        self.absurd_row = absurd_row
        self.covariance_requested = []

    def predict_(self, input_sets, multitask_cov=False, **kwargs):
        self.covariance_requested.append(bool(multitask_cov))
        x = np.atleast_2d(input_sets)
        output = np.column_stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1], x[:, 0] * x[:, 1]])
        std = (0.02 + 0.3 * np.abs(x[:, [0]] - 0.5)) * np.ones((1, 3))
        if self.absurd_row is not None:
            output[self.absurd_row] = 1.0e200  # a prediction no likelihood can be computed for
        result = {"output": output, "std": std}
        if multitask_cov:
            result["multitask_cov"] = [[np.diag(row ** 2)] for row in std]  # one 3 x 3 block per candidate
        return result


class _ErrorEmulator:
    """Secondary emulator of the model error: a constant offset with a constant standard deviation."""
    OFFSET, STD = 0.1, 0.04

    def predict_(self, input_sets, **kwargs):
        n = np.atleast_2d(input_sets).shape[0]
        return {"output": np.full((n, 3), self.OFFSET), "std": np.full((n, 3), self.STD)}


RNG = np.random.default_rng(1)
PRIOR = RNG.uniform(size=(5000, 2))
TRAINING_POINTS = PRIOR[:6].copy()
OBSERVATIONS = np.array([[1.0, 0.1, 0.25]])
VARIANCES = np.full(3, 0.05 ** 2)


def _design(do_tradeoff=False, prior=PRIOR, training_points=TRAINING_POINTS, mc_samples=300, design=_Design,
            absurd_row=None, **keywords):
    emulator = _Emulator(training_points, absurd_row)
    sequential_design = SequentialDesign(
        exp_design=design(), sm_object=emulator, obs=OBSERVATIONS.copy(), errors=VARIANCES.copy(),
        do_tradeoff=do_tradeoff, mc_samples=mc_samples, mc_exploration=400, n_jobs=1, parallel=False, **keywords)
    return sequential_design, emulator, prior


def _select(*arguments, **keywords):
    sequential_design, emulator, prior = _design(*arguments, **keywords)
    np.random.seed(0)  # the sampler of the utility draws from numpy's global generator
    new_point, utility_name = sequential_design.run_sequential_design(prior_samples=prior)
    return sequential_design, emulator, new_point, utility_name


def _constant_utility(monkeypatch, value=1.0):
    """Let every candidate have the same utility, so that only the exploration score can tell them apart."""
    monkeypatch.setattr(
        SequentialDesign, "run_al_functions",
        lambda self, exploit_method, candidates, index, m_error, utility_func: (index, np.full(len(candidates), value)))


def _distance_to_training_points(candidates, training_points, prior):
    lower, span = prior.min(axis=0), np.ptp(prior, axis=0)
    difference = (candidates[:, None, :] - training_points[None, :, :]) / span
    return np.sqrt((difference ** 2).sum(axis=-1)).min(axis=1)


# --------------------------------------------------------------------------- #
# exploration / exploitation trade-off
# --------------------------------------------------------------------------- #
def test_the_trade_off_gives_every_candidate_an_exploration_score():
    """With the trade-off on, the exploration score has to say something.

    All drivers pass the prior sample as the candidate pool. For that case the exploration score was a vector of zeros, so that the trade-off that the log and ``BAL_dictionary.pkl`` reported as active ranked the candidates by exploitation alone.
    """
    sequential_design, _, _, _ = _select(do_tradeoff=True)

    score = sequential_design.explore_score
    assert score.shape == (len(sequential_design.candidates),)
    assert np.all(np.isfinite(score)) and np.all(score >= 0.0)
    assert score.sum() == pytest.approx(1.0), "normalized like the exploitation score it is combined with"
    assert np.ptp(score) > 0.0


def test_the_trade_off_selects_a_remote_candidate_when_the_utility_does_not_discriminate(monkeypatch):
    _constant_utility(monkeypatch)

    sequential_design, _, new_point, _ = _select(do_tradeoff=True)

    distance = _distance_to_training_points(sequential_design.candidates, TRAINING_POINTS, PRIOR)
    selected = int(np.where((sequential_design.candidates == new_point[0]).all(axis=1))[0][0])
    assert selected == int(np.argmax(sequential_design.explore_score))
    assert distance[selected] >= np.percentile(distance, 95), "the exploration score rewards distance from the training runs"


def test_the_trade_off_changes_the_ranking():
    exploitation_only, _, _, _ = _select(do_tradeoff=False)
    with_trade_off, _, _, _ = _select(do_tradeoff=True)

    assert np.allclose(exploitation_only.exploit_score, with_trade_off.exploit_score), "same utilities in both runs"
    assert not np.array_equal(np.argsort(exploitation_only.exploit_score_norm), np.argsort(with_trade_off.total_score))


def test_exploration_distances_do_not_depend_on_the_units_of_the_parameters(monkeypatch):
    """A friction coefficient of 0.02 to 0.06 and a parameter of 0 to 1000 have to count alike.

    The distances are therefore taken in the unit hypercube spanned by the prior sample. In physical units, the parameter with the widest range would decide the exploration score alone.
    """
    _constant_utility(monkeypatch)
    units = np.array([1.0, 1000.0])

    reference, _, _, _ = _select(do_tradeoff=True)
    rescaled, _, _, _ = _select(do_tradeoff=True, prior=PRIOR * units, training_points=TRAINING_POINTS * units)

    assert np.allclose(reference.explore_score, rescaled.explore_score)


def test_a_prior_sample_smaller_than_the_candidate_budget_is_accepted():
    """Fewer unused prior samples than ``mc_samples`` used to end in a broadcast error once the trade-off was on."""
    sequential_design, _, new_point, _ = _select(do_tradeoff=True, prior=PRIOR[:150], mc_samples=300)

    assert len(sequential_design.candidates) == 144  # 150 prior samples minus the 6 training runs
    assert sequential_design.explore_score.shape == (144,)
    assert new_point.shape == (1, 2)


def test_without_the_trade_off_the_exploration_score_is_zero():
    sequential_design, _, _, _ = _select(do_tradeoff=False)

    assert sequential_design.explore_score.shape == (len(sequential_design.candidates),)
    assert np.all(sequential_design.explore_score == 0.0)


# --------------------------------------------------------------------------- #
# candidates without a usable utility
# --------------------------------------------------------------------------- #
def test_a_candidate_without_a_finite_utility_does_not_change_the_selection():
    """One failed candidate must be ranked last, and nothing else may change.

    A failed candidate carries the score ``-inf``. Dividing the scores by the sum of their absolute values then gave 0 for every other candidate, and the next full-complexity run went to a parameter set that no criterion had chosen.
    """
    healthy, _, expected_point, _ = _select()
    with_failure, _, new_point, _ = _select(absurd_row=17)

    assert with_failure.exploit_score[17] == -np.inf
    assert np.array_equal(new_point, expected_point)
    assert np.isnan(with_failure.exploit_score_norm[17])
    others = np.delete(with_failure.exploit_score_norm, 17)
    assert np.all(np.isfinite(others)) and np.abs(others).sum() == pytest.approx(1.0)
    assert np.allclose(others * np.abs(np.delete(with_failure.exploit_score, 17)).sum(), np.delete(healthy.exploit_score, 17))


@pytest.mark.parametrize("do_tradeoff", [False, True])
@pytest.mark.parametrize("value", [-np.inf, np.nan, 0.0])
def test_selection_is_space_filling_when_no_candidate_has_a_usable_utility(monkeypatch, value, do_tradeoff):
    """Without any information from the utility, the next run goes where the training runs are farthest away."""
    _constant_utility(monkeypatch, value)

    sequential_design, _, new_point, utility_name = _select(do_tradeoff=do_tradeoff)

    distance = _distance_to_training_points(sequential_design.candidates, TRAINING_POINTS, PRIOR)
    selected = int(np.where((sequential_design.candidates == new_point[0]).all(axis=1))[0][0])
    assert utility_name == "global_mc"
    assert distance[selected] >= np.percentile(distance, 95)


# --------------------------------------------------------------------------- #
# what the utility receives from the emulator(s)
# --------------------------------------------------------------------------- #
def test_the_sampling_utility_receives_the_covariance_between_the_targets():
    _, emulator, new_point, _ = _select(multitask=True)

    assert emulator.covariance_requested == [True]
    assert new_point.shape == (1, 2)


def test_the_analytic_utility_of_a_multitask_emulator_receives_standard_deviations():
    """``gaussian_assumption=True`` flattens the observations and uses the closed-form utility, which takes a standard deviation per output. The covariance between the targets was read there without having been requested, which raised ``UnboundLocalError``."""
    _, emulator, new_point, _ = _select(multitask=True, gaussian_assumption=True)

    assert emulator.covariance_requested == [False]
    assert new_point.shape == (1, 2)


def test_an_exploitation_method_that_contains_bal_is_evaluated_as_bal():
    class _Named(_Design):
        exploit_method = "BAL_global"

    reference, _, expected_point, _ = _select()
    _, _, new_point, _ = _select(design=_Named)

    assert np.array_equal(new_point, expected_point)


def test_an_unknown_exploitation_method_is_refused_by_name():
    sequential_design, _, _ = _design()

    with pytest.raises(ValueError, match="space_time"):
        sequential_design.run_al_functions("space_time", PRIOR[:5], 1, VARIANCES, "dkl")


@pytest.mark.parametrize("multitask", [False, True])
def test_the_error_emulator_adds_its_variance_to_the_emulator_variance(monkeypatch, multitask):
    """Emulator and error emulator are independent, so their variances add, not their standard deviations.

    With a multitask emulator the uncertainty of the error emulator was dropped altogether, because the covariance between the targets was passed on unchanged.
    """
    received = {}

    def record(self, y_mean, y_std, observations, error, utility_function="dkl"):
        received.setdefault("mean", np.array(y_mean, dtype=float))
        received.setdefault("spread", y_std)
        return 1.0

    monkeypatch.setattr(SequentialDesign, "bayesian_active_learning", record)
    sequential_design, emulator, prior = _design(multitask=multitask, secondary_sm=_ErrorEmulator())
    sequential_design.run_sequential_design(prior_samples=prior)

    first = sequential_design.candidates[:1]
    expected = emulator.predict_(first)
    variance = expected["std"][0] ** 2 + _ErrorEmulator.STD ** 2
    assert received["mean"] == pytest.approx(expected["output"][0] + _ErrorEmulator.OFFSET)
    if multitask:
        assert np.diag(received["spread"][0]) == pytest.approx(variance)
    else:
        assert np.asarray(received["spread"]) ** 2 == pytest.approx(variance)
