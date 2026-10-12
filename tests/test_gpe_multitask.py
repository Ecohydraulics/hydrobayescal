"""Tests for the multi-output (multitask) GPE, :class:`MultiGPyTraining`.

These run on small synthetic data with a handful of training iterations, so no
solver and no realistic GP fit is involved. They pin the output column order and
the multitask-covariance contract that Bayesian active learning depends on.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from scipy.linalg import block_diag

from hydroBayesCal.surrogate.gpe_gpytorch import MultiGPyTraining


N_TRAIN = 14
N_PARAMS = 2
TRAINING_ITER = 5


def _outputs(x, n_locations, n_quantities):
    """Distinct, well-separated analytic function per (location, quantity) column.

    Column ``i * n_quantities + q`` is the interleaved layout used throughout the
    package (see ``HydroSimulations.model_evaluations``).

    The coefficients deliberately vary in sign and direction. Columns that are all
    increasing functions of the same inputs come out almost perfectly correlated with
    each other, and then no test can tell which column is which.
    """
    columns = []
    for i in range(n_locations):
        for q in range(n_quantities):
            k = i * n_quantities + q
            columns.append(np.cos(k) * x[:, 0] + np.sin(2.0 * k) * x[:, 1] + k)
    return np.column_stack(columns)


def _column_match(predicted, truth):
    """Index of the true column each predicted column is closest to.

    Absolute distance, not correlation: correlation is blind to offset and scale, so
    two columns of the same shape but different level look identical to it.
    """
    distance = np.array([[np.mean(np.abs(predicted[:, k] - truth[:, j]))
                          for j in range(truth.shape[1])]
                         for k in range(predicted.shape[1])])
    return np.argmin(distance, axis=1)


def _train(n_locations, n_quantities, mode, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(N_TRAIN, N_PARAMS))
    y = _outputs(x, n_locations, n_quantities)

    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=N_PARAMS))
    num_tasks = {"variables": n_quantities,
                 "locations": n_locations,
                 "all": n_locations * n_quantities}[mode]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6))

    sm = MultiGPyTraining(x, y, kernel, training_iter=TRAINING_ITER,
                          likelihood=likelihood, optimizer="adam", lr=0.1,
                          number_quantities=n_quantities)
    getattr(sm, f"train_tasks_{mode}")()
    return sm, x, y


@pytest.mark.parametrize("mode", ["variables", "locations", "all"])
def test_task_mode_is_recorded(mode):
    sm, _, _ = _train(3, 2, mode)
    assert sm.task_mode == mode


@pytest.mark.parametrize("mode", ["variables", "locations", "all"])
def test_predict_column_order_nloc_equals_nq(mode):
    """Regression: dispatch used to key off len(gp_list), which is ambiguous here.

    With ``n_locations == n_quantities`` a "variables"-trained model previously took
    the "locations" writer and produced silently mis-ordered output columns.
    """
    n_locations = n_quantities = 3
    sm, x, y = _train(n_locations, n_quantities, mode)

    predicted = sm.predict_(input_sets=x)["output"]
    assert predicted.shape == y.shape
    assert not np.isnan(predicted).any()

    # Each predicted column must be closest to its own true column.
    assert np.array_equal(_column_match(predicted, y), np.arange(y.shape[1]))


def test_column_order_test_detects_the_legacy_mis_dispatch():
    """The check above must actually be able to fail, or it guards nothing.

    Dropping ``task_mode`` reinstates the old length-based heuristic. With
    ``n_locations == n_quantities`` it resolves a "variables"-trained model to the
    "locations" writer, which is exactly the bug, so the columns come out permuted.
    """
    n_locations = n_quantities = 3
    sm, x, y = _train(n_locations, n_quantities, "variables")

    correct = _column_match(sm.predict_(input_sets=x)["output"], y)
    assert np.array_equal(correct, np.arange(y.shape[1]))

    del sm.task_mode  # legacy pickle: falls back to len(gp_list), ambiguous here
    mis_dispatched = _column_match(sm.predict_(input_sets=x)["output"], y)
    assert not np.array_equal(mis_dispatched, np.arange(y.shape[1]))


@pytest.mark.parametrize("mode", ["variables", "locations", "all"])
def test_multitask_cov_shape_contract(mode):
    """BAL passes multitask_cov=True for every multi-output run.

    Regression: multitask_cov_list was bound only in the "variables" branch, so
    "locations" and "all" raised UnboundLocalError at the first BAL iteration.
    """
    n_locations, n_quantities = 4, 2
    sm, x, _ = _train(n_locations, n_quantities, mode)

    output = sm.predict_(input_sets=x[:3], multitask_cov=True)
    cov = output["multitask_cov"]

    assert len(cov) == 3
    for per_sample in cov:
        assert len(per_sample) == n_locations
        for block in per_sample:
            assert block.shape == (n_quantities, n_quantities)
            assert np.all(np.isfinite(block))
        # This is exactly what SequentialDesign.bayesian_active_learning does.
        assert block_diag(*per_sample).shape == (n_locations * n_quantities,) * 2


def test_three_quantities_locations_mode():
    """Regression: train_tasks_locations hard-coded two quantities and dropped the rest."""
    n_locations, n_quantities = 4, 3
    sm, x, y = _train(n_locations, n_quantities, "locations")

    assert len(sm.gp_list) == n_quantities
    predicted = sm.predict_(input_sets=x)["output"]
    assert predicted.shape == y.shape
    assert not np.isnan(predicted).any()


def test_locations_mode_rejects_ragged_output():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(N_TRAIN, N_PARAMS))
    y = rng.uniform(size=(N_TRAIN, 5))  # 5 outputs cannot split into 2 quantities
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    sm = MultiGPyTraining(x, y, kernel, training_iter=1, likelihood=likelihood,
                          number_quantities=2)
    with pytest.raises(ValueError, match="not a multiple"):
        sm.train_tasks_locations()


@pytest.mark.parametrize("mode", ["variables", "locations"])
def test_per_model_likelihood_is_independent(mode):
    """Regression: one shared likelihood meant the last sub-model's noise won."""
    sm, _, _ = _train(3, 2, mode)

    likelihoods = [entry["likelihood"] for entry in sm.gp_list]
    assert len({id(obj) for obj in likelihoods}) == len(likelihoods)
    assert all(obj is not sm.likelihood for obj in likelihoods)

    noises = [float(np.mean(obj.noise.detach().cpu().numpy())) for obj in likelihoods]
    assert len(set(noises)) > 1, "sub-model noises are still tied together"


def test_legacy_pickle_without_task_mode_still_predicts():
    """Surrogates pickled before task_mode existed must keep working, with a warning."""
    n_locations, n_quantities = 4, 2
    sm, x, y = _train(n_locations, n_quantities, "variables")
    del sm.task_mode

    predicted = sm.predict_(input_sets=x)["output"]
    assert predicted.shape == y.shape
    assert not np.isnan(predicted).any()


def test_unwritten_columns_raise():
    """A task layout inconsistent with n_obs must fail loudly, not return zeros."""
    n_locations, n_quantities = 4, 2
    sm, x, _ = _train(n_locations, n_quantities, "variables")
    sm.gp_list = sm.gp_list[:-1]  # drop one location's model
    sm.task_mode = "variables"
    with pytest.raises(RuntimeError, match="unwritten"):
        sm.predict_(input_sets=x)


# --------------------------------------------------------------------------- #
# the covariance handed to Bayesian active learning
# --------------------------------------------------------------------------- #
FAR_FROM_THE_DESIGN = [[5.0, 5.0]]


@pytest.mark.parametrize("mode", ["variables", "all"])
def test_multitask_cov_diagonal_is_the_predictive_variance(mode):
    """BAL samples the emulator from this covariance, so it has to be the predictive one.

    The blocks used to be the prior term, the task kernel times the input kernel at the prediction point. That term does not depend on the training data, while the standard deviation returned by the same call does, so the two described different distributions.
    """
    n_locations, n_quantities = 3, 2
    sm, x, _ = _train(n_locations, n_quantities, mode)
    probe = np.vstack([x[:2], FAR_FROM_THE_DESIGN])

    output = sm.predict_(input_sets=probe, multitask_cov=True)

    for j in range(len(probe)):
        diagonal = np.concatenate([np.diag(block) for block in output["multitask_cov"][j]])
        assert diagonal == pytest.approx(output["std"][j] ** 2, rel=1e-3)
        for block in output["multitask_cov"][j]:
            assert np.allclose(block, block.T, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("mode", ["variables", "all"])
def test_multitask_cov_is_smaller_where_the_emulator_was_trained(mode):
    """A candidate next to a training point is less uncertain than one far outside the design.

    This difference is what lets the BAL utility tell an explored region from an unexplored one. The prior term gave every candidate the same covariance.
    """
    sm, x, _ = _train(3, 2, mode)

    output = sm.predict_(input_sets=np.vstack([x[:1], FAR_FROM_THE_DESIGN]), multitask_cov=True)

    at_training_point = sum(np.trace(block) for block in output["multitask_cov"][0])
    far_away = sum(np.trace(block) for block in output["multitask_cov"][1])
    assert at_training_point < 0.9 * far_away


@pytest.mark.parametrize("budget", [2.0e7, 1.0])
def test_task_blocks_match_the_dense_predictive_covariance(budget):
    """The block-wise evaluation must return what the full covariance holds at those entries.

    With the tasks spanning all outputs the full covariance per prediction point does not fit into memory for a real calibration, so only the quantity x quantity blocks are evaluated, in chunks of prediction points. ``budget=1.0`` forces one point per chunk.
    """
    n_locations, n_quantities, n_points = 3, 2, 5
    sm, x, _ = _train(n_locations, n_quantities, "all")
    model_info = sm.gp_list[0]
    gp, likelihood = model_info["gp"], sm._model_likelihood(model_info)
    gp.eval()
    likelihood.eval()
    n_tasks = n_locations * n_quantities

    with torch.no_grad():
        predictions = likelihood(gp(torch.tensor(x[:n_points], dtype=torch.float32)))
        blocks = MultiGPyTraining._posterior_task_blocks(
            predictions, n_quantities, n_train=N_TRAIN, budget=budget)
        dense = predictions.covariance_matrix.numpy().reshape(n_points, n_tasks, n_points, n_tasks)

    assert blocks.shape == (n_points, n_locations, n_quantities, n_quantities)
    for j in range(n_points):
        for i in range(n_locations):
            sl = slice(i * n_quantities, (i + 1) * n_quantities)
            assert blocks[j, i] == pytest.approx(dense[j, sl, j, sl], rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------- #
# the blocks have to be covariances, also where round-off dominates
# --------------------------------------------------------------------------- #
class _Prediction:
    """What ``_posterior_task_blocks`` reads from a prediction, with a covariance that can be set entry by entry."""

    _interleaved = True

    def __init__(self, covariance, n_points, n_tasks):
        self.mean = torch.zeros(n_points, n_tasks)
        self.lazy_covariance_matrix = torch.as_tensor(covariance)


def test_task_blocks_are_covariances_even_where_round_off_made_them_indefinite():
    """A block whose covariance term exceeds its variances must not reach the BAL sampler.

    Where the emulator is nearly certain, its posterior variance is orders of magnitude below its prior variance, and the single-precision subtraction that yields it leaves an error of the size of the result. The block then has a negative eigenvalue, the sampler of the BAL utility refuses it, and the calibration stops in its active-learning loop.
    """
    from scipy import stats

    indefinite = np.array([[1.0e-6, 1.1e-6], [1.1e-6, 1.0e-6]])  # eigenvalues 2.1e-6 and -1.0e-7
    definite = np.array([[2.0e-6, 0.5e-6], [0.5e-6, 1.0e-6]])
    joint = np.zeros((4, 4))  # two prediction points, two tasks, interleaved
    joint[:2, :2], joint[2:, 2:] = indefinite, definite

    blocks = MultiGPyTraining._posterior_task_blocks(_Prediction(joint, 2, 2), block_size=2)

    for block in blocks[:, 0]:
        stats.multivariate_normal(mean=np.zeros(2), cov=block, allow_singular=True)  # must not raise
    assert np.linalg.eigvalsh(blocks[0, 0]).min() >= 0.0
    assert blocks[1, 0] == pytest.approx(definite, rel=1e-12), "a definite block is returned as it is"


def test_multitask_cov_next_to_training_points_is_accepted_by_the_bal_sampler():
    """The case that ended an end-to-end run of the TELEMAC driver, with the driver's emulator settings.

    Whether a trained emulator produces indefinite blocks depends on its hyperparameters. With this seed, on the machine the test was written on, the correlation lengths at one calibration point come out at about ten times the parameter ranges, the posterior variance there is six orders of magnitude below the prior variance, and one in five of the candidates below was refused before the blocks were projected. A double-precision evaluation of the same emulator gives a valid covariance at every candidate.
    """
    from scipy import stats

    rng = np.random.default_rng(2)
    torch.manual_seed(2)
    stations = np.linspace(0.2, 1.0, 3)
    x = rng.uniform(size=(9, 2))
    y = np.empty((9, 6))
    y[:, 0::2] = [[np.sin(3.0 * a * s) + b * s ** 2 for s in stations] for a, b in x]
    y[:, 1::2] = [[2.0 + a * s - 0.5 * b * np.cos(2.0 * s) for s in stations] for a, b in x]

    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=2, noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    sm = MultiGPyTraining(x, y, kernel, training_iter=150, likelihood=likelihood,
                          optimizer="adam", lr=0.07, number_quantities=2)
    sm.train_tasks_variables()

    candidates = np.clip(np.repeat(x, 10, axis=0) + rng.normal(scale=0.01, size=(90, 2)), 0.0, 1.0)
    covariance = sm.predict_(input_sets=candidates, multitask_cov=True)["multitask_cov"]

    for per_candidate in covariance:
        stats.multivariate_normal(mean=np.zeros(6), cov=block_diag(*per_candidate), allow_singular=True)


def test_task_blocks_stay_covariances_when_converted_to_physical_units():
    """The projection has to be the last operation on a block.

    The normalization constants of the model are single-precision numbers. Scaling an already projected, hence nearly singular, block with them leaves the form D B D by round-off and can make the block indefinite again: with the scales used here the smallest eigenvalue came out at -4e-9 times the largest, which the sampler refuses.
    """
    from scipy import stats

    indefinite = np.array([[1.0e-6, 1.1e-6], [1.1e-6, 1.0e-6]])
    scale = np.array([1.929037, 0.8458708], dtype=np.float32)

    blocks = MultiGPyTraining._posterior_task_blocks(_Prediction(indefinite, 1, 2), block_size=2, scale=scale)

    stats.multivariate_normal(mean=np.zeros(2), cov=blocks[0, 0], allow_singular=True)  # must not raise
    assert np.linalg.eigvalsh(blocks[0, 0]).min() >= 0.0
    # The variances are the normalized ones times the squared scales. Clipping moves them by the size of the negative eigenvalue, which is 5 % of the largest one in this deliberately bad block.
    assert np.diag(blocks[0, 0]) == pytest.approx(np.diag(indefinite) * scale.astype(float) ** 2, rel=0.2)


def test_clipping_of_an_indefinite_block_is_reported_once(monkeypatch):
    """A block that had to be clipped tells that the emulator is at the resolution of its arithmetic; a silent repair would hide that."""
    from hydroBayesCal.surrogate import gpe_gpytorch

    monkeypatch.setattr(gpe_gpytorch, "_REPORTED_CLIPPED_COVARIANCE", False)
    definite = np.array([[2.0e-6, 0.5e-6], [0.5e-6, 1.0e-6]])
    MultiGPyTraining._posterior_task_blocks(_Prediction(definite, 1, 2), block_size=2)
    assert not gpe_gpytorch._REPORTED_CLIPPED_COVARIANCE, "a valid covariance is not reported"

    indefinite = np.array([[1.0e-6, 1.1e-6], [1.1e-6, 1.0e-6]])
    MultiGPyTraining._posterior_task_blocks(_Prediction(indefinite, 1, 2), block_size=2)
    assert gpe_gpytorch._REPORTED_CLIPPED_COVARIANCE
