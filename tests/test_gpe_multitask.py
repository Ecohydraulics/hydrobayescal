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
