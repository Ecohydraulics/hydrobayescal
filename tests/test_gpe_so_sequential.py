"""Tests for :class:`SOSequentialGPyTraining`, independent GPEs for several calibration targets.

The container trains one single-output GPE per calibration target and has to hand its predictions back in the interleaved (location, quantity) column order every other part of the package assumes. A swap of two targets would not raise anywhere; the likelihood would simply compare each prediction with the measurement of the other target.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, SOSequentialGPyTraining

N_TRAIN, N_PARAMS, N_LOCATIONS = 16, 2, 3
TARGETS = ["WATER DEPTH", "SCALAR VELOCITY"]


def _outputs(x):
    """Interleaved columns [h_P1, u_P1, h_P2, u_P2, ...] with clearly separated levels per target."""
    columns = []
    for i in range(N_LOCATIONS):
        columns.append(2.0 + 0.5 * i + x[:, 0])            # WATER DEPTH around 2 to 4
        columns.append(-5.0 - 0.5 * i + 0.3 * x[:, 1])     # SCALAR VELOCITY around -5 to -6
    return np.column_stack(columns)


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(N_TRAIN, N_PARAMS))
    y = _outputs(x)
    surrogate = SOSequentialGPyTraining.from_training_data(
        x, y, TARGETS, training_iter=40, lr=0.07)
    surrogate.train_()
    return surrogate, x, y


def test_one_gpe_per_target_is_built_from_the_interleaved_outputs(trained):
    surrogate, x, y = trained

    assert list(surrogate.models) == TARGETS
    assert all(isinstance(model, GPyTraining) for model in surrogate.models.values())
    assert np.array_equal(surrogate["WATER DEPTH"].model_evaluations, y[:, 0::2])
    assert np.array_equal(surrogate["SCALAR VELOCITY"].model_evaluations, y[:, 1::2])
    assert np.array_equal(surrogate.model_evaluations, y)
    assert surrogate.n_obs == N_LOCATIONS * len(TARGETS)


def test_predictions_come_back_in_the_interleaved_column_order(trained):
    surrogate, x, y = trained

    prediction = surrogate.predict_(input_sets=x, get_conf_int=True)

    assert prediction["output"].shape == y.shape
    assert prediction["std"].shape == y.shape
    # The two targets live on different levels, so a swapped column cannot pass.
    assert np.abs(prediction["output"] - y).max() < 0.5
    assert np.all(prediction["upper_ci"] >= prediction["output"])
    assert np.all(prediction["lower_ci"] <= prediction["output"])


def test_the_targets_are_independent_so_there_is_no_task_covariance(trained):
    surrogate, x, _ = trained

    with pytest.raises(ValueError, match="multitask=False"):
        surrogate.predict_(input_sets=x[:2], multitask_cov=True)


def test_outputs_that_do_not_divide_into_the_targets_are_rejected():
    x = np.random.default_rng(0).uniform(size=(N_TRAIN, N_PARAMS))

    with pytest.raises(ValueError, match="multiple of"):
        SOSequentialGPyTraining.from_training_data(x, _outputs(x)[:, :5], TARGETS)
