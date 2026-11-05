"""Tests for the scaling of the calibration parameters to [0, 1] before an emulator is trained (``tp_normalization``)."""
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining, SOSequentialGPyTraining

RNG = np.random.default_rng(3)
#: Two parameters six orders of magnitude apart: a roughness in m and a discharge in m3/s.
X = np.column_stack([RNG.uniform(0.01, 0.1, 25), RNG.uniform(100.0, 1000.0, 25)])
#: Two calibration targets at one location, interleaved as the drivers order them.
Y = np.column_stack([np.log(X[:, 0]) + X[:, 1] / 500.0, np.sqrt(X[:, 1]) * (1.0 + X[:, 0])])


def _single(normalize, outputs=Y[:, [0]]):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    return GPyTraining(collocation_points=X, model_evaluations=outputs, likelihood=likelihood, kernel=kernel,
                       training_iter=60, lr=0.07, verbose=False, tp_normalization=normalize)


def _multi(normalize):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=2, noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    return MultiGPyTraining(X, Y, kernel, training_iter=60, likelihood=likelihood, optimizer="adam", lr=0.07,
                            number_quantities=2, tp_normalization=normalize)


def _train(emulator):
    if isinstance(emulator, MultiGPyTraining):
        emulator.train_tasks_variables()
    else:
        emulator.train_()


def test_the_parameters_are_used_as_they_are_by_default():
    for cls in (GPyTraining, MultiGPyTraining, SOSequentialGPyTraining.from_training_data):
        import inspect
        assert inspect.signature(cls).parameters["tp_normalization"].default is False


@pytest.mark.parametrize("build", [_single, _multi], ids=["single-output", "multitask"])
def test_a_normalized_emulator_scales_its_prediction_inputs_like_its_training_points(build):
    torch.manual_seed(0)
    emulator = build(True)
    _train(emulator)

    assert emulator.x_min == pytest.approx(X.min(axis=0))
    prediction = emulator.predict_(X)["output"]
    reference = Y[:, [0]] if prediction.shape[1] == 1 else Y
    assert np.all(np.abs(prediction - reference).max(axis=0) < 0.2 * reference.std(axis=0)), (
        "at its training runs the emulator follows the smooth response")

    # The same emulator, given the scaled parameters by hand, predicts the same.
    emulator.tp_norm = False
    by_hand = emulator.predict_((X - emulator.x_min) / emulator.x_range)["output"]
    assert prediction == pytest.approx(by_hand)


def test_a_constant_parameter_is_left_unscaled():
    emulator = _single(True)
    X_const = X.copy()
    X_const[:, 1] = 500.0
    emulator.training_points = X_const
    torch.manual_seed(0)
    emulator.train_()

    assert emulator.x_range[1] == 1.0
    assert np.all(np.isfinite(emulator.predict_(X_const)["output"]))


@pytest.mark.parametrize("build", [_single, _multi], ids=["single-output", "multitask"])
def test_an_emulator_stored_before_the_option_existed_still_predicts(build):
    """An emulator pickled by 1.11 has no input scale: it was trained on the parameters as they are."""
    torch.manual_seed(0)
    emulator = build(False)
    _train(emulator)
    expected = emulator.predict_(X[:5])["output"]

    old = pickle.loads(pickle.dumps(emulator))
    for name in ("x_min", "x_range"):
        delattr(old, name)
    old.tp_norm = True  # set but without effect in GPyTraining up to 1.11
    if isinstance(old, MultiGPyTraining):
        del old.tp_norm

    assert old.predict_(X[:5])["output"] == pytest.approx(expected)
