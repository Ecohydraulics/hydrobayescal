"""Tests for the training options of the single-output emulator (:class:`GPyTraining`): restarts, optimizer, and training data that cannot be used.

The drivers train with one restart of Adam, so these options are reached only by scripts that build the emulator themselves.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining

RNG = np.random.default_rng(0)
X = RNG.uniform(size=(12, 2))
Y = np.column_stack([np.sin(3.0 * X[:, 0]) + X[:, 1], 2.0 + X[:, 0] * X[:, 1]])


def _emulator(outputs=Y, training_iter=30, **options):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    return GPyTraining(collocation_points=X, model_evaluations=outputs, likelihood=likelihood, kernel=kernel,
                       training_iter=training_iter, lr=0.07, verbose=False, **options)


@pytest.mark.parametrize("n_restarts", [1, 2, 3])
def test_every_restart_after_the_first_starts_from_new_hyperparameters(monkeypatch, n_restarts):
    """Restarts exist to escape a poor optimum, so each one needs its own starting point.

    The hyperparameters were re-initialized from the third restart on only. With two restarts, the second one repeated the first exactly and doubled the training time for nothing.
    """
    calls = []
    original = GPyTraining.init_model_params
    monkeypatch.setattr(GPyTraining, "init_model_params", staticmethod(lambda model: (calls.append(1), original(model))[1]))
    torch.manual_seed(0)

    _emulator(outputs=Y[:, [0]], n_restarts=n_restarts).train_()

    assert len(calls) == n_restarts - 1


def test_the_lbfgs_optimizer_trains_the_emulator():
    """``optimizer='lbfgs'`` is documented but named a class that does not exist in torch, and L-BFGS needs the loss as a closure."""
    torch.manual_seed(0)
    emulator = _emulator(optimizer="lbfgs", training_iter=15)

    emulator.train_()
    prediction = emulator.predict_(input_sets=X)["output"]

    assert np.all(np.isfinite(prediction))
    assert np.abs(prediction - Y).max() < 0.05 * Y.std(axis=0).min(), "a noise-free smooth response is reproduced at its training runs"


def test_an_unknown_optimizer_is_refused_by_name():
    with pytest.raises(ValueError, match="sgd"):
        _emulator(optimizer="sgd").train_()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_training_outputs_are_refused_with_their_position(bad_value):
    """A failed full-complexity run must be named, not surface as a NaN in a Cholesky factorization."""
    outputs = Y.copy()
    outputs[7, 1] = bad_value

    with pytest.raises(ValueError, match=r"run\(s\) \[7\].*column\(s\) \[1\]"):
        _emulator(outputs=outputs).train_()

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    for mode in ("variables", "locations", "all"):
        emulator = MultiGPyTraining(X, outputs, kernel, training_iter=3, likelihood=likelihood, optimizer="adam",
                                    lr=0.07, number_quantities=1 if mode == "locations" else 2)
        with pytest.raises(ValueError, match=r"run\(s\) \[7\].*column\(s\) \[1\]"):
            getattr(emulator, f"train_tasks_{mode}")()


def test_training_without_a_finite_loss_says_so(monkeypatch):
    """If no restart ends with a finite loss, there are no hyperparameters to keep. That used to end in ``load_state_dict(None)``."""
    monkeypatch.setattr(gpytorch.mlls.ExactMarginalLogLikelihood, "forward",
                        lambda self, function_dist, target, *params, **kwargs:
                        sum(parameter.sum() for parameter in self.model.parameters()) * float("nan"))

    with pytest.raises(RuntimeError, match="finite"):
        _emulator(outputs=Y[:, [0]], training_iter=1, n_restarts=2).train_()
