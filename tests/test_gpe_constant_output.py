"""An output that is the same in every training run must not stop the emulators.

A calibration point that stays dry in all runs gives a water depth of 0.0 in every one of them, and a calibration target that does not respond to the calibration parameters at a point does the same with another value. The spread of such a column is zero. The emulators normalize every output column by its spread, which turned the column into NaN and stopped the training with ``NanError: cholesky_cpu: ... elements ... are NaN``, after the initial design had been simulated.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from scipy import stats
from scipy.linalg import block_diag

from hydroBayesCal.surrogate.bal_functions import BayesianInference
from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining

N_TRAIN = 12
RNG = np.random.default_rng(0)
X = RNG.uniform(size=(N_TRAIN, 2))
PRIOR = RNG.uniform(size=(200, 2))


def _outputs(constant):
    """Two calibration points with two targets each; the first target of the second point never responds."""
    return np.column_stack([np.sin(3.0 * X[:, 0]) + X[:, 1], 2.0 + X[:, 0] * X[:, 1],
                            np.full(N_TRAIN, constant), 1.0 + X[:, 1] ** 2])


def _kernel():
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))


@pytest.mark.parametrize("constant", [0.0, 0.3])
def test_single_output_emulators_predict_a_constant_output_without_uncertainty(constant):
    y = _outputs(constant)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    emulator = GPyTraining(collocation_points=X, model_evaluations=y, likelihood=likelihood, kernel=_kernel(),
                           training_iter=40, optimizer="adam", lr=0.07, verbose=False)
    torch.manual_seed(0)

    emulator.train_()
    prediction = emulator.predict_(input_sets=PRIOR, get_conf_int=True)

    assert np.all(np.isfinite(prediction["output"])) and np.all(np.isfinite(prediction["std"]))
    assert np.all(prediction["output"][:, 2] == constant), "the value of every training run, exactly"
    assert np.all(prediction["std"][:, 2] == 0.0), "no run gave any reason for an emulator uncertainty"
    assert np.all(prediction["upper_ci"][:, 2] == constant) and np.all(prediction["lower_ci"][:, 2] == constant)
    assert np.all(prediction["std"][:, [0, 1, 3]] > 0.0), "the other outputs keep their emulator"


@pytest.mark.parametrize("constant", [0.0, 0.3])
@pytest.mark.parametrize("mode, num_tasks", [("variables", 2), ("locations", 2), ("all", 4)])
def test_multitask_emulators_predict_a_constant_output_without_uncertainty(mode, num_tasks, constant):
    y = _outputs(constant)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks, noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    emulator = MultiGPyTraining(X, y, _kernel(), training_iter=40, likelihood=likelihood,
                                optimizer="adam", lr=0.07, number_quantities=2)
    torch.manual_seed(0)

    getattr(emulator, f"train_tasks_{mode}")()
    prediction = emulator.predict_(input_sets=PRIOR, get_conf_int=True, multitask_cov=True)

    assert np.all(np.isfinite(prediction["output"])) and np.all(np.isfinite(prediction["std"]))
    assert np.all(prediction["output"][:, 2] == constant), "the value of every training run, exactly"
    assert np.all(prediction["std"][:, 2] == 0.0)
    assert np.all(prediction["std"][:, [0, 1, 3]] > 0.0)
    # The covariance between the targets has a zero row and column for the constant target and is still a covariance.
    for blocks in prediction["multitask_cov"]:
        assert np.all(blocks[1][0, :] == 0.0) and np.all(blocks[1][:, 0] == 0.0)
        stats.multivariate_normal(mean=np.zeros(4), cov=block_diag(*blocks), allow_singular=True)  # must not raise


def test_bayesian_inference_works_on_the_predictions_of_a_constant_output():
    """The whole point: the calibration continues, and the constant output takes part with its measurement error only."""
    y = _outputs(0.0)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    emulator = GPyTraining(collocation_points=X, model_evaluations=y, likelihood=likelihood, kernel=_kernel(),
                           training_iter=40, optimizer="adam", lr=0.07, verbose=False)
    torch.manual_seed(0)
    emulator.train_()
    prediction = emulator.predict_(input_sets=PRIOR)

    inference = BayesianInference(
        model_predictions=prediction["output"], observations=np.array([[1.2, 2.2, 0.0, 1.3]]),
        error=np.array([0.05, 0.05, 0.01, 0.05]) ** 2, prior=PRIOR, model_error=prediction["std"],
        sampling_method="rejection_sampling")
    inference.estimate_bme()

    assert np.isfinite(inference.log_BME) and np.isfinite(inference.RE)
    assert inference.posterior is not None and len(inference.posterior) > 0


def test_a_constant_output_is_reported_once(monkeypatch):
    from hydroBayesCal.surrogate import gpe_gpytorch

    monkeypatch.setattr(gpe_gpytorch, "_REPORTED_CONSTANT_OUTPUTS", set())
    messages = []
    monkeypatch.setattr(gpe_gpytorch.logger_warn, "warning", lambda message, *a, **k: messages.append(str(message)))

    for _ in range(2):  # the emulator is trained again in every BAL iteration
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=2, noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        emulator = MultiGPyTraining(X, _outputs(0.0), _kernel(), training_iter=3, likelihood=likelihood,
                                    optimizer="adam", lr=0.07, number_quantities=2)
        emulator.train_tasks_variables()

    reports = [m for m in messages if "same in all" in m]
    assert len(reports) == 1
    assert "calibration point 2" in reports[0] and "target 1" in reports[0]
