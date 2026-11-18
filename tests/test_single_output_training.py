"""The single-output emulator trains its hyperparameters and is as accurate as the multitask one on the same runs.

Measured over six scramblings of the 16-run Sobol design below, with the training settings of the drivers (``SINGLE_OUTPUT_ITERATIONS`` at ``SINGLE_OUTPUT_LEARNING_RATE``): the largest length scale of the single-output emulators was 10.3 to 13.1 (0.42 to 0.45 with the StepLR schedule of 1.13), and its validation error 0.31 to 0.91 times that of the multitask emulator (0.70 to 1.44 with 1.13), the same at 1 and 4 threads. The design used here is one where 1.13 was clearly less accurate (1.26 times).
"""
import numpy as np
import pytest
from analytic_models import wall_layer

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
from scipy.stats import qmc

from hydroBayesCal.surrogate import gpe_gpytorch
from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining, SOSequentialGPyTraining, multitask_likelihood

LOWER, UPPER = np.array([0.06, 0.001]), np.array([0.12, 0.05])
#: Above the largest length scale the single-output emulators reached with the StepLR schedule of 1.13.
MAX_LENGTHSCALE_UP_TO_1_13 = 2.0


@pytest.fixture(scope="module")
def emulators():
    x = qmc.scale(qmc.Sobol(2, seed=2).random(16), LOWER, UPPER)
    y = wall_layer(x)
    torch.manual_seed(0)
    single = SOSequentialGPyTraining.from_training_data(x, y, ["U_x", "TKE"])  # the drivers' settings
    single.train_()
    torch.manual_seed(0)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    multi = MultiGPyTraining(x, y, kernel, 150, multitask_likelihood(2), lr=0.07, number_quantities=2)
    multi.train_tasks_variables()
    test = LOWER + (UPPER - LOWER) * np.random.default_rng(2).random((300, 2))
    return single, multi, test, wall_layer(test)


def test_the_single_output_hyperparameters_leave_their_initial_values(emulators):
    """Up to 1.13 a StepLR schedule halved the learning rate every 5 iterations: the length scales could not leave about [0.4, 1.1] around their initial 0.69."""
    single = emulators[0]
    lengthscales = np.concatenate([g["cl_hp"] for q in single.calibration_quantities for g in single[q].gp_list])
    assert lengthscales.max() > MAX_LENGTHSCALE_UP_TO_1_13, lengthscales


def test_the_single_output_emulator_is_as_accurate_as_the_multitask_one(emulators):
    single, multi, test, truth = emulators
    spread = truth.std(axis=0)
    error = {name: np.mean(np.sqrt(np.mean((emulator.predict_(test)["output"] - truth) ** 2, axis=0)) / spread)
             for name, emulator in (("single", single), ("multi", multi))}
    assert error["single"] <= error["multi"], error


@pytest.mark.parametrize("driver", ["bal_telemac.py", "bal_openfoam.py", "bal_delft3d.py"])
def test_every_driver_trains_the_single_output_emulator_alike(driver):
    """With a constant learning rate, 150 steps at 0.07 (TELEMAC) and 100 at 0.5 (OpenFOAM, Delft3D) gave different emulators for the same calibration, and 0.5 without input scaling an error of 0.6 to 0.9."""
    import ast
    import pathlib

    source = (pathlib.Path(gpe_gpytorch.__file__).parents[1] / "drivers" / driver).read_text()
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and getattr(node.func, "id", getattr(node.func, "attr", None)) == "GPyTraining"]
    assert calls
    for call in calls:
        options = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        assert options.get("training_iter") == "SINGLE_OUTPUT_ITERATIONS", options
        assert options.get("lr") == "SINGLE_OUTPUT_LEARNING_RATE", options


def test_the_sequential_emulators_train_like_the_single_output_one_of_the_drivers():
    import inspect

    defaults = inspect.signature(SOSequentialGPyTraining.from_training_data).parameters
    assert defaults["training_iter"].default == gpe_gpytorch.SINGLE_OUTPUT_ITERATIONS
    assert defaults["lr"].default == gpe_gpytorch.SINGLE_OUTPUT_LEARNING_RATE


def test_the_loss_of_the_kept_hyperparameters_is_the_one_compared_between_restarts():
    """The loss of the last iteration is computed before the last optimizer step; with a constant learning rate that step is not negligible."""
    x = qmc.scale(qmc.Sobol(2, seed=1).random(12), LOWER, UPPER)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    torch.manual_seed(0)
    emulator = GPyTraining(x, wall_layer(x)[:, :1], kernel, 20, likelihood, lr=0.5, verbose=False, n_restarts=2)
    emulator.train_()
    entry = emulator.gp_list[0]
    gp, likelihood = entry["gp"], entry["likelihood"]
    gp.train()
    likelihood.train()
    with torch.no_grad():
        loss = -gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)(gp(*gp.train_inputs), gp.train_targets)
    assert entry["loss"] == pytest.approx(float(loss), rel=1e-9)
