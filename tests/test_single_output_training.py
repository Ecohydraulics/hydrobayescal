"""The single-output emulator trains its hyperparameters and is as accurate as the multitask one on the same runs.

Measured over six scramblings of the 16-run Sobol design below: the largest length scale of the single-output emulators was 5.8 to 6.9 (0.42 to 0.45 with the StepLR schedule of 1.13), and its validation error 0.35 to 1.02 times that of the multitask emulator (0.70 to 1.44 with 1.13). The design used here is one where 1.13 was clearly less accurate (1.26 times).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
from scipy.stats import qmc

from hydroBayesCal.surrogate.gpe_gpytorch import MultiGPyTraining, SOSequentialGPyTraining, multitask_likelihood

LOWER, UPPER = np.array([0.06, 0.001]), np.array([0.12, 0.05])
#: Above the largest length scale the single-output emulators reached with the StepLR schedule of 1.13.
MAX_LENGTHSCALE_UP_TO_1_13 = 2.0


def _wall_layer(points, heights=(0.1, 0.4, 0.8), depth=0.3, bulk_velocity=0.5):
    """Log layer of a k-epsilon flume at fixed discharge: C_mu and k_s -> [u, k] at three heights, interleaved."""
    points = np.atleast_2d(points)
    cmu, ks = points[:, :1], points[:, 1:]
    ustar = 0.41 * bulk_velocity / (np.log(30.0 * depth / ks) - 1.0)
    z = np.asarray(heights) * depth
    out = np.empty((points.shape[0], 2 * len(heights)))
    out[:, 0::2] = ustar / 0.41 * np.log(30.0 * z / ks)
    out[:, 1::2] = ustar ** 2 / np.sqrt(cmu) * np.exp(-2.0 * z / depth)
    return out


@pytest.fixture(scope="module")
def emulators():
    x = qmc.scale(qmc.Sobol(2, seed=2).random(16), LOWER, UPPER)
    y = _wall_layer(x)
    torch.manual_seed(0)
    single = SOSequentialGPyTraining.from_training_data(x, y, ["U_x", "TKE"])  # the drivers' settings
    single.train_()
    torch.manual_seed(0)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    multi = MultiGPyTraining(x, y, kernel, 150, multitask_likelihood(2), lr=0.07, number_quantities=2)
    multi.train_tasks_variables()
    test = LOWER + (UPPER - LOWER) * np.random.default_rng(2).random((300, 2))
    return single, multi, test, _wall_layer(test)


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
