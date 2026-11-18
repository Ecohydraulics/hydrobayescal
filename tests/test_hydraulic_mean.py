"""Tests of the trend means of the emulators (``sampling['gp_mean']``)."""
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from hydroBayesCal.surrogate import hydraulic_mean as hm
from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining, SOSequentialGPyTraining, multitask_likelihood

SLOPE, UNIT_DISCHARGE = 1e-3, 4.0


def _normal_flow(points):
    """Wide channel at normal flow: Manning's n and a discharge factor -> [depth, velocity]."""
    points = np.atleast_2d(points)
    q = UNIT_DISCHARGE * points[:, 1]
    depth = (points[:, 0] * q / np.sqrt(SLOPE)) ** 0.6
    return np.column_stack([depth, q / depth])


def _log_law(points):
    """Wide channel at normal flow with the log law of a sand roughness: k_s and a discharge factor -> [depth, velocity]."""
    from scipy.optimize import brentq

    rows = []
    for ks, factor in np.atleast_2d(points):
        q = UNIT_DISCHARGE * factor
        depth = brentq(lambda h: h * np.sqrt(9.81 * h * SLOPE) / 0.41 * np.log(12.0 * h / ks) - q, 0.5, 50.0)
        rows.append([depth, q / depth])
    return np.array(rows)


def _emulator(x, y, exponents, layout="single"):
    torch.manual_seed(0)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=x.shape[1]))
    if layout == "single":
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        likelihood.noise = 1e-5
        emulator = GPyTraining(x, y, kernel, 150, likelihood, lr=0.07, verbose=False, tp_normalization=True,
                               mean_exponents=exponents)
        emulator.train_()
    else:
        emulator = MultiGPyTraining(x, y, kernel, 150, multitask_likelihood(2), lr=0.07, number_quantities=2,
                                    tp_normalization=True, mean_exponents=exponents)
        emulator.train_tasks_variables()
    return emulator


@pytest.mark.parametrize("law, quantity, expected", [
    ("MANN", "WATER DEPTH", 0.6), ("MANN", "SCALAR VELOCITY", -0.6),
    ("STRI", "FREE SURFACE", -0.6), ("STRI", "VELOCITY U", 0.6),
    ("CHEZ", "WATER_DEPTH", -2.0 / 3.0), ("CHEZ", "U_MAG", 2.0 / 3.0),
    ("NIKU", "WATER DEPTH", hm.LOG_FEATURE), ("COWH", "U_x", hm.LOG_FEATURE),
    ("NOFR", "WATER DEPTH", 1.0), (None, "WATER DEPTH", 1.0), ("MANN", "TURBULENT ENERG", 1.0),
])
def test_the_trend_follows_the_normal_flow_scaling_of_the_friction_law(law, quantity, expected):
    assert hm.trend_exponent(law, quantity) == pytest.approx(expected)


def test_the_exponents_have_one_row_per_output_column_in_the_interleaved_order():
    exponents = hm.mean_exponents("hydraulic", ["zone1", "gaiaX"], ["WATER DEPTH", "SCALAR VELOCITY"], 3,
                                  [[0.02, 0.05], [0.04, 0.07]], {"zone1": "MANN"})
    assert exponents.shape == (6, 2)
    np.testing.assert_allclose(exponents[0::2, 0], 0.6)
    np.testing.assert_allclose(exponents[1::2, 0], -0.6)
    np.testing.assert_allclose(exponents[:, 1], 1.0)
    np.testing.assert_allclose(hm.mean_exponents("linear", ["zone1", "gaiaX"], ["WATER DEPTH"], 2), 1.0)
    assert hm.mean_exponents("constant", ["zone1"], ["WATER DEPTH"], 2) is None
    with pytest.raises(ValueError, match="Unknown sampling"):
        hm.mean_exponents("quadratic", ["zone1"], ["WATER DEPTH"], 2)


@pytest.mark.parametrize("driver", ["bal_telemac.py", "bal_openfoam.py", "bal_delft3d.py", "bal_telemac_multiflow.py"])
def test_every_driver_checks_gp_mean_right_after_loading_the_configuration(driver):
    """An unknown value has to stop a calibration before the initial runs, not when the first emulator is trained."""
    import pathlib

    source = (pathlib.Path(hm.__file__).parents[1] / "drivers" / driver).read_text()
    main = source[source.index("def main("):]
    assert main.index("check_gp_mean(") < main.index("load_config(args.config)") + 200


def test_a_range_that_reaches_zero_gets_the_linear_trend():
    exponents = hm.mean_exponents("hydraulic", ["zone1", "zone2"], ["WATER DEPTH"], 1,
                                  [[0.0, 0.5], [0.002, 0.6]], {"zone1": "NIKU", "zone2": "NIKU"})
    np.testing.assert_allclose(exponents, [[1.0, hm.LOG_FEATURE]])


def test_the_mean_reproduces_a_response_that_is_its_trend_also_on_scaled_inputs():
    rng = np.random.default_rng(0)
    theta = 0.02 + 0.04 * rng.random((20, 2))
    response = 1.5 + 2.0 * theta[:, 0] ** 0.6 - 0.3 * np.log(theta[:, 1])
    x_min, x_range = theta.min(axis=0), np.ptp(theta, axis=0)
    scaled = torch.tensor((theta - x_min) / x_range)
    mean = hm.HydraulicMean([0.6, hm.LOG_FEATURE], scaled, torch.tensor(response), x_min, x_range)
    new = 0.02 + 0.04 * rng.random((50, 2))
    expected = 1.5 + 2.0 * new[:, 0] ** 0.6 - 0.3 * np.log(new[:, 1])
    with torch.no_grad():
        predicted = mean(torch.tensor((new - x_min) / x_range)).numpy()
    np.testing.assert_allclose(predicted, expected, rtol=1e-3)


def test_too_few_runs_for_the_trend_keep_the_constant_mean():
    x = torch.rand(5, 2, dtype=torch.float64)
    assert hm.trend_mean([0.6, 1.0], x, torch.rand(5, dtype=torch.float64)) is None
    assert isinstance(hm.trend_mean([0.6, 1.0], torch.rand(6, 2, dtype=torch.float64)), hm.HydraulicMean)


@pytest.mark.parametrize("layout", ["single", "variables"])
def test_the_hydraulic_mean_emulates_a_sand_roughness_response_better_than_the_constant_one(layout):
    """A sand roughness over two orders of magnitude: with 8 designs of 10 runs, the hydraulic mean had 6 to 42 % of the error of the constant mean, in both layouts."""
    lo, hi = np.array([0.002, 0.8]), np.array([0.6, 1.2])
    x = lo + (hi - lo) * np.random.default_rng(10).random((10, 2))
    test = lo + (hi - lo) * np.random.default_rng(2).random((400, 2))
    exponents = hm.mean_exponents("hydraulic", ["zone1", "Q"], ["WATER DEPTH", "SCALAR VELOCITY"], 1,
                                  [[0.002, 0.6], [0.8, 1.2]], {"zone1": "NIKU"})
    errors = {}
    for name, exps in (("constant", None), ("hydraulic", exponents)):
        emulator = _emulator(x, _log_law(x), exps, layout)
        prediction = emulator.predict_(test)["output"]
        errors[name] = np.sqrt(np.mean((prediction - _log_law(test)) ** 2, axis=0)).mean()
    assert errors["hydraulic"] < 0.6 * errors["constant"], errors


def test_an_emulator_with_the_hydraulic_mean_predicts_the_same_after_pickling(tmp_path):
    x = np.random.default_rng(3).random((8, 2)) * 0.04 + 0.02
    exponents = np.array([[0.6, 0.6], [-0.6, -0.6]])
    emulator = _emulator(x, _normal_flow(np.column_stack([x[:, 0], 1 + x[:, 1]])), exponents)
    path = tmp_path / "gpe.pkl"
    path.write_bytes(pickle.dumps(emulator))
    restored = pickle.loads(path.read_bytes())
    np.testing.assert_allclose(restored.predict_(x)["output"], emulator.predict_(x)["output"])


def test_so_sequential_hands_every_target_the_rows_of_its_columns():
    x = np.random.default_rng(4).random((8, 2))
    outputs = np.random.default_rng(5).random((8, 4))
    exponents = np.array([[0.6, 1.0], [-0.6, 1.0]] * 2)
    container = SOSequentialGPyTraining.from_training_data(x, outputs, ["WATER DEPTH", "SCALAR VELOCITY"],
                                                           mean_exponents=exponents)
    np.testing.assert_allclose(container["WATER DEPTH"].mean_exponents, [[0.6, 1.0]] * 2)
    np.testing.assert_allclose(container["SCALAR VELOCITY"].mean_exponents, [[-0.6, 1.0]] * 2)


def test_telemac_reads_the_law_of_every_friction_zone_and_of_the_steering_file(tmp_path):
    from hydroBayesCal.telemac.control_telemac import TelemacModel

    (tmp_path / "friction.tbl").write_text(
        "* no typeB rB\n*\n\n2  NIKU  0.05  NULL\n3  mann  0.03  NULL\n4  STRI  30.0  NULL\n")
    (tmp_path / "run.cas").write_text("/ steering\nLAW OF BOTTOM FRICTION : 4\nFRICTION COEFFICIENT = 0.03\n")
    model = object.__new__(TelemacModel)
    model.friction_file = "friction.tbl"
    model.fr_tbl, model.tm_cas = str(tmp_path / "friction.tbl"), str(tmp_path / "run.cas")
    model.calibration_parameters = ["zone2", "zone3", "zone4", "zone9", "FRICTION COEFFICIENT",
                                    "gaiaCLASSES SHIELDS PARAMETERS 1"]
    assert model.friction_laws() == {"zone2": "NIKU", "zone3": "MANN", "zone4": "STRI",
                                     "FRICTION COEFFICIENT": "MANN"}


def test_delft3d_and_openfoam_name_the_law_of_their_roughness_parameter():
    from hydroBayesCal.delft3d.control_delft3d import Delft3DModel
    from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel

    delft3d = object.__new__(Delft3DModel)
    delft3d.calibration_parameters, delft3d.roughness_formulation = ["roughness", "Vicouv"], "Chezy"
    assert delft3d.friction_laws() == {"roughness": "CHEZ"}
    openfoam = object.__new__(OpenFOAMModel)
    openfoam.calibration_parameters = ["ks", "Cmu"]
    assert openfoam.friction_laws() == {"ks": "NIKU"}
