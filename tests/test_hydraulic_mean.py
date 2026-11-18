"""Tests of the trend means of the emulators (``sampling['gp_mean']``)."""
import pathlib
import pickle

import numpy as np
import pytest
from analytic_models import wall_layer

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from hydroBayesCal.surrogate import hydraulic_mean as hm
from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining, SOSequentialGPyTraining, multitask_likelihood

SLOPE, UNIT_DISCHARGE = 1e-3, 4.0
#: Largest ratio of the validation errors with the hydraulic and with the constant mean in the emulation tests; see their docstrings for the measured ratios.
THRESHOLD = 0.6


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


def _emulator(x, y, features, layout="single"):
    torch.manual_seed(0)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=x.shape[1]))
    if layout == "single":
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        likelihood.noise = 1e-5
        emulator = GPyTraining(x, y, kernel, 150, likelihood, lr=0.07, verbose=False, tp_normalization=True,
                               mean_features=features)
        emulator.train_()
    else:
        emulator = MultiGPyTraining(x, y, kernel, 150, multitask_likelihood(2), lr=0.07, number_quantities=2,
                                    tp_normalization=True, mean_features=features)
        emulator.train_tasks_variables()
    return emulator


def _relative_errors(x, test, response, features, layout):
    errors = {}
    for name, feats in (("constant", None), ("hydraulic", features)):
        error = _emulator(x, response(x), feats, layout).predict_(test)["output"] - response(test)
        errors[name] = (np.sqrt(np.mean(error ** 2, axis=0)) / response(test).std(axis=0)).mean()
    return errors


@pytest.mark.parametrize("law, quantity, expected", [
    ("MANN", "WATER DEPTH", ("power", 0, 0.6)), ("MANN", "SCALAR VELOCITY", ("power", 0, -0.6)),
    ("STRI", "FREE SURFACE", ("power", 0, -0.6)), ("STRI", "VELOCITY U", ("power", 0, 0.6)),
    ("CHEZ", "WATER_DEPTH", ("power", 0, -2.0 / 3.0)), ("CHEZ", "U_MAG", ("power", 0, 2.0 / 3.0)),
    ("MANN", "TURBULENT ENERG.", ("power", 0, 0.6)), ("MANN", "u_fluct", ("power", 0, 0.3)),
    ("NIKU", "WATER DEPTH", ("log", 0)), ("COWH", "U_x", ("log", 0)), ("NIKU", "TKE", ("log", 0)),
    ("NOFR", "WATER DEPTH", None), (None, "WATER DEPTH", None), ("MANN", "ALPHA_WATER", None),
])
def test_the_trend_follows_the_physical_scaling_of_the_friction_law(law, quantity, expected):
    feature = hm.friction_feature(0, law, hm.target_kind(quantity))
    if expected is None:
        assert feature is None
    else:
        assert feature[:2] == expected[:2] and feature[2:] == pytest.approx(expected[2:])


def test_the_features_have_one_list_per_output_column_in_the_interleaved_order():
    features = hm.trend_features("hydraulic", ["zone1", "gaiaX"], ["WATER DEPTH", "SCALAR VELOCITY"], 3,
                                 [[0.02, 0.05], [0.04, 0.07]], {"zone1": "MANN"})
    assert len(features) == 6
    assert all(f == [("power", 0, 0.6), ("power", 1, 1.0)] for f in features[0::2])
    assert all(f == [("power", 0, -0.6), ("power", 1, 1.0)] for f in features[1::2])
    assert hm.trend_features("linear", ["zone1", "gaiaX"], ["WATER DEPTH"], 2) == [[("power", 0, 1.0), ("power", 1, 1.0)]] * 2
    assert hm.trend_features("constant", ["zone1"], ["WATER DEPTH"], 2) is None
    with pytest.raises(ValueError, match="Unknown sampling"):
        hm.trend_features("quadratic", ["zone1"], ["WATER DEPTH"], 2)


def test_openfoam_targets_get_the_log_layer_scaling_of_ks_and_cmu():
    """k = u*^2/sqrt(C_mu) in the log layer; the wall roughness enters every target through the log law."""
    features = hm.trend_features("hydraulic", ["Cmu", "ks", "C1"], ["U_x", "TKE", "u_fluct", "ALPHA_WATER"], 1,
                                 [[0.06, 0.12], [0.001, 0.05], [1.3, 1.6]], {"ks": "NIKU"}, {"Cmu": hm.CMU})
    u, k, fluct, alpha = features
    assert u == [("power", 0, 1.0), ("log", 1), ("power", 2, 1.0)]
    assert k == [("power", 0, -0.5), ("log", 1), ("power", 2, 1.0)]
    assert fluct == [("power", 0, -0.25), ("log", 1), ("power", 2, 1.0)]
    assert alpha == [("power", 0, 1.0), ("power", 1, 1.0), ("power", 2, 1.0)]


@pytest.mark.parametrize("driver", ["bal_telemac.py", "bal_openfoam.py", "bal_delft3d.py", "bal_telemac_multiflow.py"])
def test_every_driver_checks_gp_mean_right_after_loading_the_configuration(driver):
    """An unknown value has to stop a calibration before the initial runs, not when the first emulator is trained."""
    source = (pathlib.Path(hm.__file__).parents[1] / "drivers" / driver).read_text()
    main = source[source.index("def main("):]
    assert main.index("check_gp_mean(") < main.index("load_config(args.config)") + 200


def test_a_range_that_reaches_zero_gets_the_linear_trend():
    features = hm.trend_features("hydraulic", ["zone1", "zone2"], ["WATER DEPTH"], 1,
                                 [[0.0, 0.5], [0.002, 0.6]], {"zone1": "NIKU", "zone2": "NIKU"})
    assert features == [[("power", 0, 1.0), ("log", 1)]]


def test_the_mean_reproduces_a_response_that_is_its_trend_also_on_scaled_inputs():
    rng = np.random.default_rng(0)
    theta = 0.02 + 0.04 * rng.random((20, 2))
    response = 1.5 + 2.0 * theta[:, 0] ** 0.6 - 0.3 * np.log(theta[:, 1])
    x_min, x_range = theta.min(axis=0), np.ptp(theta, axis=0)
    scaled = torch.tensor((theta - x_min) / x_range)
    mean = hm.HydraulicMean([("power", 0, 0.6), ("log", 1)], scaled, torch.tensor(response), x_min, x_range)
    new = 0.02 + 0.04 * rng.random((50, 2))
    expected = 1.5 + 2.0 * new[:, 0] ** 0.6 - 0.3 * np.log(new[:, 1])
    with torch.no_grad():
        predicted = mean(torch.tensor((new - x_min) / x_range)).numpy()
    np.testing.assert_allclose(predicted, expected, rtol=1e-3)


def test_too_few_runs_for_the_trend_keep_the_constant_mean():
    features = [("power", 0, 0.6), ("power", 1, 1.0)]
    x = torch.rand(5, 2, dtype=torch.float64)
    assert hm.trend_mean(features, x, torch.rand(5, dtype=torch.float64)) is None
    assert isinstance(hm.trend_mean(features, torch.rand(6, 2, dtype=torch.float64)), hm.HydraulicMean)


@pytest.mark.parametrize("feature", [("power", 2, 0.5), ("log",), ("cube", 0), [("power", 0, 1.0)]])
def test_an_invalid_feature_is_refused(feature):
    with pytest.raises(ValueError):
        hm.HydraulicMean([feature], torch.rand(6, 2, dtype=torch.float64))


@pytest.mark.parametrize("layout", ["single", "variables"])
def test_the_hydraulic_mean_emulates_a_sand_roughness_response_better_than_the_constant_one(layout):
    """A sand roughness over two orders of magnitude: with 8 designs of 10 runs, the hydraulic mean had 6 to 42 % of the error of the constant mean, in both layouts."""
    lo, hi = np.array([0.002, 0.8]), np.array([0.6, 1.2])
    x = lo + (hi - lo) * np.random.default_rng(10).random((10, 2))
    test = lo + (hi - lo) * np.random.default_rng(2).random((400, 2))
    features = hm.trend_features("hydraulic", ["zone1", "Q"], ["WATER DEPTH", "SCALAR VELOCITY"], 1,
                                 [[0.002, 0.6], [0.8, 1.2]], {"zone1": "NIKU"})
    errors = _relative_errors(x, test, _log_law, features, layout)
    assert errors["hydraulic"] < THRESHOLD * errors["constant"], errors


@pytest.mark.parametrize("layout", ["single", "variables"])
def test_the_hydraulic_mean_emulates_the_log_layer_of_a_k_epsilon_flume_better_than_the_constant_one(layout):
    """C_mu and k_s with the ranges of the shipped OpenFOAM configuration, velocity and turbulent kinetic energy at three heights. Over 8 scramblings of a 16-run Sobol design, the hydraulic mean had 26 to 51 % (single output) and 12 to 33 % (variables) of the error of the constant mean."""
    from scipy.stats import qmc

    lo, hi = np.array([0.06, 0.001]), np.array([0.12, 0.05])
    x = qmc.scale(qmc.Sobol(2, seed=4).random(16), lo, hi)
    test = lo + (hi - lo) * np.random.default_rng(2).random((400, 2))
    features = hm.trend_features("hydraulic", ["Cmu", "ks"], ["U_x", "TKE"], 3, [[0.06, 0.12], [0.001, 0.05]],
                                 {"ks": "NIKU"}, {"Cmu": hm.CMU})
    errors = _relative_errors(x, test, wall_layer, features, layout)
    assert errors["hydraulic"] < THRESHOLD * errors["constant"], errors


def test_an_emulator_with_the_hydraulic_mean_predicts_the_same_after_pickling(tmp_path):
    x = np.random.default_rng(3).random((8, 2)) * 0.04 + 0.02
    features = [[("power", 0, 0.6), ("power", 1, 0.6)], [("power", 0, -0.6), ("log", 1)]]
    emulator = _emulator(x, _normal_flow(np.column_stack([x[:, 0], 1 + x[:, 1]])), features)
    path = tmp_path / "gpe.pkl"
    path.write_bytes(pickle.dumps(emulator))
    restored = pickle.loads(path.read_bytes())
    np.testing.assert_allclose(restored.predict_(x)["output"], emulator.predict_(x)["output"])


def test_so_sequential_hands_every_target_the_features_of_its_columns():
    x = np.random.default_rng(4).random((8, 2))
    outputs = np.random.default_rng(5).random((8, 4))
    depth, velocity = [("power", 0, 0.6), ("power", 1, 1.0)], [("power", 0, -0.6), ("power", 1, 1.0)]
    container = SOSequentialGPyTraining.from_training_data(x, outputs, ["WATER DEPTH", "SCALAR VELOCITY"],
                                                           mean_features=[depth, velocity] * 2)
    assert container["WATER DEPTH"].mean_features == [depth] * 2
    assert container["SCALAR VELOCITY"].mean_features == [velocity] * 2


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


def test_delft3d_and_openfoam_name_the_roles_of_their_parameters():
    from hydroBayesCal.delft3d.control_delft3d import Delft3DModel
    from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel

    delft3d = object.__new__(Delft3DModel)
    delft3d.calibration_parameters, delft3d.roughness_formulation = ["roughness", "Vicouv"], "Chezy"
    assert delft3d.friction_laws() == {"roughness": "CHEZ"}
    assert delft3d.turbulence_constants() == {}
    openfoam = object.__new__(OpenFOAMModel)
    openfoam.calibration_parameters = ["ks", "Cmu", "C1"]
    assert openfoam.friction_laws() == {"ks": "NIKU"}
    assert openfoam.turbulence_constants() == {"Cmu": hm.CMU}


# Findings of the review of PR 33.

def test_telemac_3d_names_its_bottom_friction_coefficient_and_defaults_to_nikuradse(tmp_path):
    """TELEMAC-3D calls the keyword FRICTION COEFFICIENT FOR THE BOTTOM, and LAW OF BOTTOM FRICTION defaults to 5 (telemac3d.dico); TELEMAC-2D has no default."""
    from hydroBayesCal.telemac.control_telemac import TelemacModel

    (tmp_path / "run.cas").write_text("FRICTION COEFFICIENT FOR THE BOTTOM = 0.01\n")
    model = object.__new__(TelemacModel)
    model.friction_file, model.fr_tbl, model.tm_cas = None, str(tmp_path / "none.tbl"), str(tmp_path / "run.cas")
    model.calibration_parameters = ["FRICTION COEFFICIENT FOR THE BOTTOM"]
    model.tm_xd = "Telemac3d"
    assert model.friction_laws() == {"FRICTION COEFFICIENT FOR THE BOTTOM": "NIKU"}
    model.tm_xd = "Telemac2d"
    assert model.friction_laws() == {}
    (tmp_path / "run.cas").write_text("LAW OF BOTTOM FRICTION = 6\n")
    model.calibration_parameters = ["FRICTION COEFFICIENT"]
    assert model.friction_laws() == {"FRICTION COEFFICIENT": "COWH"}


@pytest.mark.parametrize("quantity, kind", [("3D VELOCITY MAGNITUDE", "velocity"), ("U_magnitude", "velocity"),
                                            ("TKE_MODELED", "tke"), ("TURBULENT ENERG", "tke")])
def test_the_target_names_of_the_bindings_are_recognized(quantity, kind):
    assert hm.target_kind(quantity) == kind


def test_a_prior_sample_at_or_below_zero_does_not_blow_up_the_trend():
    """With a normal prior, param_values are [mean, std]: prior samples can reach zero or below, where a power or logarithm is undefined."""
    theta = torch.tensor([[0.005], [0.01], [0.02], [0.03], [0.04], [0.05]], dtype=torch.float64)
    mean = hm.HydraulicMean([("log", 0)], theta, torch.log(theta[:, 0]))
    with torch.no_grad():
        values = mean(torch.tensor([[-0.01], [0.0], [1e-12]], dtype=torch.float64))
    assert torch.all(torch.isfinite(values))
    with torch.no_grad():
        floor = mean(torch.tensor([[hm.FLOOR_FRACTION * 0.005]], dtype=torch.float64))
    assert torch.allclose(values[:3], floor)  # flat below the floor


def test_a_shaped_feature_of_a_parameter_with_nonpositive_training_values_becomes_linear():
    theta = torch.tensor([[-0.01], [0.01], [0.02], [0.03], [0.04], [0.05]], dtype=torch.float64)
    mean = hm.trend_mean([("log", 0)], theta, theta[:, 0])
    assert mean.features == [("power", 0, 1.0)]


def test_the_emulators_scale_the_parameters_whenever_they_have_a_trend():
    x = np.random.default_rng(0).random((8, 2))
    features = [[("power", 0, 1.0), ("power", 1, 1.0)]] * 2
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    single = GPyTraining(x, np.ones((8, 2)), kernel, 1, gpytorch.likelihoods.GaussianLikelihood(),
                         tp_normalization=False, mean_features=features)
    multi = MultiGPyTraining(x, np.ones((8, 2)), kernel, 1, multitask_likelihood(2), number_quantities=2,
                             tp_normalization=False, mean_features=features)
    assert single.tp_norm and multi.tp_norm
    assert not GPyTraining(x, np.ones((8, 2)), kernel, 1, gpytorch.likelihoods.GaussianLikelihood(),
                           tp_normalization=False).tp_norm


def test_a_trend_that_does_not_fit_the_output_columns_falls_back_to_the_constant_mean():
    class Model:
        calibration_parameters, calibration_quantities, nloc, parameter_ranges = ["ks"], ["U_x"], 2, [[0.001, 0.05]]

        def trend_features(self, gp_mean):
            return hm.trend_features(gp_mean, self.calibration_parameters, self.calibration_quantities, self.nloc,
                                     self.parameter_ranges, {"ks": "NIKU"})

    assert len(hm.driver_trend_features(Model(), "hydraulic", "gpy", n_columns=2)) == 2
    assert hm.driver_trend_features(Model(), "hydraulic", "gpy", n_columns=3) is None
    assert hm.driver_trend_features(Model(), "hydraulic", "skl", n_columns=2) is None
    assert hm.driver_trend_features(Model(), "constant", "gpy", n_columns=2) is None


def test_a_multiflow_model_reports_its_trend_once_for_all_flows(caplog):
    from hydroBayesCal.telemac.multiflow_telemac import MultiflowTelemacModel

    flow = type("Flow", (), {"friction_laws": lambda self: {"zone1": "MANN"}, "turbulence_constants": lambda self: {},
                             "nloc": 2})()
    model = object.__new__(MultiflowTelemacModel)
    model.models, model.nloc = [flow, flow, flow], 6
    model.calibration_parameters, model.calibration_quantities = ["zone1"], ["WATER DEPTH"]
    model.parameter_ranges = [[0.02, 0.05]]
    with caplog.at_level("INFO", logger="HydroBayesCal"):
        features = model.trend_features("hydraulic")
    assert features == [[("power", 0, 0.6)]] * 6
    assert sum("Hydraulic GP mean" in record.getMessage() for record in caplog.records) == 1
