"""Tests for the floating-point precision of the gpytorch emulators.

The emulators compute in double precision. Up to release 1.10.1 they computed in single precision, which resolves about seven digits. That is not enough where the result is a small difference of large terms: the posterior covariance of an emulator that is nearly certain lost all its digits, and the covariance matrix of 30 to 40 training runs of a smooth model response could not be factorized, which stopped the training with ``NotPSDError`` in a quarter to a third of the attempts.
"""
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from scipy import stats
from scipy.linalg import block_diag

from hydroBayesCal.surrogate import gpe_gpytorch
from hydroBayesCal.surrogate.gpe_gpytorch import GPyTraining, MultiGPyTraining, SOSequentialGPyTraining

STATIONS = np.linspace(0.2, 1.0, 3)


def _response(points):
    """Two calibration targets at three calibration points, smooth in two parameters (the model of the end-to-end tests)."""
    first = np.array([[np.sin(3.0 * a * s) + b * s ** 2 for s in STATIONS] for a, b in points])
    second = np.array([[2.0 + a * s - 0.5 * b * np.cos(2.0 * s) for s in STATIONS] for a, b in points])
    out = np.empty((len(points), 6))
    out[:, 0::2], out[:, 1::2] = first, second
    return out


def _kernel():
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))


def _single_output(x, y, training_iter=20, **options):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    emulator = GPyTraining(collocation_points=x, model_evaluations=y, likelihood=likelihood, kernel=_kernel(),
                           training_iter=training_iter, optimizer="adam", lr=0.07, verbose=False, **options)
    emulator.train_()
    return emulator


def _multitask(x, y, mode="variables", training_iter=20, **options):
    """The emulator of the drivers for several calibration targets."""
    num_tasks = {"variables": 2, "locations": 3, "all": 6}[mode]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks, noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5
    emulator = MultiGPyTraining(x, y, _kernel(), training_iter=training_iter, likelihood=likelihood,
                                optimizer="adam", lr=0.07, number_quantities=2, **options)
    getattr(emulator, f"train_tasks_{mode}")()
    return emulator


def _models(emulator):
    return [entry["gp"] for entry in emulator.gp_list]


def _types(emulator):
    types = set()
    for model in _models(emulator):
        types.update(parameter.dtype for parameter in model.parameters())
        types.update(tensor.dtype for tensor in model.train_inputs)
        types.add(model.train_targets.dtype)
    return types


RNG = np.random.default_rng(0)
X = RNG.uniform(size=(9, 2))
Y = _response(X)
PRIOR = RNG.uniform(size=(50, 2))


# --------------------------------------------------------------------------- #
# the precision of the emulators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("build", [lambda **o: _single_output(X, Y, **o),
                                   lambda **o: _multitask(X, Y, "variables", **o),
                                   lambda **o: _multitask(X, Y, "locations", **o),
                                   lambda **o: _multitask(X, Y, "all", **o)],
                         ids=["single-output", "tasks-variables", "tasks-locations", "tasks-all"])
def test_emulators_compute_in_double_precision_by_default(build):
    emulator = build()

    assert _types(emulator) == {torch.float64}, "parameters and training data of every model"
    prediction = emulator.predict_(input_sets=PRIOR)
    assert prediction["output"].dtype == np.float64 and prediction["std"].dtype == np.float64
    assert np.all(np.isfinite(prediction["output"])) and np.all(prediction["std"] > 0.0)


@pytest.mark.parametrize("build", [lambda **o: _single_output(X, Y, **o),
                                   lambda **o: _multitask(X, Y, "variables", **o)],
                         ids=["single-output", "tasks-variables"])
def test_single_precision_can_be_requested(build):
    """For comparison with the results of releases up to 1.10.1."""
    assert _types(build(precision="single")) == {torch.float32}


def test_the_sequential_single_output_emulator_passes_the_precision_on():
    default = SOSequentialGPyTraining.from_training_data(X, Y, ["WATER DEPTH", "SCALAR VELOCITY"], training_iter=3)
    single = SOSequentialGPyTraining.from_training_data(X, Y, ["WATER DEPTH", "SCALAR VELOCITY"], training_iter=3,
                                                        precision="single")
    default.train_()
    single.train_()

    assert all(_types(default[quantity]) == {torch.float64} for quantity in ("WATER DEPTH", "SCALAR VELOCITY"))
    assert all(_types(single[quantity]) == {torch.float32} for quantity in ("WATER DEPTH", "SCALAR VELOCITY"))


@pytest.mark.parametrize("emulator_class", [GPyTraining, MultiGPyTraining])
def test_an_unknown_precision_is_refused_by_name(emulator_class):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    with pytest.raises(ValueError, match="half"):
        emulator_class(X, Y, _kernel(), 5, likelihood, precision="half")


@pytest.mark.parametrize("build", [lambda **o: _single_output(X, Y, **o),
                                   lambda **o: _multitask(X, Y, "variables", **o)],
                         ids=["single-output", "tasks-variables"])
def test_double_precision_resolves_a_parameter_step_that_single_precision_does_not(build):
    """What the precision means for everything that differentiates or subtracts emulator outputs.

    Two parameter sets that differ by 1e-9 are the same number in single precision, so the emulator returns the same output for both. In double precision the outputs differ, by the slope of the emulator times the step.
    """
    point = np.array([[0.4, 0.6]])
    shifted = point + np.array([[1.0e-9, 0.0]])

    def difference(precision):
        torch.manual_seed(0)
        emulator = build(precision=precision)
        return emulator.predict_(input_sets=shifted)["output"] - emulator.predict_(input_sets=point)["output"]

    assert np.all(difference("single") == 0.0)
    step = difference("double")
    assert np.all(step != 0.0) and np.all(np.abs(step) < 1.0e-6)


# --------------------------------------------------------------------------- #
# what double precision removes
# --------------------------------------------------------------------------- #
def test_the_covariance_between_targets_is_valid_without_clipping(monkeypatch):
    """The emulator whose single-precision covariance stopped BAL in release 1.10.0, now in double precision.

    Its correlation lengths are about ten times the parameter ranges at one calibration point, so that its posterior variance is six orders of magnitude below its prior variance. Single precision cannot resolve that difference, and 3 to 8 % of the covariance blocks came out indefinite; since 1.10.1 they are clipped. Double precision resolves it with nine digits to spare: every block is a covariance as it is computed.
    """
    monkeypatch.setattr(gpe_gpytorch, "_REPORTED_CLIPPED_COVARIANCE", False)
    rng = np.random.default_rng(2)
    torch.manual_seed(2)
    x = rng.uniform(size=(9, 2))
    emulator = _multitask(x, _response(x), "variables", training_iter=150)

    covariance = emulator.predict_(input_sets=rng.uniform(size=(300, 2)), multitask_cov=True)["multitask_cov"]

    assert not gpe_gpytorch._REPORTED_CLIPPED_COVARIANCE, "no block had to be clipped"
    for blocks in covariance:
        assert min(np.linalg.eigvalsh(block).min() for block in blocks) > 0.0
        stats.multivariate_normal(mean=np.zeros(6), cov=block_diag(*blocks), allow_singular=True)


@pytest.mark.parametrize("seed", [2, 7, 13])
def test_training_on_forty_runs_of_a_smooth_response_succeeds(seed):
    """The training that failed in single precision, for three of the data sets it failed on.

    With the settings of the drivers and 40 training runs of this model, the covariance matrix of the training runs has a condition number beyond what single precision can factorize, and training or prediction stopped with ``NotPSDError: Matrix not positive definite after repeatedly adding jitter`` in 6 of 20 attempts. An emulator is trained in every BAL iteration, so a calibration of that size was unlikely to reach its end. In double precision none of 20 attempts failed.
    """
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(seed)
    x = rng.uniform(size=(40, 2))
    y = _response(x)

    emulator = _multitask(x, y, "variables", training_iter=150)
    prediction = emulator.predict_(input_sets=rng.uniform(size=(100, 2)), multitask_cov=True)

    assert np.all(np.isfinite(prediction["output"])) and np.all(np.isfinite(prediction["std"]))
    error = np.abs(emulator.predict_(input_sets=x)["output"] - y).max(axis=0) / y.std(axis=0)
    assert error.max() < 0.02, "a noise-free smooth response is reproduced at its training runs"


def test_eighty_training_runs_of_a_smooth_response_are_trained():
    """The number of training runs a calibration can reach.

    With 80 training runs of this model, the emulator of the drivers could not be trained in single precision in any of 20 attempts on the machine this was measured on: every one ended with ``NotPSDError``. In double precision all 20 were trained, with an error at the training runs of 0.2 % of the spread of the outputs. Only the latter is asserted, because that single precision fails is a limit of the arithmetic, not a requirement.
    """
    rng = np.random.default_rng(1000)
    torch.manual_seed(0)
    x = rng.uniform(size=(80, 2))
    y = _response(x)

    emulator = _multitask(x, y, "variables", training_iter=150)

    error = np.abs(emulator.predict_(input_sets=x)["output"] - y).max(axis=0) / y.std(axis=0)
    assert error.max() < 0.02


# --------------------------------------------------------------------------- #
# emulators stored by earlier releases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("build", [lambda **o: _single_output(X, Y, **o),
                                   lambda **o: _multitask(X, Y, "variables", **o),
                                   lambda **o: _multitask(X, Y, "locations", **o),
                                   lambda **o: _multitask(X, Y, "all", **o)],
                         ids=["single-output", "tasks-variables", "tasks-locations", "tasks-all"])
def test_an_emulator_pickled_by_an_earlier_release_still_predicts(build):
    """The drivers pickle the emulator of every iteration, and the post-processing scripts load it again.

    An emulator stored by release 1.10.1 or earlier holds single-precision models and knows nothing of a precision setting. It has to predict as it did, in the precision of its models.
    """
    emulator = build(precision="single")
    reference = emulator.predict_(input_sets=PRIOR, get_conf_int=True)
    del emulator.precision  # an object of an earlier release never had the attribute

    restored = pickle.loads(pickle.dumps(emulator))
    prediction = restored.predict_(input_sets=PRIOR, get_conf_int=True)

    assert not hasattr(restored, "precision")
    assert _types(restored) == {torch.float32}
    for key in ("output", "std", "upper_ci", "lower_ci"):
        assert np.array_equal(prediction[key], reference[key])


def test_a_double_precision_emulator_survives_pickling():
    emulator = _multitask(X, Y, "variables")
    reference = emulator.predict_(input_sets=PRIOR, multitask_cov=True)

    restored = pickle.loads(pickle.dumps(emulator))
    prediction = restored.predict_(input_sets=PRIOR, multitask_cov=True)

    assert _types(restored) == {torch.float64}
    assert np.array_equal(prediction["output"], reference["output"])
    assert np.array_equal(np.asarray(prediction["multitask_cov"]), np.asarray(reference["multitask_cov"]))


# --------------------------------------------------------------------------- #
# memory of a prediction
# --------------------------------------------------------------------------- #
LAYOUTS = {"single-output": lambda **o: _single_output(X, Y, **o),
           "tasks-variables": lambda **o: _multitask(X, Y, "variables", **o),
           "tasks-locations": lambda **o: _multitask(X, Y, "locations", **o),
           "tasks-all": lambda **o: _multitask(X, Y, "all", **o)}


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_a_prediction_in_pieces_returns_the_numbers_of_a_prediction_in_one_piece(monkeypatch, layout):
    """The emulators use the mean, the variance and the covariance between the targets at each prediction point, nothing between two points.

    gpytorch evaluates the joint distribution of all points it is given, with a covariance matrix of the square of their number: 25,000 prior samples needed 15 GB in single precision. The predictions are therefore evaluated in pieces, which must not change a single result.
    """
    emulator = LAYOUTS[layout]()
    inputs = np.random.default_rng(4).uniform(size=(101, 2))
    options = dict(get_conf_int=True) if layout == "single-output" else dict(get_conf_int=True, multitask_cov=True)

    monkeypatch.setattr(gpe_gpytorch, "PREDICTION_BUDGET", 1.0e12)  # everything in one piece, as up to release 1.10.1
    reference = emulator.predict_(input_sets=inputs, **options)
    monkeypatch.setattr(gpe_gpytorch, "PREDICTION_BUDGET", 42.0 ** 2)  # pieces of 42 points (single output) down to 7 (six tasks)
    in_pieces = emulator.predict_(input_sets=inputs, **options)

    assert set(in_pieces) == set(reference)
    for key in ("output", "std", "upper_ci", "lower_ci"):
        assert in_pieces[key].shape == reference[key].shape == (101, 6)
        assert in_pieces[key] == pytest.approx(reference[key], rel=1e-9, abs=1e-12)
    if "multitask_cov" in reference:
        assert len(in_pieces["multitask_cov"]) == 101
        assert np.asarray(in_pieces["multitask_cov"]) == pytest.approx(np.asarray(reference["multitask_cov"]), rel=1e-7, abs=1e-14)


@pytest.mark.parametrize("layout, tasks", [("single-output", 1), ("tasks-variables", 2), ("tasks-locations", 3), ("tasks-all", 6)])
def test_no_model_is_asked_for_more_points_than_the_budget_allows(monkeypatch, layout, tasks):
    """The memory of a prediction depends on the budget, not on the number of prior samples."""
    emulator = LAYOUTS[layout]()
    sizes = []
    for model_class in (gpe_gpytorch.MyExactGPyModel, gpe_gpytorch.MultitaskGPModel):
        original = model_class.__call__
        monkeypatch.setattr(model_class, "__call__",
                            lambda self, *args, _original=original, **kwargs:
                            (sizes.append(args[0].shape[0]) if not self.training else None, _original(self, *args, **kwargs))[1])

    prediction = emulator.predict_(input_sets=np.random.default_rng(5).uniform(size=(5000, 2)))

    limit = gpe_gpytorch._prediction_chunk(tasks)
    assert limit == 2000 // tasks
    assert prediction["output"].shape == (5000, 6)
    assert sizes and max(sizes) <= limit


def test_an_emulator_without_a_stored_task_layout_warns_once_per_prediction(monkeypatch):
    """Emulators of early releases do not store which task layout they were trained with. The layout is then inferred, with a warning, once per prediction and not once per piece."""
    emulator = _multitask(X, Y, "variables")
    del emulator.task_mode
    messages = []
    monkeypatch.setattr(gpe_gpytorch.logger_warn, "warning", lambda message, *a, **k: messages.append(str(message)))
    monkeypatch.setattr(gpe_gpytorch, "PREDICTION_BUDGET", 20.0 ** 2)  # pieces of 10 points

    prediction = emulator.predict_(input_sets=np.random.default_rng(6).uniform(size=(95, 2)))

    assert prediction["output"].shape == (95, 6)
    assert len([message for message in messages if "task_mode" in message]) == 1
