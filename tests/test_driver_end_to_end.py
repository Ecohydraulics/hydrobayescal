"""End-to-end runs of the calibration drivers on an analytic model.

The drivers are orchestration scripts whose ``run_bal_model`` functions are near copies of each other, and the other tests check them by parsing their sources. That finds a line missing from one copy, but not a name that is undefined on a path, an argument that no longer fits, or a surrogate layout that is built and then not trained. Here the three ``run_bal_model`` are executed for real, with a cheap analytic model in place of the solver, once per surrogate layout, so every branch on ``multitask_selection`` is run at least once in every driver.
"""
import importlib.util
import pathlib
import pickle

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")
pytest.importorskip("bayesvalidrox")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from hydroBayesCal.hysim import HydroSimulations

DRIVERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydroBayesCal" / "drivers"
STATIONS = np.linspace(0.2, 1.0, 3)
TRUTH = np.array([0.4, 0.7])
# Sized for the test suite: one BAL iteration trains every emulator twice and runs the sequential design once.
INIT_RUNS, MAX_RUNS = 8, 9


def _response(points, n_quantities, dry_point=False):
    """Smooth analytic stand-in for a solver: interleaved columns [Q1_P1, Q2_P1, Q1_P2, ...].

    With ``dry_point``, a further calibration point follows whose targets are 0.0 for every parameter set, as at a point that stays dry in all runs.
    """
    points = np.atleast_2d(points)
    q1 = np.array([[np.sin(3.0 * a * s) + b * s ** 2 for s in STATIONS] for a, b in points])
    q2 = np.array([[2.0 + a * s - 0.5 * b * np.cos(2.0 * s) for s in STATIONS] for a, b in points])
    if dry_point:
        q1, q2 = (np.column_stack([q, np.zeros(len(points))]) for q in (q1, q2))
    if n_quantities == 1:
        return q1
    out = np.empty((points.shape[0], 2 * q1.shape[1]))
    out[:, 0::2], out[:, 1::2] = q1, q2
    return out


class _AnalyticModel(HydroSimulations):
    """A binding without a solver, recording how the driver called it."""

    def __init__(self, folder, quantities, multitask_selection, dry_point=False):
        self.dry_point = dry_point
        observations = _response(TRUTH, len(quantities), dry_point)[0]
        points_file = folder / "calibration-points.csv"
        header = ["Point", "X", "Y"] + [f"{q}_{suffix}" for q in quantities for suffix in ("DATA", "ERROR")]
        rows = []
        for i, station in enumerate(list(STATIONS) + [1.2] * dry_point):
            values = observations[i * len(quantities):(i + 1) * len(quantities)]
            row = [f"P{i + 1}", f"{station:.3f}", "0.0"]
            for value in values:
                # a measured 0.0 needs an absolute error: a relative one would give the observation a variance of zero
                row += [f"{value:.6f}", f"{max(0.02 * abs(value), 0.01):.6f}"]
            rows.append(",".join(row))
        points_file.write_text(",".join(header) + "\n" + "\n".join(rows) + "\n")

        super().__init__(
            model_dir=str(folder), res_dir=str(folder), calibration_pts_file_path=str(points_file),
            init_runs=INIT_RUNS, max_runs=MAX_RUNS,
            calibration_parameters=["a", "b"], param_values=[[0.0, 1.0], [0.0, 1.0]],
            calibration_quantities=list(quantities), extraction_quantities=list(quantities),
            complete_bal_mode=True, only_bal_mode=False, delete_complex_outputs=False,
            multitask_selection=multitask_selection)
        self.calls = []

    def run_multiple_simulations(self, collocation_points=None, bal_new_set_parameters=None,
                                 start_index=0, **kwargs):
        self.calls.append(dict(kwargs, bal=bal_new_set_parameters is not None))
        if bal_new_set_parameters is None:
            self.model_evaluations = _response(collocation_points, len(self.calibration_quantities), self.dry_point)
        else:
            new = _response(bal_new_set_parameters, len(self.calibration_quantities), self.dry_point)
            self.model_evaluations = np.vstack([self.model_evaluations, new])
        return self.model_evaluations

    def output_processing(self, *args, **kwargs):
        return self.model_evaluations


def _driver(name):
    spec = importlib.util.spec_from_file_location(name, DRIVERS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibrate(driver_name, folder, quantities, multitask_selection, dry_point=False, **bal_options):
    """Initial design and one BAL iteration through the functions of one driver."""
    matplotlib.pyplot.rcParams.update({"text.usetex": False})
    driver = _driver(driver_name)
    model = _AnalyticModel(folder, quantities, multitask_selection, dry_point)
    design = driver.setup_experiment_design(
        complex_model=model, tp_selection_criteria="dkl",
        parameter_distribution="uniform", parameter_sampling_method="sobol")

    extraction = dict(output_extraction_time="index", n_last=3) if driver_name == "bal_telemac" else {}
    points, outputs = driver.run_complex_model(
        complex_model=model, experiment_design=design, adaptive_init_runs=False, **extraction)
    driver.run_bal_model(
        collocation_points=points, model_outputs=outputs, complex_model=model,
        experiment_design=design, eval_steps=1, prior_samples=300, mc_samples_al=30,
        mc_exploration=60, gp_library="gpy", bal_n_jobs=1, **extraction, **bal_options)

    with open(pathlib.Path(model.calibration_folder) / "BAL_dictionary.pkl", "rb") as handle:
        return model, pickle.load(handle)


LAYOUTS = [
    (("WATER DEPTH", "SCALAR VELOCITY"), "variables"),
    (("WATER DEPTH", "SCALAR VELOCITY"), "locations"),
    (("WATER DEPTH", "SCALAR VELOCITY"), "all"),
    (("WATER DEPTH", "SCALAR VELOCITY"), "SO_sequential"),
    (("WATER DEPTH",), "variables"),
    (("WATER DEPTH",), "locations"),
]


@pytest.mark.parametrize("driver_name", ["bal_telemac", "bal_openfoam", "bal_delft3d"])
@pytest.mark.parametrize("quantities, multitask_selection", LAYOUTS,
                         ids=[f"{len(q)}target-{m}" for q, m in LAYOUTS])
def test_a_calibration_runs_through_for_every_surrogate_layout(tmp_path, driver_name, quantities,
                                                               multitask_selection):
    model, bal = _calibrate(driver_name, tmp_path, quantities, multitask_selection)
    n_bal = MAX_RUNS - INIT_RUNS

    # the design grew by one run per BAL iteration, and every iteration has its scores
    assert model.model_evaluations.shape == (MAX_RUNS, len(STATIONS) * len(quantities))
    assert bal["N_tp"].tolist() == list(range(INIT_RUNS, MAX_RUNS + 1))
    assert np.all(np.isfinite(bal["log_BME"]))
    assert len(bal["util_func"]) == n_bal and all(bal["util_func"])

    # the log-likelihood of the posterior samples is stored per iteration, aligned with the posterior
    assert len(bal["post_loglikelihood"]) == n_bal + 1
    for posterior, log_likelihood in zip(bal["posterior"], bal["post_loglikelihood"]):
        if posterior is not None and len(posterior):
            assert len(log_likelihood) == len(posterior)


def test_the_runs_bal_selects_are_extracted_with_the_window_of_the_initial_design(tmp_path):
    """The initial design and the BAL runs have to be read from the solver output in the same way."""
    model, _ = _calibrate("bal_telemac", tmp_path, ("WATER DEPTH",), "variables")

    initial = [call for call in model.calls if not call["bal"]]
    selected = [call for call in model.calls if call["bal"]]
    assert initial and len(selected) == MAX_RUNS - INIT_RUNS
    for call in initial + selected:
        assert call["output_extraction_time"] == "index"
        assert call["n"] == 3


@pytest.mark.parametrize("quantities, multitask_selection", LAYOUTS,
                         ids=[f"{len(q)}target-{m}" for q, m in LAYOUTS])
def test_a_calibration_runs_through_when_a_calibration_point_never_responds(tmp_path, quantities,
                                                                           multitask_selection):
    """A calibration point that stays dry gives the same output in every run, which must not stop the emulator.

    The emulators normalize every output by its spread over the training runs. For a constant output the spread is zero, the normalized output was NaN, and the training stopped with ``NanError`` after the initial design had been simulated.
    """
    model, bal = _calibrate("bal_telemac", tmp_path, quantities, multitask_selection, dry_point=True)

    n_outputs = (len(STATIONS) + 1) * len(quantities)
    assert model.model_evaluations.shape == (MAX_RUNS, n_outputs)
    assert np.all(model.model_evaluations[:, -len(quantities):] == 0.0)
    assert np.all(np.isfinite(bal["log_BME"]))
    assert all(bal["util_func"])


def test_a_calibration_runs_through_with_the_exploration_trade_off(tmp_path):
    """With the trade-off on, every candidate of the prior sample gets a space-filling score from the experimental design of the driver."""
    model, bal = _calibrate("bal_telemac", tmp_path, ("WATER DEPTH", "SCALAR VELOCITY"), "variables",
                            bal_exploration_tradeoff=True)

    assert model.model_evaluations.shape[0] == MAX_RUNS
    assert np.all(bal["exploration_tradeoff_active"])
    assert np.all(np.isfinite(bal["log_BME"]))
