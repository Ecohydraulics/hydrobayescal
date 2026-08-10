"""Tests for ``HydroSimulations.save_calibration_data``.

The method lives on the base class so that every binding writes its BAL bookkeeping
the same way. It used to be duplicated verbatim in the OpenFOAM and Delft3D bindings,
which is how a crash fixed in one of them survived untouched in the other.

It is exercised through the base class with a stub carrying only the attributes it
reads, so the tests need no solver stack and no trained surrogate.
"""
import numpy as np
import pandas as pd
import pytest

from hydroBayesCal.hysim import HydroSimulations


@pytest.fixture
def saver(tmp_path):
    """A minimal stand-in exposing exactly what save_calibration_data reads."""
    return type("Stub", (), {
        "calibration_folder": str(tmp_path),
        "calibration_parameters": ["ks", "n"],
        "calibration_quantities": ["U_x"],
        "nloc": 2,
        "model_evaluations": np.zeros((3, 2)),
    })()


def _bayesian_dict(n_iter=1, **extra):
    """A scores dict shaped the way ``BayesianInference`` actually returns one.

    The scalar scores are numpy arrays, not lists. That is the whole point: the
    previous ``d.get(key) or [None] * (it + 1)`` idiom called ``bool()`` on them.
    """
    d = {
        "BME": np.zeros(n_iter + 1),
        "RE": np.zeros(n_iter + 1),
        "IE": np.zeros(n_iter + 1),
        "ELPD": np.zeros(n_iter + 1),
        "post_size": np.ones(n_iter + 1),
        "posterior": [np.array([[0.1, 0.2]])] * (n_iter + 1),
        "log_BME": np.zeros(n_iter + 1),
    }
    d.update(extra)
    return d


def test_a_numpy_log_bme_does_not_raise(saver, tmp_path):
    """The first BAL iteration used to die on log_BME being an array.

    ``bayesian_dict.get('log_BME') or [None] * (it + 1)`` evaluates the truth value
    of a multi-element numpy array, which is a ValueError, so BAL crashed before it
    could write anything.
    """
    HydroSimulations.save_calibration_data(saver, 0, np.zeros((3, 2)), _bayesian_dict())

    scores = pd.read_csv(tmp_path / "bayesian_scores.csv")
    assert len(scores) == 1
    assert scores["log_BME"][0] == 0.0


def test_a_missing_log_bme_is_recorded_as_empty(saver, tmp_path):
    """The fallback for a genuinely absent key still has to work."""
    d = _bayesian_dict()
    del d["log_BME"]

    HydroSimulations.save_calibration_data(saver, 0, np.zeros((3, 2)), d)

    scores = pd.read_csv(tmp_path / "bayesian_scores.csv")
    assert pd.isna(scores["log_BME"][0])


def test_numpy_diagnostics_do_not_raise(saver, tmp_path):
    """Every optional diagnostic goes through the same accessor, arrays included."""
    d = _bayesian_dict(
        marginal_optima=np.array([[0.5, 1.5]]),
        marginal_hdi=np.array([[[0.4, 0.6], [1.4, 1.6]]]),
        variance_reduction=np.array([[0.3, 0.7]]),
    )

    HydroSimulations.save_calibration_data(saver, 0, np.zeros((3, 2)), d)

    optima = pd.read_csv(tmp_path / "marginal_optima.csv")
    assert list(optima["parameter"]) == ["ks", "n"]
    assert list(optima["marginal_peak"]) == [0.5, 1.5]
    assert list(optima["hdi_low"]) == [0.4, 1.4]


def test_scores_accumulate_one_row_per_iteration(saver, tmp_path):
    """bayesian_scores.csv is rewritten, not appended, but must still grow."""
    d = _bayesian_dict(n_iter=2)
    for it in range(3):
        HydroSimulations.save_calibration_data(saver, it, np.zeros((3 + it, 2)), d)

    scores = pd.read_csv(tmp_path / "bayesian_scores.csv")
    assert list(scores["iteration"]) == [0, 1, 2]
    assert list(scores["N_tp"]) == [3, 4, 5]


def test_late_diagnostics_do_not_shift_earlier_columns(saver, tmp_path):
    """Columns appearing only from iteration 1 must not corrupt iteration 0's row.

    to_csv in append mode writes in DataFrame order but emits no header for an
    existing file, so a row with extra columns would land under the wrong names.
    """
    plain = _bayesian_dict(n_iter=1)
    HydroSimulations.save_calibration_data(saver, 0, np.zeros((3, 2)), plain)

    with_gap = _bayesian_dict(
        n_iter=1,
        marginal_joint_gap=[{"density_percentile": 90.0, "max_abs_correlation": 0.2,
                             "verdict": "identifiable"}] * 2,
    )
    HydroSimulations.save_calibration_data(saver, 1, np.zeros((4, 2)), with_gap)

    scores = pd.read_csv(tmp_path / "bayesian_scores.csv")
    assert list(scores["iteration"]) == [0, 1]
    assert pd.isna(scores["equifinality_verdict"][0])
    assert scores["equifinality_verdict"][1] == "identifiable"
    assert scores["marginal_peak_density_percentile"][1] == 90.0


def test_model_results_columns_follow_the_interleaving_contract(saver, tmp_path):
    """model_evaluations is [n_runs, nloc * n_quantities], quantities per location."""
    saver.calibration_quantities = ["U_x", "TKE"]
    saver.model_evaluations = np.arange(12).reshape(3, 4)

    HydroSimulations.save_calibration_data(saver, 0, np.zeros((3, 2)), _bayesian_dict())

    results = pd.read_csv(tmp_path / "model_results_N003.csv")
    assert list(results.columns) == ["run_idx", "U_x_z0", "TKE_z0", "U_x_z1", "TKE_z1"]


# --------------------------------------------------------------------------------
# drift guard
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("binding", ["telemac", "openfoam", "delft3d"])
@pytest.mark.parametrize("method", ["save_calibration_data", "_save_all_results"])
def test_the_shared_bookkeeping_is_not_re_duplicated(binding, method):
    """Both methods must resolve to HydroSimulations for every binding.

    They were duplicated verbatim across bindings twice, and both times a fix
    applied to one copy left the other broken. Overriding either in a binding is
    almost certainly that drift starting again.
    """
    import importlib

    module = importlib.import_module(f"hydroBayesCal.{binding}.control_{binding}")
    model = next(obj for name, obj in vars(module).items()
                 if isinstance(obj, type) and name.endswith("Model")
                 and issubclass(obj, HydroSimulations) and obj is not HydroSimulations)

    owner = next(k for k in model.__mro__ if method in k.__dict__)
    assert owner is HydroSimulations, (
        f"{model.__name__} overrides {method}; it is shared bookkeeping and belongs "
        f"on HydroSimulations so every binding gets the same fixes"
    )
