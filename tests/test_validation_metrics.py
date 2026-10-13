"""Tests for the validation criteria of an emulator against full-complexity runs (``validation_error``)."""
import numpy as np
import pytest

from hydroBayesCal.surrogate import gpe_skl, validation_metrics
from hydroBayesCal.surrogate.validation_metrics import save_valid_criteria, validation_error

RNG = np.random.default_rng(0)
TRUE = RNG.normal(loc=2.0, size=(40, 4))        # two output types with two calibration points each
PREDICTED = TRUE + RNG.normal(scale=0.1, size=TRUE.shape)
STD = np.full(TRUE.shape, 0.1)
NAMES = ["WATER DEPTH", "SCALAR VELOCITY"]


def test_the_root_mean_square_error_is_computed_with_the_installed_scikit_learn():
    """The criteria used ``mean_squared_error(squared=False)``, which scikit-learn removed in version 1.6, so every call raised ``TypeError``."""
    rmse, criteria = validation_error(TRUE, PREDICTED, NAMES, n_per_type=2)

    expected = np.sqrt(np.mean((TRUE - PREDICTED) ** 2, axis=0))
    assert rmse == pytest.approx(expected)
    assert criteria["rmse"]["WATER DEPTH"] == pytest.approx(expected[:2])
    assert criteria["rmse"]["SCALAR VELOCITY"] == pytest.approx(expected[2:])
    assert criteria["mse"]["SCALAR VELOCITY"] == pytest.approx(expected[2:] ** 2)


def test_the_criteria_have_one_value_per_calibration_point_and_output_type():
    _, criteria = validation_error(TRUE, PREDICTED, NAMES, n_per_type=2)

    assert set(criteria) == {"rmse", "mse", "nse", "r2", "mean_error", "std_error"}
    for criterion in criteria.values():
        assert list(criterion) == NAMES
        assert all(np.shape(values) == (2,) for values in criterion.values())
    variance = np.var(TRUE, axis=0)
    assert criteria["nse"]["WATER DEPTH"] == pytest.approx((1.0 - np.mean((TRUE - PREDICTED) ** 2, axis=0) / variance)[:2])
    assert criteria["r2"]["WATER DEPTH"][0] == pytest.approx(np.corrcoef(TRUE[:, 0], PREDICTED[:, 0])[0, 1])


def test_a_prediction_with_confidence_bounds_adds_the_uncertainty_criteria():
    prediction = {"output": PREDICTED, "std": STD, "upper_ci": PREDICTED + 2 * STD, "lower_ci": PREDICTED - 2 * STD}

    _, criteria = validation_error(TRUE, prediction, NAMES, n_per_type=2)

    assert criteria["norm_error"]["WATER DEPTH"] == pytest.approx(np.mean(((PREDICTED - TRUE) / STD) ** 2, axis=0)[:2])
    inside = (TRUE <= PREDICTED + 2 * STD) & (TRUE >= PREDICTED - 2 * STD)
    assert criteria["P95"]["SCALAR VELOCITY"] == pytest.approx(inside.mean(axis=0)[2:])


def test_a_prediction_without_confidence_bounds_is_accepted():
    """``predict_`` returns the bounds only with ``get_conf_int=True``; without them the criteria that need them are left out instead of raising ``KeyError``."""
    _, criteria = validation_error(TRUE, {"output": PREDICTED, "std": STD}, NAMES, n_per_type=2)

    assert "norm_error" in criteria and "P95" not in criteria
    _, criteria = validation_error(TRUE, {"output": PREDICTED}, NAMES, n_per_type=2)
    assert "norm_error" not in criteria and "P95" not in criteria


def test_both_emulator_modules_share_one_implementation():
    """The functions existed twice, word for word, so that a correction to one copy left the other one behind."""
    gpe_gpytorch = pytest.importorskip("hydroBayesCal.surrogate.gpe_gpytorch")

    for name in ("validation_error", "save_valid_criteria"):
        assert getattr(gpe_skl, name) is getattr(validation_metrics, name)
        assert getattr(gpe_gpytorch, name) is getattr(validation_metrics, name)


def test_the_criteria_of_successive_iterations_are_stacked():
    _, first = validation_error(TRUE, PREDICTED, NAMES, n_per_type=2)
    _, second = validation_error(TRUE, PREDICTED + 0.05, NAMES, n_per_type=2)

    history = save_valid_criteria(first, {}, n_tp=10)
    history = save_valid_criteria(second, history, n_tp=11)

    assert history["N_tp"] == [10, 11]
    assert history["rmse"]["WATER DEPTH"].shape == (2, 2)
    assert history["rmse"]["WATER DEPTH"][1] == pytest.approx(second["rmse"]["WATER DEPTH"])
