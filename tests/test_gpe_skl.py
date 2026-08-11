"""Tests for the scikit-learn surrogate path, ``sampling['gp_library'] = "skl"``.

That path was unreachable in two independent places at once, which is why it needs its
own tests: the drivers never bound ``surrogate_object`` in the ``skl`` branch, so the
first log line raised ``UnboundLocalError``, and ``SklTraining.predict_`` called
``predict_`` on the raw scikit-learn regressor, which only has ``predict``. Neither is
visible to a test that merely imports the module.
"""
import ast
import pathlib

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF

from hydroBayesCal.surrogate.gpe_skl import SklTraining

DRIVERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydroBayesCal" / "drivers"
DRIVERS = [DRIVERS_DIR / name for name in
           ("bal_telemac.py", "bal_openfoam.py", "bal_delft3d.py")]


def _train(n_runs=12, ndim=2, n_obs=3):
    rng = np.random.default_rng(0)
    points = rng.random((n_runs, ndim))
    outputs = np.column_stack([np.sin(3 * points[:, 0]) + k * points[:, 1]
                               for k in range(1, n_obs + 1)])
    surrogate = SklTraining(collocation_points=points, model_evaluations=outputs,
                            noise=True, kernel=1 * RBF(length_scale=np.full(ndim, 0.5)),
                            alpha=1e-6, n_restarts=0, parallelize=False)
    surrogate.train_()
    return surrogate, points, outputs


def test_skl_training_predicts_mean_and_standard_deviation():
    surrogate, points, outputs = _train()

    prediction = surrogate.predict_(input_sets=points, get_conf_int=True)

    assert prediction["output"].shape == outputs.shape
    assert prediction["std"].shape == outputs.shape
    assert np.all(np.isfinite(prediction["output"]))
    assert np.all(prediction["std"] >= 0.0)
    # An interpolating GP reproduces its own training points.
    assert np.allclose(prediction["output"], outputs, atol=1e-2)
    assert np.all(prediction["lower_ci"] <= prediction["output"])
    assert np.all(prediction["upper_ci"] >= prediction["output"])


def test_skl_prediction_is_uncertain_away_from_the_training_points():
    surrogate, points, _ = _train()
    at_training = surrogate.predict_(input_sets=points)["std"]
    far_away = surrogate.predict_(input_sets=np.full((1, points.shape[1]), 5.0))["std"]
    assert np.max(far_away) > np.max(at_training)


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_every_driver_binds_the_surrogate_in_the_skl_branch(driver):
    """Everything downstream reads ``surrogate_object``, including the first log line."""
    tree = ast.parse(driver.read_text())
    run_bal_model = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.FunctionDef) and node.name == "run_bal_model")

    # The innermost `if` that builds an SklTraining, i.e. the skl branch itself rather
    # than any block that merely contains it.
    branches = [node for node in ast.walk(run_bal_model)
                if isinstance(node, ast.If)
                and any("SklTraining" in ast.dump(statement) for statement in node.body)]
    assert branches, f"{driver.name} has no SklTraining branch"
    branch = min(branches, key=lambda node: len(ast.dump(node)))

    assigned = {target.id for statement in ast.walk(branch)
                if isinstance(statement, ast.Assign)
                for target in statement.targets if isinstance(target, ast.Name)}
    assert "surrogate_object" in assigned, (
        f"{driver.name} builds an SklTraining without binding surrogate_object, so "
        f"gp_library='skl' raises UnboundLocalError on the first log line")
