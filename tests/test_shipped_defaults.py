"""Tests for the defaults that actually ship.

This is a set of literals scattered across ten files, and the one that decides the
behaviour of an *existing* configuration file is not the value in the config at all,
it is the fallback in ``config.<block>.get(key, <fallback>)`` inside the driver. Both
are checked here, by parsing the sources, so nothing has to import the solver stack.
"""
import ast
import difflib
import importlib.util
import inspect
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The drivers ship as package data, so they live inside the package, not in a
#: top-level templates/ directory.
DRIVERS_DIR = REPO / "src" / "hydroBayesCal" / "drivers"

CONFIGS = [
    DRIVERS_DIR / "config_Telemac.py",
    DRIVERS_DIR / "config_OpenFOAM.py",
    DRIVERS_DIR / "config_Delft3D.py",
    REPO / "examples" / "Telemac" / "Hydromorphodynamic" / "Ering" / "config_Ering.py",
]
DRIVERS = [
    DRIVERS_DIR / "bal_telemac.py",
    DRIVERS_DIR / "bal_openfoam.py",
    DRIVERS_DIR / "bal_delft3d.py",
    DRIVERS_DIR / "bal_telemac_multiflow.py",
    REPO / "examples" / "Telemac" / "Hydromorphodynamic" / "Ering" / "bal_telemac.py",
]

#: Expected fallback for every ``.get(key, fallback)`` in the drivers.
EXPECTED_FALLBACKS = {
    "include_surrogate_error": True,
    "gpe_error": 0.0,
    "measurement_error": 0.10,
    "model_structural_error": 0.0,
}


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_call_fallbacks(tree):
    """Every ``<something>.get('<key>', <literal>)`` in a module, as {key: [values]}."""
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)):
            key = node.args[0].value
            if key in EXPECTED_FALLBACKS:
                found.setdefault(key, []).append(ast.literal_eval(node.args[1]))
    return found


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_ships_the_surrogate_error_defaults(path):
    config = _load(path)
    assert config.sampling["include_surrogate_error"] is True
    assert config.calibration["gpe_error"] == 0.0
    assert config.calibration["measurement_error"] == 0.10
    assert config.calibration["model_structural_error"] == 0.0


@pytest.mark.parametrize("path", DRIVERS, ids=lambda p: str(p.relative_to(REPO)))
def test_driver_get_fallbacks_match_the_shipped_defaults(path):
    """The fallback governs what an OLD config file does, so it matters most."""
    fallbacks = _get_call_fallbacks(ast.parse(path.read_text()))
    for key, values in fallbacks.items():
        for value in values:
            assert value == EXPECTED_FALLBACKS[key], (
                f"{path.name}: .get('{key}', {value!r}) should fall back to "
                f"{EXPECTED_FALLBACKS[key]!r}")


@pytest.mark.parametrize("path", DRIVERS[:3] + DRIVERS[4:],
                         ids=lambda p: str(p.relative_to(REPO)))
def test_run_bal_model_signature_default(path):
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "run_bal_model"]
    assert len(functions) == 1, f"{path.name}: expected one run_bal_model"

    args = functions[0].args
    defaults = dict(zip([a.arg for a in args.args][-len(args.defaults):],
                        args.defaults))
    assert "include_surrogate_error" in defaults, f"{path.name}: parameter missing"
    assert ast.literal_eval(defaults["include_surrogate_error"]) is True


def test_hydrosimulations_error_budget_defaults():
    from hydroBayesCal.hysim import HydroSimulations

    for function in (HydroSimulations.__init__,
                     HydroSimulations.set_observations_and_variances):
        parameters = inspect.signature(function).parameters
        assert parameters["gpe_error"].default == 0.0
        assert parameters["measurement_error"].default == 0.10
        assert parameters["model_structural_error"].default == 0.0


def test_ering_driver_matches_the_template():
    """The Ering copy must differ from the template only in its two known hunks.

    Keeps the "keep in sync" convention from drifting back into a stale copy.
    """
    template = (DRIVERS_DIR / "bal_telemac.py").read_text().splitlines()
    example = (REPO / "examples" / "Telemac" / "Hydromorphodynamic" / "Ering"
               / "bal_telemac.py").read_text().splitlines()

    added = [line[1:].strip() for line in difflib.unified_diff(template, example, n=0)
             if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:].strip() for line in difflib.unified_diff(template, example, n=0)
               if line.startswith("-") and not line.startswith("---")]

    assert all("config_Ering.py" in line or "Ering case driver" in line
               or "Keep in sync" in line or line == ""
               for line in added), added
    assert all("config_Telemac.py" in line for line in removed), removed


# --------------------------------------------------------------------------- #
# the extraction window has to reach every driver that has one
# --------------------------------------------------------------------------- #
def _main_source(path):
    """Source of the module-level ``main()`` in a driver, parsed not imported."""
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return ast.get_source_segment(pathlib.Path(path).read_text(), node)
    raise AssertionError(f"{pathlib.Path(path).name} has no module-level main()")


@pytest.mark.parametrize("driver", [
    DRIVERS_DIR / "bal_telemac.py",
    DRIVERS_DIR / "bal_telemac_multiflow.py",
])
def test_telemac_drivers_pass_the_configured_extraction_window(driver):
    """Both TELEMAC drivers must read ``config.extraction`` and pass it on.

    ``bal_telemac_multiflow.py`` imports ``run_complex_model`` from
    ``bal_telemac.py``, so omitting the argument does not fail - it silently takes
    that function's ``mean_last`` default. The result was that the same
    configuration calibrated single-flow and multi-flow fitted *different data*:
    one honoured the requested extraction window, the other averaged the last
    frames, which on a run marching to steady state folds the residual transient
    into the values the surrogate is trained on.
    """
    source = _main_source(driver)
    assert "getattr(config, 'extraction'" in source, (
        f"{driver.name}'s main() does not read the config's extraction block")
    assert "output_extraction_time=" in source, (
        f"{driver.name}'s main() does not pass output_extraction_time to "
        "run_complex_model, so it falls back to the function default")
    assert "n_last=" in source, (
        f"{driver.name}'s main() does not pass n_last, so the averaging window "
        "would be the function default even when the config sets one")
