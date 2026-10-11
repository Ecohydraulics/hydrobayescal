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
    REPO / "examples" / "Telemac" / "Telemac2d" / "Ering" / "hydrodynamic_calibration" / "bal_telemac.py",
    REPO / "examples" / "Telemac" / "Telemac3d" / "cylinderFlume" / "bal_telemac.py",
]

#: Expected fallback for every ``.get(key, fallback)`` in the drivers.
EXPECTED_FALLBACKS = {
    "include_surrogate_error": True,
    "gpe_error": 0.0,
    "measurement_error": 0.10,
    "model_structural_error": 0.0,
    "multitask_selection": "variables",
    "bal_n_jobs": -1,
    "posterior_sampling_method": "rejection_sampling",
    "user_param_values": False,
}

#: Keys a configuration file may leave out. A driver that reads one of them without a fallback hands ``None`` to the library, or fails on the missing key, and does so after the initial design has been simulated.
OPTIONAL_KEYS = sorted(EXPECTED_FALLBACKS) + ["bal_exploration_tradeoff", "adaptive_init_runs",
                                              "init_runs_min", "gaia_layer_average"]


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


def _sync_script():
    """``env-scripts/sync-example-drivers.py``, which generates the example copies of the TELEMAC driver."""
    spec = importlib.util.spec_from_file_location(
        "sync_example_drivers", REPO / "env-scripts" / "sync-example-drivers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative_path", sorted(_sync_script().EXAMPLE_DRIVERS))
def test_example_drivers_match_the_template(relative_path):
    """An example copy differs from the template in its header line and its default configuration only.

    Keeps the "keep in sync" convention from drifting back into stale copies. A copy that carries its own version of the workflow is how a fix of the template does not reach a case, and how a feature written into a copy reaches no other case.
    """
    sync = _sync_script()
    case, config = sync.EXAMPLE_DRIVERS[relative_path]
    template = (DRIVERS_DIR / "bal_telemac.py").read_text().splitlines()
    example = (REPO / relative_path).read_text().splitlines()

    added = [line[1:].strip() for line in difflib.unified_diff(template, example, n=0)
             if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:].strip() for line in difflib.unified_diff(template, example, n=0)
               if line.startswith("-") and not line.startswith("---")]

    hint = "run: python env-scripts/sync-example-drivers.py"
    assert all(config in line or "case driver: copy of" in line for line in added), (added, hint)
    assert all("config_Telemac.py" in line for line in removed), (removed, hint)
    assert (REPO / relative_path).read_text() == sync.expected_copy(case, config), hint


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



# --------------------------------------------------------------------------- #
# optional configuration keys keep their fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", DRIVERS, ids=lambda p: str(p.relative_to(REPO)))
def test_optional_keys_are_never_read_without_a_fallback(path):
    """``config.sampling.get('posterior_sampling_method')`` returns ``None`` for every configuration written before the key existed."""
    source = path.read_text()
    without_fallback = [
        node.args[0].value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get" and len(node.args) == 1 and not node.keywords
        and isinstance(node.args[0], ast.Constant) and node.args[0].value in OPTIONAL_KEYS
    ]
    assert not without_fallback, f"{path.name}: read without a fallback: {without_fallback}"
    assert "config.extraction[" not in source, (
        f"{path.name} indexes config.extraction, but the extraction block is optional")


# --------------------------------------------------------------------------- #
# the runs BAL selects are extracted like the initial design
# --------------------------------------------------------------------------- #
def _function(path, name):
    tree = ast.parse(pathlib.Path(path).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(functions) == 1, f"{pathlib.Path(path).name}: expected one {name}()"
    return functions[0]


def _calls(node, attribute):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Attribute, ast.Name))
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == attribute]


TELEMAC_TEMPLATE = DRIVERS_DIR / "bal_telemac.py"


def test_the_bal_loop_uses_the_extraction_window_it_is_given():
    """A literal window in the BAL loop trains the surrogate on two differently defined quantities.

    The initial design was extracted with the configured window while every run selected by BAL was averaged over the last 40 time steps, whatever the configuration said.
    """
    run_bal_model = _function(TELEMAC_TEMPLATE, "run_bal_model")
    parameters = [a.arg for a in run_bal_model.args.args]
    assert {"output_extraction_time", "n_last"} <= set(parameters)

    simulations = _calls(run_bal_model, "run_multiple_simulations")
    assert len(simulations) == 1
    keywords = {k.arg: k.value for k in simulations[0].keywords}
    for keyword, variable in (("output_extraction_time", "output_extraction_time"), ("n", "n_last")):
        value = keywords[keyword]
        assert isinstance(value, ast.Name) and value.id == variable, (
            f"run_bal_model passes {ast.unparse(value)} as {keyword}; it has to pass {variable}")


@pytest.mark.parametrize("driver", [
    DRIVERS_DIR / "bal_telemac.py",
    DRIVERS_DIR / "bal_telemac_multiflow.py",
])
def test_telemac_drivers_hand_the_extraction_window_to_the_bal_loop(driver):
    """``bal_telemac_multiflow.py`` imports ``run_bal_model``, so leaving the window out there falls back to its default without any error."""
    bal_calls = _calls(_function(driver, "main"), "run_bal_model")
    assert len(bal_calls) == 1
    keywords = {k.arg for k in bal_calls[0].keywords}
    assert {"output_extraction_time", "n_last"} <= keywords


# --------------------------------------------------------------------------- #
# settings that have to reach every driver
# --------------------------------------------------------------------------- #
ALL_DRIVER_TEMPLATES = [DRIVERS_DIR / name for name in
                        ("bal_telemac.py", "bal_telemac_multiflow.py", "bal_openfoam.py", "bal_delft3d.py")]


@pytest.mark.parametrize("driver", ALL_DRIVER_TEMPLATES, ids=lambda p: p.name)
def test_every_driver_reads_the_surrogate_layout_from_the_configuration(driver):
    """``multitask_selection`` existed on the model but no driver passed it, so only editing a driver could change it."""
    source = driver.read_text()
    assert ".get('multitask_selection'" in source or '.get("multitask_selection"' in source
    assert ".get('bal_n_jobs'" in source or '.get("bal_n_jobs"' in source


@pytest.mark.parametrize("driver", [DRIVERS_DIR / name for name in
                                    ("bal_telemac.py", "bal_openfoam.py", "bal_delft3d.py")],
                         ids=lambda p: p.name)
def test_every_bal_loop_is_complete(driver):
    """The three ``run_bal_model`` are near copies, and a line missing from one of them fails late.

    ``do_tradeoff`` is read in every iteration and was assigned before the loop in one driver only, so the first BAL iteration of the others raised after the initial design had been simulated. The independent GPEs and the worker count are checked for the same reason.
    """
    run_bal_model = _function(driver, "run_bal_model")
    loop = next(n for n in run_bal_model.body
                if isinstance(n, ast.For) and getattr(n.target, "id", None) == "it")
    assigned_before = {
        target.id
        for statement in run_bal_model.body[:run_bal_model.body.index(loop)]
        for node in ast.walk(statement) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert "do_tradeoff" in assigned_before, f"{driver.name}: do_tradeoff is not set before the BAL loop"

    source = ast.get_source_segment(driver.read_text(), run_bal_model)
    assert "SOSequentialGPyTraining.from_training_data(" in source
    assert "n_jobs=bal_n_jobs" in source
    assert "bayesian_dict['post_loglikelihood'][it]" in source


def test_an_unknown_surrogate_layout_is_rejected_at_construction():
    """The drivers compare the value only when they build the surrogate, i.e. after the initial design."""
    from hydroBayesCal.hysim import MULTITASK_SELECTIONS, HydroSimulations

    class _Model(HydroSimulations):
        def run_multiple_simulations(self, *args, **kwargs):
            pass

        def output_processing(self, *args, **kwargs):
            pass

    assert set(MULTITASK_SELECTIONS) == {"variables", "locations", "all", "SO_sequential"}
    with pytest.raises(ValueError, match="multitask_selection"):
        _Model(multitask_selection="variable")



@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_plot_labels_cover_every_calibration_parameter(path):
    """The labels are assigned by position, so a shorter list labels a parameter with the name of another one."""
    config = _load(path)
    plotting = getattr(config, "plotting", None)
    if plotting is None:
        pytest.skip("no plotting block")
    n_parameters = len(config.calibration["parameters"])

    assert len(plotting["parameter_names"]) == n_parameters
    assert len(plotting["parameter_units"]) == n_parameters
    assert all(0 <= index < n_parameters for index in plotting["parameter_indices"])
