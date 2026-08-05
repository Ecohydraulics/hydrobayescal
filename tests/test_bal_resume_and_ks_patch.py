"""Tests for the BAL resume modes and for the OpenFOAM ks-patch auto-detection.

The resume logic is a three-line contract repeated verbatim in five near-duplicate
driver scripts, and the drivers have drifted apart before, so it is checked by parsing
the sources rather than by running a calibration: that keeps the check honest for every
driver at once without importing a solver stack.

The ks-patch detection is exercised directly against ``0/nut`` fixtures, because the
value it returns is written into an OpenFOAM boundary condition and a silent ``None``
there used to surface much later as a failed simulation.
"""
import ast
import pathlib
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Every driver that owns a copy of ``run_bal_model``. ``bal_telemac_multiflow.py`` is
#: deliberately absent: it imports ``run_bal_model`` from ``bal_telemac.py`` verbatim.
DRIVERS = [
    REPO / "templates" / "bal_telemac.py",
    REPO / "templates" / "bal_openfoam.py",
    REPO / "templates" / "bal_delft3d.py",
    REPO / "examples" / "Telemac" / "Hydromorphodynamic" / "Ering" / "bal_telemac.py",
]


def _run_bal_model(path):
    """Return the ``run_bal_model`` function node of a driver script."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_bal_model":
            return node
    raise AssertionError(f"{path.name} defines no run_bal_model()")


def _conditions(func):
    """Return every ``if`` test in a function, unparsed and whitespace-normalised."""
    return [" ".join(ast.unparse(n.test).split())
            for n in ast.walk(func) if isinstance(n, ast.If)]


# The three conditions that together define the resume contract. `complete_bal_mode`
# alone decides whether new simulations run, so a resume run (both flags True) still
# simulates the training points BAL selects, while pure re-analysis never does.
ZERO_ITER = "complex_model.only_bal_mode and (not complex_model.complete_bal_mode)"
RUN_SIMS = "complex_model.complete_bal_mode"
UPDATE_COLLOCATION = "not complex_model.only_bal_mode or complex_model.complete_bal_mode"


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_iterations_are_skipped_only_for_pure_reanalysis(driver):
    """n_iter is zeroed iff only_bal_mode is set without complete_bal_mode.

    Guarding on ``only_bal_mode`` alone made a resume run (both flags True) do zero
    iterations, so continuing a campaign silently did nothing.
    """
    assert ZERO_ITER in _conditions(_run_bal_model(driver))


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_new_simulations_are_gated_on_complete_bal_mode_alone(driver):
    """The simulation call is guarded by complete_bal_mode and nothing else."""
    conditions = _conditions(_run_bal_model(driver))
    assert RUN_SIMS in conditions
    # The two forms that skipped, or forced, the simulations of a resume run.
    assert "complex_model.complete_bal_mode and (not complex_model.only_bal_mode)" not in conditions
    assert "complex_model.complete_bal_mode or complex_model.only_bal_mode" not in conditions


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_collocation_points_grow_on_a_resume_run(driver):
    """New training points are appended unless the run is pure re-analysis."""
    conditions = _conditions(_run_bal_model(driver))
    assert UPDATE_COLLOCATION in conditions
    assert "not complex_model.only_bal_mode" not in conditions


def test_all_drivers_agree_on_the_resume_contract():
    """The five drivers are near-duplicates, so the contract must be identical."""
    per_driver = {
        driver.name: {c for c in _conditions(_run_bal_model(driver))
                      if "only_bal_mode" in c or "complete_bal_mode" in c}
        for driver in DRIVERS
    }
    assert len(set(map(frozenset, per_driver.values()))) == 1, per_driver


# --------------------------------------------------------------------------------
# ks patch auto-detection
# --------------------------------------------------------------------------------

def _detect(tmp_path, nut_contents):
    """Run OpenFOAMModel._detect_ks_patch against a case template holding 0/nut."""
    from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel

    if nut_contents is not None:
        nut = tmp_path / "0" / "nut"
        nut.parent.mkdir(parents=True, exist_ok=True)
        nut.write_text(textwrap.dedent(nut_contents))

    stub = type("Stub", (), {"case_template_dir": str(tmp_path)})()
    return OpenFOAMModel._detect_ks_patch(stub)


BOUNDARY_FIELD = """\
    boundaryField
    {{
        inlet
        {{
            type            calculated;
            value           uniform 0;
        }}
        {patch}
        {{
            type            nutkRoughWallFunction;
            Ks              uniform 0.01;
            Cs              uniform 0.5;
            value           uniform 0;
        }}
        atmosphere
        {{
            type            calculated;
            value           uniform 0;
        }}
    }}
"""


@pytest.mark.parametrize("patch", ["bottom", "base", "bed", "wall_1"])
def test_roughness_patch_is_read_from_the_case_template(tmp_path, patch):
    """The patch name is whatever the template calls it, not a hardcoded 'bottom'.

    Templates disagree on the name of the bed patch, and writing ``ks`` to a patch
    that does not exist is what the hardcoded name used to cause.
    """
    assert _detect(tmp_path, BOUNDARY_FIELD.format(patch=patch)) == patch


def test_a_patch_without_the_roughness_wall_function_is_not_selected(tmp_path):
    """Only nutkRoughWallFunction counts; a plain wall function is not a ks patch."""
    assert _detect(tmp_path, """\
        boundaryField
        {
            bottom
            {
                type            nutkWallFunction;
                value           uniform 0;
            }
        }
    """) is None


def test_missing_nut_file_is_reported_as_no_patch(tmp_path):
    """A case template without 0/nut yields None rather than raising."""
    assert _detect(tmp_path, None) is None


# --------------------------------------------------------------------------------
# boundary-condition writing
# --------------------------------------------------------------------------------

def _write_bc(tmp_path, contents, patch, bc_type, value):
    """Run update_boundary_condition over a field file and return the result."""
    from hydroBayesCal.openfoam.control_openfoam import OpenFOAMController

    field = tmp_path / "0" / "nut"
    field.parent.mkdir(parents=True, exist_ok=True)
    field.write_text(textwrap.dedent(contents))

    OpenFOAMController(str(tmp_path)).update_boundary_condition(
        file="0/nut", patch=patch, field_type="scalar", bc_type=bc_type, value=value,
    )
    return field.read_text()


def test_roughness_patch_gets_the_new_ks(tmp_path):
    """The calibrated ks reaches the rough-wall patch, and Cs is written alongside it."""
    out = _write_bc(
        tmp_path, BOUNDARY_FIELD.format(patch="bottom"),
        patch="bottom", bc_type="nutkRoughWallFunction", value=0.042,
    )
    assert "Ks uniform 0.04200;" in out
    assert "Cs uniform 0.5;" in out
    assert "uniform 0.01" not in out          # the template value is gone


def test_ks_is_not_injected_into_a_non_roughness_patch(tmp_path):
    """Ks/Cs are nutkRoughWallFunction settings and must not leak into other patches.

    They are inserted ahead of a patch's ``value`` line, so without a gate on the
    wall-function type any boundary condition updated with a non-None value would
    silently acquire roughness entries that OpenFOAM does not expect there.
    """
    out = _write_bc(
        tmp_path, BOUNDARY_FIELD.format(patch="bottom"),
        patch="inlet", bc_type="fixedValue", value=1.5,
    )
    inlet = out.split("inlet")[1].split("}")[0]
    assert "Ks" not in inlet
    assert "Cs" not in inlet
    assert "type fixedValue;" in inlet


def test_a_multiline_nonuniform_value_list_is_consumed(tmp_path):
    """A nonuniform value list is replaced wholesale, not left dangling.

    Hot-started fields carry ``value nonuniform List<scalar>`` blocks spanning many
    lines. Rewriting only the first line used to leave the list body behind, which
    OpenFOAM then read as stray tokens.
    """
    out = _write_bc(tmp_path, """\
        boundaryField
        {
            bottom
            {
                type            nutkRoughWallFunction;
                Ks              uniform 0.01;
                Cs              uniform 0.5;
                value           nonuniform List<scalar>
        3
        (
        0.1
        0.2
        0.3
        )
        ;
            }
        }
    """, patch="bottom", bc_type="nutkRoughWallFunction", value=0.02)

    assert "value uniform 0;" in out
    assert "nonuniform" not in out
    assert "0.1" not in out and "0.3" not in out   # list body consumed
    assert "Ks uniform 0.02000;" in out
