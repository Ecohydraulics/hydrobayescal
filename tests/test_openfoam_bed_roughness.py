"""Tests for the bed roughness ``ks`` of the OpenFOAM binding: the patch it is written to and the roughness length z0 = ks / 30 of the atmospheric rough wall functions."""
import pytest

from hydroBayesCal.openfoam.control_openfoam import (
    ATM_ROUGH_WALL, FOUNDATION_ATM_ROUGH_WALL, KS_ROUGH_WALL, OpenFOAMController, _find_ks_patch, _set_patch_entry)


def _field(patches):
    body = "".join(f"    {name}\n    {{\n" + "".join(f"        {k:<16}{v};\n" for k, v in entries.items()) + "    }\n"
                   for name, entries in patches.items())
    return ("FoamFile\n{\n    class volScalarField;\n    object nut;\n}\n"
            "internalField uniform 0;\n\nboundaryField\n{\n" + body + "}\n")


def _write(case, time_dir, name, patches):
    folder = case / time_dir
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(_field(patches))
    return folder / name


@pytest.mark.parametrize("wall", [ATM_ROUGH_WALL, FOUNDATION_ATM_ROUGH_WALL, KS_ROUGH_WALL])
def test_the_bed_patch_is_found_for_every_rough_wall_function(tmp_path, wall):
    _write(tmp_path, "0", "nut", {"walls": {"type": "nutkWallFunction"}, "bottom": {"type": wall}})

    assert _find_ks_patch(str(tmp_path)) == ("bottom", wall)


def test_an_atmospheric_wall_function_has_priority(tmp_path):
    _write(tmp_path, "0", "nut", {"sides": {"type": KS_ROUGH_WALL}, "bed": {"type": FOUNDATION_ATM_ROUGH_WALL}})

    assert _find_ks_patch(str(tmp_path)) == ("bed", FOUNDATION_ATM_ROUGH_WALL)


def test_of_several_rough_patches_the_bed_is_chosen_or_the_choice_refused(tmp_path):
    _write(tmp_path, "0", "nut", {"left": {"type": KS_ROUGH_WALL}, "base": {"type": KS_ROUGH_WALL}})
    assert _find_ks_patch(str(tmp_path)) == ("base", KS_ROUGH_WALL)

    _write(tmp_path, "0", "nut", {"left": {"type": KS_ROUGH_WALL}, "right": {"type": KS_ROUGH_WALL}})
    with pytest.raises(ValueError, match="cannot tell"):
        _find_ks_patch(str(tmp_path))


def test_only_the_named_patch_is_changed(tmp_path):
    path = _write(tmp_path, "0", "nut", {"bottom": {"type": KS_ROUGH_WALL, "Ks": "uniform 0.1"},
                                         "sides": {"type": KS_ROUGH_WALL, "Ks": "uniform 0.2"}})

    assert _set_patch_entry(str(path), "bottom", "Ks", "uniform 0.05")
    text = path.read_text()
    assert "uniform 0.05" in text and "uniform 0.2" in text and "uniform 0.1" not in text
    assert not _set_patch_entry(str(path), "bottom", "z0", "uniform 0.05")


@pytest.mark.parametrize("wall, form", [(ATM_ROUGH_WALL, "constant"), (FOUNDATION_ATM_ROUGH_WALL, "uniform")])
def test_z0_keeps_the_form_its_solver_reads_in_every_time_directory(tmp_path, wall, form):
    """ESI reads z0 as a Function1 (constant), Foundation as a field (uniform)."""
    for time_dir in ("0", "600"):
        _write(tmp_path, time_dir, "nut", {"bottom": {"type": wall, "z0": f"{form} 0.1"}})
    epsilon = _write(tmp_path, "0", "epsilon", {"bottom": {"type": "epsilonWallFunction"}})

    OpenFOAMController(str(tmp_path)).update_roughness_length("bottom", 0.06 / 30.0, ks=0.06)

    for time_dir in ("0", "600"):
        assert f"{form} 0.002;" in (tmp_path / time_dir / "nut").read_text()
    assert "z0" not in epsilon.read_text(), "a field without z0 entry is left alone"


def test_a_nut_without_z0_entry_is_refused(tmp_path):
    _write(tmp_path, "0", "nut", {"bottom": {"type": ATM_ROUGH_WALL}})

    with pytest.raises(ValueError, match="No 'z0' entry"):
        OpenFOAMController(str(tmp_path)).update_roughness_length("bottom", 0.002)
