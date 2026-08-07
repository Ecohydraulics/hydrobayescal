"""Tests for the packaged-driver API.

The drivers are shipped as package *data*, not as importable modules, so nothing in
the import machinery would notice if one stopped being packaged. These tests pin the
two properties a downstream caller depends on: every driver in the source tree is
reachable through the API, and ``copy_driver`` produces a directory a driver can
actually run from, which for ``bal_telemac_multiflow.py`` means its single-flow
sibling has to come along.
"""
import pathlib

import pytest

from hydroBayesCal import available_drivers, copy_driver, driver_path, drivers_dir

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_every_driver_in_the_tree_is_reachable():
    """A driver added to the folder is packaged, or this fails loudly."""
    on_disk = {p.name for p in (REPO / "src" / "hydroBayesCal" / "drivers").glob("*.py")
               if p.name != "__init__.py"}
    assert set(available_drivers()) == on_disk
    assert "bal_telemac.py" in on_disk        # sanity: the glob found something real


@pytest.mark.parametrize("name", ["bal_telemac.py", "bal_openfoam.py", "bal_delft3d.py"])
def test_driver_path_resolves_the_shipped_drivers(name):
    assert driver_path(name) == drivers_dir() / name
    assert driver_path(name).is_file()


def test_the_py_suffix_is_optional():
    assert driver_path("bal_telemac") == driver_path("bal_telemac.py")


def test_an_unknown_driver_names_the_ones_that_exist():
    """A typo has to surface here, not later as a missing-file error in a run."""
    with pytest.raises(FileNotFoundError, match="bal_telemac.py"):
        driver_path("bal_telmac.py")


def test_copy_driver_brings_the_sibling_it_imports(tmp_path):
    """bal_telemac_multiflow.py imports bal_telemac.py from beside itself."""
    copied = copy_driver("bal_telemac_multiflow.py", tmp_path)

    assert copied == tmp_path / "bal_telemac_multiflow.py"
    assert copied.is_file()
    assert (tmp_path / "bal_telemac.py").is_file()
    assert copied.read_text() == driver_path("bal_telemac_multiflow.py").read_text()


def test_copy_driver_creates_a_missing_destination(tmp_path):
    dest = tmp_path / "does" / "not" / "exist"
    assert copy_driver("bal_telemac.py", dest).is_file()


def test_overwrite_false_leaves_an_edited_driver_alone(tmp_path):
    """A user who adapted their copy should not lose it to a second call."""
    copy_driver("bal_telemac.py", tmp_path)
    (tmp_path / "bal_telemac.py").write_text("# adapted locally\n")

    copy_driver("bal_telemac.py", tmp_path, overwrite=False)
    assert (tmp_path / "bal_telemac.py").read_text() == "# adapted locally\n"

    copy_driver("bal_telemac.py", tmp_path)
    assert (tmp_path / "bal_telemac.py").read_text() != "# adapted locally\n"
