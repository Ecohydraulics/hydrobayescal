"""Tests for launching TELEMAC: the steering file name, the working directory and the launcher on PATH."""
import os
import stat
import sys

import pytest

pytest.importorskip("tqdm")

from hydroBayesCal.telemac.control_telemac import TelemacModel


def _model(model_dir, nproc=4):
    model = TelemacModel.__new__(TelemacModel)
    model.model_dir = str(model_dir)
    model.num_run = 1
    model.nproc = nproc
    model.tm_xd = "Telemac2d"
    model.tm_xd_dict = {"Telemac2d": "telemac2d.py ", "Telemac3d": "telemac3d.py "}
    return model


@pytest.mark.skipif(os.name == "nt", reason="the fake launcher is a POSIX script")
def test_the_launcher_runs_in_the_model_directory_with_the_file_name(tmp_path, monkeypatch):
    """Relative paths in the steering file (FORTRAN FILE, ...) are resolved against the model directory."""
    bin_dir, model_dir = tmp_path / "bin", tmp_path / "model"
    bin_dir.mkdir()
    model_dir.mkdir()
    (model_dir / "run.cas").write_text("/ steering file\n")
    launcher = bin_dir / "telemac2d.py"
    launcher.write_text(f"#!{sys.executable}\nimport os, sys\n"
                        "open('call.txt', 'w').write(os.getcwd() + '\\n' + ' '.join(sys.argv[1:]))\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    _model(model_dir).run_single_simulation(control_file="run.cas")

    cwd, arguments = (model_dir / "call.txt").read_text().splitlines()
    assert os.path.samefile(cwd, model_dir)
    assert arguments == "run.cas --ncsize=4"


def test_a_missing_launcher_or_steering_file_is_named(tmp_path, monkeypatch):
    (tmp_path / "run.cas").write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(FileNotFoundError, match="telemac2d.py"):
        _model(tmp_path).run_single_simulation(control_file="run.cas")
    with pytest.raises(FileNotFoundError, match="other.cas"):
        _model(tmp_path).run_single_simulation(control_file="other.cas")


def test_on_windows_a_python_launcher_is_found_without_pathext(tmp_path, monkeypatch):
    """shutil.which matches on Windows only the extensions in PATHEXT, which has no .PY by default."""
    from hydroBayesCal.telemac.control_telemac import find_telemac_launcher

    (tmp_path / "telemac2d.py").write_text("")  # not executable: shutil.which skips it, as on Windows without .PY
    monkeypatch.setenv("PATH", f"{tmp_path / 'empty'}{os.pathsep}{tmp_path}")

    assert find_telemac_launcher("telemac2d.py", windows=False) is None
    assert os.path.samefile(find_telemac_launcher("telemac2d.py", windows=True), tmp_path / "telemac2d.py")
    assert find_telemac_launcher("telemac3d.py", windows=True) is None
