"""Tests for the results folder of a calibration case and for the base-class parameters the bindings forward."""
import os

import pytest

from hydroBayesCal.hysim import RESULTS_FOLDER, results_folder_name


@pytest.mark.parametrize("case", [None, "", "   "])
def test_without_a_case_the_folder_keeps_its_name(case):
    """Results of earlier releases stay restartable with only_bal_mode."""
    assert results_folder_name(case) == RESULTS_FOLDER == "auto-saved-results-HydroBayesCal"


def test_a_case_is_appended():
    assert results_folder_name(" ks_and_shields ") == "auto-saved-results-HydroBayesCal_ks_and_shields"


@pytest.mark.parametrize("case", ["a/b", "..", os.path.join("x", "y")])
def test_a_path_is_refused(case):
    with pytest.raises(ValueError, match="folder name"):
        results_folder_name(case)


def _delft3d_case(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "river.mdf").write_text("Runid = #river#\n")
    return case


def test_delft3d_forwards_the_error_terms_and_the_case(tmp_path):
    """The three error terms were accepted but not forwarded, so the base-class defaults replaced them."""
    from hydroBayesCal.delft3d.control_delft3d import Delft3DModel

    model = Delft3DModel(case_template_dir=str(_delft3d_case(tmp_path)), model_dir=str(tmp_path / "model"),
                         res_dir=str(tmp_path / "results"), measurement_error=0.3, gpe_error=0.02,
                         model_structural_error=0.05, calibration_case="roughness")

    assert (model.measurement_error, model.gpe_error, model.model_structural_error) == (0.3, 0.02, 0.05)
    assert model.asr_dir == str(tmp_path / "results" / "auto-saved-results-HydroBayesCal_roughness")
    assert os.path.isdir(model.restart_data_folder)


def test_openfoam_forwards_the_case(tmp_path):
    from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel

    case = tmp_path / "case"
    (case / "0").mkdir(parents=True)
    model = OpenFOAMModel(case_template_dir=str(case), model_dir=str(tmp_path / "model"),
                          res_dir=str(tmp_path / "results"), calibration_case="cmu")

    assert model.asr_dir == str(tmp_path / "results" / "auto-saved-results-HydroBayesCal_cmu")
