"""Tests for sand ratios averaged over GAIA bed layers, and for the result-file routing of the extraction.

GAIA writes the ratio of a sand class in a bed layer as ``LAY<layer> SAND RAT<class>``. The variable name therefore carries two numbers, and a layer average that takes the trailing one for the layer reads the sand classes of a single layer instead. The result still lies between 0 and 1, so nothing but the variable naming gives the error away. The naming is pinned here against small SELAFIN files written for the purpose, read back through ``TelemacModel.extract_data_point`` itself.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tqdm")

from hydroBayesCal.telemac.config_telemac import (
    gaia_sand_ratio_variable,
    resolve_gaia_layer_average,
)
from hydroBayesCal.telemac.control_telemac import TelemacModel
from hydroBayesCal.telemac.pputils.ppmodules.selafin_io_pp import ppSELAFIN

#: Two sand classes in two bed layers, the ratios of the classes adding up to one in each layer.
SAND_RATIOS = {
    "LAY1 SAND RAT1": 0.20,
    "LAY2 SAND RAT1": 0.50,
    "LAY1 SAND RAT2": 0.80,
    "LAY2 SAND RAT2": 0.50,
}
FINE_FRACTION = {"FINE SEDIMENT FRACTION": {"sand_classes": [1], "layers": [1, 2],
                                             "thicknesses": [0.08, 0.60]}}


# --------------------------------------------------------------------------- #
# variable naming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer, sand_class, expected", [
    (1, 1, "LAY1 SAND RAT1"),
    (2, 1, "LAY2 SAND RAT1"),      # second layer, first class
    (1, 2, "LAY1 SAND RAT2"),      # first layer, second class
    (10, 1, "LAY10SAND RAT1"),     # a two-digit layer number takes the place of the blank
    (1, 10, "LAY1 SAND RAT10"),
])
def test_sand_ratio_variable_names_follow_gaia(layer, sand_class, expected):
    assert gaia_sand_ratio_variable(layer, sand_class) == expected


def test_variable_names_match_a_gaia_reference_result():
    """The names agree with what GAIA itself writes, where a TELEMAC installation is at hand."""
    hometel = os.environ.get("HOMETEL", "")
    reference = os.path.join(hometel, "examples", "gaia", "cvsm-t2d", "gai_ref_cvsm.slf")
    if not hometel or not os.path.isfile(reference):
        pytest.skip("needs the GAIA examples of a TELEMAC installation (HOMETEL)")

    slf = ppSELAFIN(reference)
    slf.readHeader()
    written = {" ".join(name.split()) for name in slf.getVarNames()}

    # The example has five sand classes in two bed layers.
    expected = {gaia_sand_ratio_variable(layer, sand_class)
                for layer in (1, 2) for sand_class in (1, 2, 3, 4, 5)}
    assert expected <= written
    assert gaia_sand_ratio_variable(3, 1) not in written


def test_a_layer_average_names_one_variable_per_layer_and_class():
    resolved = resolve_gaia_layer_average(
        {"SAND": {"sand_classes": [1, 2], "layers": [1, 2, 3], "thicknesses": [0.1, 0.2, 0.3]}})

    assert resolved["SAND"]["variables"] == [
        ["LAY1 SAND RAT1", "LAY1 SAND RAT2"],
        ["LAY2 SAND RAT1", "LAY2 SAND RAT2"],
        ["LAY3 SAND RAT1", "LAY3 SAND RAT2"],
    ]


def test_an_entry_without_sand_classes_is_rejected_with_the_naming_rule():
    """This is the form that read sand classes for layers. It must not be guessed past."""
    with pytest.raises(ValueError, match=r"LAY<layer> SAND RAT<class>"):
        resolve_gaia_layer_average(
            {"LAY1 SAND RAT": {"layers": [1, 2], "thicknesses": [0.08, 0.60]}})


@pytest.mark.parametrize("entry, message", [
    ({"sand_classes": [1], "layers": [1, 2], "thicknesses": [0.08]}, "2 layers but 1 thicknesses"),
    ({"sand_classes": [1], "layers": [1], "thicknesses": [0.0]}, "> 0"),
    ({"sand_classes": [1], "layers": [1]}, "'layers' and 'thicknesses'"),
    ("LAY1 SAND RAT1", "must be a dictionary"),
])
def test_inconsistent_entries_are_rejected(entry, message):
    with pytest.raises(ValueError, match=message):
        resolve_gaia_layer_average({"SAND": entry})


def test_only_requested_targets_are_resolved():
    spec = dict(FINE_FRACTION, UNUSED={"layers": [1], "thicknesses": [1.0]})  # incomplete, but not requested
    assert list(resolve_gaia_layer_average(spec, requested=["FINE SEDIMENT FRACTION"])) == [
        "FINE SEDIMENT FRACTION"]


# --------------------------------------------------------------------------- #
# extraction from a result file
# --------------------------------------------------------------------------- #
def _write_selafin(path, variables, n_steps=2):
    """A two-triangle SELAFIN file with spatially uniform variables, so any interpolation returns the value itself.

    The last time step holds the given values, earlier steps hold zeros.
    """
    x = np.array([0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    ikle = np.array([[1, 2, 3], [1, 3, 4]])
    slf = ppSELAFIN(str(path))
    slf.setPrecision("f", 4)
    slf.setTitle("hydroBayesCal test")
    slf.setVarNames(list(variables))
    slf.setVarUnits([""] * len(variables))
    slf.setIPARAM([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    slf.setMesh(2, 4, 3, ikle, np.array([1, 2, 3, 4]), x, y)
    slf.writeHeader()
    values = np.array([[value] * 4 for value in variables.values()], dtype=float)
    for step in range(n_steps):
        slf.writeVariables(float(step), values if step == n_steps - 1 else np.zeros_like(values))
    slf.close()


def _model(tmp_path, gaia_layer_average=None, telemac_file=None, gaia_variables=None):
    """A model object carrying what ``extract_data_point`` reads, with its result files on disk."""
    model_dir, results = tmp_path / "simulation", tmp_path / "results"
    restart = results / "restart_data"
    for folder in (model_dir, restart):
        folder.mkdir(parents=True)

    model = TelemacModel.__new__(TelemacModel)
    model.restart_data_folder = str(restart)
    model.asr_dir = str(results)
    model.init_runs = 5
    model.gaia_cas = str(model_dir / "gaia.cas")
    model.gaia_results_filename = "gaia_1.slf"
    model.tm_results_filename = "telemac_1.slf"
    model.gaia_layer_average = gaia_layer_average
    _write_selafin(model_dir / "gaia_1.slf",
                   dict(SAND_RATIOS if gaia_variables is None else gaia_variables,
                        **{"CUMUL BED EVOL": -0.10}))
    if telemac_file is not None:
        _write_selafin(model_dir / "telemac_1.slf", telemac_file)
    return model, model_dir, results


def _extract(model, model_dir, results, quantities, **kwargs):
    points = pd.DataFrame({"Point": ["P1", "P2"], "X": [2.0, 7.0], "Y": [3.0, 6.0], "Z": [0.0, 0.0]})
    model.extract_data_point("telemac_1.slf", points, "extraction", quantities, 1,
                             str(model_dir), str(results), output_extraction="nearest", **kwargs)
    with open(results / "extraction-detailed.json") as handle:
        return json.load(handle)


def test_layer_average_reads_one_sand_class_in_the_configured_layers(tmp_path):
    """Sand class 1 over the layers 1 and 2: (0.08 * 0.20 + 0.60 * 0.50) / 0.68.

    Reading the classes 1 and 2 of layer 1 instead would give (0.08 * 0.20 + 0.60 * 0.80) / 0.68 = 0.73.
    """
    model, model_dir, results = _model(tmp_path, FINE_FRACTION)

    output = _extract(model, model_dir, results, ["CUMUL BED EVOL", "FINE SEDIMENT FRACTION"])

    for point in ("P1", "P2"):
        assert output[point][0]["FINE SEDIMENT FRACTION"] == pytest.approx(0.316 / 0.68, rel=1e-5)
        assert output[point][0]["CUMUL BED EVOL"] == pytest.approx(-0.10, rel=1e-5)
        assert "LAY1 SAND RAT1" not in output[point][0], "only the derived quantity is stored"


def test_the_ratios_of_several_classes_add_up_within_a_layer(tmp_path):
    spec = {"SAND": {"sand_classes": [1, 2], "layers": [1, 2], "thicknesses": [0.08, 0.60]}}
    model, model_dir, results = _model(tmp_path, spec)

    output = _extract(model, model_dir, results, ["SAND"])

    assert output["P1"][0]["SAND"] == pytest.approx(1.0, rel=1e-5)


def test_a_configuration_passed_to_the_call_has_priority(tmp_path):
    model, model_dir, results = _model(tmp_path, FINE_FRACTION)
    top_layer_only = {"FINE SEDIMENT FRACTION": {"sand_classes": [1], "layers": [1], "thicknesses": [0.08]}}

    output = _extract(model, model_dir, results, ["FINE SEDIMENT FRACTION"],
                      gaia_layer_average=top_layer_only)

    assert output["P1"][0]["FINE SEDIMENT FRACTION"] == pytest.approx(0.20, rel=1e-5)


def test_a_layer_the_model_does_not_have_is_named_in_the_error(tmp_path):
    spec = {"SAND": {"sand_classes": [1], "layers": [1, 2, 3], "thicknesses": [0.1, 0.2, 0.3]}}
    model, model_dir, results = _model(tmp_path, spec)

    with pytest.raises(ValueError, match="LAY3 SAND RAT1"):
        _extract(model, model_dir, results, ["SAND"])


def test_a_ratio_requested_without_its_class_points_to_the_configuration(tmp_path):
    """The name the removed implementation used is not a GAIA variable and not a configured average."""
    model, model_dir, results = _model(tmp_path)

    with pytest.raises(ValueError, match="gaia_layer_average"):
        _extract(model, model_dir, results, ["LAY1 SAND RAT"])


def test_gaia_quantities_alone_do_not_need_the_telemac_result(tmp_path):
    """A calibration on GAIA quantities only must not fail on a hydrodynamic file it never reads."""
    model, model_dir, results = _model(tmp_path, FINE_FRACTION)  # no TELEMAC file written

    output = _extract(model, model_dir, results, ["FINE SEDIMENT FRACTION"])

    assert set(output) == {"P1", "P2"}


def test_result_files_leave_the_model_directory_also_when_they_were_not_read(tmp_path):
    """Nothing else removes them, so a file left behind is a file per run for the whole calibration."""
    model, model_dir, results = _model(tmp_path, FINE_FRACTION,
                                       telemac_file={"WATER DEPTH": 1.5, "SCALAR VELOCITY": 0.7})
    stale = results / "saved_slf_files"
    stale.mkdir()
    (stale / "gaia_1.slf").write_text("left over from an earlier run with the same number")

    _extract(model, model_dir, results, ["FINE SEDIMENT FRACTION"])  # reads the GAIA file only

    assert sorted(os.listdir(model_dir)) == []
    assert sorted(os.listdir(stale)) == ["gaia_1.slf", "telemac_1.slf"]
    assert (stale / "gaia_1.slf").stat().st_size > 100, "the stale file was replaced by the new result"


def test_telemac_and_gaia_quantities_are_read_from_their_own_files(tmp_path):
    model, model_dir, results = _model(tmp_path, FINE_FRACTION,
                                       telemac_file={"WATER DEPTH": 1.5, "SCALAR VELOCITY": 0.7})

    output = _extract(model, model_dir, results,
                      ["WATER DEPTH", "SCALAR VELOCITY", "CUMUL BED EVOL", "FINE SEDIMENT FRACTION"])

    assert output["P2"][0] == pytest.approx({"WATER DEPTH": 1.5, "SCALAR VELOCITY": 0.7,
                                             "CUMUL BED EVOL": -0.10,
                                             "FINE SEDIMENT FRACTION": 0.316 / 0.68}, rel=1e-5)


def test_an_unknown_quantity_is_refused_instead_of_dropped(tmp_path):
    model, model_dir, results = _model(tmp_path, telemac_file={"WATER DEPTH": 1.5})

    with pytest.raises(ValueError, match="WATER DEPHT"):
        _extract(model, model_dir, results, ["WATER DEPHT"])


# --------------------------------------------------------------------------- #
# validation attempts
# --------------------------------------------------------------------------- #
def _bare_model(tmp_path):
    model = TelemacModel.__new__(TelemacModel)
    model.restart_data_folder = str(tmp_path)
    return model


def _names(paths):
    return [os.path.basename(path) for path in paths]


def test_validation_files_are_found_without_a_run_in_this_session(tmp_path):
    """Post-processing builds a model and reads validation results it did not produce itself."""
    for name in ("collocation-points-validation_1.csv", "model-results-validation_1.json",
                 "collocation-points-validation_2.csv", "model-results-validation_2.json"):
        (tmp_path / name).write_text("")

    assert _names(_bare_model(tmp_path).validation_paths()) == [
        "collocation-points-validation_2.csv", "model-results-validation_2.json",
        "model-results-validation_2.csv"]
    assert _names(_bare_model(tmp_path).validation_paths(attempt=1))[0] == \
        "collocation-points-validation_1.csv"


def test_unnumbered_validation_files_of_earlier_versions_stay_readable(tmp_path):
    (tmp_path / "collocation-points-validation.csv").write_text("")

    assert _names(_bare_model(tmp_path).validation_paths()) == [
        "collocation-points-validation.csv", "model-results-validation.json",
        "model-results-validation.csv"]


def test_a_new_validation_attempt_does_not_reuse_the_files_of_the_previous_one(tmp_path):
    model = _bare_model(tmp_path)

    first = model.validation_paths(new_attempt=True)
    for path in first[:2]:
        open(path, "w").close()
    assert model.validation_paths() == first, "the running attempt is kept until a new one starts"
    second = model.validation_paths(new_attempt=True)

    assert _names(first)[0] == "collocation-points-validation_1.csv"
    assert _names(second)[0] == "collocation-points-validation_2.csv"
    assert not set(first) & set(second)
