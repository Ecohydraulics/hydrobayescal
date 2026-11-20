"""Tests for the multiflow wrapper shared by TELEMAC and OpenFOAM.

The wrapper is exercised with stub single-flow models carrying only what it reads, so the tests need no solver. The OpenFOAM subclass is checked for what differs from TELEMAC: each flow is its own case template, the flows need distinct run folders, and the OpenFOAM ``run_multiple_simulations`` takes fewer keywords than TELEMAC's.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from hydroBayesCal.drivers import copy_driver, driver_path
from hydroBayesCal.multiflow import MultiflowModel
from hydroBayesCal.openfoam.multiflow_openfoam import MultiflowOpenFOAMModel
from hydroBayesCal.telemac.multiflow_telemac import MultiflowTelemacModel


class StubFlowModel:
    """A single-flow model: two locations, one quantity, output = location + 10 x first parameter + flow offset."""

    created = []

    def __init__(self, calibration_pts_file_path, res_dir, offset=0.0, **kwargs):
        self.kwargs = dict(kwargs, calibration_pts_file_path=calibration_pts_file_path, res_dir=res_dir)
        StubFlowModel.created.append(self)
        self.offset = float(offset)
        self.observations = np.array([[1.0, 2.0]]) + self.offset
        self.variances = np.array([0.01, 0.01])
        self.measurement_errors = np.array([0.1, 0.1])
        self.nloc = 2
        self.ndim = 1
        self.calibration_parameters = ["ks"]
        self.param_values = [[0.01, 0.3]]
        self.parameter_ranges = [[0.01, 0.3]]
        self.calibration_quantities = ["U_x"]
        self.extraction_quantities = ["U_x"]
        self.num_calibration_quantities = 1
        self.multitask_selection = "variables"
        self.init_runs = kwargs.get("init_runs", 4)
        self.max_runs = 8
        self.complete_bal_mode = True
        self.only_bal_mode = False
        self.validation = False
        self.delete_complex_outputs = False
        self.user_param_values = False
        self.user_collocation_points = None
        self.restart_collocation_points = None
        self.restart_data_folder = str(Path(res_dir) / "restart_data")
        self.model_evaluations = None
        self.calls = []

    def friction_laws(self):
        return {"ks": "NIKU"}

    def turbulence_constants(self):
        return {}

    # the OpenFOAM signature: no output_extraction, n or kill_process
    def run_multiple_simulations(self, collocation_points, complete_bal_mode=True, validation=False,
                                 bal_iteration=None, bal_new_set_parameters=None, start_index=0):
        self.calls.append(dict(start_index=start_index, bal_new_set_parameters=bal_new_set_parameters))
        points = np.atleast_2d(collocation_points if bal_new_set_parameters is None
                               else bal_new_set_parameters)
        rows = np.array([[1.0 + 10 * p[0] + self.offset, 2.0 + 10 * p[0] + self.offset] for p in points])
        if bal_new_set_parameters is None:
            self.model_evaluations = rows
        else:
            self.model_evaluations = np.vstack([self.model_evaluations, rows])


class StubMultiflowOpenFOAM(MultiflowOpenFOAMModel):
    model_class = StubFlowModel


def _flows(tmp_path):
    return [
        {"name": "august", "case_template_dir": str(tmp_path / "template-august"),
         "calibration_pts_file_path": str(tmp_path / "august.csv"), "offset": 0.0},
        {"name": "march", "case_template_dir": str(tmp_path / "template-march"),
         "calibration_pts_file_path": str(tmp_path / "march.csv"), "offset": 5.0},
    ]


@pytest.fixture(autouse=True)
def _reset():
    StubFlowModel.created.clear()


def test_flows_get_their_own_run_folder_and_template(tmp_path):
    model = StubMultiflowOpenFOAM(flows=_flows(tmp_path), res_dir=str(tmp_path / "res"),
                                  model_dir=str(tmp_path / "runs"))
    kw = [m.kwargs for m in model.models]
    assert [k["model_dir"] for k in kw] == [str(tmp_path / "runs" / "flow-august"),
                                            str(tmp_path / "runs" / "flow-march")]
    assert [k["case_template_dir"] for k in kw] == [str(tmp_path / "template-august"),
                                                    str(tmp_path / "template-march")]
    assert [k["res_dir"] for k in kw] == [str(tmp_path / "res" / "flow-august"),
                                          str(tmp_path / "res" / "flow-march")]
    assert [k["dict_output_name"] for k in kw] == ["extraction-data-august", "extraction-data-march"]


def test_a_flow_may_name_its_run_folder(tmp_path):
    flows = _flows(tmp_path)
    flows[1]["model_dir"] = str(tmp_path / "elsewhere")
    model = StubMultiflowOpenFOAM(flows=flows, res_dir=str(tmp_path / "res"), model_dir=str(tmp_path / "runs"))
    assert model.models[1].kwargs["model_dir"] == str(tmp_path / "elsewhere")


def test_flows_sharing_a_run_folder_are_refused(tmp_path):
    flows = _flows(tmp_path)
    for f in flows:
        f["model_dir"] = str(tmp_path / "same")
    with pytest.raises(ValueError, match="model_dir must be distinct"):
        StubMultiflowOpenFOAM(flows=flows, res_dir=str(tmp_path / "res"))


def test_a_flow_without_template_is_refused(tmp_path):
    flows = _flows(tmp_path)
    del flows[0]["case_template_dir"]
    with pytest.raises(ValueError, match="lacks case_template_dir"):
        StubMultiflowOpenFOAM(flows=flows, res_dir=str(tmp_path / "res"))


def test_observations_and_runs_are_stacked_in_flow_order(tmp_path):
    model = StubMultiflowOpenFOAM(flows=_flows(tmp_path), res_dir=str(tmp_path / "res"))
    assert model.nloc == 4 and model.nloc_per_flow == [2, 2]
    np.testing.assert_allclose(model.observations, [[1.0, 2.0, 6.0, 7.0]])
    design = np.array([[0.1], [0.2]])
    # TELEMAC-only keywords the stock initial design may pass are dropped for OpenFOAM
    out = model.run_multiple_simulations(collocation_points=design, complete_bal_mode=True,
                                         validation=False, start_index=0,
                                         output_extraction_time="mean_last", n=80)
    np.testing.assert_allclose(out, [[2.0, 3.0, 7.0, 8.0], [3.0, 4.0, 8.0, 9.0]])
    assert (Path(model.calibration_folder) / "model-results-multiflow.csv").is_file()
    # a BAL iteration appends one row per flow
    out = model.run_multiple_simulations(collocation_points=design, bal_new_set_parameters=np.array([[0.3]]),
                                         bal_iteration=1, complete_bal_mode=True, validation=False)
    assert out.shape == (3, 4)


def test_identical_initial_runs_fail_early(tmp_path):
    model = StubMultiflowOpenFOAM(flows=_flows(tmp_path), res_dir=str(tmp_path / "res"))
    with pytest.raises(RuntimeError, match="identical outputs"):
        model.run_multiple_simulations(collocation_points=np.array([[0.1], [0.1]]))


def test_the_initial_design_size_reaches_every_flow(tmp_path):
    model = StubMultiflowOpenFOAM(flows=_flows(tmp_path), res_dir=str(tmp_path / "res"))
    model.init_runs = 12
    assert [m.init_runs for m in model.models] == [12, 12]


def test_the_wrapper_has_every_attribute_the_openfoam_driver_reads(tmp_path):
    """bal_openfoam.py's run_complex_model and run_bal_model are reused unchanged; an attribute missing here fails after the initial design."""
    model = StubMultiflowOpenFOAM(flows=_flows(tmp_path), res_dir=str(tmp_path / "res"))
    used = set(re.findall(r"complex_model\.([A-Za-z_]+)", driver_path("bal_openfoam.py").read_text()))
    missing = sorted(a for a in used if not hasattr(model, a))
    assert not missing, f"MultiflowOpenFOAMModel lacks {missing}"


def test_the_base_class_is_abstract(tmp_path):
    with pytest.raises(TypeError, match="abstract"):
        MultiflowModel(flows=_flows(tmp_path), res_dir=str(tmp_path))


def test_telemac_flows_keep_their_rules():
    assert MultiflowTelemacModel.required_keys == (
        "control_file", "results_filename_base", "calibration_pts_file_path")
    assert MultiflowTelemacModel.distinct_keys == ("results_filename_base",)


def test_the_openfoam_multiflow_driver_brings_its_sibling(tmp_path):
    copied = copy_driver("bal_openfoam_multiflow.py", tmp_path)
    assert copied == tmp_path / "bal_openfoam_multiflow.py"
    assert (tmp_path / "bal_openfoam.py").is_file()


def test_the_hydraulic_mean_has_one_trend_per_combined_column(tmp_path):
    """sampling['gp_mean'] (PR 33): the trend of every output column of every flow, the roles of the parameters read from the first flow."""
    model = StubMultiflowOpenFOAM(flows=_flows(tmp_path), res_dir=str(tmp_path / "res"))
    assert model.trend_features("constant") is None
    features = model.trend_features("hydraulic")
    assert len(features) == model.nloc * len(model.calibration_quantities) == 4
    assert features[0] == features[2]          # the same point of another flow, the same trend
