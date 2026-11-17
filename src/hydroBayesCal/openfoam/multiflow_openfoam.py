"""Multi-discharge (multi-flow) OpenFOAM model for HydroBayesCal.

Calibrates one shared parameter set (bed roughness ``ks`` and k-epsilon coefficients) against velocity measurements taken at SEVERAL steady discharges. Each flow is its own OpenFOAM case template (built for that discharge, with its own inflow and outlet values and its own initial fields) and its own calibration-points CSV; every collocation point is run once per flow, and the outputs are concatenated by :class:`hydroBayesCal.multiflow.MultiflowModel`.

Each flow copies its template into ``<model_dir>/run_NNNN`` for every run, so the flows need distinct run folders: a flow entry may give ``model_dir``; otherwise it is ``<model_dir>/flow-<name>`` below the shared ``model_dir`` (or below ``res_dir`` without one).

Example flow specification (see ``src/hydroBayesCal/drivers/bal_openfoam_multiflow.py``)::

    flows = [
        {"name": "august-2026", "case_template_dir": "/path/template-august-2026",
         "calibration_pts_file_path": "/path/measurements-august-2026.csv"},
        {"name": "march-2026", "case_template_dir": "/path/template-march-2026",
         "calibration_pts_file_path": "/path/measurements-march-2026.csv"},
    ]
"""

import os

from hydroBayesCal.multiflow import MultiflowModel
from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel


class MultiflowOpenFOAMModel(MultiflowModel):
    """One shared calibration across several steady-flow OpenFOAM cases.

    Each flow entry gives ``name``, ``case_template_dir`` and ``calibration_pts_file_path``, and optionally ``model_dir``; every further keyword of :class:`~hydroBayesCal.openfoam.control_openfoam.OpenFOAMModel` is shared by all flows.
    """

    model_class = OpenFOAMModel
    required_keys = ("case_template_dir", "calibration_pts_file_path")
    distinct_keys = ("case_template_dir", "model_dir")

    def _flow_kwargs(self, name, spec, res_dir, shared):
        kwargs = super()._flow_kwargs(name, spec, res_dir, shared)
        if not spec.get("model_dir"):
            root = shared.get("model_dir") or res_dir or os.getcwd()
            kwargs["model_dir"] = os.path.join(root, f"flow-{name}")
        kwargs["model_dir"] = os.path.abspath(kwargs["model_dir"])
        kwargs["case_template_dir"] = os.path.abspath(kwargs["case_template_dir"])
        return kwargs

    def _describe(self, kwargs):
        return os.path.basename(str(kwargs["case_template_dir"]))
