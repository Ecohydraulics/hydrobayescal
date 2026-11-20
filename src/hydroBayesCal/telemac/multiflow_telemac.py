"""Multi-discharge (multi-flow) TELEMAC model for HydroBayesCal.

Calibrates one shared parameter set (e.g. friction-zone roughness) against measurements taken at SEVERAL steady discharges: every collocation point is evaluated by running TELEMAC once per flow (each flow has its own steering ``.cas`` and its own calibration-points CSV), and the per-flow results are concatenated into one combined observation and output space. The mechanics are in :class:`hydroBayesCal.multiflow.MultiflowModel`; this module names the TELEMAC model and what each flow must give.

Requirements on the flows:

* all steering files live in the same ``model_dir`` (same mesh, boundary and friction files) and differ in the hydraulic forcing (e.g. PRESCRIBED FLOWRATES / ELEVATIONS);
* each flow must use a distinct ``results_filename_base``.

Example flow specification (see ``src/hydroBayesCal/drivers/bal_telemac_multiflow.py``)::

    flows = [
        {"name": "q47-3", "control_file": "steady2d.cas",
         "results_filename_base": "r2d-q47-3",
         "calibration_pts_file_path": "/path/measurements-q47.3.csv"},
        {"name": "q168", "control_file": "steady2d-q168.cas",
         "results_filename_base": "r2d-q168",
         "calibration_pts_file_path": "/path/measurements-q168.csv"},
    ]
"""

from hydroBayesCal.multiflow import MultiflowModel, _check_run_matrix  # noqa: F401 - _check_run_matrix kept importable from here
from hydroBayesCal.telemac.control_telemac import TelemacModel


class MultiflowTelemacModel(MultiflowModel):
    """One shared calibration across several steady-flow TELEMAC cases.

    Each flow entry gives ``name``, ``control_file`` (steering ``.cas`` in ``model_dir``), ``results_filename_base`` (distinct base for the per-run SELAFIN renames) and ``calibration_pts_file_path``; every further keyword of :class:`~hydroBayesCal.telemac.control_telemac.TelemacModel` is shared by all flows.
    """

    model_class = TelemacModel
    required_keys = ("control_file", "results_filename_base", "calibration_pts_file_path")
    distinct_keys = ("results_filename_base",)
