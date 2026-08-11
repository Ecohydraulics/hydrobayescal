"""Multi-discharge (multi-flow) TELEMAC model for HydroBayesCal.

Calibrates one shared parameter set (e.g. friction-zone roughness) against
measurements taken at SEVERAL steady discharges: every collocation point is
evaluated by running the full-complexity model once per flow (each flow has its
own steering ``.cas`` and its own calibration-points CSV), and the per-flow
results are concatenated into one combined observation/output space that the
existing surrogate + Bayesian Active Learning machinery consumes unchanged.

Design: **composition, not modification**. :class:`MultiflowTelemacModel` wraps
one stock :class:`~hydroBayesCal.telemac.control_telemac.TelemacModel` per flow
(each with its own results subfolder ``<res_dir>/flow-<name>``), so all
single-flow behaviour (steering/friction updates, solver launch, extraction,
output processing, restart files) is exactly the code current users run. The
wrapper only

* concatenates ``observations`` / ``variances`` / ``measurement_errors`` along
  the location axis (order = order of the ``flows`` list),
* runs the flows sequentially per collocation point set and horizontally stacks
  the per-flow ``model_evaluations`` ([n_runs x sum(nloc_i * n_quantities)]),
* mirrors the bookkeeping attributes the ``bal_telemac`` driver reads.

Requirements on the flows:

* identical ``calibration_parameters`` / ``param_values`` (the whole point: one
  shared parameter vector),
* identical ``calibration_quantities`` / ``extraction_quantities``,
* all steering files live in the same ``model_dir`` (same mesh/boundary/friction
  files) and differ in the hydraulic forcing (e.g. PRESCRIBED FLOWRATES /
  ELEVATIONS); each flow must use a distinct ``results_filename_base``.

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

import os

import numpy as np

from hydroBayesCal.function_pool import *  # noqa: F401,F403 - provides logging/logger
from hydroBayesCal.telemac.control_telemac import TelemacModel

try:
    logger  # provided by function_pool
except NameError:  # pragma: no cover - fallback if function_pool changes
    import logging
    logger = logging.getLogger(__name__)


def _check_run_matrix(matrix, flow_name, init_runs, is_init_phase):
    """Fail fast on a degenerate per-flow evaluation matrix.

    Guards against the silent extraction/accumulation failures that otherwise
    only surface after days of compute: a row of all zeros (a run whose outputs
    never made it into the results, e.g. a JSON that failed to accumulate), or
    initial-design rows that are byte-identical across every run (the results
    did not vary with the calibration parameter - so either the parameter is
    not being applied or every run read the same file). Only checked for the
    initial design (``is_init_phase``); BAL rows are single new runs.
    """
    if not is_init_phase:
        return
    n = min(init_runs, matrix.shape[0])
    init_rows = matrix[:n]
    zero_rows = [i + 1 for i, r in enumerate(init_rows) if not np.any(r)]
    if zero_rows:
        raise RuntimeError(
            f"multiflow flow '{flow_name}': run(s) {zero_rows} extracted all "
            "zeros - the model outputs were not captured (check the per-flow "
            "detailed JSON accumulation and that the result SELAFIN was found).")
    if n > 1 and np.all(init_rows == init_rows[0]):
        raise RuntimeError(
            f"multiflow flow '{flow_name}': all {n} initial-design runs returned "
            "identical outputs - the calibration parameter is not changing the "
            "results (check the friction .tbl update / per-run result files).")


class MultiflowTelemacModel:
    """One shared calibration across several steady-flow TELEMAC cases.

    Parameters
    ----------
    flows : list of dict
        One entry per steady flow with keys ``name`` (short tag used for the
        per-flow results subfolder and output files), ``control_file`` (steering
        ``.cas`` in ``model_dir``), ``results_filename_base`` (distinct base for
        the per-run SELAFIN renames) and ``calibration_pts_file_path`` (that
        flow's calibration-points CSV). Optional: ``dict_output_name``.
    res_dir : str
        Root results directory. The combined artifacts (surrogate, BAL
        dictionary) land in ``<res_dir>/auto-saved-results-HydroBayesCal``;
        every flow gets its own stock results tree in ``<res_dir>/flow-<name>``.
    **shared
        Every further keyword accepted by
        :class:`~hydroBayesCal.telemac.control_telemac.TelemacModel` /
        ``HydroSimulations`` (``model_dir``, ``friction_file``, ``n_cpus``,
        ``init_runs``, ``max_runs``, ``calibration_parameters``,
        ``param_values``, ``calibration_quantities``,
        ``extraction_quantities``, ``complete_bal_mode``, ``only_bal_mode``,
        ``delete_complex_outputs``, ``validation``, ...). They are passed to
        every per-flow model unchanged.
    """

    def __init__(self, flows, res_dir="", **shared):
        if not flows:
            raise ValueError("multiflow model needs at least one flow spec")
        names = [str(f.get("name", i)) for i, f in enumerate(flows)]
        if len(set(names)) != len(names):
            raise ValueError(f"flow names must be unique, got {names}")
        bases = [f.get("results_filename_base") for f in flows]
        if len(set(bases)) != len(bases):
            raise ValueError(
                f"results_filename_base must be distinct per flow, got {bases} "
                "(per-run result files would overwrite each other)")

        self.flow_specs = list(flows)
        self.flow_names = names
        self.res_dir = res_dir
        self.models = []
        for name, spec in zip(names, flows):
            kwargs = dict(shared)
            kwargs.update(
                control_file=spec["control_file"],
                calibration_pts_file_path=spec["calibration_pts_file_path"],
                results_filename_base=spec["results_filename_base"],
                dict_output_name=spec.get("dict_output_name",
                                          f"extraction-data-{name}"),
                res_dir=os.path.join(res_dir, f"flow-{name}"),
            )
            logger.info(f"multiflow: initialising flow '{name}' "
                        f"({spec['control_file']}, "
                        f"{os.path.basename(str(spec['calibration_pts_file_path']))})")
            self.models.append(TelemacModel(**kwargs))

        first = self.models[0]
        for m, name in zip(self.models[1:], names[1:]):
            if m.calibration_quantities != first.calibration_quantities:
                raise ValueError(f"flow '{name}': calibration_quantities differ")
            if m.num_calibration_quantities != first.num_calibration_quantities:
                raise ValueError(f"flow '{name}': quantity count differs")

        # ---- combined observation space (location axis = flows in order) ----
        self.observations = np.hstack([m.observations for m in self.models])
        self.variances = np.concatenate([np.asarray(m.variances).ravel()
                                         for m in self.models])
        self.measurement_errors = np.concatenate(
            [np.asarray(m.measurement_errors).ravel() for m in self.models])
        self.nloc = int(sum(m.nloc for m in self.models))
        self.nloc_per_flow = [int(m.nloc) for m in self.models]

        # ---- bookkeeping attributes the driver reads (mirror the first flow) --
        self.ndim = first.ndim
        self.calibration_parameters = first.calibration_parameters
        self.param_values = first.param_values
        self.parameter_ranges = getattr(first, "parameter_ranges",
                                        list(first.param_values))
        self.calibration_quantities = first.calibration_quantities
        self.extraction_quantities = first.extraction_quantities
        self.num_calibration_quantities = first.num_calibration_quantities
        self.multitask_selection = first.multitask_selection
        self._init_runs = first.init_runs
        self.max_runs = first.max_runs
        self.complete_bal_mode = first.complete_bal_mode
        self.only_bal_mode = first.only_bal_mode
        self.validation = first.validation
        self.delete_complex_outputs = first.delete_complex_outputs
        self.user_param_values = first.user_param_values
        self.gpe_error = getattr(first, "gpe_error", 0.0)
        self.measurement_error = getattr(first, "measurement_error", 0.10)
        self.model_structural_error = getattr(first, "model_structural_error", 0.0)
        self.user_collocation_points = first.user_collocation_points
        self.restart_collocation_points = first.restart_collocation_points
        self.dict_output_name = "extraction-data-multiflow"
        self.model_evaluations = None

        # ---- combined artifact tree (mirrors the hysim conventions) ---------
        self.asr_dir = os.path.join(res_dir, "auto-saved-results-HydroBayesCal")
        self.calibration_folder = os.path.join(
            self.asr_dir, "calibration-data", "_".join(self.calibration_quantities))
        self.restart_data_folder = os.path.join(self.asr_dir, "restart_data")
        for d in (self.asr_dir, self.calibration_folder, self.restart_data_folder,
                  os.path.join(self.asr_dir, "plots"),
                  os.path.join(self.asr_dir, "surrogate-gpe")):
            os.makedirs(d, exist_ok=True)

        logger.info(
            f"multiflow model ready: {len(self.models)} flows "
            f"({', '.join(names)}), {self.nloc} combined calibration points "
            f"({' + '.join(str(n) for n in self.nloc_per_flow)}), "
            f"{self.ndim} calibration parameter(s)")

    @property
    def init_runs(self):
        """Size of the initial design, kept identical across every flow.

        A staged initial design grows ``init_runs`` between blocks
        (:mod:`~hydroBayesCal.surrogate.initial_design`). Each flow runs the design
        through its own ``TelemacModel``, which reads its *own* ``init_runs``, so
        assigning here has to reach them too; otherwise the flows would keep running the
        first block while the combined model believes the design has grown.
        """
        return self._init_runs

    @init_runs.setter
    def init_runs(self, value):
        self._init_runs = int(value)
        for model in self.models:
            model.init_runs = int(value)

    # ------------------------------------------------------------------ runs
    def run_multiple_simulations(self, collocation_points=None,
                                 bal_new_set_parameters=None,
                                 bal_iteration=int(),
                                 complete_bal_mode=True,
                                 output_extraction="interpolated",
                                 output_extraction_time="last",
                                 n=40,
                                 validation=False,
                                 kill_process=True,
                                 start_index=0):
        """Run every flow for the given collocation points and stack outputs.

        Same signature/semantics as ``TelemacModel.run_multiple_simulations``;
        each flow runs the full set sequentially through its stock model (so
        the per-flow trees carry the standard collocation/extraction files),
        then ``model_evaluations`` becomes the horizontal stack
        ``[n_runs x sum(nloc_i * n_quantities)]`` in flow order.
        """
        per_flow = []
        for name, m in zip(self.flow_names, self.models):
            logger.info(f"multiflow: === flow '{name}' "
                        f"({m.control_file}) ===")
            m.run_multiple_simulations(
                collocation_points=collocation_points,
                bal_new_set_parameters=bal_new_set_parameters,
                bal_iteration=bal_iteration,
                complete_bal_mode=complete_bal_mode,
                output_extraction=output_extraction,
                output_extraction_time=output_extraction_time,
                n=n,
                validation=validation,
                kill_process=kill_process,
                start_index=start_index,
            )
            per_flow.append(np.atleast_2d(m.model_evaluations))

        rows = {p.shape[0] for p in per_flow}
        if len(rows) != 1:
            raise RuntimeError(
                "multiflow: flows returned different run counts "
                f"{[p.shape for p in per_flow]} - a flow's solver run or "
                "extraction failed; check the per-flow logs")
        for name, p in zip(self.flow_names, per_flow):
            _check_run_matrix(p, name, self.init_runs, bal_new_set_parameters is None)
        self.model_evaluations = np.hstack(per_flow)
        logger.info(f"multiflow: combined model evaluations "
                    f"{self.model_evaluations.shape} "
                    f"(runs x locations*quantities)")
        # persist the combined matrix (columns grouped by flow, in flow order)
        header = ",".join(
            f"{name}_PT{i + 1}_{q}"
            for name, m in zip(self.flow_names, self.models)
            for i in range(m.nloc)
            for q in self.calibration_quantities)
        np.savetxt(os.path.join(self.calibration_folder,
                                "model-results-multiflow.csv"),
                   self.model_evaluations, delimiter=",", fmt="%.8f",
                   header=header)
        return self.model_evaluations

    # ------------------------------------------------------- restart support
    def output_processing(self, output_data_path="", **kwargs):
        """Combine per-flow restart outputs (``only_bal_mode`` restarts).

        The stock driver passes the path of the *combined* model's restart
        JSON; a multiflow restart instead re-reads every flow's own
        ``initial-model-outputs.json`` (written by the per-flow models during
        the initial runs) and stacks them. The ``output_data_path`` argument
        is therefore ignored beyond logging.
        """
        if output_data_path:
            logger.info("multiflow: output_processing ignores the combined "
                        f"path {output_data_path!r}; reading per-flow restart "
                        "files instead")
        per_flow = []
        for m in self.models:
            path = os.path.join(m.restart_data_folder, "initial-model-outputs.json")
            result = m.output_processing(output_data_path=path, **kwargs)
            m.model_evaluations = result
            per_flow.append(np.atleast_2d(result))
        rows = {p.shape[0] for p in per_flow}
        if len(rows) != 1:
            raise RuntimeError(
                "multiflow restart: flows have different stored run counts "
                f"{[p.shape for p in per_flow]}")
        self.model_evaluations = np.hstack(per_flow)
        return self.model_evaluations
