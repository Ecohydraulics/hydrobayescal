"""One calibration across several steady flow situations, for any model binding.

Calibrates one shared parameter set against measurements taken at SEVERAL steady discharges: every collocation point is evaluated by running the full-complexity model once per flow (each flow with its own case and its own calibration-points CSV), and the per-flow results are concatenated into one combined observation and output space that the existing surrogate and Bayesian Active Learning machinery consumes unchanged.

Design: **composition, not modification**. :class:`MultiflowModel` wraps one stock single-flow model per flow (each with its own results subfolder ``<res_dir>/flow-<name>``), so all single-flow behavior (parameter updates, solver launch, extraction, output processing, restart files) is exactly the code that single-flow users run. The wrapper only

* concatenates ``observations``, ``variances`` and ``measurement_errors`` along the location axis (order = order of the ``flows`` list),
* runs the flows one after the other per set of collocation points and stacks the per-flow ``model_evaluations`` horizontally ([n_runs x sum(nloc_i * n_quantities)]),
* mirrors the bookkeeping attributes the drivers read.

A binding subclasses it and names its single-flow class (:attr:`MultiflowModel.model_class`), the keys every flow must give (:attr:`MultiflowModel.required_keys`) and the keys whose values must differ between flows (:attr:`MultiflowModel.distinct_keys`). See :class:`hydroBayesCal.telemac.multiflow_telemac.MultiflowTelemacModel` and :class:`hydroBayesCal.openfoam.multiflow_openfoam.MultiflowOpenFOAMModel`.

Requirements on the flows: identical ``calibration_parameters`` and ``param_values`` (one shared parameter vector) and identical calibration and extraction quantities.
"""

import inspect
import os

import numpy as np

from hydroBayesCal.function_pool import *  # noqa: F401,F403 - provides logging/logger
from hydroBayesCal.hysim import HydroSimulations, results_folder_name
from hydroBayesCal.utils.config_logging import setup_file_logging

try:
    logger  # provided by function_pool
except NameError:  # pragma: no cover - fallback if function_pool changes
    import logging
    logger = logging.getLogger(__name__)


def _check_run_matrix(matrix, flow_name, init_runs, is_init_phase):
    """Fail fast on a degenerate per-flow evaluation matrix.

    Guards against the silent extraction or accumulation failures that otherwise only surface after days of compute: a row of all zeros (a run whose outputs never made it into the results, e.g. a JSON that failed to accumulate), or initial-design rows that are identical across every run (the results did not vary with the calibration parameter, so either the parameter is not applied or every run read the same file). Only checked for the initial design (``is_init_phase``); BAL rows are single new runs.
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
            "detailed JSON accumulation and that the result files were found).")
    if n > 1 and np.all(init_rows == init_rows[0]):
        raise RuntimeError(
            f"multiflow flow '{flow_name}': all {n} initial-design runs returned "
            "identical outputs - the calibration parameter is not changing the "
            "results (check that the parameter reaches the model and the per-run "
            "result files).")


def _accepted(method, kwargs):
    """The subset of ``kwargs`` that ``method`` accepts (all of them if it takes ``**kwargs``)."""
    params = inspect.signature(method).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


class MultiflowModel:
    """One shared calibration across several steady flows of one model binding.

    Parameters
    ----------
    flows : list of dict
        One entry per steady flow. ``name`` is a short tag for the per-flow results subfolder and output files; ``calibration_pts_file_path`` is that flow's calibration-points CSV. Every further key is passed to the flow's single-flow model and overrides the shared keyword of the same name (e.g. ``control_file`` for TELEMAC, ``case_template_dir`` for OpenFOAM). Optional: ``dict_output_name``.
    res_dir : str
        Root results directory. The combined artifacts (surrogate, BAL dictionary) land in ``<res_dir>/auto-saved-results-HydroBayesCal`` (or ``..._<calibration_case>``); every flow gets its own stock results tree in ``<res_dir>/flow-<name>``.
    **shared
        Every further keyword accepted by the single-flow model, passed to every flow unchanged.
    """

    #: The single-flow model class of the binding.
    model_class = None
    #: Keys every flow entry must give.
    required_keys = ("calibration_pts_file_path",)
    #: Keys whose values must differ between flows, because per-run files would overwrite each other otherwise.
    distinct_keys = ()

    def __init__(self, flows, res_dir="", **shared):
        if self.model_class is None:
            raise TypeError("MultiflowModel is abstract; use the subclass of a model binding")
        if not flows:
            raise ValueError("multiflow model needs at least one flow spec")
        names = [str(f.get("name", i)) for i, f in enumerate(flows)]
        if len(set(names)) != len(names):
            raise ValueError(f"flow names must be unique, got {names}")
        for name, spec in zip(names, flows):
            missing = [k for k in self.required_keys if not spec.get(k)]
            if missing:
                raise ValueError(f"flow '{name}' lacks {', '.join(missing)}")

        self.flow_specs = list(flows)
        self.flow_names = names
        self.res_dir = res_dir
        per_flow = [self._flow_kwargs(name, spec, res_dir, shared)
                    for name, spec in zip(names, flows)]
        for key in self.distinct_keys:
            values = [kw.get(key) for kw in per_flow]
            if len({str(v) for v in values}) != len(values):
                raise ValueError(
                    f"{key} must be distinct per flow, got {values} "
                    "(per-run result files would overwrite each other)")

        self.models = []
        for name, kwargs in zip(names, per_flow):
            logger.info(f"multiflow: initialising flow '{name}' ({self._describe(kwargs)}, "
                        f"{os.path.basename(str(kwargs['calibration_pts_file_path']))})")
            self.models.append(self.model_class(**kwargs))

        first = self.models[0]
        for m, name in zip(self.models[1:], names[1:]):
            if list(m.calibration_quantities) != list(first.calibration_quantities):
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
        self.parameter_ranges = first.parameter_ranges
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
        self.asr_dir = os.path.join(res_dir, results_folder_name(shared.get("calibration_case")))
        self.calibration_folder = os.path.join(
            self.asr_dir, "calibration-data", "_".join(self.calibration_quantities))
        self.restart_data_folder = os.path.join(self.asr_dir, "restart_data")
        for d in (self.asr_dir, self.calibration_folder, self.restart_data_folder,
                  os.path.join(self.asr_dir, "plots"),
                  os.path.join(self.asr_dir, "surrogate-gpe")):
            os.makedirs(d, exist_ok=True)
        # The per-flow models moved the log files into their own folders; the calibration logs into the combined one.
        setup_file_logging(self.asr_dir)

        logger.info(
            f"multiflow model ready: {len(self.models)} flows "
            f"({', '.join(names)}), {self.nloc} combined calibration points "
            f"({' + '.join(str(n) for n in self.nloc_per_flow)}), "
            f"{self.ndim} calibration parameter(s)")

    #: Per-iteration BAL bookkeeping (``calibration-data/<quantities>/``). The base class method reads only attributes this wrapper mirrors, and the OpenFOAM driver calls it.
    save_calibration_data = HydroSimulations.save_calibration_data

    # ------------------------------------------------------- per-flow setup
    def _flow_kwargs(self, name, spec, res_dir, shared):
        """Keywords of the single-flow model of one flow: the shared ones, overridden by the flow entry."""
        kwargs = dict(shared)
        kwargs.update({k: v for k, v in spec.items() if k != "name"})
        kwargs.setdefault("dict_output_name", f"extraction-data-{name}")
        kwargs["res_dir"] = os.path.join(res_dir, f"flow-{name}")
        return kwargs

    def _describe(self, kwargs):
        """A short label of the flow's case for the log."""
        return str(kwargs.get("control_file", ""))

    @property
    def init_runs(self):
        """Size of the initial design, kept identical across every flow.

        A staged initial design grows ``init_runs`` between blocks (:mod:`~hydroBayesCal.surrogate.initial_design`). Each flow runs the design through its own model, which reads its *own* ``init_runs``, so assigning here has to reach them too; otherwise the flows would keep running the first block while the combined model believes the design has grown.
        """
        return self._init_runs

    @init_runs.setter
    def init_runs(self, value):
        self._init_runs = int(value)
        for model in self.models:
            model.init_runs = int(value)

    # ------------------------------------------------------------------ runs
    def trend_features(self, gp_mean="constant"):
        """Trend features of the emulator mean for the combined output columns, ``None`` for ``"constant"``.

        All flows share the calibration parameters, the targets and the files that define the roles of the parameters (the friction data file, the turbulence model), so the features of a calibration point do not depend on its flow; the roles are read from the first flow.
        """
        from hydroBayesCal.surrogate.hydraulic_mean import trend_features
        hydraulic = str(gp_mean).strip().lower() == "hydraulic"
        first = self.models[0]
        return trend_features(gp_mean, self.calibration_parameters, self.calibration_quantities, self.nloc,
                              parameter_ranges=self.parameter_ranges,
                              friction_laws=first.friction_laws() if hydraulic else None,
                              turbulence_constants=first.turbulence_constants() if hydraulic else None)

    def run_multiple_simulations(self, collocation_points=None, bal_new_set_parameters=None, **kwargs):
        """Run every flow for the given collocation points and stack the outputs.

        Same keywords as the single-flow ``run_multiple_simulations`` of the binding; each flow receives the keywords its own method accepts. Each flow runs the full set through its stock model (so the per-flow trees hold the standard collocation and extraction files), then ``model_evaluations`` becomes the horizontal stack ``[n_runs x sum(nloc_i * n_quantities)]`` in flow order.
        """
        per_flow = []
        for name, m in zip(self.flow_names, self.models):
            logger.info(f"multiflow: === flow '{name}' ===")
            call = dict(kwargs, collocation_points=collocation_points,
                        bal_new_set_parameters=bal_new_set_parameters)
            m.run_multiple_simulations(**_accepted(m.run_multiple_simulations, call))
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
        np.savetxt(os.path.join(self.calibration_folder, "model-results-multiflow.csv"),
                   self.model_evaluations, delimiter=",", fmt="%.8f", header=header)
        return self.model_evaluations

    # ------------------------------------------------------- restart support
    def output_processing(self, output_data_path="", **kwargs):
        """Combine per-flow restart outputs (``only_bal_mode`` restarts).

        The stock driver passes the path of the *combined* model's restart JSON; a multiflow restart instead re-reads every flow's own ``initial-model-outputs.json`` (written by the per-flow models during the initial runs) and stacks them. The ``output_data_path`` argument is therefore ignored beyond logging.
        """
        if output_data_path:
            logger.info("multiflow: output_processing ignores the combined "
                        f"path {output_data_path!r}; reading per-flow restart "
                        "files instead")
        per_flow = []
        for m in self.models:
            path = os.path.join(m.restart_data_folder, "initial-model-outputs.json")
            result = m.output_processing(**_accepted(m.output_processing, dict(kwargs, output_data_path=path)))
            m.model_evaluations = result
            per_flow.append(np.atleast_2d(result))
        rows = {p.shape[0] for p in per_flow}
        if len(rows) != 1:
            raise RuntimeError(
                "multiflow restart: flows have different stored run counts "
                f"{[p.shape for p in per_flow]}")
        self.model_evaluations = np.hstack(per_flow)
        return self.model_evaluations
