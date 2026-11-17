"""Multi-discharge (multi-flow) surrogate-assisted Bayesian calibration for OpenFOAM.

Same workflow as ``bal_openfoam.py`` (initial design -> GP surrogate -> Bayesian Active Learning), but every collocation point is evaluated at SEVERAL steady discharges: one OpenFOAM run per flow, each from its own case template and with its own calibration-points CSV. The per-flow observations and model outputs are concatenated into one combined space, so the surrogate learns ``quantity(location, flow; parameters)`` and the Bayesian inference uses the measurements of all flows at once.

This driver is additive: it reuses ``bal_openfoam.py`` (which must sit next to it) for the experiment design, the initial runs and the BAL loop, and only swaps the single-flow ``OpenFOAMModel`` for a :class:`hydroBayesCal.openfoam.multiflow_openfoam.MultiflowOpenFOAMModel`.

Config: a standard ``config_OpenFOAM.py`` (paths / simulation / interfoam / calibration / sampling / execution) **plus** a ``multiflow`` block::

    multiflow = {
        'flows': [
            {'name': 'august-2026',
             'case_template_dir': '/path/template-august-2026',
             'calibration_pts_file_path': '/path/measurements-august-2026.csv'},
            {'name': 'march-2026',
             'case_template_dir': '/path/template-march-2026',
             'calibration_pts_file_path': '/path/measurements-march-2026.csv'},
        ],
    }

``paths['case_template_dir']`` and ``paths['calibration_pts_file_path']`` are ignored in favor of the per-flow entries; each flow runs in ``<paths['model_dir']>/flow-<name>`` unless its entry gives a ``model_dir``. Everything else keeps its single-flow meaning.

Run:  python bal_openfoam_multiflow.py --config config_OpenFOAM_multiflow.py
"""
import argparse
import os
import sys

# bal_openfoam.py lives next to this driver (the packaged drivers folder or a staged copy of both files); make it importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bal_openfoam import (  # noqa: E402  (reuses the stock driver logic)
    load_config,
    run_bal_model,
    run_complex_model,
    setup_experiment_design,
)
from hydroBayesCal.function_pool import *  # noqa: E402,F401,F403 - logger
from hydroBayesCal.surrogate.hydraulic_mean import check_gp_mean  # noqa: E402
from hydroBayesCal.openfoam.multiflow_openfoam import MultiflowOpenFOAMModel  # noqa: E402


def build_multiflow_model(config, calibration_quantities=None, only_bal_mode=None,
                          complete_bal_mode=None):
    """Instantiate the multiflow model from a config module with a ``multiflow`` block."""
    flows = config.multiflow["flows"]
    if not isinstance(flows, (list, tuple)) or not flows:
        raise ValueError("config.multiflow['flows'] must be a non-empty list")
    return MultiflowOpenFOAMModel(
        flows=flows,
        res_dir=config.paths["res_dir"],
        # OpenFOAM-specific (shared by all flows)
        solver_name=config.simulation["solver_name"],
        n_processors=config.simulation["n_processors"],
        results_filename_base=config.simulation["results_filename_base"],
        alpha_water_name=config.interfoam["alpha_water_name"],
        water_surface_alpha=config.interfoam["water_surface_alpha"],
        reference_z=config.interfoam["reference_z"],
        control_file=config.simulation["control_file"],
        n_avg_timesteps=config.simulation.get("n_avg_timesteps", 1),
        # general settings (shared by all flows)
        model_dir=config.paths["model_dir"],
        n_cpus=config.sampling["n_cpus"],
        init_runs=config.sampling["init_runs"],
        calibration_parameters=config.calibration["parameters"],
        param_values=config.calibration["param_values"],
        extraction_quantities=config.calibration["extraction_quantities"],
        calibration_quantities=calibration_quantities or config.calibration["calibration_quantities"],
        gpe_error=config.calibration.get("gpe_error", 0.0),
        measurement_error=config.calibration.get("measurement_error", 0.10),
        model_structural_error=config.calibration.get("model_structural_error", 0.0),
        user_param_values=config.execution.get("user_param_values", False),
        max_runs=config.sampling["max_runs"],
        complete_bal_mode=(config.execution["complete_bal_mode"]
                           if complete_bal_mode is None else complete_bal_mode),
        only_bal_mode=(config.execution["only_bal_mode"]
                       if only_bal_mode is None else only_bal_mode),
        delete_complex_outputs=config.execution["delete_complex_outputs"],
        validation=config.execution["validation"],
        multitask_selection=config.sampling.get("multitask_selection", "variables"),
        calibration_case=config.calibration.get("calibration_case", None),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-discharge OpenFOAM surrogate-assisted Bayesian calibration.")
    parser.add_argument("--config", type=str, default="config_OpenFOAM_multiflow.py",
                        help="Python configuration file with a `multiflow` block "
                             "(default: config_OpenFOAM_multiflow.py)")
    parser.add_argument("--only_bal_mode", type=str,
                        help="Override only_bal_mode from config (True/False).")
    parser.add_argument("--complete_bal_mode", type=str,
                        help="Override complete_bal_mode from config (True/False).")
    parser.add_argument("--only_init_mode", type=str, default="False",
                        help="Run only the initial simulations, then stop without BAL (True/False).")
    args = parser.parse_args()
    config = load_config(args.config)
    # refused before any run, as in bal_openfoam.py
    check_gp_mean(config.sampling.get("gp_mean", "constant"))

    def flag(value):
        return None if value is None else value.lower() == "true"

    multiflow_model = build_multiflow_model(
        config, only_bal_mode=flag(args.only_bal_mode),
        complete_bal_mode=flag(args.complete_bal_mode))

    exp_design = setup_experiment_design(
        complex_model=multiflow_model,
        tp_selection_criteria=config.sampling["tp_selection_criteria"],
        parameter_distribution=config.sampling["parameter_distribution"],
        parameter_sampling_method=config.sampling["parameter_sampling_method"],
    )
    init_collocation_points, model_evaluations = run_complex_model(
        complex_model=multiflow_model,
        experiment_design=exp_design,
        adaptive_init_runs=config.sampling.get("adaptive_init_runs", True),
        init_runs_min=config.sampling.get("init_runs_min", None),
    )
    if args.only_init_mode.lower() == "true" or not (
            multiflow_model.complete_bal_mode or multiflow_model.only_bal_mode):
        logger.info("Initial multiflow runs finished (only-init mode): "
                    "skipping surrogate training and BAL.")
        return
    run_bal_model(
        collocation_points=init_collocation_points,
        model_outputs=model_evaluations,
        complex_model=multiflow_model,
        experiment_design=exp_design,
        eval_steps=config.sampling["eval_steps"],
        prior_samples=config.sampling["prior_samples"],
        mc_samples_al=config.sampling["mc_samples_al"],
        mc_exploration=config.sampling["mc_exploration"],
        gp_library=config.sampling["gp_library"],
        include_surrogate_error=config.sampling.get("include_surrogate_error", True),
        bal_exploration_tradeoff=config.sampling.get("bal_exploration_tradeoff", "auto"),
        bal_n_jobs=config.sampling.get("bal_n_jobs", -1),
        tp_normalization=config.sampling.get("tp_normalization", False),
        gp_mean=config.sampling.get("gp_mean", "constant"),
    )


if __name__ == "__main__":
    main()
