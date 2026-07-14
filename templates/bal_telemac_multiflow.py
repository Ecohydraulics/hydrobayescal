"""Multi-discharge (multi-flow) surrogate-assisted Bayesian calibration for TELEMAC.

Same workflow as ``bal_telemac.py`` (initial DoE -> GP surrogate -> Bayesian
Active Learning), but every collocation point is evaluated at SEVERAL steady
discharges: one TELEMAC run per flow, each with its own steering file and its
own calibration-points CSV. The per-flow observations and model outputs are
concatenated into one combined space, so the surrogate learns
``quantity(location, flow; parameters)`` and the Bayesian inference uses all
flows' measurements simultaneously.

This driver is additive: it reuses ``bal_telemac.py`` (which must sit next to
it) for the experiment design, the initial-runs loop and the BAL loop, and only
swaps the single-flow ``TelemacModel`` for a
:class:`hydroBayesCal.telemac.multiflow_telemac.MultiflowTelemacModel`.
Existing single-flow configs and workflows are untouched.

Config: a standard ``config_Telemac.py`` (paths / hydrodynamic_simulation /
morphodynamic_simulation / calibration / sampling / execution) **plus** a
``multiflow`` block::

    multiflow = {
        'flows': [
            {'name': 'q47-3',
             'control_file': 'steady2d.cas',
             'results_filename_base': 'r2d-q47-3',
             'calibration_pts_file_path': '/path/measurements-q47.3.csv'},
            {'name': 'q168',
             'control_file': 'steady2d-q168.cas',
             'results_filename_base': 'r2d-q168',
             'calibration_pts_file_path': '/path/measurements-q168.csv'},
        ],
    }

``paths['calibration_pts_file_path']`` and
``hydrodynamic_simulation['control_file']`` are ignored in favour of the
per-flow entries; everything else keeps its single-flow meaning.

Run:  python bal_telemac_multiflow.py --config config_Telemac_multiflow.py
"""
import argparse
import os
import sys

# bal_telemac.py lives next to this driver (hydrobayescal/templates/ or a
# staged copy of both files); make it importable regardless of the CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bal_telemac import (  # noqa: E402  (reuses the stock driver logic)
    load_config,
    run_bal_model,
    run_complex_model,
    setup_experiment_design,
)
from hydroBayesCal.function_pool import *  # noqa: E402,F401,F403 - logger
from hydroBayesCal.telemac.multiflow_telemac import MultiflowTelemacModel  # noqa: E402


def build_multiflow_model(config):
    """Instantiate the multiflow model from a config module with a ``multiflow`` block."""
    flows = config.multiflow["flows"]
    if not isinstance(flows, (list, tuple)) or not flows:
        raise ValueError("config.multiflow['flows'] must be a non-empty list")
    return MultiflowTelemacModel(
        flows=flows,
        res_dir=config.paths["res_dir"],
        # Telemac-specific (shared by all flows)
        friction_file=config.hydrodynamic_simulation["friction_file"],
        tm_xd=config.hydrodynamic_simulation["solver_name"],
        gaia_steering_file=config.morphodynamic_simulation["gaia_cas"],
        gaia_results_filename_base=config.morphodynamic_simulation[
            "gaia_results_filename_base"],
        fortran_file=config.hydrodynamic_simulation["fortran_file"],
        # general hydro-simulation settings (shared by all flows)
        model_dir=config.paths["model_dir"],
        n_cpus=config.hydrodynamic_simulation["n_processors"],
        init_runs=config.sampling["init_runs"],
        calibration_parameters=config.calibration["parameters"],
        param_values=config.calibration["param_values"],
        extraction_quantities=config.calibration["extraction_quantities"],
        calibration_quantities=config.calibration["calibration_quantities"],
        user_param_values=config.execution["user_param_values"],
        max_runs=config.sampling["max_runs"],
        complete_bal_mode=config.execution["complete_bal_mode"],
        only_bal_mode=config.execution["only_bal_mode"],
        delete_complex_outputs=config.execution["delete_complex_outputs"],
        validation=config.execution["validation"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-discharge TELEMAC surrogate-assisted Bayesian calibration.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_Telemac_multiflow.py",
        help="Python configuration file with a `multiflow` block "
             "(default: config_Telemac_multiflow.py)",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    multiflow_model = build_multiflow_model(config)

    exp_design = setup_experiment_design(
        complex_model=multiflow_model,
        tp_selection_criteria=config.sampling["tp_selection_criteria"],
        parameter_distribution=config.sampling["parameter_distribution"],
        parameter_sampling_method=config.sampling["parameter_sampling_method"],
    )
    init_collocation_points, model_evaluations = run_complex_model(
        complex_model=multiflow_model,
        experiment_design=exp_design,
    )
    if not (multiflow_model.complete_bal_mode or multiflow_model.only_bal_mode):
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
    )


if __name__ == "__main__":
    main()
