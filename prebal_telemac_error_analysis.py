"""
Code that trains a Gaussian Process Emulator (GPE) for any deterministic numerical model (i.e., hydrodynamic models) of Telemac
Possible to couple with any other open source hydrodynamic software.
Can use normal training (once) or sequential training (BAL)

Author: Andres Heredia Hidalgo MSc
"""
import sys
import os
import time
import argparse
import importlib.util
import bayesvalidrox as bvr


# Import own scripts
from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.surrogate.bal_functions import BayesianInference, SequentialDesign
from src.hydroBayesCal.surrogate.gpe_skl import *
from src.hydroBayesCal.surrogate.gpe_gpytorch import *
from src.hydroBayesCal.function_pool import *



def load_config(config_path):
    """
    Load configuration from Python file.

    Parameters
    ----------
    config_path : str
        Path to the Python configuration file

    Returns
    -------
    module
        Configuration module with all variables
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def setup_experiment_design(
        complex_model,
        tp_selection_criteria='dkl',
        parameter_distribution = 'uniform',
        parameter_sampling_method = 'sobol',

):
    """
    Sets up the experimental design for running the initial simulations of the hydrodynamic model. Samples with BayesValidRox: https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/

    Parameters
    ----------
    complex_model : object
        An instance representing the hydrodynamic model to be used in the experiment.
    tp_selection_criteria : str, optional
        The criteria for selecting new training points (TP) during the Bayesian Active Learning process.
        Default is 'dkl' (relative entropy).
    parameter_distribution: str, optional
        The criteria for selecting the parameter distribution.
        Default: 'uniform' (uniform distribution)
    parameter_sampling: str, optional
        The criteria for selecting the parameter sampling.
        Default: 'sobol'

    Returns
    -------
    exp_design : object
        An instance of the experiment design object configured with the specified model and selection criteria.
    """
    Inputs = bvr.Input()
    # # One "Marginal" for each parameter.
    for i in range(complex_model.ndim):
        Inputs.add_marginals()  # Create marginal for parameter "i"
        Inputs.marginals[i].name = complex_model.calibration_parameters[i]  # Parameter name
        Inputs.marginals[i].dist_type = parameter_distribution  # Parameter distribution (see exp_design.py --> build_dist()
        Inputs.marginals[i].parameters = complex_model.param_values[i]  # Inputs needed for distribution

    # # Experimental design: ....................................................................
    exp_design = bvr.ExpDesigns(Inputs)
    exp_design.n_init_samples = complex_model.init_runs
    # Sampling methods
    # 1) random 2) latin_hypercube 3) sobol 4) halton 5) hammersley
    # 6) chebyshev(FT) 7) grid(FT) 8) User
    exp_design.sampling_method = parameter_sampling_method
    exp_design.n_new_samples = 1
    exp_design.x = complex_model.user_collocation_points
    exp_design.n_max_samples = complex_model.max_runs
    # 1)'Voronoi' 2)'random' 3)'latin_hypercube' 4)'LOOCV' 5)'dual annealing'
    exp_design.explore_method = 'random'
    exp_design.util_func = tp_selection_criteria  # 'bme' 'dkl'
    exp_design.exploit_method = 'bal'
    samples = exp_design.generate_ed()
    return exp_design


def run_complex_model(complex_model,
                      experiment_design
                      ):
    """
    Executes the hydrodynamic model for a given experiment design and returns the collocation points,
    model outputs.

    Parameters
    ----------
    complex_model : obj
        Instance representing the hydrodynamic model to be evaluated.
    experiment_design : obj
        Instance of the experiment design object that specifies the settings for the experimental runs.

    Returns
    -------
    collocation_points : array
        Contains the collocation points (parameter combination sets) with shape [number of runs x number of calibration
        parameters] used for model evaluations.
    model_outputs : array
        Contains the model outputs. The shape of the array depends on the number of quantities:
        - For 1 quantity: [number of runs x number of locations]
        - For 2 quantities: [number of runs x 2 * number of locations]
          (Each pair of columns contains the two quantities for each location.)

    """
    collocation_points = None
    model_outputs = None
    if not complex_model.only_bal_mode:
        logger.info(
            f"Sampling {complex_model.init_runs} collocation points for the selected calibration parameters with {experiment_design.sampling_method} sampling method.")
        collocation_points = experiment_design.x
        complex_model.run_multiple_simulations(collocation_points=collocation_points,
                                               complete_bal_mode=complex_model.complete_bal_mode,
                                               validation=complex_model.validation,
                                               output_extraction_time="mean_last", n=80)
        model_outputs = complex_model.model_evaluations
    else:
        try:
            model_outputs = complex_model.output_processing(output_data_path=os.path.join(complex_model.restart_data_folder,
                                                                                          f'initial-model-outputs.json'),
                                                            delete_slf_files=complex_model.delete_complex_outputs,
                                                            validation=complex_model.validation,
                                                            filter_outputs=True,
                                                            save_extraction_outputs=True,
                                                            run_range_filtering=(1, complex_model.init_runs))
            collocation_points = complex_model.restart_collocation_points

        except FileNotFoundError:
            logger.info('Saved collocation points or model results as numpy arrays not found. '
                        'Please run initial runs first to execute only Bayesian Active Learning.')

    return collocation_points, model_outputs#, observations, errors, nloc

def main():
    parser = argparse.ArgumentParser(description="Run TELEMAC (2D/3D) model surrogate-assisted Bayesian calibration.")
    parser.add_argument(
        '--config',
        type=str,
        default='config_Telemac.py',
        help='Path to Python configuration file (default: config_Telemac.py)'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    full_complexity_model = TelemacModel(
            # Telemac parameters
            friction_file=config.hydrodynamic_simulation['friction_file'],
            tm_xd=config.hydrodynamic_simulation['solver_name'],
            gaia_steering_file=config.morphodynamic_simulation['gaia_cas'],
            gaia_results_filename_base = config.morphodynamic_simulation['gaia_results_filename_base'],
            fortran_file = config.hydrodynamic_simulation['fortran_file'],
            # General hydrosimulation parameters
            results_filename_base=config.hydrodynamic_simulation['results_filename_base'],
            control_file=config.hydrodynamic_simulation['control_file'],
            model_dir=config.paths['model_dir'],
            res_dir=config.paths['res_dir'],
            calibration_pts_file_path=config.paths['calibration_pts_file_path'],
            n_cpus=config.hydrodynamic_simulation['n_processors'],
            init_runs=config.sampling['init_runs'],
            calibration_parameters=config.calibration['parameters'],
            param_values=config.calibration['param_values'],
            extraction_quantities=config.calibration['extraction_quantities'],
            calibration_quantities=config.calibration['calibration_quantities'],
            dict_output_name=config.calibration['dict_output_name'],
            user_param_values=config.execution['user_param_values'],
            max_runs=config.sampling['max_runs'],
            complete_bal_mode=config.execution['complete_bal_mode'],
            only_bal_mode=config.execution['only_bal_mode'],
            delete_complex_outputs=config.execution['delete_complex_outputs'],
            validation=config.execution['validation']
    )

    # Setup and run the experiment
    exp_design = setup_experiment_design(
        complex_model=full_complexity_model,
        tp_selection_criteria=config.sampling['tp_selection_criteria'],
        parameter_distribution=config.sampling['parameter_distribution'],
        parameter_sampling_method=config.sampling['parameter_sampling_method']
    )
    init_collocation_points, model_evaluations= run_complex_model(
        complex_model=full_complexity_model,
        experiment_design=exp_design,
    )
    print(init_collocation_points.shape)
    print(model_evaluations.shape)
    print(full_complexity_model.calibration_quantities)
    print(full_complexity_model.observations)
if __name__ == "__main__":
    main()

