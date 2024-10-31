"""
Code that runs full complexity model (i.e., hydrodynamic models) of Telemac multiple times
using experiment design from BayesValidRox.
Possible to couple with any other open source hydrodynamic software (in the future OpenFoam).

Author: Andres Heredia Hidalgo
"""
import sys
import os
import bayesvalidrox as bvr

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# Import own scripts
from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.function_pool import *

def initialize_model(complex_model=None):
    """
    Initializes the full complex model instance.

    Parameters
    ----------
    complex_model : object (Example: Telemacmodel)
        The complex model to initialize. Default is `None`.

    Returns
    -------
    object or None
        Returns the `complex_model` if provided; otherwise, returns `None`.
    """
    return complex_model

def setup_experiment_design(
        complex_model,
        tp_selection_criteria='dkl'
):
    """
    Sets up the experimental design for running the initial simulations of the hydrodynamic model.

    Parameters
    ----------
    complex_model : object
        An instance representing the hydrodynamic model to be used.
    tp_selection_criteria : str, optional
        The criteria for selecting new training points (TP) during the Bayesian Active Learning process.
        Default is 'dkl' (relative entropy).

    Returns
    -------
    exp_design : object
        An instance of the experiment design object configured with the specified model and selection criteria.
    """
    Inputs = bvr.Input()
    # One "Marginal" for each parameter.
    for i in range(complex_model.ndim):
        Inputs.add_marginals()  # Create marginal for parameter "i"
        Inputs.Marginals[i].name = complex_model.calibration_parameters[i]  # Parameter name
        Inputs.Marginals[i].dist_type = 'uniform'  # Parameter distribution (see exp_design.py --> build_dist()
        Inputs.Marginals[i].parameters = complex_model.param_values[i]  # Inputs needed for distribution

    # # Experimental design: ....................................................................
    exp_design = bvr.ExpDesigns(Inputs)
    exp_design.n_init_samples = complex_model.init_runs
    # Sampling methods
    # 1) random 2) latin_hypercube 3) sobol 4) halton 5) hammersley
    # 6) chebyshev(FT) 7) grid(FT) 8)user
    exp_design.sampling_method = complex_model.parameter_sampling_method
    exp_design.n_new_samples = 1
    #exp_design.X=
    exp_design.n_max_samples = complex_model.max_runs
    # 1)'Voronoi' 2)'random' 3)'latin_hypercube' 4)'LOOCV' 5)'dual annealing'
    exp_design.explore_method = 'random'
    exp_design.exploit_method = 'bal'
    exp_design.util_func = tp_selection_criteria
    exp_design.generate_ED(n_samples=exp_design.n_init_samples, max_pce_deg=1)

    return exp_design


def run_complex_model(complex_model,
                      experiment_design
                      ):
    """
    Executes the hydrodynamic model for a given experiment design and returns the collocation points,
    model outputs, observations, and measurement errors.

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
    observations : array
        A 1D array containing the observation values at each measurement location. If there are two calibration
        quantities, the values are stacked horizontally.
    errors : array
        A 1D array containing the measurement errors at each measurement location. If there are two calibration
        quantities, the values are stacked horizontally.
    nloc : int
        The number of calibration locations (i.e., calibration points).

    """
    collocation_points = None
    model_outputs = None

    if not complex_model.only_bal_mode:
        logger.info(
            f"Sampling {complex_model.init_runs} collocation points for the selected calibration parameters with {complex_model.parameter_sampling_method} sampling method.")
        collocation_points = experiment_design.generate_samples(n_samples=experiment_design.n_init_samples,
                                                                sampling_method=experiment_design.sampling_method)

        complex_model.run_multiple_simulations(collocation_points=collocation_points,
                                               complete_bal_mode=complex_model.complete_bal_mode,
                                               validation=complex_model.validation)
        model_outputs = complex_model.model_evaluations
    else:
        try:
            path_np_collocation_points = os.path.join(complex_model.asr_dir, 'collocation-points.csv')
            path_np_model_results = os.path.join(complex_model.asr_dir, 'model-results.csv')
            # Load the collocation points and model results if they exist
            collocation_points = _np.loadtxt(path_np_collocation_points, delimiter=',', skiprows=1)
            model_outputs = _np.loadtxt(path_np_model_results, delimiter=',', skiprows=1)


        except FileNotFoundError:
            logger.info('Saved collocation points or model results as numpy arrays not found. '
                        'Please run initial runs first to execute only Bayesian Active Learning.')
    # Importing observations and erros at calibration points.
    observations = complex_model.observations
    errors = complex_model.measurement_errors

    # number of output locations (i.e., calibration points) / Surrogates to train. (One surrogate per calibration point)
    nloc = complex_model.nloc

    return collocation_points, model_outputs, observations, errors, nloc
if __name__ == "__main__":
    full_complexity_model = initialize_model(
        TelemacModel(
            # TelemacModel specific parameters
            friction_file="friction_ering_MU.tbl",
            tm_xd="1",  # Either 'Telemac2d' or 'Telemac3d', or their corresponding indicator
            gaia_steering_file="",
            results_filename_base="results2m3_mu",
            stdout=6,
            python_shebang="#!/usr/bin/env python3",
            # HydroSimulations class parameters
            control_file="tel_ering_mu_restart.cas",
            model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/",
            res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
            calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurementsWDEPTH_filtered.csv",
            n_cpus=8,
            init_runs=5,
            calibration_parameters=["zone3",
                                    "zone4",
                                    "zone10",
                                    "zone12",
                                    "zone13",
                                    "zone14",
                                    "zone15",
                                    "zone16",
                                    "zone17"],
            param_values=[[0.8, 1.5],
                          [0.005, 0.01],
                          [0.04, 0.1],
                          [0.04, 0.1],
                          [0.04, 0.1],
                          [0.04, 0.1],
                          [0.04, 0.1],
                          [0.04, 0.1],
                          [0.04, 0.1]],

            calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
            dict_output_name="model-outputs-wd",
            parameter_sampling_method="sobol",
            complete_bal_mode=False,
            only_bal_mode=False,
            delete_complex_outputs=True,
        )
    )
    exp_design = setup_experiment_design(complex_model=full_complexity_model,
                                         tp_selection_criteria='dkl'
                                         )
    init_collocation_points, model_evaluations, obs, error_pp, n_loc = run_complex_model(
        complex_model=full_complexity_model,
        experiment_design=exp_design,
    )
