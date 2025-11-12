"""
Code that trains a Gaussian Process Emulator (GPE) for any full complexity model (i.e., hydrodynamic models) of Telemac
Possible to couple with any other open source hydrodynamic software.
Can use normal training (once) or sequential training (BAL, SF, Sobol)

Author: Andres Heredia Hidalgo MSc
"""
import pdb
import sys
import os
import time
import argparse
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
from src.hydroBayesCal.bayesvalidrox.metamodel.bal_functions import BayesianInference, SequentialDesign
from src.hydroBayesCal.bayesvalidrox.metamodel.gpe_skl import *
from src.hydroBayesCal.bayesvalidrox.metamodel.gpe_gpytorch import *
from src.hydroBayesCal.function_pool import *

def initialize_model(complex_model=None):
    return complex_model

def setup_experiment_design(
        complex_model,
        tp_selection_criteria='dkl',
        parameter_distribution = 'uniform',
        parameter_sampling_method = 'sobol'

):
    """
    Sets up the experimental design for running the initial simulations of the hydrodynamic model.

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
        Inputs.Marginals[i].name = complex_model.calibration_parameters[i]  # Parameter name
        Inputs.Marginals[i].dist_type = parameter_distribution  # Parameter distribution (see exp_design.py --> build_dist()
        Inputs.Marginals[i].parameters = complex_model.param_values[i]  # Inputs needed for distribution

    # # Experimental design: ....................................................................
    exp_design = bvr.ExpDesigns(Inputs)
    exp_design.n_init_samples = complex_model.init_runs
    # Sampling methods
    # 1) random 2) latin_hypercube 3) sobol 4) halton 5) hammersley
    # 6) chebyshev(FT) 7) grid(FT) 8) User
    exp_design.sampling_method = parameter_sampling_method
    exp_design.n_new_samples = 1
    exp_design.X = complex_model.user_collocation_points
    exp_design.n_max_samples = complex_model.max_runs
    # 1)'Voronoi' 2)'random' 3)'latin_hypercube' 4)'LOOCV' 5)'dual annealing'
    exp_design.explore_method = 'random'
    exp_design.exploit_method = 'bal'
    exp_design.util_func = tp_selection_criteria
    exp_design.generate_ED(n_samples=exp_design.n_init_samples)
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
        collocation_points = experiment_design.X
        complex_model.run_multiple_simulations(collocation_points=collocation_points,
                                               complete_bal_mode=complex_model.complete_bal_mode,
                                               validation=complex_model.validation)
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


#@log_actions
def run_bal_model(collocation_points,
                  model_outputs,
                  complex_model,
                  experiment_design,
                  eval_steps=1,  # By default
                  prior_samples=10000,  # By default
                  mc_samples_al=5000,  # By default
                  mc_exploration=1000,  # By default
                  gp_library="gpy",  # By default
                  ):
    """
    Executes the Bayesian Active Learning (BAL) model to select new training points and evaluate the hydrodynamic model.

    Parameters
    ----------
    collocation_points : array
        An array containing the collocation points used for model evaluations, with shape [number of runs x number of
        calibration parameters].
    model_outputs : array
        Contains the outputs from the hydrodynamic model, with shape dependent on the number of quantities
        and locations.
    complex_model : obj
        An instance representing the hydrodynamic model to be evaluated.
    experiment_design : obj
        Contains the experiment design object specifying the settings for the experimental runs.
    eval_steps : int, optional
        Every ow many iterations the surrogate model is evaluated and saved in surrogate model folder.
        Default is 1. Every BAL iteration the surrogate model will be evaluated.
    prior_samples : int, optional
        The number of samples drawn from the prior distribution.
        Default is 10,000.
    mc_samples_al : int, optional
        The number of Monte Carlo samples used for the Bayesian inference process.
        Default is 5,000.
    mc_exploration : int, optional
        The number of samples used for exploring the parameter space during the Bayesian Active Learning process.
        Default is 1,000.
    gp_library : str, optional
        The Gaussian Process library to be used for modeling. Options may include "gpy" or "skl".
        Default is "gpy" for GPyTorch or "skl" for SciKitLearn.

    Returns
    -------
    None
        BAL_dictionary: Dictionary and .pkl file containing the data from Bayesian Active Learning
        updated_collocation_points: array and .csv file containing all the collocation points (Initial + BAL-added)
        model-outputs: Files .csv and .jason containing all model output obtained from the collocation points and required model variables.

        *These files are saved in the user-defined results directory res_dir as auto-saved-results-HydroBayesCal

    """

    #Prior sampling
    prior = experiment_design.generate_samples(prior_samples)
    prior_logpdf = np.log(experiment_design.JDist.pdf(prior.T)).reshape(-1)
    # Number of BAL (Bayesian Active Learning iterations)
    n_iter = experiment_design.n_max_samples - experiment_design.n_init_samples
    # Number of evaluations:
    if eval_steps == 1 or experiment_design.exploit_method == 'sobol':
        n_evals = n_iter + 1
    else:
        n_evals = math.ceil(n_iter / eval_steps) + 1
    new_tp = None
    sm = None
    multi_sm = None
    # INITIALIZATION GPE AND RESULTS FOLDERS
    # Creates folder for specific case ....................................................................
    logger.info(f"<<< Will run ({n_iter + 1}) GP training iterations and ({n_evals}) GP evaluations. >>> ")

    # Auto-saved-results folder
    gpe_results_folder = os.path.join(complex_model.asr_dir, 'surrogate-gpe')
    if not os.path.exists(gpe_results_folder):
        logger.info(f'Creating folder {gpe_results_folder}')
        os.makedirs(gpe_results_folder, exist_ok=True)

    #     # Create a folder for the exploration method: needed only if BAL is to be used
    gpe_results_folder_bal = os.path.join(gpe_results_folder,
                                          f'{experiment_design.exploit_method}_{experiment_design.util_func}')
    if not os.path.exists(gpe_results_folder_bal):
        logger.info(f'Creating folder {gpe_results_folder_bal}')
        os.makedirs(gpe_results_folder_bal, exist_ok=True)
    #
    #     # Arrays to save results ---------------------------------------------------------------------------- #
    #
    # Arrays to save results ---------------------------------------------------------------------------- #
    bayesian_dict = {'N_tp': np.zeros(n_iter + 1), 'BME': np.zeros(n_iter + 1), 'ELPD': np.zeros(n_iter + 1),
                     'RE': np.zeros(n_iter + 1), 'IE': np.zeros(n_iter + 1), 'post_size': np.zeros(n_iter + 1),
                     'posterior': [None] * (n_iter + 1),
                     f'{experiment_design.exploit_method}_{experiment_design.util_func}': np.zeros(n_iter),
                     'util_func': np.empty(n_iter, dtype=object), 'prior': prior, 'observations': complex_model.observations,
                     'errors': complex_model.measurement_errors}
    # SURROGATE MODEL
    # Train the GPE a maximum of "iteration_limit" times
    for it in range(0, n_iter + 1):

        # 1. Train surrogate
        if gp_library == 'skl':
            # 1.1. Set up the kernel
            # Setting up initial length scales and length scales bounds for Radial Basis Function (RBF) kernel
            # Length scale: Assumed to be the midpoint of the minimum and maximum range for each of the calibration parameters.
            # Length scale bounds: Assumed to be the range for each calibration parameter.
            if it == 0:
                length_scales = []
                length_scales_bounds = []

                # Calculate length scales and bounds
                for param_range in complex_model.parameter_ranges:
                    # Ensure the calculated length scale is positive
                    length_scale = max(sum(param_range) / len(param_range),
                                       1e-5)  # Prevent negative or zero length scales
                    length_scales.append(length_scale)

                    # Ensure bounds are positive and finite
                    lower_bound, upper_bound = max(param_range[0], 1e-5), max(param_range[1], 1e-5)
                    length_scales_bounds.append((lower_bound, upper_bound))

            kernel = 1 * RBF(length_scale=length_scales, length_scale_bounds=length_scales_bounds)

            # 1.2. Setup a GPR: initialize the general SKL class
            sm = SklTraining(collocation_points=collocation_points, model_evaluations=model_outputs,
                             noise=True,
                             kernel=kernel,
                             alpha=1e-6,
                             n_restarts=10,
                             parallelize=False)

        elif gp_library == 'gpy':
            # 1.1. Set up the kernel

            # 1.2. Set up Likelihood
            if complex_model.num_calibration_quantities == 1:
                kernel = gpytorch.kernels.ScaleKernel(
                                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=complex_model.ndim)
                                        )
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
                # Modify default kernel/likelihood values:
                likelihood.noise = 1e-5  # Initialize the noise with a very small value.
                # 1.3. Train a GPE, which consists of a gpe for each location being evaluated
                sm = GPyTraining(collocation_points=collocation_points, model_evaluations=model_outputs,
                                 likelihood=likelihood, kernel=kernel,
                                 training_iter=150,
                                 optimizer="adam", lr=0.07,
                                 verbose=False)
            else:
                kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=complex_model.ndim))
                # *
                #     gpytorch.kernels.RBFKernel(ard_num_dims=complex_model.ndim)
                # )

                if complex_model.multitask_selection == "variables":
                    multi_likelihood_var = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                        num_tasks=complex_model.num_calibration_quantities,
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)  # Allow smaller noise
                    )
                    multi_likelihood_var.noise = 1e-5
                    multi_sm_var = MultiGPyTraining(collocation_points,
                                                    model_outputs,
                                                    kernel,
                                                    training_iter=150,
                                                    likelihood=multi_likelihood_var,
                                                    optimizer="adam", lr=0.07,
                                                    number_quantities=complex_model.num_calibration_quantities,
                                                    )
                if complex_model.multitask_selection == "locations":

                    multi_likelihood_loc = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                        num_tasks=complex_model.nloc,
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-3)  # Allow smaller noise
                    )

                    multi_sm_loc = MultiGPyTraining(collocation_points,
                                                model_outputs,
                                                kernel,
                                                training_iter=150,
                                                likelihood=multi_likelihood_loc,
                                                optimizer="adam", lr=0.01, number_quantities=complex_model.num_calibration_quantities,
                                                )
                if complex_model.multitask_selection == "all":
                    multi_likelihood_all = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                        num_tasks=model_outputs.shape[1],
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-3)  # Allow smaller noise
                    )
                    multi_sm_all = MultiGPyTraining(collocation_points,
                                                    model_outputs,
                                                    kernel,
                                                    training_iter=150,
                                                    likelihood=multi_likelihood_all,
                                                    optimizer="adam", lr=0.01,
                                                    number_quantities=complex_model.num_calibration_quantities,
                                                    )

        # Trains the GPR
        if it == n_iter:
            if n_iter < 0:
                logger.info(
                    f'------------ Number of initial runs init_runs and n_tp_max are the same. Conditions: n_tp_max > init_runs.     -------------------')
                # exit()
            else:
                logger.info(f'------------ Training final surrogate model    -------------------')
        elif it > 0:
            logger.info(f'------------ Training model with new training point: {new_tp}   -------------------')
        elif it == 0:
            logger.info(
                'Starting surrogate model training with the initial collocation points. Please check the .csv file if information required.')
            #logger.info(collocation_points)

        if complex_model.num_calibration_quantities == 1:
            sm.train_()
            surrogate_object = sm
        else:
            if complex_model.multitask_selection == "variables":
                multi_sm_var.train_tasks_variables()
                surrogate_object = multi_sm_var
            if complex_model.multitask_selection == "locations":
                multi_sm_loc.train_tasks_locations()
                surrogate_object = multi_sm_loc
            if complex_model.multitask_selection == "all":
                multi_sm_all.train_tasks_all()
                surrogate_object = multi_sm_all


        # 2. Validate GPR
        if it % eval_steps == 0:
            if complex_model.num_calibration_quantities == 1:
                # Construct the save_name path for single quantity
                save_name = os.path.join(gpe_results_folder_bal,
                                         f'gpr_{gp_library}_TP{collocation_points.shape[0]:02d}_'
                                         f'{experiment_design.exploit_method}_quantities_{complex_model.calibration_quantities}.pkl')
                sm.exp_design = experiment_design
                with open(save_name, "wb") as file:
                    pickle.dump(sm, file)
            else:
                # Construct the save_name path for multiple quantities
                save_name = os.path.join(gpe_results_folder_bal,
                                         f'gpr_{gp_library}_TP{collocation_points.shape[0]:02d}_'
                                         f'{experiment_design.exploit_method}_quantities_{complex_model.calibration_quantities}_{complex_model.multitask_selection}.pkl')
                surrogate_object.exp_design = experiment_design
                with open(save_name, "wb") as file:
                    pickle.dump(surrogate_object, file)

        # 3. Compute Bayesian scores in parameter space ----------------------------------------------------------
        # Surrogate outputs for prior samples
        if complex_model.num_calibration_quantities == 1:
            multitask = False
            start_time_prediction = time.time()
            surrogate_output = sm.predict_(input_sets=prior,
                                           get_conf_int=True)
            end_time_prediction = time.time()
            print(f"Surrogate model predictions took {end_time_prediction - start_time_prediction:.2f} seconds.")
            model_predictions = surrogate_output['output']
            total_error = complex_model.variances
            if it == 0 or it == n_iter:
                try:
                    # Open the file and save the dictionary
                    with open(os.path.join(complex_model.asr_dir, f'surrogate_output_iter_{it}.pkl'),
                              'wb') as pickle_file:
                        pickle.dump(surrogate_output, pickle_file)
                    print(f"Surrogate output data for iteration {it} successfully saved.")
                except Exception as e:
                    print(f"An error occurred while saving the dictionary: {e}")

        else:
            multitask=True
            start_time_prediction = time.time()
            surrogate_output = surrogate_object.predict_(input_sets=prior,get_conf_int=True)
            end_time_prediction = time.time()
            print(f"Surrogate model predictions took {end_time_prediction - start_time_prediction:.2f} seconds.")
            total_error = complex_model.variances
            model_predictions = surrogate_output['output']
            if it == 0 or it == n_iter:
                try:
                    # Open the file and save the dictionary
                    with open(os.path.join(complex_model.asr_dir, f'surrogate_output_iter_{it}.pkl'),
                              'wb') as pickle_file:
                        pickle.dump(surrogate_output, pickle_file)
                    print(f"Surrogate output data for iteration {it} successfully saved..")
                except Exception as e:
                    print(f"An error occurred while saving the dictionary: {e}")

        bi_gpe = BayesianInference(model_predictions=model_predictions,
                                   observations=complex_model.observations,
                                   error=total_error,
                                   sampling_method='rejection_sampling',
                                   prior=prior,
                                   prior_log_pdf=prior_logpdf)
        bi_gpe.estimate_bme()
        bayesian_dict['N_tp'][it] = collocation_points.shape[0]
        bayesian_dict['BME'][it], bayesian_dict['RE'][it] = bi_gpe.BME, bi_gpe.RE
        bayesian_dict['ELPD'][it], bayesian_dict['IE'][it] = bi_gpe.ELPD, bi_gpe.IE
        bayesian_dict['post_size'][it] = bi_gpe.posterior_output.shape[0]
        bayesian_dict['posterior'][it] = bi_gpe.posterior

        # 4. Sequential Design --------------------------------------------------------------------------------------
        if it < n_iter:
            logger.info(
                f'Selecting {experiment_design.n_new_samples} additional TP using {experiment_design.exploit_method}')

            # gaussian_assumption = True (Assumes Analytical Function Bayesian Active Learning )
            # gaussian assumption = False (General Bayesian Active Learning)

            SD = SequentialDesign(exp_design=experiment_design,
                                  sm_object=surrogate_object,
                                  obs=complex_model.observations,
                                  errors=total_error,
                                  do_tradeoff=False,
                                  gaussian_assumption=False,
                                  mc_samples=mc_samples_al,
                                  mc_exploration=mc_exploration,
                                  multitask=multitask)  # multiprocessing=parallelize

            new_tp, util_fun = SD.run_sequential_design(prior_samples=prior)
            logger.info(f"The new collocation point after rejection sampling is {new_tp} obtained with {util_fun}")
            bayesian_dict['util_func'][it] = util_fun

            # Evaluate model in new TP

            if complex_model.complete_bal_mode or complex_model.only_bal_mode:
                bal_iteration = it + 1
                complex_model.run_multiple_simulations(collocation_points=collocation_points,
                                                       bal_iteration=bal_iteration,
                                                       bal_new_set_parameters=new_tp,
                                                       complete_bal_mode=complex_model.complete_bal_mode,
                                                       validation=complex_model.validation)

                model_outputs = complex_model.model_evaluations

            # -------------------------------------------------------
            # Update collocation points:
            if experiment_design.exploit_method == 'sobol':
                collocation_points = new_tp
            else:
                collocation_points = np.vstack((collocation_points, new_tp))
                logger.info(f'------------ Finished iteration {it + 1}/{n_iter} -------------------')
        try:
            with open(os.path.join(complex_model.calibration_folder, 'BAL_dictionary.pkl'), 'wb') as pickle_file:
                pickle.dump(bayesian_dict, pickle_file)
            print("BAL data successfully saved.")
        except Exception as e:
            print(f"An error occurred while saving the dictionary: {e}")
    updated_collocation_points = collocation_points
    return bayesian_dict, updated_collocation_points

# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Run Telemac Model with calibration parameters.")
#     parser.add_argument(
#         '--calibration_quantities',
#         type=str,
#         nargs='+',  # Accept multiple arguments as a list
#         required=True,
#         help='Calibration quantities as a list of strings, e.g., "WATER DEPTH" "SCALAR VELOCITY".'
#     )
#     parser.add_argument(
#         '--only_bal_mode',
#         type=str,
#         default=False,  # Default value is False
#         help='Set to True if only Bayesian Active Learning mode is to be used.'
#     )
#     parser.add_argument(
#         '--complete_bal_mode',
#         type=str,
#         default=True,  # Default value is True
#         help='Set to False if only initial runs are required.'
#     )
#
#     args = parser.parse_args()
#
#     # Extract arguments
#     calibration_quantities = args.calibration_quantities
#     only_bal_mode = args.only_bal_mode
#     complete_bal_mode = args.complete_bal_mode
#
#     print(f"Calibration Quantities: {calibration_quantities}")
#     # print(f"Only BAL Mode: {only_bal_mode}")
#     # print(f"Complete BAL Mode: {complete_bal_mode}")

    #Initialize the model with arguments
    # full_complexity_model = initialize_model(
    #     TelemacModel(
    #         # Telemac parameters
    #         friction_file="friction_ering_MU.tbl",
    #         tm_xd="1",
    #         gaia_steering_file="",
    #         # General hydrosimulation parameters
    #         results_filename_base="results2m3",
    #         control_file="tel_ering_mu_restart.cas",
    #         model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/",
    #         res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
    #         calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
    #         n_cpus=8,
    #         init_runs=30,
    #         calibration_parameters=["zone11", "zone12", "zone13", "zone14", "zone15"],
    #         param_values = [[0.010, 0.1], [0.050, 0.79], [0.0020, 0.1], [0.002, 0.1], [0.050, 0.79]], # coarse-coarse -fine -fine -coarse
    #         extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V"],
    #         calibration_quantities=calibration_quantities,
    #         dict_output_name="extraction-data",
    #         user_param_values=False,
    #         max_runs=170,
    #         complete_bal_mode=complete_bal_mode,
    #         only_bal_mode=only_bal_mode,
    #         delete_complex_outputs=True,
    #         validation=False
    #     )
    # )
    #
    # # Setup and run the experiment
    # exp_design = setup_experiment_design(
    #     complex_model=full_complexity_model,
    #     tp_selection_criteria='dkl',
    #     parameter_distribution='uniform',
    #     parameter_sampling_method='sobol'
    # )
    # init_collocation_points, model_evaluations = run_complex_model(
    #     complex_model=full_complexity_model,
    #     experiment_design=exp_design,
    # )
    # run_bal_model(
    #     collocation_points=init_collocation_points,
    #     model_outputs=model_evaluations,
    #     complex_model=full_complexity_model,
    #     experiment_design=exp_design,
    #     eval_steps=20,
    #     prior_samples=20000,
    #     mc_samples_al=3000,
    #     mc_exploration=1000,
    #     gp_library="gpy"
    # )


if __name__ == "__main__":
    #main()
    full_complexity_model = initialize_model(
        TelemacModel(
            # Telemac parameters
            friction_file="friction_ering_MU_initial_NIKU.tbl",
            tm_xd="1",
            gaia_steering_file="gaia_ering_initial_NIKU.cas",
            gaia_results_filename_base = "resultsGAIA",
            # General hydrosimulation parameters
            results_filename_base="results2m3",
            control_file="tel_ering_initial_NIKU.cas",
            model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation-folder-telemac-gaia",
            res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
            calibration_pts_file_path = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
            n_cpus=16,
            init_runs=30,
            calibration_parameters=["gaiaCLASSES SHIELDS PARAMETERS 1",
                                    "gaiaCLASSES SHIELDS PARAMETERS 2",
                                    # "gaiaCLASSES SHIELDS PARAMETERS 3",
                                    # "zone0",
                                    # "zone1",
                                    "zone2",
                                    # "zone3",
                                    "zone4",
                                    "zone5",
                                    "zone6",
                                    # "zone7",
                                    "zone8",
                                    "zone9",
                                    #"zone10",
                                    # "zone11",
                                    #"zone12",
                                    "zone13"],
            param_values = [[0.047,0.070], # critical shields parameter class 1
                            [0.047, 0.070], # critical shields parameter class 2
                            # [0.047, 0.070], # critical shields parameter class 3
                            [0.008, 0.4], # zone2 Pool
                            # [0.008, 0.6], # zone3 Slackwater
                            [0.002, 0.4], # zone4 Glide
                            [0.002, 0.4], # zone5 Riffle
                            [0.030, 0.4], # zone6 Run
                            [0.002, 0.4], # zone8 Backwater
                            [0.030, 0.4], # zone9 Wake
                            [0.040, 1.8]], # zone 13 LW
            extraction_quantities = ["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V","CUMUL BED EVOL"],

            calibration_quantities=["WATER DEPTH","SCALAR VELOCITY","CUMUL BED EVOL"],
            # calibration_quantities=["SCALAR VELOCITY","WATER DEPTH","CUMUL BED EVOL"],
            # calibration_quantities=["CUMUL BED EVOL"],
            # calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
            # calibration_quantities=["WATER DEPTH"],
            # calibration_quantities=["WATER DEPTH"],
            dict_output_name="extraction-data",
            user_param_values = False,
            max_runs=100,
            complete_bal_mode=True,
            only_bal_mode=False,
            delete_complex_outputs=True,
            validation=False
        )
    )

    # Setup and run the experiment
    exp_design = setup_experiment_design(
        complex_model=full_complexity_model,
        tp_selection_criteria='dkl',
        parameter_distribution='uniform',
        parameter_sampling_method = 'sobol'
    )
    init_collocation_points, model_evaluations= run_complex_model(
        complex_model=full_complexity_model,
        experiment_design=exp_design,
    )
    run_bal_model(
        collocation_points=init_collocation_points,
        model_outputs=model_evaluations,
        complex_model=full_complexity_model,
        experiment_design=exp_design,
        eval_steps=5,
        prior_samples=22000,
        mc_samples_al=2000,
        mc_exploration=1000,
        gp_library="gpy"
    )

    # # TODO: Why is this in a __main__ namespace? This should be refactored into functions and the function call - Refactored into functions
    # # TODO  sequence should self-explain the workflow. - Done