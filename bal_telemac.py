"""
Code that trains a Gaussian Process Emulator (GPE) for any full complexity model (i.e., hydrodynamic models) of Telemac
Possible to couple with any other open source hydrodynamic software.
Can use normal training (once) or sequential training (BAL, SF, Sobol)

Author: Andres Heredia Hidalgo MSc
"""
import sys
import os
import pdb
import pickle
import bayesvalidrox as bvr

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# from src.hydroBayesCal.bayesvalidrox.surrogate_models.inputs import Input
# from src.hydroBayesCal.bayesvalidrox.surrogate_models.exp_designs import ExpDesigns
from src.hydroBayesCal.function_pool import *
from src.hydroBayesCal.plots.plots import BayesianPlotter


# TODO: handle the following line differently. this template should be able
# TODO  to access this line - all parameters assigned in TelemacModel instance
# TODO: the following import has no target - done importing BayesianPlotter


# TODO: Tidy up - here are many things that should go into a config file

def initialize_model(complex_model=None):
    return complex_model


def setup_experiment_design(
        complex_model,
        tp_selection_criteria='dkl'
):
    """

    Parameters
    ----------
    complex_model
    tp_selection_criteria

    Returns
    -------

    """
    Inputs = bvr.Input()
    # # One "Marginal" for each parameter.
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
    collocation_points = None
    model_outputs = None

    if not complex_model.only_bal_mode:
        logger.info(
            f"Sampling {complex_model.init_runs} collocation points for the selected calibration parameters with {complex_model.parameter_sampling_method} sampling method.")
        collocation_points = experiment_design.generate_samples(n_samples=experiment_design.n_init_samples,
                                                                sampling_method=experiment_design.sampling_method)
        # # bal_mode = True : Activates Bayesian Active Learning after finishing the initial runs of the full complexity model
        # # bal_mode = False : Only runs the full complexity model the number of times indicated in init_runs
        complex_model.run_multiple_simulations(collocation_points=collocation_points,
                                          # bal_iteration=0,
                                          # bal_new_set_parameters=None,
                                          complete_bal_mode=complex_model.complete_bal_mode)
        model_outputs = complex_model.model_evaluations
    else:
        try:
            path_np_collocation_points = os.path.join(complex_model.asr_dir, 'collocation_points.csv')
            path_np_model_results = os.path.join(complex_model.asr_dir, 'model_results.csv')
            # Load the collocation points and model results if they exist
            collocation_points = np.loadtxt(path_np_collocation_points, delimiter=',',skiprows=1)
            model_outputs = np.loadtxt(path_np_model_results, delimiter=',',skiprows=1)

        except FileNotFoundError:
            logger.info('Saved collocation points or model results as numpy arrays not found. '
                        'Please run initial runs first to execute only Bayesian Active Learning.')
    # Importing measured values of the calibration quantities at the calibration points as a 2D numpy array shape [No.quantities , No.calibration points].
    # Importing errors at calibration points as a numpy array shape [No. calibration points,No.calibration quantities]
    observations = complex_model.observations
    errors = complex_model.measurement_errors

    # number of output locations (i.e., calibration points) / Surrogates to train. (One surrogate per calibration point)
    nloc = complex_model.nloc

    return collocation_points, model_outputs, observations, errors, nloc


#@log_actions
def run_bal_model(collocation_points,
                  model_outputs,
                  complex_model,
                  experiment_design,
                  eval_steps=1,  # By default
                  prior_samples=25000,  # By default
                  mc_samples=10000,  # By default
                  mc_exploration=1000,  # By default
                  gp_library="gpy",  # By default
                  ):
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
                     'util_func': np.empty(n_iter, dtype=object), 'prior': prior, 'observations': obs,
                     'errors': error_pp}
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
            if complex_model.num_quantities == 1:
                kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=complex_model.ndim))
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
                # Modify default kernel/likelihood values:
                likelihood.noise = 1e-5  # Initialize the noise with a very small value.
                # 1.3. Train a GPE, which consists of a gpe for each location being evaluated
                sm = GPyTraining(collocation_points=collocation_points, model_evaluations=model_outputs,
                                 likelihood=likelihood, kernel=kernel,
                                 training_iter=100,
                                 optimizer="adam",
                                 verbose=False)
            else:
                combined_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=complex_model.ndim))
                kernel = (combined_kernel, combined_kernel)
                multi_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
                multi_sm = MultiGPyTraining(collocation_points,
                                            model_outputs,
                                            kernel,
                                            training_iter=150,
                                            likelihood=multi_likelihood,
                                            optimizer="adam", lr=0.01, number_quantities=2,
                                            )

        # Trains the GPR
        if it == n_iter:
            if n_iter < 0:
                logger.info(
                    f'------------ Number of initial runs init_runs and n_tp_max are the same. Conditions: n_tp_max > init_runs.     -------------------')
                exit()
            else:
                logger.info(f'------------ Training final surrogate model    -------------------')
        elif it > 0:
            logger.info(f'------------ Training model with new training point: {new_tp}   -------------------')
        elif it == 0:
            logger.info('Starting surrogate model training with the initial collocation points')
            logger.info(collocation_points)

        if complex_model.num_quantities == 1:
            sm.train_()
            surrogate_object = sm
        else:
            multi_sm.train()
            surrogate_object = multi_sm

        # 2. Validate GPR
        if it % eval_steps == 0:
            if complex_model.num_quantities == 1:
                # Construct the save_name path for single quantity
                save_name = os.path.join(gpe_results_folder_bal,
                                         f'gpr_{gp_library}_TP{collocation_points.shape[0]:02d}_'
                                         f'{experiment_design.exploit_method}_quantities{complex_model.num_quantities}.pkl')
                sm.Exp_Design = experiment_design
                with open(save_name, "wb") as file:
                    pickle.dump(sm, file)
            else:
                # Construct the save_name path for multiple quantities
                save_name = os.path.join(gpe_results_folder,
                                         f'gpr_{gp_library}_TP{collocation_points.shape[0]:02d}_'
                                         f'{experiment_design.exploit_method}_quantities{complex_model.num_quantities}.pkl')
                with open(save_name, "wb") as file:
                    pickle.dump(multi_sm, file)

        # 3. Compute Bayesian scores in parameter space ----------------------------------------------------------
        # Surrogate outputs for prior samples
        if complex_model.num_quantities == 1:

            surrogate_output = sm.predict_(input_sets=prior,
                                           get_conf_int=True)
            model_predictions = surrogate_output['output']
            total_error = complex_model.measurement_errors
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
            surrogate_output = multi_sm.predict_(input_sets=prior)
            total_error = complex_model.measurement_errors
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
            # gaussian assumption = True (General Bayesian Active Learning)

            SD = SequentialDesign(exp_design=experiment_design,
                                  sm_object=surrogate_object,
                                  obs=complex_model.observations,
                                  errors=total_error,
                                  do_tradeoff=False,
                                  gaussian_assumption=False,
                                  mc_samples=mc_samples,
                                  mc_exploration=mc_exploration)  #multiprocessing=parallelize

            new_tp, util_fun = SD.run_sequential_design(prior_samples=prior)
            logger.info(f"The new collocation point after rejection sampling is {new_tp} obtained with {util_fun}")
            bayesian_dict['util_func'][it] = util_fun

            # Evaluate model in new TP

            if complex_model.complete_bal_mode or complex_model.only_bal_mode:
                bal_iteration = it + 1
                complex_model.run_multiple_simulations(collocation_points=None,
                                                  bal_iteration=bal_iteration,
                                                  bal_new_set_parameters=new_tp,
                                                  complete_bal_mode=complex_model.complete_bal_mode)

                model_outputs=complex_model.model_evaluations

            #-------------------------------------------------------
            # Update collocation points:
            if experiment_design.exploit_method == 'sobol':
                collocation_points = new_tp
            else:
                collocation_points = np.vstack((collocation_points, new_tp))
                logger.info(f'------------ Finished iteration {it + 1}/{n_iter} -------------------')
            try:
                with open(os.path.join(complex_model.asr_dir, 'BAL_dictionary.pkl'), 'wb') as pickle_file:
                    pickle.dump(bayesian_dict, pickle_file)
                print("BAL data successfully saved.")
            except Exception as e:
                print(f"An error occurred while saving the dictionary: {e}")
        try:
            with open(os.path.join(complex_model.asr_dir, 'BAL_dictionary.pkl'), 'wb') as pickle_file:
                pickle.dump(bayesian_dict, pickle_file)
            print("BAL data successfully saved.")
        except Exception as e:
            print(f"An error occurred while saving the dictionary: {e}")
    updated_collocation_points = collocation_points
    return bayesian_dict, updated_collocation_points


# def plots():
#     pass
if __name__ == "__main__":
    full_complexity_model = initialize_model(TelemacModel(
        # HydroSimulations class parameters
        control_file="tel_ering_restart0.5.cas",
        model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering_1quantity/simulation_folder_telemac/",
        res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering_1quantity/",
        calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering_1quantity/measurements_VITESSE_WDEPTH_filtered.csv",
        n_cpus=4,
        init_runs=3,
        calibration_parameters=["zone11",
                                "zone9",
                                "zone8",
                                "zone10",
                                "zone1",
                                "zone4",
                                "zone5",
                                "zone7"],
        param_values=[[0.01, 0.18],
                      [0.01, 0.18],
                      [0.01, 0.18],
                      [0.01, 0.18],
                      [0.01, 0.18],
                      [0.01, 0.03],
                      [0.15, 0.30],
                      [0.02, 0.10]],
        calibration_quantities=["SCALAR VELOCITY"
                                ,"WATER DEPTH"],
        # ,
        #                         "WATER DEPTH"],
        dict_output_name="model-outputsS-tm",
        parameter_sampling_method="sobol",
        max_runs=5,
        # TelemacModel class parameters
        friction_file="friction_ering.tbl",
        tm_xd="1",
        gaia_steering_file="",
        results_filename_base="Results_ering2m3",
        stdout=6,
        python_shebang="#!/usr/bin/env python3",
        complete_bal_mode=True,
        only_bal_mode=False,
    ))
    exp_design = setup_experiment_design(complex_model=full_complexity_model,
                                         tp_selection_criteria='dkl'
                                         )
    init_collocation_points, model_evaluations, obs, error_pp, n_loc = run_complex_model(
        complex_model=full_complexity_model,
        experiment_design=exp_design,
    )
    bal_dict, updated_collocation_points = run_bal_model(collocation_points=init_collocation_points,
                                                         model_outputs=model_evaluations,
                                                         complex_model=full_complexity_model,
                                                         experiment_design=exp_design,
                                                         eval_steps=1,
                                                         prior_samples=2000,
                                                         mc_samples=500,
                                                         mc_exploration=100,
                                                         gp_library="gpy")
    plotter = BayesianPlotter(results_folder_path=full_complexity_model.asr_dir)
    plotter.plot_bme_re(bayesian_dict=bal_dict,
                        num_bal_iterations=2,
                        plot_type='both')
    plotter.plot_combined_bal(collocation_points=updated_collocation_points,
                              n_init_tp=full_complexity_model.init_runs,
                              bayesian_dict=bal_dict)
    plotter.plot_posterior_updates(posterior_arrays=bal_dict['posterior'],
                                   parameter_names=full_complexity_model.calibration_parameters,
                                   prior=bal_dict['prior'],
                                   iterations_to_plot=[2])
    plotter.plot_bme_3d(param_sets=updated_collocation_points,
                        param_ranges=full_complexity_model.param_values,
                        param_names=full_complexity_model.calibration_parameters,
                        bme_values=bal_dict['BME'],
                        param_indices=(1, 3),
                        grid_size=200,
                        last_iterations=2
                        )

    # # TODO: Why is this in a __main__ namespace? This should be refactored into functions and the function call - Refactored into functions
    # # TODO  sequence should self-explain the workflow. - Done
