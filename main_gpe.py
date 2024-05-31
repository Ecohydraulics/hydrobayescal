"""
Code that trains a Gaussian Process Emulator (GPE) for any full complex model (i.e., hydrodynamic models).
Possible to couple with any other open source hydrodynamic software.
Can use normal training (once) or sequential training (BAL, SF, Sobol)
"""
import time
from pathlib import Path
import sys
import pickle
import warnings
import pandas as _pd
import pdb

par_path = Path.cwd().parent.parent  # Main directory
sys.path.append(str(par_path))
sys.path.append(str(par_path / 'src'))

sys.path.append(str(par_path / 'src/hyBayesCal/surrogate_modelling'))
sys.path.append(str(par_path / 'src/hyBayesCal/utils'))

#from src.hyBayesCal.plots.plots_1d_2d import *

from user_settings import user_inputs
from src.hydroBayesCal.hydro_simulations import HydroSimulations

from src.hydroBayesCal.surrogate_modelling.bal_functions import BayesianInference, SequentialDesign
from src.hydroBayesCal.surrogate_modelling.gpe_skl import *
from src.hydroBayesCal.surrogate_modelling.gpe_gpytorch import *
from src.hydroBayesCal.surrogate_modelling.inputs import Input
from src.hydroBayesCal.surrogate_modelling.exp_design_ import ExpDesign

# from src.plots.plots_convergence import *
# from src.plots.plots_1d_2d import *
# from src.plots.plots_validation import *

from src.hydroBayesCal.utils.log import *

warnings.filterwarnings("ignore")
#matplotlib.use("Qt5Agg")

if __name__ == "__main__":

    ## Uncomment these lines if you need to retrieve the collocation points and model results as numpy arrays for surrogate mo
    # el construction.

    # path_np_collocation_points='/home/IWS/hidalgo/Documents/hybayescal/examples/donau/auto-saved-results/colocation_points.npy'
    # path_np_model_results='/home/IWS/hidalgo/Documents/hybayescal/examples/donau/auto-saved-results/model_results.npy'
    # np_model_results = np.load(path_np_model_results)
    # np_collocation_points = np.load(path_np_collocation_points)

    #  INPUT DATA
    # paths ..........................................................................
    # Folder where to save results. a folder called auto-saved-results will be automatically created to store all code outputs.
    results_path_folder = Path(user_inputs['results_folder_path'])/ 'auto-saved-results' # Folder where to save results

    # surrogate data .................................................................
    parallelize = False  # to parallelize surrogate training, BAL
    # GP library .....................................................................
    gp_library = user_inputs['gp_library']  # gpy: GPyTorch or skl: Scikit-Learn

    # Importing measurement quantities and measurement errors at calibration points.............................
    calibration_pts_df = _pd.read_csv(user_inputs['calib_pts_file_path'])

    # Importing measured values of the calibration quantity at the calibration points as a 2D numpy array shape [1 , No.calibration points].
    np_observations = calibration_pts_df.iloc[:, 3].to_numpy().reshape(1, -1)

    # Importing errors at calibration points as a numpy arra shape [No. calibration points,]
    np_error = calibration_pts_df.iloc[:, 4].to_numpy()

    # number of output locations (i.e., calibration points) / Surrogates to train. (One surrogate per calibration point)
    n_loc = np_observations.size
    input_data_path = None

    # number of calibration parameters
    ndim = len(user_inputs['calib_parameter_list'])

    # Name for the different calibration quantities. Model outputs
    output_names = user_inputs['calib_quantity_list']

    # Redefining observations and errors variables for surrogate model construction and BAL
    obs = np_observations
    error_pp = np_error

    # EXPERIMENT DESIGN

    # # Probabilistic model input: ....................................................
    Inputs = Input()
    # # One "Marginal" for each parameter.
    for i in range(ndim):
        Inputs.add_marginals()  # Create marginal for parameter "i"
        Inputs.Marginals[i].name = user_inputs['calib_parameter_list'][i]  # Parameter name
        Inputs.Marginals[i].dist_type = 'uniform'  # Parameter distribution (see exp_design.py --> build_dist()
        Inputs.Marginals[i].parameters = user_inputs['parameter_ranges_list'][i]  # Inputs needed for distribution

    # # Experimental design: ....................................................................

    exp_design = ExpDesign(input_object=Inputs,
                           exploit_method='bal',  # bal, space_filling, sobol
                           explore_method='random',  # method to sample from parameter set for active learning
                           training_step=1,  # No. of training points to sample in each iteration
                           sampling_method=user_inputs['parameter_sampling_method'],  # how to sample the initial training points
                           main_meta_model='gpr',  # main surrogate method: 'gpr' or 'apce'
                           n_initial_tp=user_inputs['init_runs'],  # Number of initial training points (min = n_trunc*2)
                           n_max_tp=user_inputs['n_max_tp'],  # max number of tp to use
                           training_method='sequential',  # normal (train only once) or sequential (Active Learning)
                           util_func='dkl',  # criteria for bal (dkl, bme, ie, dkl_bme) or SF (default: global_mc)
                           eval_step=2,  # every how many iterations to evaluate the surrogate
                           secondary_meta_model=False  # only gpr is available
                           )

    exp_design.setup_ED_()

    # setup surrogate model data:
    # sets gpy (GpyTorch) or skl(Scikitlearn) as libraries for GPE.
    if exp_design.main_meta_model == 'gpr':
        exp_design.setup_gpe(library=gp_library
                             )

    # COLLOCATION POINTS
    # This parts generates the initial collocation points for GPE based on the number of init_runs, calibration parameters ranges
    # and sampling method.
    collocation_points = exp_design.generate_samples(n_samples=exp_design.n_init_tp,
                                                        sampling_method=exp_design.sampling_method)

    # RUN FULL COMPLEXITY MODEL: ============

    #     """NOTE: The collocation points and model results should be in numpy-array form, and in the order needed by
    #     the gpe_skl.py or gpe_gpytorh.py classes"""
    # shape model_results: [No. runs x No. calibration points]
    # shape collocation_points: [No. runs x No. calibration parameters]

    full_complexity_model = HydroSimulations(user_inputs, user_inputs['bal_mode'])
    model_evaluations = full_complexity_model.run(collocation_points=collocation_points, bal_iteration=0,
                                              bal_new_set_parameters=None, )

    print(
        f"<<< Will run ({exp_design.n_iter + 1}) GP training iterations and ({exp_design.n_evals}) GP evaluations. >>> ")

    # Reference data .......................................................
    prior = exp_design.generate_samples(user_inputs['n_samples'])

    prior_logpdf = np.log(exp_design.JDist.pdf(prior.T)).reshape(-1)


    # INITIALIZATION GPE AND RESULTS FOLDERS

    #     """This part can be ommited, if the files can be saved directly in the results_path folder"""
    #     # Creates folder for specific case ....................................................................
    results_folder = results_path / f'surrogate_{user_inputs["results_file_name_base"]}_ndim_{ndim}_nout_{n_loc}'
    if not results_folder.exists():
        logger.info(f'Creating folder {results_folder}')
        results_folder.mkdir(parents=True)

    #     # Create a folder for the exploration method: needed only if BAL is to be used
    results_folder = results_folder / f'{exp_design.exploit_method}_{exp_design.util_func}'
    if not results_folder.exists():
        logger.info(f'Creating folder {results_folder}')
        results_folder.mkdir(parents=True)
    #
    #     # Arrays to save results ---------------------------------------------------------------------------- #
    #
    bayesian_dict = {
        'N_tp': np.zeros(exp_design.n_iter + 1),
        'BME': np.zeros(exp_design.n_iter + 1),  # To save BME value for each GPE, after training
        'ELPD': np.zeros(exp_design.n_iter + 1),
        'RE': np.zeros(exp_design.n_iter + 1),  # To save RE value for each GPE, after training
        'IE': np.zeros(exp_design.n_iter + 1),
        'post_size': np.zeros(exp_design.n_iter + 1),
        f'{exp_design.exploit_method}_{exp_design.util_func}': np.zeros(exp_design.n_iter),
        'util_func': np.empty(exp_design.n_iter, dtype=object)
    }

    #     # ========================================================================================================= #
    # Set up the exploration samples for Bayesian Active Learning
    exploration_set = exp_design.generate_samples(user_inputs["n_samples_exploration_bal"])

    # SURROGATE MODEL TRAINING
    # --------------------------------
    t_start = time.time()
    # --------------------------------

    print('Starting surrogate training with initial collocation points')
    print (collocation_points)

    # Train the GPE a maximum of "iteration_limit" times
    for it in range(0, exp_design.n_iter + 1):

        # 1. Train surrogate
        if exp_design.gpr_lib == 'skl':
            # 1.1. Set up the kernel

            # kernel = 1 * RBF(length_scale=exp_design.kernel_data['length_scale'],
            #                  length_scale_bounds=exp_design.kernel_data['bounds'])

            # Setting up initial length scales and length scales bounds for Radial Basis Function (RBF) kernel
            # Length scale: Assumed to be the midpoint of the minimum and maximum range for each of the calibration parameters.
            # Length scale bounds: Assumed to be the range for each calibration parameter.
            if it == 0:
                length_scales = []
                for param_range in user_inputs['parameter_ranges_list']:
                    length_scale = sum(param_range) / len(param_range)
                    length_scales.append(length_scale)
                length_scales_bounds = [tuple(param_range) for param_range in user_inputs['parameter_ranges_list']]

            kernel = 1 * RBF(length_scale=length_scales, length_scale_bounds=length_scales_bounds)

            # 1.2. Setup a GPR: initialize the general SKL class
            sm = SklTraining(collocation_points=collocation_points, model_evaluations=model_evaluations,
                             noise=True,
                             kernel=kernel,
                             alpha=1e-6,
                             n_restarts=10,
                             parallelize=parallelize)

        elif exp_design.gpr_lib == 'gpy':
            # 1.1. Set up the kernel
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ndim))

            # 1.2. Set up Likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
            # Modify default kernel/likelihood values:
            likelihood.noise = 1e-5  # Initialize the noise with a very small value.

            # 1.3. Train a GPE, which consists of a gpe for each location being evaluated
            sm = GPyTraining(collocation_points=collocation_points, model_evaluations=model_evaluations,
                             likelihood=likelihood, kernel=kernel,
                             training_iter=100,
                             optimizer="adam",
                             verbose=False)

        # Trains the GPR
        if it == exp_design.n_iter:
            logger.info(f'------------ Training final model with the last training point: {new_tp}     -------------------')
        elif it > 0:
            logger.info(f'------------ Training model with new training point: {new_tp}   -------------------')

        sm.train_()

        # 2. Validate GPR
        if it % exp_design.eval_step == 0:
            save_name = results_folder / f'gpr_{exp_design.gpr_lib}_TP{collocation_points.shape[0]:02d}_{exp_design.exploit_method}.pickle'
            sm.Exp_Design = exp_design
            with open(save_name, "wb") as file:
                pickle.dump(sm, file)

        # 3. Compute Bayesian scores in parameter space ----------------------------------------------------------
        # Surrogate outputs for prior samples
        surrogate_output = sm.predict_(input_sets=prior, get_conf_int=True)
        total_error = error_pp  #(error_pp ** 2)
        #print(np.ndim(total_error))
        bi_gpe = BayesianInference(model_predictions=surrogate_output['output'], observations=obs, error=total_error,
                                   sampling_method='rejection_sampling', prior=prior, prior_log_pdf=prior_logpdf)
        bi_gpe.estimate_bme()
        bayesian_dict['N_tp'][it] = collocation_points.shape[0]
        bayesian_dict['BME'][it], bayesian_dict['RE'][it] = bi_gpe.BME, bi_gpe.RE
        bayesian_dict['ELPD'][it], bayesian_dict['IE'][it] = bi_gpe.ELPD, bi_gpe.IE
        bayesian_dict['post_size'][it] = bi_gpe.posterior_output.shape[0]

        # 4. Sequential Design --------------------------------------------------------------------------------------
        if it < exp_design.n_iter:
            logger.info(f'Selecting {exp_design.training_step} additional TP using {exp_design.exploit_method}')
            SD = SequentialDesign(exp_design=exp_design, sm_object=sm, obs=obs, errors=total_error,
                                  do_tradeoff=False)  #multiprocessing=parallelize

            #new_tp, util_fun = SD.run_sequential_design(prior_samples=prior)
            SD.gaussian_assumption = True
            new_tp, util_fun = SD.run_sequential_design(prior_samples=exploration_set)
            print(f"The new collocation point after rejection sampling is {new_tp}")

            bayesian_dict[f'{exp_design.exploit_method}_{exp_design.util_func}'][it] = SD.selected_criteria[0]
            bayesian_dict['util_func'][it] = util_fun

            # Evaluate model in new TP: This is specific to each problem --------
            bal_iteration = it + 1
            model_evaluations = full_complexity_model.run(collocation_points=None, bal_iteration=bal_iteration,
                                                      bal_new_set_parameters=new_tp)

            #new_output = nonlinear_model(params=new_tp, loc=pt_loc)
            #-------------------------------------------------------
            # Update collocation points:
            if exp_design.exploit_method == 'sobol':
                collocation_points = new_tp
            else:
                collocation_points = np.vstack((collocation_points, new_tp))

            # collocation_points = np_collocation_points
            # model_evaluations = np_model_results
        #             # Plot process
        #             if it % exp_design.eval_step == 0:
        #                 if ndim == 1:
        #                     if SD.do_tradeoff:
        #                         plot_1d_bal_tradeoff(prior, surrogate_output['output'], surrogate_output['lower_ci'],
        #                                              surrogate_output['upper_ci'], SD.candidates,
        #                                              SD, collocation_points, model_evaluations, it, obs)
        #                     else:
        #                         plot_1d_gpe_bal(prior, surrogate_output['output'], surrogate_output['lower_ci'],
        #                                         surrogate_output['upper_ci'], SD.candidates,
        #                                         SD.total_score, collocation_points, model_evaluations, it, obs)
        #                 if ndim < 3:
        #                     plot_likelihoods(prior, bi_gpe.likelihood, prior, ref_scores.likelihood,
        #                                      n_iter=it + 1)
        #
        #                 stop=1

            logger.info(f'------------ Finished iteration {it + 1}/{exp_design.n_iter} -------------------')

#     # Plot all TP:
#     if ndim == 1:
#         plot_1d_gpe_final(prior, surrogate_output['output'],
#                           surrogate_output['lower_ci'], surrogate_output['upper_ci'],
#                           collocation_points, model_evaluations,
#                           exp_design.n_init_tp, exp_design.n_iter, obs)
#
#     if ndim < 3:
#         plot_likelihoods(prior, bi_gpe.likelihood, prior, ref_scores.likelihood,
#                          n_iter=exp_design.n_iter)
#         if exp_design.exploit_method == 'bal' or exp_design.exploit_method == 'space_filling':
#             plot_combined_bal(collocation_points=collocation_points, n_init_tp=exp_design.n_init_tp,
#                               bayesian_dict=bayesian_dict)

#     # Plot evaluation criteria:
#     plot_correlation(sm_out=valid_sm['output'], valid_eval=exp_design.val_y,
#                      output_names=output_names,
#                      label_list=[f"{np.mean(eval_dict['r2']['Z']):0.3f}"],
#                      fig_title=f'Outputs', n_loc_=n_loc)
#
#     plot_validation_loc(eval_dict_list=[run_valid], label_list=[''],
#                         output_names=output_names, n_loc=n_loc,
#                         criteria=['mse', 'nse', 'r2', 'mean_error', 'std_error'],
#                         fig_title=f'Validation criteria: TP: {exp_design.n_max_tp}')
#
#     plot_validation_loc(eval_dict_list=[run_valid], label_list=[''],
#                         output_names=output_names, n_loc=n_loc,
#                         criteria=['norm_error', 'P95'],
#                         fig_title=f'Validation criteria with error consideration: TP: {exp_design.n_max_tp}')
#
#     plot_validation_tp(eval_dict_list=[eval_dict], label_list=[''],
#                        output_names=output_names, n_loc=n_loc,
#                        criteria=['mse', 'nse', 'r2', 'mean_error', 'std_error'], plot_loc=True,
#                        fig_title='Validation criteria for increasing number of TP')
#
#     # # Plot Bayesian criteria
#     plot_gpe_scores(bayesian_dict['BME'], bayesian_dict['RE'], exp_design.n_init_tp,
#                     ref_bme=ref_scores.BME, ref_re=ref_scores.RE)
#
#     if exp_design.exploit_method == 'BAL':
#         plot_bal_criteria(bayesian_dict[f'{exp_design.exploit_method}_{exp_design.util_func}'], exp_design.util_func)
#     # Plot training points
#     plot_parameters(prior_samples, collocation_points, post=bi_gpe.posterior)

#
#
#     end = 0
#
# """
# TO-DO:
# - FOR 1D-2D plots:
# # 1) To plot GPE + BAL for single and multiple observation points (and only one N_p), do separate functions (CONTINUE)
# # 2) Plot GPE and Ref likelihoods
#
# - For general case:
# # 1) Plot RE, ELPD and BME for GPE and reference model with increasing number of iterations
# # 2) Plot BAl criteria with increasing number of iterations
# 3) Save time for each run, changing number of data points and parameters (one at a time) to plot later
# 4) Plot 1 and 2 for case 3
# 5) Add loop for number of parameters?
#
# """
