# import sys
# import pandas as pd
# import joblib
# import emcee
import sys
import os
import time
import emcee
import numpy as np
import pandas as pd
import logging
import bayesvalidrox as bvr

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# from bayesvalidrox import PyLinkForwardModel
from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.function_pool import *

# --- Configure logging ---
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# --- Setup paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')

sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# Trick BayesValidRox into recognizing this as an importable module
sys.modules['bal_telemac_bvr'] = sys.modules[__name__]

# -------------------------------------------------------------------------
# Helper function to run Telemac model and return results
# -------------------------------------------------------------------------
def run_complex_model(complex_model, input_parameters):
    """
    Runs the Telemac model for a given set of input parameters and returns results.
    """
    collocation_points = None
    model_outputs = None

    if not complex_model.only_bal_mode:
        logger.info(
            f"Sampling {complex_model.init_runs} collocation points for the selected "
        )
        collocation_points = input_parameters
        complex_model.run_multiple_simulations(
            collocation_points=collocation_points,
            complete_bal_mode=complex_model.complete_bal_mode,
            validation=complex_model.validation
        )
        model_outputs = complex_model.model_evaluations
    else:
        try:
            path_np_collocation_points = os.path.join(complex_model.asr_dir, 'collocation-points.csv')
            path_np_model_results = os.path.join(complex_model.asr_dir, 'model-results.csv')
            collocation_points = np.loadtxt(path_np_collocation_points, delimiter=',', skiprows=1)
            model_outputs = np.loadtxt(path_np_model_results, delimiter=',', skiprows=1)
        except FileNotFoundError:
            logger.warning(
                'Saved collocation points or model results not found. '
                'Please run initial simulations first to use only Bayesian Active Learning.'
            )

    observations = complex_model.observations
    errors = complex_model.measurement_errors
    nloc = complex_model.nloc

    return collocation_points, model_outputs, observations, errors, nloc


# -------------------------------------------------------------------------
# Function to initialize and run Telemac
# -------------------------------------------------------------------------
def bal_telemac_bvr(collocation_points_runs=None, complex_model=None):
    """
    Flexible wrapper for Telemac simulations.

    Modes:
    -------
    1. Instance-only mode: collocation_points_runs=None and complex_model=None
       -> returns only the model instance.
    2. BayesValidRox mode: collocation_points_runs provided
       -> runs simulation and returns model evaluations dict only.
    """
    # --- Instantiate model if needed ---
    if complex_model is None:
        complex_model = TelemacModel(
            friction_file="friction_ering_MU_initial_NIKU.tbl",
            tm_xd="1",
            gaia_steering_file="gaia_ering_initial_NIKU.cas",
            gaia_results_filename_base="resultsGAIA",
            results_filename_base="results2m3",
            control_file="tel_ering_initial_NIKU.cas",
            model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation-folder-telemac-gaia",
            res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
            calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
            n_cpus=16,
            calibration_parameters=[
                "gaiaCLASSES SHIELDS PARAMETERS 1",
                "gaiaCLASSES SHIELDS PARAMETERS 2",
                "zone2", "zone4", "zone5", "zone6", "zone8", "zone9", "zone13"
            ],
            param_values=[
                [0.047, 0.070],
                [0.047, 0.070],
                [0.008, 0.4],
                [0.002, 0.4],
                [0.002, 0.4],
                [0.030, 0.4],
                [0.002, 0.4],
                [0.030, 0.4],
                [0.040, 1.8],
            ],
            extraction_quantities=[
                "WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG",
                "VELOCITY U", "VELOCITY V", "CUMUL BED EVOL"
            ],
            calibration_quantities=["WATER DEPTH"],
            complete_bal_mode=True,
            delete_complex_outputs=True,
        )

    # --- Instance-only mode ---
    if collocation_points_runs is None:
        return complex_model  # only return the instance

    # --- BayesValidRox mode: run simulations ---
    collocation_points, model_evaluations, _, _, _ = run_complex_model(
        complex_model=complex_model,
        input_parameters=collocation_points_runs
    )

    model_evaluations_dic = {
        'x_values': np.arange(model_evaluations.shape[0]),
        'H': model_evaluations.flatten()
    }

    return model_evaluations_dic  # only return the evaluations


# -------------------------------------------------------------------------
# Entry point for manual testing or BayesValidRox linkage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    complex_model = bal_telemac_bvr()
    inputs = bvr.Input()
    # # One "Marginal" for each parameter.
    for i in range(complex_model.ndim):
        inputs.add_marginals()  # Create marginal for parameter "i"
        inputs.marginals[i].name = complex_model.calibration_parameters[i]  # Parameter name
        inputs.marginals[i].dist_type = 'unif' # Parameter distribution (see exp_design.py --> build_dist()
        inputs.marginals[i].parameters = complex_model.param_values[i]  # Inputs needed for distribution

    # Create BayesValidRox-compatible callable
    model = bvr.PyLinkForwardModel()
    model.link_type = 'Function'  # This file acts as an external function
    model.py_file = 'bal_telemac_bvr'  # So BayesValidRox imports this as module
    model.name = 'bal_telemac_bvr'
    model.output.names = ['H']  # Output variable name
    calibration_pts_file_path = complex_model.calibration_pts_file_path
    test_Data = pd.read_csv(calibration_pts_file_path)
    model.observations = {}
    # ✅ Make sure it's a DataFrame with shape (36, 1)
    model.observations = {
        'H': test_Data['WATER DEPTH_DATA'].values  # or .to_numpy()
    }
    # model.observations['U_scalar'] = test_Data['U_scalar']
    # %% test forward run from the bayesvalidrox if necessary
    # Step 3: Create the experimental design
    # exp_design = ExpDesigns(inputs)  # Latin Hypercube Sampling
    # exp_design.sampling_method = 'sobol'#'latin_hypercube'
    # samples_test = exp_design.generate_samples(1)
    # output_test, samples_test2 = model.run_model_parallel(samples_test)
    # ass = model.run_forwardmodel(samples_test)
    # %%
    # =====================================================
    # ==========  DEFINITION OF THE METAMODEL  ============
    # =====================================================
    meta_model = bvr.GPESkl(inputs)
    # !%%
    # #------------------------------------------------
    # # ------------- GPE Specification ----------------
    # # ------------------------------------------------
    # # Select the solver for solving for the GP Hyperparameters using the ML approach
    # # ToDo: Remove this as a user-defined parameter, since only one is available?
    # # 1)LBFGS: only option for Scikit Learn
    meta_model._gpe_reg_method = "LBFGS"

    # Kernel options ----------------------------
    # Loop over different Kernels:
    # 1) True to loop over the different kernel types and select the best one
    meta_model._auto_select = False

    # Select Kernel type:
    # 1) RBF: Gaussian/squared exponential kernel
    # 2) Matern Kernel
    # 3) RQ: Rational Quadratic kernel
    meta_model._kernel_type = "RBF"

    meta_model.normalize_x_method = "norm"  # Input data transformation
    meta_model._kernel_isotropy = True  # Kernel isotropy, False for anisotropy
    meta_model._nugget = 1e-6  # Set regularization parameter (constant)
    meta_model._kernel_noise = (
        False  # Optimize regularization parameter. True to consider WhiteKernel
    )
    meta_model.n_restarts = 20

    # Bootstraping
    # 1) normal 2) fast
    meta_model.n_bootstrap_itrs = 1

    # ------------------------------------------------
    # ------ Experimental Design Configuration -------
    # ------------------------------------------------
    exp_design = bvr.ExpDesigns(inputs)

    # Number of initial (static) training samples
    exp_design.n_init_samples = 5

    # Sampling methods
    # 1) random 2) latin_hypercube 3) sobol 4) halton 5) hammersley
    # 6) chebyshev(FT) 7) grid(FT) 8)user
    exp_design.sampling_method = 'sobol'  # "latin_hypercube"

    # Provide the experimental design object with a hdf5 file
    # exp_design.hdf5_file = 'exp_design_AnalyticFunc.hdf5'

    # Set the sampling parameters
    exp_design.n_new_samples = 1
    exp_design.n_max_samples = 6  # sum of init + sequential
    exp_design.mod_loo_threshold = 1e-16

    # Tradeoff scheme
    # 1) None 2) 'equal' 3)'epsilon-decreasing' 4) 'adaptive'
    exp_design.tradeoff_scheme = 'explore_only'
    # exp_design.n_replication = 5

    # -------- Exploration ------
    # 1)'Voronoi' 2)'random' 3)'latin_hypercube' 4)'LOOCV' 5)'dual annealing'
    exp_design.explore_method = "random"

    # Use when 'dual annealing' chosen
    exp_design.max_func_itr = 1000

    # Use when 'Voronoi' or 'random' or 'latin_hypercube' chosen
    exp_design.n_canddidate = 1000
    exp_design.n_cand_groups = 4

    # -------- Exploitation ------
    # 1)'BayesOptDesign' 2)'BayesActDesign' 3)'VarOptDesign' 4)'alphabetic'
    # 5)'Space-filling'
    # exp_design.exploit_method = "Space-filling"
    exp_design.exploit_method = "BayesActDesign"
    exp_design.util_func = "DKL"

    # BayesOptDesign/BayesActDesign -> when data is available
    # 1) MI (Mutual information) 2) ALC (Active learning McKay)
    # 2)DKL (Kullback-Leibler Divergence) 3)DPP (D-Posterior-percision)
    # 4)APP (A-Posterior-percision)  # ['DKL', 'BME', 'infEntropy']
    # exp_design.util_func = 'DKL'

    # BayesActDesign -> when data is available
    # 1) BME (Bayesian model evidence) 2) infEntropy (Information entropy)
    # 2)DKL (Kullback-Leibler Divergence)
    # exp_design.util_func = 'DKL'

    # VarBasedOptDesign -> when data is not available
    # 1)ALM 2)EIGF, 3)LOOCV
    # or a combination as a list
    # exp_design.util_func = 'EIGF'

    # alphabetic
    # 1)D-Opt (D-Optimality) 2)A-Opt (A-Optimality)
    # 3)K-Opt (K-Optimality) or a combination as a list
    # exp_design.util_func = 'D-Opt'

    # Defining the measurement error, if it's known a priori

    # obs_uncert =  pd.DataFrame(model.observations, columns=model.output.names) ** 2
    obs_uncert = test_Data[['WATER DEPTH_ERROR']].copy()  # test_Data['U_scalar_error']
    obs_uncert.columns = model.output.names
    # obs_uncert2.columns=model.output.names #pd.DataFrame(model.observations, columns=model.output.names) ** 2

    discrepancy = bvr.Discrepancy(parameters=obs_uncert, disc_type="Gaussian")

    # Plot the posterior snapshots for SeqDesign
    # exp_design.max_a_post = [0] * NDIM

    # NDIM = 9

    # # For calculation of validation error for SeqDesign
    # test_prior = np.load('Prior_2.npy')
    # prior_outputs = np.load(f"data/origModelOutput_{NDIM}.npy")
    # likelihood = np.load(f"data/validLikelihoods_{NDIM}.npy")
    # exp_design.valid_samples = prior[:500]
    # exp_design.valid_model_runs = {"Z": prior_outputs[:500]}

    # Run using the engine
    engine = bvr.Engine(meta_model, model, exp_design, discrepancy=discrepancy)
    # %%
    engine.train_sequential()
    print('Surrogate has been trained')
    end_time = time.time()  # Record the end time

    # Calculate the elapsed time
    run_time = end_time - start_time
    print(f"BAL is finished after {run_time:.6f} seconds.")
    # engine.train_normal()
    # %%

    # Base folder for GPE results
    gpe_results_folder = os.path.join(complex_model.asr_dir, "surrogate-gpe")
    # Create subfolder based on calibration parameters
    subfolder_name = f"gpe_bvr_{complex_model.calibration_parameters}"
    output_folder = os.path.join(gpe_results_folder, subfolder_name)

    # Create subfolder
    os.makedirs(output_folder, exist_ok=True)
    # Paths for saving
    engine_path = os.path.join(output_folder, "engine.pkl")
    runtime_path = os.path.join(output_folder, "runTime.pkl")

    # Save engine
    with open(engine_path, "wb") as f:
        pickle.dump(engine, f, protocol=2)

    # Save runtime
    with open(runtime_path, "wb") as f:
        pickle.dump(run_time, f, protocol=2)

    # %%
    # with open("/home/ran-wei/Documents/coding2025/hydrodynamic_model_surrogate/hydrobayesian_2dhydrodynamic/coupling_bayesvalidrox/bal/results_bayesvalidrox/GP_WD_longRun/engine.pkl", "rb") as input_:
    #     engine = joblib.load(input_)
    # %%
    # =====================================================
    # =========  POST PROCESSING OF METAMODELS  ===========
    # =====================================================
    post = bvr.PostProcessing(engine)
    # %%
    # # Plot to check validation visually.
    # post.valid_metamodel(n_samples=1)
    # # Compute and print RMSE error
    # post.check_accuracy(n_samples=10)
    # #%% Compute the moments and compare with the Monte-Carlo reference
    # post.plot_moments()
    # %%
    # Plot the evolution of the KLD,BME, and Modified LOOCV error
    # if engine.exp_design.method == "sequential":
    #     refBME_KLD = np.load("data/refBME_KLD_" + str(NDIM) + ".npy")
    # post.plot_seq_design_diagnostics(refBME_KLD)
    # %%
    # # # =====================================================
    # # # ========  Bayesian inference with Emulator ==========
    # # # =====================================================
    # engine.model.observations_valid = model.observations
    bayes_path = os.path.join(output_folder, f"Bayes_{model.name}.pkl")
    bayes = bvr.BayesInference(engine)

    # # Basic settings
    # bayes.use_emulator = False
    bayes.n_prior_samples = 100

    # # Reference data choice and perturbation
    # bayes.selected_indices = [0, 3, 5, 7, 9]

    # bayes = BayesInference(Engine_)
    bayes.use_emulator = True
    bayes.discrepancy = discrepancy
    bayes.plot = False

    # BME bootstrapping
    # bayes.bootstrap_method = "normal"
    # bayes.n_bootstrap_itrs = 500
    # bayes.bootstrap_noise = 100

    # !%% plot bayesian stats
    # Step 1: Extract the 'U_scalar' column from test_Data (ensure the correct column is used)
    # observation_bayesStats = test_Data[['U_scalar']].copy()  # Make sure to use the correct column

    # # Step 2: Convert it into a dictionary where the key is 'U_scalar' and the value is the corresponding data
    # model.observation = {
    #     'U_scalar': observation_bayesStats['U_scalar'].values  # Convert the 'U_scalar' column to a NumPy array
    # }

    # # Generate additional validation observations
    # observations_valid = model.run_model_parallel(
    #     engine.exp_design.generate_samples(1), key_str="U_scalar"
    # )[0]
    # %%
    # # for key in observations_valid:
    #     # observations_valid[key] = observations_valid[key][0]
    # bayes.engine.model.n_obs_valid = 1

    # # Step 3: Set the validation observations
    # bayes.discrepancy = discrepancy

    # #  # Reference data choice and perturbation
    # bayes.selected_indices = [0, 3, 5, 7, 9]

    # bayes.engine.model.observations_valid = model.observations
    # bayes.sampler.engine.model.observations_valid = model.observations
    # #
    # bayes.sampler.observation = model.observations

    # # Step 4: Specify the metrics for validation
    # bayes.valid_metrics = ['kld', 'inf_entropy']

    # log_bme = bayes.run_validation()
    # %%
    # Select the inference method - either 'rejection' or 'MCMC'
    bayes.inference_method = "rejection"
    # bayes.inference_method = "MCMC"

    # Set the MCMC parameters passed to self.mcmc_params
    # bayes.mcmc_params = {
    #     "n_steps": 1e4,
    #     "n_walkers": 30,
    #     "moves": emcee.moves.KDEMove(),
    #     "multiprocessing": False,
    #     "verbose": False,
    # }

    # Perform inference<
    bayesInference_result = bayes.run_inference()

    # posterior = bayes.run_inference()
    # posterior = bayes.create_inference()
    # %%
    # Save BayesInference object
    with open(bayes_path, "wb") as f:
        pickle.dump(bayes, f, protocol=2)

    # with open(f"/home/ran-wei/Documents/coding2025/hydrodynamic_model_surrogate/hydrobayesian_2dhydrodynamic/coupling_bayesvalidrox/bal/results_bayesvalidrox/GP_WD_longRun/Bayes_{model.name}.pkl", "wb") as output:
    #     joblib.dump(bayesInference_result, output)