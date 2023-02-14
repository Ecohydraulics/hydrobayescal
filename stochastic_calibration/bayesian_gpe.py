"""
Bayesian calibration of a numerical model using surrogate-assisted Bayesian inversion with a
Gaussian Process Emulator (GPE) as surrogate model (metamodel).

Full-complexity coupling is currently only available for the open-source TELEMAC modeling suite.
For any new model bindings create:
    * a new config_SOFTWARE.py file (read instructions in config_BASICS.py) and add it to the imports here
        The new model (Software) should then here be added as an option in __setattr__()
    * a new control_SOFTWARE.py file (read instructions in control_BASIC_MODEL.py)
    * an additional inheritance for the BALwithGPE class (e.g., UserDefsOpenFOAM) and initialize it
        in __init__(), similar to the Telemac bindings (not forget to import the user_defs_MODEL.py here)
    * a user file (e.g., a user input XLSX like the one for TELEMAC)
    * a new documentation in docs/usage-SOFTWARE.rst and bind it to docs/index.rst
Please do not forget to verify if the new Software works (i.e., provide a small showcase) before merging with main.


Method adapted from: Oladyshkin et al. (2020). Bayesian Active Learning for the Gaussian Process
Emulator Using Information Theory. Entropy, 22(8), 890.
"""
import os
import numpy as np
from datetime import datetime
from functools import wraps

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from BAL_core import BAL
from control_TELEMAC import *
from usr_defs_TELEMAC import *  # contains UserDefs and link to config and basic_functions
from doepy.doe_control import DesignOfExperiment


class BALwithGPE(UserDefsTelemac):
    """
    The BAL_GPE object is the framework for running a stochastic calibration of a deterministic model by using a
        Gaussian process emulator (GPE) - based surrogate model that is fitted through Bayesian active learning (BAL).
        The deterministic full-complexity model is defined through a coupled software (default TELEMAC).

    .. important::

        The object must be instantiated with metadata stored in the user-input.xlsx file corresponding to the software
        used. Visit https://stochastic-calibration.readthedocs.io to learn more.
    """
    def __init__(
            self,
            input_worbook_name="user-input.xlsx",
            software_coupling="telemac",
            *args,
            **kwargs
    ):
        """
        Initializer of the BAL_GPE class.

        :param str input_worbook_name: name of the user-input.xlsx file including its full directory
        :param software_coupling: name of the software of the deterministic model (default: "telemac")
        :param args: placeholder for consistency
        :param kwargs: placeholder for consistency
        """
        UserDefsTelemac.__init__(self, input_worbook_name)  # get user def file
        self.write_global_settings(self.input_xlsx_name)  # apply user defs
        self.__numerical_model = None
        self.__setattr__("numerical_model", software_coupling)
        self.observations = {}
        self.n_simulation = int()
        self.prior_distribution = np.ndarray(())  # will create and update in self.update_prior_distributions
        self.collocation_points = np.ndarray(())  # assign with self.prepare_initial_model - will be used and updated in self.run_BAL
        self.bme_csv_prior = ""
        self.re_csv_prior = ""
        self.bme_score_file = None
        self.re_score_file = None
        self.doe = DesignOfExperiment()

    def __setattr__(self, name, value):
        if name == "numerical_model":
            if value == "telemac":
                self.numerical_model = TelemacModel(
                    model_dir=self.SIM_DIR,
                    calibration_parameters=list(self.CALIB_PAR_SET.keys()),
                    control_file=self.TM_CAS,
                    gaia_steering_file=self.GAIA_CAS,
                    n_processors=self.N_CPUS,
                    tm_xd=self.tm_xD
                )
                print("Instantiated a %s model for BAL." % value.upper())

    def initialize_score_writing(self, func):
        """
        Wrapper function for writing BME and RE scores to csv files (for convergence check-ups)
        :param func: a function
        :return: the same function
        """
        self.bme_csv_prior = self.numerical_model.res_dir + os.sep + "BMEprior-%s.csv" % str(rnd.randint(1000, 9999))
        self.re_csv_prior = self.numerical_model.res_dir + os.sep + "REprior-%s.csv" % str(rnd.randint(1000, 9999))
        logger.info("> Instantiated recording of information theory scores BME and RE:")
        logger.info("  -- BME will be written to %s" % self.bme_csv_prior)
        logger.info("  -- RE will be written to %s" % self.re_csv_prior)
        with open(self.bme_csv_prior, mode="w+") as file:
            file.write("Started BME logging at %s\n" % str(datetime.now()))
        with open(self.re_csv_prior, mode="w+") as file:
            file.write("Started relative entropy (RE) logging at %s\n" % str(datetime.now()))
        self.bme_score_file = open(self.bme_csv_prior, mode="a")
        self.re_score_file = open(self.re_csv_prior, mode="a")

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

        self.bme_score_file.close()
        self.re_score_file.close()
        return wrapper

    @log_actions
    def run_calibration(self):
        """loads provided input file name as pandas dataframe - all actions called from here are written to a logfile

            Returns:
                tbd
        """
        logger.info("STARTING CALIBRATION PROCESS")
        # 1 - assign prior distributions
        self.initiate_prior_distributions()
        # 2 - load field observations
        self.load_observations()
        # 3 - run initial simulations with full-complexity numerical model
        self.run_initial_simulations()
        # 4 - get initial collocation points
        self.get_collocation_points()
        # incomplete from here on - USE EDUARDO EMAIL
        self.initialize_score_writing(self.run_BAL())

    def initiate_prior_distributions(self):
        """Assign uniform distributions for all user-defined parameters.
        Modifies self.prior_distribution (numpy.ndarray): copy of all uniform input distributions
        of the calibration parameters.
        """
        logger.info("-- updating prior distributions...")
        self.prior_distribution = np.zeros((self.MC_SAMPLES, len(self.CALIB_PAR_SET)))
        column = 0
        for par in self.CALIB_PAR_SET.keys():
            print(" * drawing {0} uniform random samples for {1}".format(par, str(self.MC_SAMPLES)))
            self.CALIB_PAR_SET[par]["distribution"] = np.random.uniform(
                self.CALIB_PAR_SET[par]["bounds"][0],
                self.CALIB_PAR_SET[par]["bounds"][1],
                self.MC_SAMPLES
            )
            self.prior_distribution[:, column] = np.copy(self.CALIB_PAR_SET[par]["distribution"])
            column += 1

    def load_observations(self):
        """Load observations stored in calibration_points.csv to self.observations
        """
        print(" * reading observations file (%s)..." % self.CALIB_PTS)
        observation_file = np.loadtxt(self.CALIB_PTS, delimiter=",")
        self.observations = {
            "no of points": observation_file.shape[0],
            "node IDs": observation_file[:, 0].reshape(-1, 1),
            "observation": observation_file[:, 1].reshape(-1, 1),
            "observation error": observation_file[:, 2]
        }

    def run_initial_simulations(self):
        """
        Launch initial full complexity simulations
        TO DO

        Create baseline for response surface
        """

        # make initial parameter value space -- PLUG IN DESIGN OF EXPERIMENTS IN LATER VERSIONS
        calib_par_value_dict = {}
        recalc_pars = {}
        for par, v in self.CALIB_PAR_SET.items():
            # list-like calibration parameter values require recalculation by a multiplier
            #  - only the multiplier can be used for doe, not the list-like values
            if not v["recalc par"]:
                calib_par_value_dict.update({par: v})
            else:
                if v["recalc par"] == "Multiplier":
                    if not ("Multiplier" in calib_par_value_dict.keys()):
                        calib_par_value_dict.update({"Multiplier": v})
                else:
                    recalc_pars.update({par: [v["initial val"]]})

        self.doe.generate_multi_parameter_space(
            parameter_dict=calib_par_value_dict,
            method=self.init_run_sampling,
            total_number_of_samples=self.init_runs
        )
        self.doe.df_parameter_spaces.to_csv(self.SIM_DIR + "/initial-run-parameters-scalars.csv")

        # update any list-like parameter values and remove multiplier from calib par values
        if "Multiplier" in calib_par_value_dict.keys():
            for list_par, v in recalc_pars.keys():
                initial_val = recalc_pars[list_par][0]  # preserve initial value
                recalc_pars[list_par] = []  # re-initialize clean empty list
                for mult in self.doe.df_parameter_spaces["Multiplier"]:
                    recalc_pars[list_par] = initial_val * mult
                self.doe.df_parameter_spaces = self.doe.df_parameter_spaces.join(
                    pd.DataFrame(data={list_par: recalc_pars[list_par]})
                )
            # remove multiplier
            self.doe.df_parameter_spaces.drop("Multiplier", axis=1, inplace=True)
        # write final initial run parameters
        self.doe.df_parameter_spaces.to_csv(self.SIM_DIR + "/initial-run-parameters-all.csv")

        # run initial simulations and update the steering file with new parameters before each run
        for init_run_it in range(self.init_runs):
            self.numerical_model.update_model_controls(
                new_parameter_values=self.doe.df_parameter_spaces.loc[init_run_it, :],
                simulation_id=init_run_it
            )
            self.numerical_model.run_simulation()

    def get_collocation_points(self):

        # Part 2. Read initial collocation points
        temp = np.loadtxt(os.path.abspath(os.path.expanduser(self.RESULTS_DIR))+"/parameter_file.txt", dtype=str, delimiter=";")
        simulation_names = temp[:, 0]
        self.collocation_points = temp[:, 1:].astype(float)
        self.n_simulation = self.collocation_points.shape[0]  # corresponds to m full-complexity runs?

        # Part 3. Read the previously computed simulations of the numerical model in the initial collocation points
        temp = np.loadtxt(os.path.abspath(os.path.expanduser(self.RESULTS_DIR)) + "/" + simulation_names[0] + "_" +
                          self.CALIB_TARGET + ".txt")
        model_results = np.zeros((self.collocation_points.shape[0], temp.shape[0])) # temp.shape=n_points
        for i, name in enumerate(simulation_names):
            model_results[i, :] = np.loadtxt(os.path.abspath(os.path.expanduser(self.RESULTS_DIR))+"/" + name + "_" +
                                             self.CALIB_TARGET + ".txt")[:, 1]

    def get_surrogate_prediction(self, model_results, number_of_points, prior=None):
        """
        TO-DO

        Create response surface: Computation of surrogate model prediction in MC points using gaussian processes
        Corresponds to PART 4 in original codes

        :param ndarray model_results: output of previous (full-complexity) model runs at collocation points of measurements
        :param int number_of_points: corresponds to observations["no of points"]
        :param ndarray prior: prior distribution of model results based on all calibration parameters (option for modular use)
        :return: tuple containing surrogate_prediction (nd.array), surrogate_std (nd.array)
        """
        if prior:
            self.prior_distribution = prior

        # initialize surrogate outputs
        surrogate_prediction = np.zeros((number_of_points, self.prior_distribution.shape[0]))
        surrogate_std = np.zeros((number_of_points, self.prior_distribution.shape[0]))

        # get length scales and their bounds
        length_scales = [np.nanmean(self.CALIB_PAR_SET[par]["bounds"]) for par in self.CALIB_PAR_SET.keys()]
        length_scale_bounds = [self.CALIB_PAR_SET[par]["bounds"] for par in self.CALIB_PAR_SET.keys()]

        logger.info("   -- creating and fitting GPE with RBF kernel...")
        kernel = RBF(
            length_scale=length_scales,  # original code: [0.05, 0.2, 150, 0.5],
            length_scale_bounds=length_scale_bounds  # [(0.001, 0.1), (0.001, 0.4), (5, 300), (0.02, 2)]
        )
        for par, model in enumerate(model_results.T):
            # construct square exponential, Radial-Basis Function kernel with means (lengths) and bounds of
            # calibration params, and multiply with variance of the model
            ikernel = kernel * np.var(model)
            # instantiate and fit GPR
            gp = GaussianProcessRegressor(kernel=ikernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
            gp.fit(self.collocation_points, model)
            surrogate_prediction[par, :], surrogate_std[par, :] = gp.predict(self.prior_distribution, return_std=True)
        return surrogate_prediction, surrogate_std

    @wraps(initialize_score_writing)
    def run_BAL(self, model_results, observations, prior=None):
        """ Bayesian iterations for updating the surrogate and calculating maximum likelihoods

        :param ndarray model_results: results of an initial full-complexity model run (before calling this function)
        :param ndarray observations: data stored in calibration_points
        :param ndarray prior: prior distributions of calibration parameters stored in columns (option for modular use)
        :return:
        """

        if prior:
            self.prior_distribution = prior

        for bal_step in range(0, self.IT_LIMIT):

            # 6.1 get response surface
            # Involves 4. and 11. (create and fit surrogate model with mc_samples points using a GPE)
            logger.info(" * retreaving surrogate predictions for BAL step {0}".format(str(bal_step)))
            surrogate_prediction, surrogate_std = self.get_surrogate_prediction(
                model_results=model_results,
                number_of_points=observations["no of points"]
            )

            # prepare 6.1.x: Read or compute the other errors to incorporate in the likelihood function
            loocv_error = np.loadtxt("loocv_error_variance.txt")[:, 1]
            total_error = (observations["observation error"] ** 2 + loocv_error) * 5

            # Part 6.1.x Bayesian inference: INSTANTIATE BAL object to evaluate initial surrogate model
            bal = BAL(observations=observations["observation"].T, error=total_error)
            # CHECK CONVERGENCE: Compute Bayesian scores of prior (in parameter space)
            logger.info(" * writing prior BME and RE for BAL step no. {0} to ".format(str(bal_step)) + self.RESULTS_DIR)
            # retrieve bayesian scores through rejection sampling
            self.BME[bal_step], self.RE[bal_step] = bal.compute_bayesian_scores(surrogate_prediction.T, method=self.score_method)
            try:
                self.bme_score_file.write("%s\n" % str(self.BME[bal_step]))
                self.re_score_file.write("%s\n" % str(self.RE[bal_step]))
            except AttributeError:
                logger.warn("NO BME / RE STORAGE FILE DEFINED - writing BME and RE to distinguished file in %s" % self.file_write_dir)
                np.savetxt(self.file_write_dir + "BMEprior_BALstep{0}.txt".format(str(bal_step)), self.BME)
                np.savetxt(self.file_write_dir + "REprior_BALstep{0}.txt".format(str(bal_step)), self.RE)

            # Bayesian active learning (in output space)
            # 6.2 extract surrogate predictions at n locations of interest for the number of observations
            # find where colloation points are not used in prior distribution
            none_use_idx = np.where((self.prior_distribution[:self.AL_SAMPLES+bal_step, :] == self.collocation_points[:, None]).all(-1))[1]
            # verify whether each element of the prior_distribution array is also present in none_use_idx
            idx = np.invert(np.in1d(
                np.arange(self.prior_distribution[:self.AL_SAMPLES+bal_step, :].shape[0]),
                none_use_idx
            ))
            # 6.3 get output space from priors
            al_unique_index = np.arange(self.prior_distribution[:self.AL_SAMPLES+bal_step, :].shape[0])[idx]

            for iAL, vAL in enumerate(al_unique_index):
                # 6.3 and 6.4: explore output subspace associated with a defined prior combination
                al_exploration = np.random.normal(
                    size=(self.MC_SAMPLES_AL, observations["no of points"])
                ) * surrogate_std[:, vAL] + surrogate_prediction[:, vAL]
                # 6.5 and 6.6: BAL scores computation through Bayesian weighting or rejection sampling
                self.al_BME[iAL], self.al_RE[iAL] = bal.compute_bayesian_scores(
                    al_exploration,
                    self.score_method
                )

            # part 7. selection criteria for next collocation point
            al_value, al_value_index = bal.selection_criteria(self.AL_STRATEGY, self.al_BME, self.al_RE)

            # part 9. Selection of new collocation points
            self.collocation_points = np.vstack(
                (self.collocation_points, self.prior_distribution[al_unique_index[al_value_index], :])
            )

            # Part 10. Computation of the numerical model in the newly defined collocation point
            # Update steering files
            update_steering_file(
                self.collocation_points[-1, :],
                list(self.CALIB_PAR_SET.keys()),
                self.CALIB_ID_PAR_SET[list(self.CALIB_ID_PAR_SET.keys()[0])]["classes"],
                list(self.CALIB_ID_PAR_SET.keys()[0]),
                self.n_simulation + 1 + bal_step
            )

            # Run telemac
            run_telemac(self.TM_CAS, self.N_CPUS)

            # Extract values of interest
            # RESULT_NAME now manages in telemac_core.update_steering file
            updated_string = RESULT_NAME_GAIA[1:] + str(self.n_simulation+1+bal_step) + ".slf"
            save_name = self.RESULTS_DIR + "/PC" + str(self.n_simulation+1+bal_step) + "_" + self.CALIB_TARGET + ".txt"
            results = get_variable_value(updated_string, self.CALIB_TARGET, observations["node IDs"], save_name)
            model_results = np.vstack((model_results, results[:, 1].T))

            # Move the created files to their respective folders
            shutil.move(RESULT_NAME_GAIA[1:] + str(self.n_simulation+1+bal_step) + ".slf", self.SIM_DIR)
            shutil.move(RESULT_NAME_TM[1:] + str(self.n_simulation+1+bal_step) + ".slf", self.SIM_DIR)

            # Append the parameter used to a file
            new_line = "; ".join(map("{:.3f}".format, self.collocation_points[-1, :]))
            new_line = "PC" + str(self.n_simulation+1+bal_step) + "; " + new_line
            append_new_line(self.RESULTS_DIR + "/parameter_file.txt", new_line)

            # Progress report
            print("Bayesian iteration: " + str(bal_step + 1) + "/" + str(self.IT_LIMIT))

    def sample_collocation_points(self, method="uniform"):
        """Sample initial collocation points

        :param str method: experimental design method for sampling initial collocation points
                            default is 'uniform'; other options
                            (NOT YET IMPLEMENTED)
        """
        collocation_points = np.zeros((self.init_runs, self.n_calib_pars))
        # assign minimum and maximum values of parameters to the first two tests
        par_minima = []
        par_maxima = []
        for par in self.CALIB_PAR_SET.keys():
            par_minima.append(self.CALIB_PAR_SET[par]["bounds"][0])
            par_maxima.append(self.CALIB_PAR_SET[par]["bounds"][1])
        collocation_points[:, 0] = np.array(par_minima)
        collocation_points[:, 1] = np.array(par_maxima)
        #collocation_points[:, 0] = np.random.uniform(-5, 5, n_cp)
        #collocation_points[:, 1] = np.random.uniform(-5, 5, n_cp)

    def __call__(self, *args, **kwargs):
        # no effective action: print class in system
        print("Class Info: <type> = BAL_GPE (%s)" % os.path.dirname(__file__))
        print(dir(self))

""" part 10 only used with earlier versiions
# Part 10. Compute solution in final time step --------------------------------------------------------------------
surrogate_prediction = np.zeros((observations["no of points"], prior_distribution.shape[0]))
surrogate_std = np.zeros((observations["no of points"], prior_distribution.shape[0]))
for i, model in enumerate(model_results.T):
    kernel = RBF(length_scale=[1, 1], length_scale_bounds=[(0.01, 20), (0.01, 20)]) * np.var(model)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(collocation_points, model)
    surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior_distribution, return_std=True)

likelihood_final = compute_likelihood(surrogate_prediction.T, observations["observation"].T, total_error)
"""

"""plot options for graphing (implement later)
# Save final results of surrogate model to graph them later
graph_likelihood_surrogates = np.zeros((prior_distribution.shape[0], 3))
graph_likelihood_surrogates[:, :2] = prior_distribution
graph_likelihood_surrogates[:, 2] = likelihood_final
graph_list.append(np.copy(graph_likelihood_surrogates))
graph_name.append("iteration: " + str(IT_LIMIT))


# Plot comparison between surrogate model and reference solution
plot_likelihoods(graph_list, graph_name)
x=1
"""
