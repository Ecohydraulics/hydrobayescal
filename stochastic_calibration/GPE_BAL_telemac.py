"""
Stochastic calibration of a Telemac2d hydro-morphodynamic model using
Surrogate-Assisted Bayesian inversion. The surrogate model is created using
Gaussian Process Regression

Method adapt: Oladyshkin et al. (2020). Bayesian Active Learning for the Gaussian Process
Emulator Using Information Theory. Entropy, 22(8), 890.
"""
import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from BAL_fun import *
from telemac_fun import *
from usr_defs import UserDefs
from surrogate import *


class BAL_GPE(UserDefs):
    """
    The BAL_GPE object is the framework for running a stochastic calibration of a deterministic model by using a
        Gaussian process emulator (GPE) - based surrogate model that is fitted through Bayesian active learning (BAL).
        The deterministic full-complexity model is defined through a coupled software (default TELEMAC).

    .. important::

        The object must be instantiated with metadata stored in the user-input.xlsx file corresponding to the software
        used. Visit https://stochastic-calibration.readthedocs.io to learn more.
    """
    def __init__(self, input_worbook_name="user-input.xlsx", software_coupling="telemac", *args, **kwargs):
        """
        Initializer of the BAL_GPE class.

        :param str input_worbook_name: name of the user-input.xlsx file including its full directory
        :param software_coupling: name of the software of the deterministic model (default: "telemac")
        :param args: placeholder for consistency
        :param kwargs: placeholder for consistency
        """
        UserDefs.__init__(self, input_worbook_name)
        self.write_global_settings(self.input_xlsx_name)
        print("Successfully instantiated a BAL-GPE object for calibrating a TELEMAC model.")

        # global surrogate and active learning variables
        BME = np.zeros((self.IT_LIMIT, 1))
        RE = np.zeros((self.IT_LIMIT, 1))
        al_BME = np.zeros((self.AL_SAMPLES, 1))
        al_RE = np.zeros((self.AL_SAMPLES, 1))


    def run_calibration(self):
        """loads provided input file name as pandas dataframe

            Args:
                file_name (str): name of input file (default is user-input.xlsx)

            Returns:
                tbd
        """

        self.update_distributions()
        self.update_al_variables()
        self.load_observations()

    def update_distributions(self):
        """Calculate uniform distributions for all user-defined parameters

        :return numpy.ndarray prior_distribution: copy of all uniform input distributions of the calibration parameters
        """
        global CALIB_PAR_SET
        self.prior_distribution = np.zeros((MC_SAMPLES, len(CALIB_PAR_SET)))
        column = 0
        for par in CALIB_PAR_SET.keys():
            print(" * drawing {0} uniform random samples for {1}".format(par, str(MC_SAMPLES)))
            CALIB_PAR_SET[par]["distribution"] = np.random.uniform(
                CALIB_PAR_SET[par]["bounds"][0],
                CALIB_PAR_SET[par]["bounds"][1],
                MC_SAMPLES
            )
            self.prior_distribution[:, column] = np.copy(CALIB_PAR_SET[par]["distribution"])
            column += 1

    def update_al_variables(self):
        """update the global AL variables with user input"""
        print(" *  updating (re-initializing) active learning variables...")
        global BME
        global RE
        global al_BME
        global al_RE
        BME = np.zeros((IT_LIMIT, 1))
        RE = np.zeros((IT_LIMIT, 1))
        al_BME = np.zeros((AL_SAMPLES, 1))
        al_RE = np.zeros((AL_SAMPLES, 1))


    def load_observations(self):
        """Load observations stored in calibration_points.csv to self.observations
        """
        print(" * reading observations file (%s)..." % CALIB_PTS)
        observation_file = np.loadtxt(CALIB_PTS, delimiter=",")
        self.observations= {
            "no of points": observation_file.shape[0],
            "node IDs": observation_file[:, 0].reshape(-1, 1),
            "observation": observation_file[:, 1].reshape(-1, 1),
            "observation error": observation_file[:, 2]
        }

    def prepare_initial_model(self):
        """
        TO-DO
        """
        # Part 2. Read initial collocation points  ----------------------------------------------------------------------------
        temp = np.loadtxt(os.path.abspath(os.path.expanduser(RESULTS_DIR))+"/parameter_file.txt", dtype=str, delimiter=";")
        simulation_names = temp[:, 0]
        collocation_points = temp[:, 1:].astype(float)
        n_simulation = collocation_points.shape[0]

        # Part 3. Read the previously computed simulations of the numerical model in the initial collocation points -----------
        temp = np.loadtxt(os.path.abspath(os.path.expanduser(RESULTS_DIR)) + "/" + simulation_names[0] + "_" +
                          CALIB_TARGET + ".txt")
        model_results = np.zeros((collocation_points.shape[0], temp.shape[0]))
        for i, name in enumerate(simulation_names):
            model_results[i, :] = np.loadtxt(os.path.abspath(os.path.expanduser(RESULTS_DIR))+"/" + name + "_" +
                                             CALIB_TARGET + ".txt")[:, 1]


    def get_surrogate_prediction(self, model_results, number_of_points, prior_distribution):
        """
        TO-DO

        Computation of surrogate model prediction in MC points using gaussian processes
        Corresponds to PART 4 in original codes

        :param ndarray model_results: output of previous (full-complexity) model runs at collocation points of measurements
        :param int number_of_points: corresponds to observations["no of points"]
        :param ndarray prior_distribution: prior distribution of model results based on all calibration parameters
        :return: tuple containing surrogate_prediction (nd.array), surrogate_std (nd.array)
        """
        # initialize surrogate outputs
        surrogate_prediction = np.zeros((number_of_points, prior_distribution.shape[0]))
        surrogate_std = np.zeros((number_of_points, prior_distribution.shape[0]))

        for par, model in enumerate(model_results.T):
            # construct square exponential, Radial-Basis Function kernel with means (lengths) and bounds of
            # calibration params, and multiply with variance of the model
            kernel = RBF(
                length_scale=[0.05, 0.2, 150, 0.5],
                length_scale_bounds=[(0.001, 0.1), (0.001, 0.4), (5, 300), (0.02, 2)]
            ) * np.var(model)
            # instantiate and fit GPR
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
            gp.fit(collocation_points, model)
            surrogate_prediction[par, :], surrogate_std[par, :] = gp.predict(prior_distribution, return_std=True)
        return surrogate_prediction, surrogate_std


    def run_BAL(model_results, observations, prior_distribution):
        """ Bayesian iterations for updating the surrogate and calculating maximum likelihoods

        :param ndarray model_results: results of the full-complexity model responses need to be run prior to this function
        :param ndarray observations: data stored in calibration_points
        :param ndarray prior_distribution: prior distributions of calibration parameters stored in columns
        :return:
        """
        global IT_LIMIT
        global MC_SAMPLES_AL

        for bal_step in range(0, IT_LIMIT):
            # Part 4. Computation of surrogate model prediction in MC points using gaussian processes
            surrogate_prediction, surrogate_std = get_surrogate_prediction(
                model_results=model_results,
                number_of_points=observations["no of points"],
                prior_distribution=prior_distribution
            )

            # Part 5. Read or compute the other errors to incorporate in the likelihood function
            loocv_error = np.loadtxt("loocv_error_variance.txt")[:, 1]
            total_error = (observations["observation error"] ** 2 + loocv_error) * 5

            # Part 6. Compute Bayesian scores (in parameter space)
            BME[iter], RE[iter] = compute_bayesian_scores(surrogate_prediction.T, observations["observation"].T, total_error)
            np.savetxt("BME.txt", BME)
            np.savetxt("RE.txt", RE)


            # Part 7. Bayesian active learning (in output space)
            # Index of the elements of the prior distribution that have not been used as collocation points
            # ?means: get test points?
            aux1 = np.where((prior_distribution[:AL_SAMPLES+bal_step, :] == collocation_points[:, None]).all(-1))[1]
            aux2 = np.invert(np.in1d(np.arange(prior_distribution[:AL_SAMPLES+bal_step, :].shape[0]), aux1))
            al_unique_index = np.arange(prior_distribution[:AL_SAMPLES+bal_step, :].shape[0])[aux2]

            for iAL, vAL in enumerate(al_unique_index):
                # Exploration of output subspace associated with a defined prior combination.
                al_exploration = np.random.normal(size=(MC_SAMPLES_AL, observations["no of points"]))*surrogate_std[:, vAL] + surrogate_prediction[:, vAL]
                # BAL scores computation
                al_BME[iAL], al_RE[iAL] = compute_bayesian_scores(al_exploration, observations["observation"].T, total_error, AL_STRATEGY)

            # Part 8. Selection criteria for next collocation point ------------------------------------------------------
            al_value, al_value_index = BAL_selection_criteria(AL_STRATEGY, al_BME, al_RE)

            # Part 9. Selection of new collocation point
            collocation_points = np.vstack((collocation_points, prior_distribution[al_unique_index[al_value_index], :]))

            # Part 10. Computation of the numerical model in the newly defined collocation point --------------------------
            # Update steering files
            update_steering_file(collocation_points[-1, :], CALIB_PARAMETERS, CALIB_ID_PAR_SET[list(CALIB_ID_PAR_SET.keys()[0])]["classes"], list(CALIB_ID_PAR_SET.keys()[0]), GAIA_CAS,
                                 TM_CAS, RESULT_NAME_GAIA, RESULT_NAME_TM, n_simulation + 1 + bal_step)
            # Run telemac
            run_telemac(TM_CAS, N_CPUS)

            # Extract values of interest
            updated_string = RESULT_NAME_GAIA[1:] + str(n_simulation+1+bal_step) + ".slf"
            save_name = RESULTS_DIR + "/PC" + str(n_simulation+1+bal_step) + "_" + CALIB_TARGET + ".txt"
            results = get_variable_value(updated_string, CALIB_TARGET, observations["node IDs"], save_name)
            model_results = np.vstack((model_results, results[:, 1].T))

            # Move the created files to their respective folders
            shutil.move(RESULT_NAME_GAIA[1:] + str(n_simulation+1+bal_step) + ".slf", SIM_DIR)
            shutil.move(RESULT_NAME_TM[1:] + str(n_simulation+1+bal_step) + ".slf", SIM_DIR)

            # Append the parameter used to a file
            new_line = "; ".join(map("{:.3f}".format, collocation_points[-1, :]))
            new_line = "PC" + str(n_simulation+1+bal_step) + "; " + new_line
            append_new_line(RESULTS_DIR + "/parameter_file.txt", new_line)

            # Progress report
            print("Bayesian iteration: " + str(bal_step+1) + "/" + str(IT_LIMIT))


    def __call__(self, *args, **kwargs):
        # example prints class structure information to console
        print("Class Info: <type> = NewClass (%s)" % os.path.dirname(__file__))
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
