"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
import os
from pathlib import Path
import pandas as pd
import pickle
import numpy as np


# TODO: there is a log_actions wrapper function in the function pool - use this instead!
#from config_logging import *


# SETUP DIRECTORIES OR GLOBAL VARIABLES?
# TODO: GLOBAL VARIABLES SHOULD BE DEFINED IN config.py
base_dir = Path(__file__).resolve().parent.parent.parent
print("Base directory:", base_dir)
env_script_path = base_dir / 'env-scripts'

# TODO: Logging should be implemented as a wrapper that only wraps the head functions/methods; I exemplarily added a
#           logging wrapper to the run() method. For example, check the log_actions wrapper for process_adv_files in
#           https://github.com/sschwindt/TKEanalyst/blob/main/TKEanalyst/profile_analyst.py
#           Use the log_actions wrapper from utils.function_pool!
setup_logging()
logger_info = logging.getLogger("HydroBayesCal")


class HydroSimulations:
    def __init__(
            self,
            user_inputs=None,
            model_dir="",
            res_dir="",
            calibration_parameters=None,
            control_file="control.file",
            init_runs=int(),
            *args,
            **kwargs
            # TODO: The doctrings list many parameters that are not included here, like model_dir (see the PyCharm warnings)
    ):
        """
        Constructor for the HydroSimulations Class. Wraps functions for running Telemac and OpenFoam
        in the context of Bayesian Calibration using Gaussian Process Emulator (GPE).

        Parameters
        ----------
        model_dir : str
            Full complexity model directory.
        res_dir : str
            Directory of the folder where a subfolder called "auto-saved-results" will be created to store all the
            results files.
        control_file : str
            Name of the file that controls the full complexity model simulation (to be called from a terminal).
        calibration_parameters : list, optional
            Names of the considered calibration parameters (maximum 4 parameters).
        bal_mode : bool, optional
            Default is True. If True, runs Bayesian Active Learning; if False, only runs the initial collocation points.
        n_max_tp : int, optional
            Total number of model simulations, including Bayesian Active Learning iterations.
        init_runs : int, optional
            Initial runs of the full complexity model (initial surrogate model before Bayesian Active Learning).
        user_inputs : dict, optional
            User input parameters.

        Attributes
        ----------
        user_inputs : dict
            User input parameters.
        complete_bal_mode : bool
            Mode for Bayesian Active Learning. Default: True
        n_max_tp : int
            Total number of model simulations, including Bayesian Active Learning iterations.
        init_runs : int
            Initial runs of the full complexity model.
        """
        self.user_inputs = user_inputs
        self.check_inputs()
        self.model_evaluations = None
        self.observations = None
        self.measurement_errors = None
        # TODO: the following lines come from FullComplexityModel and require integration into your scheme
        self.model_dir = model_dir
        self.control_file = control_file
        self.collocation_file = "calibration-par-combinations.csv"
        self.res_dir = res_dir
        self.init_runs=init_runs
        if not os.path.exists(res_dir + os.sep + "auto-saved-results"):
            os.makedirs(res_dir + os.sep + "auto-saved-results")
        if not os.path.exists(res_dir + os.sep + "auto-saved-results" + os.sep + "plots"):
            os.makedirs(res_dir + os.sep + "auto-saved-results" + os.sep + "plots")
        self.calibration_parameters = False
        if calibration_parameters:
            self.set_calibration_parameters(calibration_parameters)
        self.supervisor_dir = os.getcwd()  # preserve directory of code that is controlling the full complexity model

    def check_inputs(
            self
    ):
        """
        TODO: add docstrings to explain what this generic method should accomplish for any code (Telemac, OF, Basement, whatever)

        :return:
        """
        # TODO: make this a generic method
        pass
        # TelemacModel.check_tm_inputs(self.user_inputs)

    def extract_data_point(
            self,
            *args,
            **kwargs
    ):
        """
        TODO: add docstrings to explain what this generic method should accomplish for any code (Telemac, OF, Basement, whatever)

        :param args:
        :param kwargs:
        :return:
        """
        pass

    def run_simulation(
            self,
            collocation_points=None,
            bal_iteration=int(),
            bal_new_set_parameters=None,
            complete_bal_mode=None
        ):
        """
        TODO: A "tm_simulations" method should not be in the HydroSimulations class,
            because it is specific to Telemac. Please rename this to simulations or what-
            ever generic, model-independent name works.
        """
        # control_tm = TelemacModel(
        #     model_dir=self.user_inputs['model_simulation_path'],
        #     res_dir=self.user_inputs['results_folder_path'],
        #     control_file=self.user_inputs['control_file_name'],
        #     friction_file=self.user_inputs['friction_file'],
        #     calibration_parameters=self.user_inputs['calibration_parameters'],
        #     calibration_pts_file_path=self.user_inputs['calib_pts_file_path'],
        #     calibration_quantities=self.user_inputs['calibration_quantities'],
        #     tm_xd=self.user_inputs['Telemac_solver'],
        #     n_processors=self.user_inputs['n_cpus'],
        #     dict_output_name=self.user_inputs['dict_output_name'],
        #     results_filename_base=self.user_inputs['results_filename_base'],
        #     init_runs=self.user_inputs['init_runs'],
        #
        # )
        #
        # control_tm.run_multiple_simulations(
        #     collocation_points,
        #     bal_new_set_parameters,
        #     bal_iteration,
        #     complete_bal_mode
        # )
        # TODO delete the above line. In this structure, this is all you need!
        self.model_evaluations = np.empty((2, 2))

        return self.model_evaluations

    def run_multiple_simulations(
            self,
            collocation_points=None,
            bal_new_set_parameters=None,
            bal_iteration=int(),
            complete_bal_mode=True,
    ):
        """
        TODO: add docstrings to explain what this generic method should accomplish for any code (Telemac, OF, Basement, whatever)
        :param collocation_points:
        :param bal_new_set_parameters:
        :param bal_iteration:
        :param complete_bal_mode:
        :return:
        """
        pass

    def run_single_simulation(
            self,
            filename="run_launcher.py",
            load_results=True
    ):
        """
        TODO: add docstrings to explain what this generic method should accomplish for any code (Telemac, OF, Basement, whatever)

        :param filename:
        :param load_results:
        :return:
        """
        pass

    def get_observations_and_errors(self, calib_pts_file_path, num_quantities):
        """
        TODO: add docstrings
        :param calib_pts_file_path:
        :param num_quantities:
        :return:
        """
        calibration_pts_df = pd.read_csv(calib_pts_file_path)
        # Calculate the column indices for observations dynamically (starting from the 3rd column)
        observation_indices = [2 * i + 3 for i in range(num_quantities)]
        # Calculate the column indices for errors dynamically (starting from the 4th column)
        error_indices = [2 * i + 4 for i in range(num_quantities)]
        # Select the observation columns and convert them to a NumPy array
        if num_quantities == 1:
            self.observations = calibration_pts_df.iloc[:, observation_indices].to_numpy().reshape(1, -1)
            self.measurement_errors = calibration_pts_df.iloc[:, error_indices].to_numpy().flatten()
        else:
            self.observations = calibration_pts_df.iloc[:, observation_indices].to_numpy().transpose().ravel().reshape(1, -1)
            self.measurement_errors = calibration_pts_df.iloc[:, error_indices].to_numpy().transpose().ravel()

            # Select the error columns and convert them to a NumPy array
            # error_columns = [calibration_pts_df.iloc[:, idx].to_numpy() for idx in error_indices]
            #
            # # Stack error columns horizontally
            # self.measurement_errors = np.hstack([col.reshape(-1, 1) for col in error_columns])

        return self.observations, self.measurement_errors

    def read_stored_data(self, file_path):
        """
        TODO: Add docstrings; no use pickle formats (i.e., no npy, pkl, nor pickle)
        :param file_path:
        :return:
        """
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
            elif file_path.endswith('.pickle'):
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
            else:
                raise ValueError("Unsupported file type. Only .npy and .pkl files are supported.")
            return data
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    def run_simulation(self):
        """
        Run a full-complexity model simulation
        TODO: Merge with run() method
        :return None:
        """
        start_time = datetime.now()
        print("DUMMY CALL")
        # implement call to run the model from command line, for example:
        # call_subroutine("openTeleFoam " + self.control_file)
        print("Full-complexity simulation time: " + str(datetime.now() - start_time))


    @logging
    def run(self, collocation_points, bal_iteration, bal_new_set_parameters,complete_bal_mode):
        """
        TODO: Add docstrings
        :param collocation_points:
        :param bal_iteration:
        :param bal_new_set_parameters:
        :param complete_bal_mode:
        :return:
        """
        if collocation_points is not None:
            logger_info.info("Running full complexity models with initial collocation points:")
            return self.tm_simulations(collocation_points=collocation_points, bal_iteration=bal_iteration,
                                       complete_bal_mode=complete_bal_mode)
        elif bal_new_set_parameters is not None:
            return self.tm_simulations(bal_iteration=bal_iteration, bal_new_set_parameters=bal_new_set_parameters,
                                       complete_bal_mode=complete_bal_mode)
        else:
            raise ValueError("Error: At least one of 'collocation_points' or 'bal_new_set_parameters' must be provided.")


    def set_calibration_parameters(self, names):
        """
        TODO: Merge this method from FulComplexityModel with your workflow
        :param names:
        :return:
        """
        self.calibration_parameters = names
        # for par in value:
        #     self.calibration_parameters.update({par: {"current value": _np.nan}})

    def update_model_controls(
            self,
            new_parameter_values,
            simulation_id=0,
    ):
        """
        TODO: Merge this method from FulComplexityModel with your workflow
        Update the model control files specifically for Bayesian calibration.

        :param dict new_parameter_values: provide a new parameter value for every calibration parameter
                    * keys correspond to Telemac or Gaia keywords in the steering file
                    * values are either scalar or list-like numpy arrays
        :param int simulation_id: optionally set an identifier for a simulation (default is 0)
        :return:
        """

        pass

    def __call__(self, *args, **kwargs):
        """
        TODO: Merge this method from FulComplexityModel with your workflow, making sure there is no __main___ statement at the bottom of the file

        Call method forwards to self.run_simulation()

        :param args:
        :param kwargs:
        :return:
        """
        self.run_simulation()

# if __name__ == "__main__":
#     # TODO: This namespace should not be within a package script. Please remove this main statement completely.
#     # TODO: This is typically in the __call__ magic method, that I now ported here.
#     full_complexity_simulation = HydroSimulations()
#     full_complexity_simulation.run()
