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
from datetime import datetime


# TODO: there is a log_actions wrapper function in the function pool - use this instead!
from src.hydroBayesCal.function_pool import *


# SETUP DIRECTORIES OR GLOBAL VARIABLES?
# TODO: GLOBAL VARIABLES SHOULD BE DEFINED IN config.py
base_dir = Path(__file__).resolve().parent.parent.parent
print("Base directory:", base_dir)
env_script_path = base_dir / 'env-scripts'

# TODO: Logging should be implemented as a wrapper that only wraps the head functions/methods; I exemplarily added a
#           logging wrapper to the run() method. For example, check the log_actions wrapper for process_adv_files in
#           https://github.com/sschwindt/TKEanalyst/blob/main/TKEanalyst/profile_analyst.py
#           Use the log_actions wrapper from utils.function_pool!


class HydroSimulations:
    def __init__(
            self,
            control_file="control.file",
            model_dir="",
            res_dir='',
            calibration_pts_file_path=None,
            n_cpus=int(),
            init_runs=int(),
            calibration_parameters=None,
            param_values=None,
            calibration_quantities=None,
            dict_output_name='',
            parameter_sampling_method='',
            max_runs=int(),
            complete_bal_mode=True,
            only_bal_mode=False,
            *args,
            **kwargs,
            # TODO: The doctrings list many parameters that are not included here, like model_dir (see the PyCharm warnings)
    ):
        """

        Constructor of HydroSimulation class to manage and run any hydrodynamic simulation within the context of Bayesian Calibration
        using a Gaussian Process Emulator (GPE). The class is designed to handle simulation setup, execution, and result
        storage while managing calibration parameters and Bayesian Active Learning (BAL) iterations.

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
        #self.user_inputs = user_inputs

        # TODO: the following lines come from FullComplexityModel and require integration into your scheme
        self.model_dir = model_dir
        self.res_dir = res_dir
        self.control_file = control_file
        self.calibration_pts_file_path = calibration_pts_file_path
        self.nproc = n_cpus
        self.param_values = param_values
        self.calibration_quantities = calibration_quantities
        self.calibration_parameters = calibration_parameters
        self.dict_output_name = dict_output_name
        self.parameter_sampling_method = parameter_sampling_method
        self.init_runs=init_runs
        self.max_runs = max_runs
        self.complete_bal_mode = complete_bal_mode
        self.only_bal_mode = only_bal_mode
        self.asr_dir = os.path.join(res_dir, "auto-saved-results")
        self.nloc = None
        self.ndim = None
        self.param_dic = None
        self.num_quantities = None
        self.observations = None
        self.measurement_errors = None
        self.calibration_pts_df = None

        if calibration_parameters:
            self.param_dic,self.ndim = self.set_calibration_parameters(calibration_parameters,param_values)
        self.supervisor_dir = os.getcwd()  # preserve directory of code that is controlling the full complexity model
        if calibration_pts_file_path:
            self.observations,self.measurement_errors,self.nloc,self.num_quantities,self.calibration_pts_df  = self.set_observations_and_errors(calibration_pts_file_path,calibration_quantities)
        #self.check_inputs()
        if not os.path.exists(self.asr_dir):
            os.makedirs(self.asr_dir)
        if not os.path.exists(os.path.join(self.asr_dir, "plots")):
            os.makedirs(os.path.join(self.asr_dir, "plots"))
        if not os.path.exists(os.path.join(self.asr_dir, "surrogate-gpe")):
            os.makedirs(os.path.join(self.asr_dir, "surrogate-gpe"))
        self.model_evaluations = None
    def check_inputs(
            self,
            user_inputs
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

    #     TODO: A "tm_simulations" method should not be in the HydroSimulations class,
    #         because it is specific to Telemac. Please rename this to simulations or what-
    #         ever generic, model-independent name works.

    #     # TODO delete the above line. In this structure, this is all you need!

    @log_actions
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
        self.model_evaluations = np.empty((2, 2))

        return self.model_evaluations
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
        start_time = datetime.now()
        print("DUMMY CALL")
        # implement call to run the model from command line, for example:
        # call_subroutine("openTeleFoam " + self.control_file)
        print("Full-complexity simulation time: " + str(datetime.now() - start_time))
        pass

    def set_observations_and_errors(self,calibration_pts_file_path, calibration_quantities):
        """
        TODO: add docstrings
        :param calib_pts_file_path:
        :param num_quantities:
        :return:
        """
        n_calib_quantities = len(calibration_quantities)
        calibration_pts_df = pd.read_csv(calibration_pts_file_path)
        # Calculate the column indices for observations dynamically (starting from the 3rd column)
        observation_indices = [2 * i + 3 for i in range(n_calib_quantities)]
        # Calculate the column indices for errors dynamically (starting from the 4th column)
        error_indices = [2 * i + 4 for i in range(n_calib_quantities)]
        # Select the observation columns and convert them to a NumPy array
        if n_calib_quantities == 1:
            observations = calibration_pts_df.iloc[:, observation_indices].to_numpy().reshape(1, -1)
            measurement_errors = calibration_pts_df.iloc[:, error_indices].to_numpy().flatten()
            n_loc = observations.size
        else:
            observations = calibration_pts_df.iloc[:, observation_indices].to_numpy().transpose().ravel().reshape(1, -1)
            measurement_errors = calibration_pts_df.iloc[:, error_indices].to_numpy().transpose().ravel()
            n_loc = int(observations.size / n_calib_quantities)

        return observations, measurement_errors,n_loc,n_calib_quantities,calibration_pts_df

    @staticmethod
    def read_data(file_path):
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


    #     TODO: Merge with run() method - merged with run method

    @log_actions
    def run_bal_simulations(self, collocation_points, bal_iteration, bal_new_set_parameters,complete_bal_mode):
        """
        TODO: Add docstrings
        :param collocation_points:
        :param bal_iteration:
        :param bal_new_set_parameters:
        :param complete_bal_mode:
        :return:
        """
        if bal_new_set_parameters is None:
            logger.info("Running full complexity models with initial collocation points:")
            return self.run_multiple_simulations(collocation_points=collocation_points, bal_iteration=bal_iteration,
                                       complete_bal_mode=complete_bal_mode)
        elif bal_new_set_parameters is not None:
            return self.run_multiple_simulations(bal_iteration=bal_iteration, bal_new_set_parameters=bal_new_set_parameters,
                                       complete_bal_mode=complete_bal_mode)
        else:
            raise ValueError("Error: At least one of 'collocation_points' or 'bal_new_set_parameters' must be provided.")

    def set_calibration_parameters(self,params, values):
        """
        Create a dictionary from calibration parameters and their value ranges.

        :param params: List of parameter names.
        :param values: List of value ranges corresponding to the parameter names.
        :return: Dictionary with parameter names as keys and value ranges as values.
        """
        if len(params) != len(values):
            logger_error.error("Mismatch between the number of parameters (%d) and values (%d)", len(params),
                               len(values))
            raise ValueError("The number of parameters and values must be the same.")
        param_dict = dict(zip(params, values))
        ndim = len(params)
        return param_dict,ndim

    def update_model_controls(
            self,
            collocation_point_values,
            calibration_parameters,
            auxiliary_file_path,
            simulation_id=0,
    ):
        """
        TODO: Merge this method from FulComplexityModel with your workflow - Done
        Update the model control files specifically for Bayesian calibration.

        :param dict new_parameter_values: provide a new parameter value for every calibration parameter
                    * keys correspond to Telemac or Gaia keywords in the steering file
                    * values are either scalar or list-like numpy arrays
        :param int simulation_id: optionally set an identifier for a simulation (default is 0)
        :return:
        """

        pass
    def output_processing(
            self,
            output_data = '',
            #complete_bal_mode=True
    ):
        """

        Retrieves data from the output data file saved as .json file

        :param args:
        :param kwargs:
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
