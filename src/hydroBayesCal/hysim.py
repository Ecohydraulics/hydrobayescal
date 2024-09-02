"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# TODO: there is a log_actions wrapper function in the function pool - use this instead!
from src.hydroBayesCal.function_pool import *


# SETUP DIRECTORIES OR GLOBAL VARIABLES?
# TODO: GLOBAL VARIABLES SHOULD BE DEFINED IN config.py


# TODO: Logging should be implemented as a wrapper that only wraps the head functions/methods; I exemplarily added a
#           logging wrapper to the run() method. For example, check the log_actions wrapper for process_adv_files in
#           https://github.com/sschwindt/TKEanalyst/blob/main/TKEanalyst/profile_analyst.py
#           Use the log_actions wrapper from utils.function_pool!


class HydroSimulations:
    def __init__(
            self,
            control_file="control.file",
            model_dir="",
            res_dir="",
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
            check_inputs=True,
            delete_complex_outputs=False,
            validation=False,
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

        # TODO: the following lines come from FullComplexityModel and require integration into your scheme
        if check_inputs:
            self.check_inputs(model_dir=model_dir,
                              res_dir=res_dir,
                              control_file=control_file,
                              init_runs=init_runs,
                              nproc=n_cpus,
                              max_runs=max_runs,
                              calibration_parameters=calibration_parameters,
                              param_values=param_values,
                              calibration_quantities=calibration_quantities
                              )
        self.model_dir = model_dir
        self.res_dir = res_dir
        self.control_file = control_file
        self.calibration_pts_file_path = calibration_pts_file_path
        self.nproc = n_cpus
        self.param_values = param_values
        self.calibration_quantities = calibration_quantities
        self.calibration_parameters = calibration_parameters
        self.parameter_sampling_method = parameter_sampling_method
        self.init_runs = init_runs
        self.max_runs = max_runs
        self.complete_bal_mode = complete_bal_mode
        self.only_bal_mode = only_bal_mode
        self.delete_complex_outputs=delete_complex_outputs
        self.validation=validation
        if self.validation:
            self.dict_output_name = dict_output_name + "-validation"
        else:
            self.dict_output_name = dict_output_name
        self.nloc = None
        self.ndim = None
        self.param_dic = None
        self.num_quantities = None
        self.observations = None
        self.measurement_errors = None
        self.calibration_pts_df = None

        if calibration_parameters:
            self.param_dic, self.ndim = self.set_calibration_parameters(calibration_parameters, param_values)
        self.supervisor_dir = os.getcwd()  # preserve directory of code that is controlling the full complexity model
        if calibration_pts_file_path:
            self.observations, self.measurement_errors, self.nloc, self.num_quantities, self.calibration_pts_df = self.set_observations_and_errors(
                calibration_pts_file_path, calibration_quantities)

        self.asr_dir = os.path.join(res_dir,
                                    f"auto-saved-results-{self.num_quantities}-quantities_{'_'.join(self.calibration_quantities)}")
        if not os.path.exists(self.asr_dir):
            os.makedirs(self.asr_dir)
        if not os.path.exists(os.path.join(self.asr_dir, "plots")):
            os.makedirs(os.path.join(self.asr_dir, "plots"))
        if not os.path.exists(os.path.join(self.asr_dir, "surrogate-gpe")):
            os.makedirs(os.path.join(self.asr_dir, "surrogate-gpe"))
        self.model_evaluations = None

    def check_inputs(
            self,
            model_dir,
            res_dir,
            control_file,
            init_runs,
            nproc,
            max_runs,
            calibration_parameters,
            param_values,
            calibration_quantities,
    ):
        """
        Validate input parameters, paths, calibration parameters, and calibration
        quantities for the model and results directories.

        Parameters
        ----------
        model_dir : str
            Full directory path where the model files are located.
        res_dir : str
            Directory path where the results will be stored. This directory should exist and be accessible.
        control_file : str
            Name of the control file to be checked within the model_dir.
        init_runs : int
            Number of initial runs to be performed. This must be an integer.
        nproc : int
            Number of processors to be used. This must be an integer.
        max_runs : int
            Maximum number of runs allowed. This must be an integer and should be greater than init_runs.
        calibration_parameters : list
            Calibration parameters that will be used in the model. This should be a list of parameter names or identifiers.
        param_values : list
            Ranges of min and max values corresponding to the calibration parameters. This should be a list and must have the same length as calibration_parameters.
        calibration_quantities : list
             Model output target quantities used for calibration. This should be a list of strings and can have a maximum of 2 elements.
        Returns
        --------
        None
        """
        # Check control_file
        control_file_path = os.path.join(model_dir, control_file)
        if os.path.exists(control_file_path):
            print(f"Control file exists: {control_file_path}")
        else:
            raise ValueError(f"Control file does not exist: {control_file_path}")

        # Check res_dir
        if os.path.isdir(res_dir):
            print(f"Results directory exists: {res_dir}")
        else:
            raise ValueError(f"Results directory does not exist: {res_dir}")

        # Check nproc
        if not isinstance(nproc, int):
            raise TypeError("Number of processors (nproc) should be an integer")

        # Check init_runs
        if not isinstance(init_runs, int):
            raise TypeError("Initial runs (init_runs) should be an integer")

        # Check max_runs
        if not isinstance(max_runs, int):
            raise TypeError("Maximum runs (max_runs) should be an integer")
        if max_runs <= init_runs:
            raise ValueError("Maximum runs (max_runs) must be greater than initial runs (init_runs)")

        print(f"Maximum runs is valid: {max_runs}")

        # Check calibration_parameters and param_ranges
        if not isinstance(calibration_parameters, list):
            raise TypeError("calibration_parameters should be a list")

        if not isinstance(param_values, list):
            raise TypeError("param_ranges should be a list")

        if len(calibration_parameters) != len(param_values):
            raise ValueError("calibration_parameters and param_ranges must have the same length")

        # Check calibration_quantities
        if not isinstance(calibration_quantities, list):
            raise TypeError("calibration_quantities should be a list of strings")

        if len(calibration_quantities) > 2:
            raise ValueError("calibration_quantities can have a maximum of 2 quantities")

        # Validate each calibration quantity
        for quantity in calibration_quantities:
            if not isinstance(quantity, str):
                raise TypeError("Each calibration quantity should be a string")

        # Optionally, check if paths exist using os (if calibration quantities were paths, for example)
        for quantity in calibration_quantities:
            if os.path.exists(quantity):
                print(f"Path exists: {quantity}")
            else:
                print(f"Path does not exist: {quantity}")

        # Output valid calibration quantities
        if calibration_quantities:
            print(f"Calibration quantities are valid: {calibration_quantities}")
        #pass

    def extract_data_point(
            self,
            *args,
            **kwargs
    ):
        """
        Extract a specific data from the desired coordinate from any hydrodynamic model.

        This is a generic method intended to be implemented for various hydrodynamic models
        (e.g., Telemac, OpenFOAM, etc.). The method can extract data based on provided
        input_output_file and .csv file containing the coordinates of the desired points.

        Parameters
        ----------
        *args :
            Arguments used to define the data extraction criteria, such as data indices,
            output file paths, etc.
        **kwargs :
            Arguments that allow for more flexible and descriptive criteria for data
            extraction, such as 'time', 'location', 'variable_name', or any other relevant
            parameters that are specific to the data or model being used.

        Returns
        -------
        None
            The method saves the extracted model outputs to JSON files in the specified result directory.

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
            bal_iteration=0,
            complete_bal_mode=True,
    ):
        """
         Executes multiple hydrodynamic simulations in the context of Bayesian Active Learning (BAL) with a set of collocation points.
         The method also considers the inclusion of a new set of calibration parameters as an array when BAL iterations are being done.
         Complete_bal_mode can be added as False when only multiple runs of the hydrodynmaic model are required.
         The number of processors to use is defined by self.nproc during the initialization of the HydroSimulation class.

         Parameters
         ----------
         collocation_points : array
             Numpy array of shape [No. init_runs x No. calibration parameters] which contains the initial
             collocation points (parameter combinations) for iterative runs. Default is None, and it
             is filled with values for the initial surrogate model phase. It remains None during the BAL phase.
         bal_new_set_parameters : array
             2D array of shape [1 x No. parameters] containing the new set of values after each BAL iteration. Default None.
             It remains None during the initial runs.
         bal_iteration : int
             The number of the BAL iteration. Default is 0.
         complete_bal_mode : bool
             Default is True when the code accounts for initial runs, surrogate construction and BAL phase. False when
             only initial runs are required.

         Returns
         -------
         model_evaluations:
         """
        self.model_evaluations = np.empty((2, 2))

        return self.model_evaluations
        pass

    def run_single_simulation(
            self,
            control_file="control_file.hydro",
            load_results=True
    ):
        """
        Executes a single simulation run using a specified script or launcher file.

        This method is intended to handle the execution of a single simulation for various models
        (e.g., Telemac, OpenFOAM, Basement) by calling the appropriate launcher script.
        It also provides an option to load and process the results of the simulation after execution.

        Parameters
        ----------
        filename : str, optional
            The name of the control file used to launch the simulation. Defaults to "control_file.hydro" as an example.
            This file should be present in the appropriate directory and executable through a terminal.
        load_results : bool, optional
            A flag indicating whether to load and process the results after the simulation run.
            If `True`, the method will attempt to read and process the output data; otherwise,
            it will only execute the simulation. Defaults to `True`.

        Returns
        -------
        result : json,files or None
            The result of the simulation, which could be a data structure, simulation output file and .json files
            depending on the simulation model and the implementation. If `load_results` is `False`,
            this may return only the simulation results file.
        """
        start_time = datetime.now()
        print("DUMMY CALL")
        # implement call to run the model from command line, for example:
        # call_subroutine("openTeleFoam " + self.control_file)
        print("Full-complexity simulation time: " + str(datetime.now() - start_time))
        pass

    def set_observations_and_errors(self,
                                    calibration_pts_file_path,
                                    calibration_quantities):
        """
        Reads and sets the observations and errors at calibration points based on the provided data file.

        It ensures that the observations and errors are properly aligned and formatted
        for Bayesian Inference.

        Parameters
        ----------
        calibration_pts_file_path : str
            Path to the file containing the calibration data points. This file should include the
            observed values and their corresponding errors.
        calibration_quantities : list of str
            Quantities for which calibration is being performed. These should correspond
            to the columns or fields in the calibration data file that contain the relevant
            observational data.


        Returns
        -------
        observations : 2D array
            Observed values extracted from the calibration data file with shape [1 x (No. calibration points * n_calib_quantities)].
        measurement_errors : 1D array
            Measurement errors associated with the observed values with shape [(No.calibration points * n_calib_quantities), ]
        n_loc : int
            Number of unique locations or data points where observations are made.
        n_calib_quantities : int
            Number of calibration quantities being processed.
        calibration_pts_df : DataFrame
            Contains the raw calibration data points read from the file.
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

        return observations, measurement_errors, n_loc, n_calib_quantities, calibration_pts_df

    @staticmethod
    def read_data(results_folder, file_name):
        """
        Reads and extracts data from various file types based on the provided file name.

        The function supports file types such as .csv, .json, .txt, .pkl, and .pickle.

        Parameters
        ----------
        results_folder : str
            The base directory where the results files are stored.
        file_name : str
            The name of the file, including its extension (e.g., 'data.csv', 'output.json').

        Returns
        -------
        data : object
            The extracted data, which can be a DataFrame, dictionary, list, or other object depending on the file type.
            Returns None if the file type is unsupported or an error occurs while reading the file.
        """
        file_path = os.path.join(results_folder, file_name)

        try:
            # Determine the file extension
            file_extension = os.path.splitext(file_name)[1]

            if file_extension == '.csv':
                data = pd.read_csv(file_path).to_numpy()
            elif file_extension == '.json':
                with open(file_path, 'r') as file:
                    data = json.load(file)
            elif file_extension == '.txt':
                with open(file_path, 'r') as file:
                    data = file.readlines()
            elif file_extension in ['.pkl', '.pickle']:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
            else:
                raise ValueError(
                    "Unsupported file type. Only .csv, .json, .txt, .pkl, and .pickle files are supported.")
            return data

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    #     TODO: Merge with run() method - merged with run method

    def set_calibration_parameters(self, params, values):
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
        return param_dict, ndim

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
            output_data="",
            delete_complex_outputs=False,
            validation=False
    ):
        """
        Extract data from a .JSON file containing model outputs to 2D array ready to use in Bayesian calibration and saves
        the results to a CSV file.

        Parameters
        ----------
        output_data_path : str
            Path to the .json file containing the model outputs. The file should be structured
            such that its keys correspond to calibration points, and its values are lists of model
            output values for each run and quantity.
        delete_complex_outputs: Boolean, Default: False
            Delete complex model outtput files from the results folder (e.g. auto-saved-results).
            Recommended when running several simulations of the full complexity model.

        Returns
        -------
        model_results : numpy.ndarray
            A 2D array containing the processed model outputs. The shape of the array is
            [No. of quantities x No. of total runs, No. of calibration points], where 'No. of quantities'
            is the number of calibration quantities being processed, and 'No. of total runs' is the sum
            of initial runs and Bayesian active learning iterations. The array is also saved to a CSV file
            in the specified directory.
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
