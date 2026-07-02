"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
import os
import csv
import json
import pickle
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime

from hydroBayesCal.utils.config_logging import logger_error


class HydroSimulations(ABC):
    def __init__(
            self,
            control_file = "control.cas",
            model_dir="",
            res_dir="",
            calibration_pts_file_path="",
            n_cpus=int(),
            init_runs=1,
            calibration_parameters=None,
            param_values=None,
            calibration_quantities=None,
            extraction_quantities=None,
            dict_output_name='extraction-data',
            user_param_values=False,
            max_runs=1,
            complete_bal_mode=False,
            only_bal_mode=False,
            check_inputs=False,
            delete_complex_outputs=True,
            validation=False,
            multitask_selection="variables",
            *args,
            **kwargs,
    ):
        """
        Constructor of the HydroSimulations class to manage and run hydrodynamic simulations within the context of
        Bayesian Calibration using a Gaussian Process Emulator (GPE). The class is designed to handle simulation setup,
        execution, and result storage while managing calibration parameters and Bayesian Active Learning (BAL) iterations.

        Parameters
        ----------
        control_file : str
            Name of the file that controls the full complexity model simulation (default is "control.cas" as an example for Telemac).
        model_dir : str
            Full complexity model directory where all simulation files (mesh, control file, boundary conditions) are located.
        res_dir : str
            Directory where a subfolder called "auto-saved-results-HydroBayesCal" will be created to store all the result files. In this directory, the results of the calibration
            process will be stored according to the calibration quantity name. Addiionally, subfolders for plots, surrogate models, and restart data will be created.
        calibration_pts_file_path : str or optional
            File path to the calibration points data file. Please check documentation for further details of the file format.
        n_cpus : int
            Number of CPUs to be used for parallel processing (if available).
        init_runs : int
            Initial runs of the full complexity model (before Bayesian Active Learning).
        calibration_parameters : list of str
            Names of the considered calibration parameters (e.g. roughness coefficients, empirical constants, turbulent viscosity, etc).
        param_values : list
            Value ranges considered for parameter sampling. Example: [[min1, max1], [min2, max2], ...].
        calibration_quantities : list of str
            Names of the calibration targets (model outputs) used for calibration. These quantities usually correspond to the measured values for calibration purposes.
            Example: ['WATER DEPTH'] for a single quantity.
            Example: ['WATER DEPTH', 'SCALAR VELOCITY'] for multiple quantities.
        extraction_quantities : list of str
            Names of the quantities to be extracted from the model output files. Generally, the same or more than the calibration_quantities. These quantities will be extracted from the model.
            Example: ``calibration_quantities = ['WATER DEPTH']`` (WATER DEPTH as calibration parameter).
            Example: ``extraction_quantities = ['WATER DEPTH', 'SCALAR VELOCITY', 'TURBULENT ENERG', 'VELOCITY U', 'VELOCITY V']``.
            Any of these additional quantities can be used for calibration purposes when restarting the
            calibration process with ``only_bal_mode = True``.
        dict_output_name : str
            Base name for output dictionary files where the outputs are saved as .json files.
            This dictionary will be saved in the calibration-data subfolder for the considered calibration target.
        parameter_sampling_method : str
            Method used for sampling parameter values during the calibration process. The available options are:
            - "random"           : Random sampling.
            - "latin_hypercube"  : Latin Hypercube Sampling (LHS).
            - "sobol"            : Sobol sequence sampling.
            - "halton"           : Halton sequence sampling.
            - "hammersley"       : Hammersley sequence sampling.
            - "chebyshev(FT)"    : Chebyshev nodes (Fourier Transform-based).
            - "grid(FT)"         : Grid-based sampling (Fourier Transform-based).
            - "user"             : User-defined sampling.

            Example::

                parameter_sampling_method = "sobol"  # Uses Sobol sequence sampling.

            If "user" is selected, a ``.csv`` file containing user-defined collocation points must be provided
            in the restart data folder. The file should follow this format::

                param1    param2    param3    param4    param5
                0.148     0.770     0.014     0.014     0.700
                0.066     0.066     0.066     0.066     0.066

        max_runs : int
            Maximum (total) number of model simulations, including initial runs and Bayesian Active Learning iterations.
        complete_bal_mode : bool, optional (Default: True)
            - If True: Bayesian Active Learning (BAL) is performed after the initial runs, enabling a complete surrogate‐assisted calibration process.
              **This option MUST be selected if you choose to perform only BAL** (i.e., when `only_bal_mode = True`).
            - If False: Only the initial runs of the full complexity model are executed, and the model outputs are stored as `.json` files.

        only_bal_mode : bool, optional (Default: False)
            - If False: The process will either execute a complete surrogate‐assisted calibration or only the initial runs, depending on the value of `complete_bal_mode`.
            - If True: Only the surrogate model construction and Bayesian Active Learning of preexisting model outputs at predefined collocation points are performed.
              **This mode can be executed only if either a complete process has already been performed** (`complete_bal_mode = True` and `only_bal_mode = True`) **or if only the initial runs have been executed** (`complete_bal_mode = False` and `only_bal_mode = False`).

        Shortcut combinations and their corresponding tasks::

            complete_bal_mode | only_bal_mode                   | task
            ------------------+---------------------------------+-----------------------------------------------------
            True              | False                           | Complete surrogate-assisted calibration
            False             | False                           | Only initial runs (no surrogate model)
            True              | True, with init_runs = max_runs | Surrogate construction with predefined runs (no BAL)
            True              | True, with init_runs > max_runs | Surrogate construction + Bayesian Active Learning
        validation : bool, optional (Default: False)
            If True, creates output files (inputs and outputs) corresponding to validation process.
        *args : tuple, optional
            Additional positional arguments.
        **kwargs : dict, optional
            Additional keyword arguments.

        Attributes
        ----------
        asr_dir : str
            Directory for auto-saved results.
        model_dir : str
            Path to the directory containing the model files.
        res_dir : str
            Path to the directory where results will be stored.
        control_file : str
            Path to the control file used in the calibration process.
        calibration_pts_file_path : str
            Path to the file containing the calibration points data.
        nproc : int
            Number of processors (CPUs) to be used.
        param_values : array
            Parameter values used in calibration.
        calibration_quantities : list
            Calibration quantities to be evaluated.
        extraction_quantities : list
           Quantities extracted from the model during calibration (calibration quantities must be included here).
        calibration_parameters : list
            Parameters involved in the calibration process.
        parameter_sampling_method : str
            Method used for sampling parameters during calibration. Options:
            - "random"
            - "latin_hypercube"
            - "sobol"
            - "halton"
            - "hammersley"
            - "chebyshev(FT)"
            - "grid(FT)"
            - "user" (requires a CSV file with user-defined collocation points in the restart data folder).
        init_runs : int
            Number of initial runs before surrogate-assisted calibration.
        max_runs : int
            Maximum number of calibration runs, including Bayesian Active Learning iterations.
        complete_bal_mode : bool
            If True, enables complete surrogate-assisted calibration with Bayesian Active Learning.
            Must be selected if `only_bal_mode = True`.
        only_bal_mode : bool
            If True, only surrogate model construction and Bayesian Active Learning are performed.
            Requires prior execution of either the full calibration process (`complete_bal_mode = True`)
            or initial runs (`complete_bal_mode = False`).
        delete_complex_outputs : bool
            If True, deletes complex model outputs after processing.
        validation : bool
            If True, the model is run in validation mode.
        multitask_selection : bool
            If True, enables multitask selection for surrogate modeling.
        dict_output_name : str
            Name of the output dictionary file. Appends "-validation" if validation mode is enabled.
        nloc : int
            Number of calibration points where measured data are available.
        ndim : int
            Number of model parameters used in the calibration.
        param_dic : dict
            Dictionary containing calibration parameters and their respective ranges.
        num_calibration_quantities : int
            Number of quantities used for calibration.
        num_extraction_quantities : int
            Number of additional quantities extracted from the model.
        observations : array
            Observed values at each calibration point.
        measurement_errors : array
            Measurement errors associated with each calibration point.
        calibration_pts_df : pandas.DataFrame
            Contains calibration point information.
            Header format::

                Point | X | Y | <quantity>_DATA | <quantity>_ERROR | ...
        user_collocation_points : array
            User-defined collocation points loaded from a CSV file (only applicable when `parameter_sampling_method="user"`).
        calibration_folder : str
            Directory where calibration data are stored.
        restart_data_folder : str
            Directory for restart data, used for resuming calibration runs.
        model_evaluations : numpy.ndarray
            2D array of processed model outputs, shape
            ``[num_runs, nloc * num_calibration_quantities]``: ``num_runs`` is the
            total number of evaluations (initial runs plus BAL iterations) and the
            columns interleave the calibration quantities per location. For example,
            with two quantities and two locations, columns 1-2 hold the two
            quantities at the first location and columns 3-4 the second.
        """
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
        self.extraction_quantities = extraction_quantities
        self.calibration_parameters = calibration_parameters
        self.init_runs = init_runs
        self.max_runs = max_runs
        self.complete_bal_mode = complete_bal_mode
        self.only_bal_mode = only_bal_mode
        self.delete_complex_outputs=delete_complex_outputs
        self.validation=validation
        self.multitask_selection = multitask_selection
        self.user_param_values = user_param_values
        if self.validation:
            self.dict_output_name = dict_output_name + "-validation"
        else:
            self.dict_output_name = dict_output_name
        self.nloc = None
        self.ndim = None
        self.param_dic = None
        self.num_calibration_quantities = None
        self.num_extraction_quantities = None
        self.observations = None
        self.measurement_errors = None
        self.variances = None
        self.calibration_pts_df = None
        self.user_collocation_points = None
        self.restart_collocation_points = None
        self.model_evaluations = None


        if calibration_parameters:
            self.param_dic, self.ndim = self.set_calibration_parameters(calibration_parameters, param_values)
        if calibration_pts_file_path:
            self.observations,self.variances, self.measurement_errors, self.nloc, self.num_calibration_quantities, self.calibration_pts_df, self.num_extraction_quantities = self.set_observations_and_variances(
                calibration_pts_file_path, calibration_quantities, extraction_quantities)

        self.asr_dir = os.path.join(res_dir,
                                    f"auto-saved-results-HydroBayesCal")
        self.calibration_folder = os.path.join(self.asr_dir,"calibration-data","_".join(self.calibration_quantities))
        self.restart_data_folder = os.path.join(self.asr_dir,"restart_data")
        if not os.path.exists(self.asr_dir):
            os.makedirs(self.asr_dir)
        if not os.path.exists(self.calibration_folder) and not self.user_param_values:
            os.makedirs(self.calibration_folder)
        if not os.path.exists(self.restart_data_folder):
            os.makedirs(self.restart_data_folder)
        if not os.path.exists(os.path.join(self.asr_dir, "plots")):
            os.makedirs(os.path.join(self.asr_dir, "plots"))
        if not os.path.exists(os.path.join(self.asr_dir, "surrogate-gpe")):
            os.makedirs(os.path.join(self.asr_dir, "surrogate-gpe"))
        if self.user_param_values:
            collocation_path = os.path.join(self.restart_data_folder, 'user-collocation-points.csv')
            self.user_collocation_points = np.loadtxt(collocation_path, delimiter=',', skiprows=1, ndmin=2)
        if self.only_bal_mode:
            collocation_path = os.path.join(self.restart_data_folder, 'initial-collocation-points.csv')
            self.restart_collocation_points = np.loadtxt(collocation_path, delimiter=',', skiprows=1, max_rows=self.init_runs,ndmin=2)

    def extract_data_point(
            self,
            input_file,
            calibration_pts_df,
            output_name,
            extraction_quantity,
            simulation_number,
            model_directory,
            results_folder_directory,
            *args,
            **kwargs
    ):
        """
        Extract data from a specified coordinate in a hydrodynamic model output file.

        This generic method is designed for use with various hydrodynamic models
        (e.g., Telemac, OpenFOAM, etc.). It extracts data from an input file
        based on a provided CSV file containing the coordinates of the target points.

        Parameters
        ----------
        input_file : str
            Path to the hydrodynamic model output file from which data will be extracted.
        calibration_pts_df : pd.DataFrame
            Contains the coordinates of the points where data extraction is required.
            It must include:
            - Point descriptions (e.g., "P1").
            - X and Y coordinates of the measurement points.
            - Measured values and errors for the calibration quantities.

            Expected columns:
            - For a single calibration quantity: `['Point Name', 'X', 'Y', 'Measured Value', 'Measured Error']`
            - For two calibration quantities: `['Point Name', 'X', 'Y', 'Measured Value 1', 'Measured Error 1', 'Measured Value 2', 'Measured Error 2']`
        output_name : str
            Base name for the output file where extracted data will be stored.
        extraction_quantity : list of str
            List of variables or quantities to be extracted.
            Example: extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG"]
        simulation_number : int
            The current simulation number, used to manage and organize data extraction (e.g. simulation number).
        model_directory : str
            Path to the directory containing the model output files.
        results_folder_directory : str
            Path to the directory where the extracted data will be saved.
        *args :
            Additional positional arguments defining specific extraction criteria, such as
            data indices or custom processing parameters.
        **kwargs :
            Additional keyword arguments for flexible data extraction criteria, such as:
            - `time`: Specific time step for extraction.
            - `location`: Specific coordinate or region of interest.
            - `variable_name`: Name of the variable to extract.
            - Any other model-specific parameters required for data extraction.

        Returns
        -------
        None
            The extracted data is saved to output files in the specified results directory.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_data_point()."
        )

    @abstractmethod
    def run_multiple_simulations(
            self,
            collocation_points=None,
            bal_new_set_parameters=None,
            bal_iteration=0,
            complete_bal_mode=True,
            validation=False,
            *args,
            **kwargs
    ):
        """Run the full-complexity model for a set of collocation points (BAL).

        Executes multiple hydrodynamic simulations in the context of Bayesian
        Active Learning (BAL). A new set of calibration parameters may be added
        as an array during BAL iterations.

        * If ``complete_bal_mode=True``, the process includes initial runs,
          surrogate-model construction and BAL iterations.
        * If ``complete_bal_mode=False``, only the initial model runs are
          performed.
        * If ``validation=True``, a separate set of runs is executed for
          validation (e.g. assessing surrogate-model performance).

        The number of processors is defined by ``self.nproc`` at initialisation.

        Parameters
        ----------
        collocation_points : numpy.ndarray, optional
            Array of shape ``[init_runs, n_parameters]`` with the initial
            collocation points for the iterative runs. ``None`` during the BAL
            phase.
        bal_new_set_parameters : numpy.ndarray, optional
            Array of shape ``[1, n_parameters]`` with the new parameter set for
            a BAL iteration. ``None`` during the initial runs.
        bal_iteration : int, optional
            BAL iteration number (default ``0``).
        complete_bal_mode : bool, optional
            ``True`` (default) to run the full process (initial runs, surrogate
            construction and BAL); ``False`` for initial runs only.
        validation : bool, optional
            ``True`` to run a separate set of validation simulations.
        *args, **kwargs
            Binding-specific options (e.g. Telemac's ``output_extraction``,
            ``output_extraction_time`` and ``n``). See the concrete subclass.

        Returns
        -------
        numpy.ndarray
            2D array of processed model outputs, shape
            ``[num_runs, nloc * num_calibration_quantities]``, where ``num_runs``
            is the total number of evaluations (initial runs plus BAL
            iterations) and the columns interleave the calibration quantities
            per location. For example, with two quantities and two locations,
            columns 1-2 hold the two quantities at the first location and
            columns 3-4 the second.
        """

    def run_single_simulation(
            self,
            control_file="control_file.hydro",
    ):
        """
        Executes a single model run using a specified script or launcher file.

        This method is intended to handle the execution of a single simulation for various models
        (e.g., Telemac, OpenFOAM, Basement) by calling the appropriate launcher script.

        Parameters
        ----------
        control_file : str
            The name of the control file used to launch the simulation. Defaults to "control_file.hydro" as an example.
            This file should be present in the appropriate directory and executable through a terminal.

        Returns
        -------
        None
            The method executes the model run using a launcher command.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_single_simulation()."
        )

    def set_observations_and_variances(
            self,
            calibration_pts_file_path,
            calibration_quantities,
            extraction_quantities,
            gpe_error=0.10,
            measurement_error=0.10):
        """
        Reads calibration point data and constructs observation variances.

        Total variance is computed as::

            variance = measurement_error**2 + gpe_error**2 + site_specific_error**2

        where:

        - ``measurement_error`` is assigned as a percentage of the measured value.
        - ``gpe_error`` is assigned as a percentage of the measured value.
        - ``site_specific_error`` is read from ``<quantity>_ERROR`` columns and should
          already be in the physical units of the corresponding calibration quantity.
        """

        calibration_pts_df = pd.read_csv(calibration_pts_file_path)

        # Resolve the required <quantity>_DATA / <quantity>_ERROR columns
        # case-insensitively, so both the Telemac (e.g. "WATER DEPTH_DATA") and
        # OpenFOAM (e.g. "U_x_DATA") naming conventions work regardless of the
        # exact case used in the CSV header.
        column_lookup = {col.lower(): col for col in calibration_pts_df.columns}

        observation_columns = []
        error_columns = []
        missing_columns = []

        for quantity in calibration_quantities:
            obs_actual = column_lookup.get(f"{quantity}_DATA".lower())
            err_actual = column_lookup.get(f"{quantity}_ERROR".lower())
            if obs_actual is not None:
                observation_columns.append(obs_actual)
            else:
                missing_columns.append(f"{quantity}_DATA")
            if err_actual is not None:
                error_columns.append(err_actual)
            else:
                missing_columns.append(f"{quantity}_ERROR")

        if missing_columns:
            raise ValueError(
                f"Missing required columns in calibration file: {missing_columns}"
            )

        observations_2d = calibration_pts_df[observation_columns].to_numpy(dtype=float)

        site_specific_errors_2d = calibration_pts_df[error_columns].to_numpy(dtype=float)

        abs_observations_2d = np.abs(observations_2d)

        measurement_errors_2d = abs_observations_2d * measurement_error
        gpe_errors_2d = abs_observations_2d * gpe_error

        observations = observations_2d.flatten().reshape(1, -1)

        measurement_errors = measurement_errors_2d.flatten()
        gpe_errors = gpe_errors_2d.flatten()
        site_specific_errors = site_specific_errors_2d.flatten()

        variances = (
                measurement_errors ** 2
                + gpe_errors ** 2
                + site_specific_errors ** 2
        )

        n_loc = len(calibration_pts_df)
        n_calib_quantities = len(calibration_quantities)

        if extraction_quantities is None:
            n_extraction_quantities = 0
        else:
            n_extraction_quantities = len(extraction_quantities)

        return (
            observations,
            variances,
            measurement_errors,
            n_loc,
            n_calib_quantities,
            calibration_pts_df,
            n_extraction_quantities
        )


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

    def set_calibration_parameters(self, params, values):
        """
        Create a dictionary from calibration parameters and their value ranges if both params and values exist.
        If only one of them exists, compute the number of dimensions.

        :param params: List of parameter names.
        :param values: List of value ranges corresponding to the parameter names.
        :return: Dictionary with parameter names as keys and value ranges as values, and the number of dimensions.
        :raises ValueError: If the number of parameters does not match the number of values when both are provided.
        """
        # Initialize default return values
        param_dict = None
        ndim = 0

        # Check if both params and values exist
        if params and values:
            if len(params) != len(values):
                logger_error.error("Mismatch between the number of parameters (%d) and values (%d)", len(params),
                                   len(values))
                raise ValueError("The number of parameters and values must be the same.")

            # Create the dictionary
            param_dict = dict(zip(params, values))
            ndim = len(params)

        # If only one of params or values exists, compute ndim
        elif params or values:
            ndim = len(params) if params else len(values)

        # Return the dictionary and number of dimensions
        return param_dict, ndim

    def update_model_controls(
            self,
            collocation_point_values,
            calibration_parameters,
            auxiliary_file_path,
            simulation_id=0,
    ):
        """
        Updates the model control files for Bayesian calibration.Incorporates new parameter values, ensuring that the model runs
        with the specified settings during Bayesian calibration.

        Parameters
        ----------
        collocation_point_values : array
            Contains values for the calibration parameters. These values are used
            to update the model control files.

        calibration_parameters : list of str
            Calibration parameter names that are to be updated in the model control files. Each
            string in the list should correspond to a parameter used in the model.

        auxiliary_file_path : str
            Path to an auxiliary file that may be required for running the model controls (i.e., .tbl file in Telemac).

        simulation_id : int
            An optional identifier for the simulation. The default is 0. This ID can be used to distinguish
            different simulations or runs.

        Returns
        -------
        None
            This method does not return any value. It modifies the model control files.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement update_model_controls()."
        )

    @abstractmethod
    def output_processing(
            self,
            output_data="",
            delete_complex_outputs=False,
            validation=False,
            *args,
            **kwargs
    ):
        """
        Extract data from a file(.txt,json,etc) containing model outputs to 2D array ready to use in Bayesian calibration
        and saves the results to a CSV file.

        Parameters
        ----------
        output_data_path : str
            Path to the file (.json) containing the model outputs. The file should be structured
            such that its keys correspond to calibration points, and its values are lists of nested dictionaries having the
            output values for each run and quantity/ies.
        delete_complex_outputs: Boolean, Default: False
            Delete complex model output files from the results folder (e.g. auto-saved-results-HydroBayesCal/<variable>).
            Recommended when running several simulations of the full complexity model.
        validation: Boolean, Default: False
            If True, new files for collocation points and model results are created. This is done to keep
            the collocation points and model results obtained during the calibration process.

        Returns
        -------
        model_results : numpy.ndarray
            A 2D array containing the processed model outputs. The shape of the array is
            [No. of total runs, No. of calibration points x No. of quantities], where 'No. of quantities'
            is the number of calibration quantities being processed, and 'No. of total runs' is the sum
            of initial runs and Bayesian active learning iterations. The array is also saved to a CSV file
            in the specified directory.
        """
