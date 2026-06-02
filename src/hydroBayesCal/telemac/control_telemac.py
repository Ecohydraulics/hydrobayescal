# coding: utf-8
"""
Functional core for controlling Telemac simulations for coupling with the Surrogate-Assisted Bayesian inversion technique.
Authors: Andres Heredia, Sebastian Schwindt
"""
from scipy import spatial
import numpy as np
from hydroBayesCal.telemac import config_telemac
from datetime import datetime
from hydroBayesCal.telemac.pputils.ppmodules.selafin_io_pp import ppSELAFIN

from collections import OrderedDict
from hydroBayesCal.hysim import HydroSimulations
from hydroBayesCal.function_pool import *  # provides os, subprocess, logging


class TelemacModel(HydroSimulations):
    def __init__(
            self,
            friction_file="",
            tm_xd="Telemac2d",
            gaia_steering_file=None,
            fortran_file=None,
            results_filename_base="",
            gaia_results_filename_base=None,
            stdout=6,
            python_shebang="#!/usr/bin/env python3",
            *args,
            **kwargs
    ):
        """
        Constructor for the TelemacModel Class. The class contains all necessary methods for Telemac simulations,extractions of simulation outputs and
        iterative updating of the control files.

        Parameters
        ----------
        friction_file : str, optional
            Name of the friction file to be used in Telemac simulations (should end with ".tbl"); do not include the directory path.
        tm_xd : str,
            Specifies the dimension of the Telemac hydrodynamic solver, either 'Telemac2d' or 'Telemac3d'.
        gaia_steering_file : str, optional
            Name of the Gaia steering file; should be provided if required. Not implemented on this HydroBayesCal version.
        results_filename_base : str, optional
            Base name for the results file, which will be iteratively updated in the .cas file.
        python_shebang : str, optional
            Shebang line for Python scripts (default is "#!/usr/bin/env python3").
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Attributes
        ----------
        friction_file : str
            Name of the Telemac friction file .tbl.
        tm_xd : str
            Dimension of the Telemac simulation ('Telemac2d' or 'Telemac3d').
        gaia_steering_file : str or None
            Gaia steering file name if provided; otherwise, None. Not implemented on this HydroBayesCal Version.
        results_filename_base : str
            Base name for the Telemac results file.
        python_shebang : str
            Shebang line for Python scripts.
        tm_cas : str
            Full path to the Telemac steering file (.cas).
        fr_tbl : str
            Full path to the friction file (.tbl).
        comm : MPI.Comm
            MPI communicator for parallel processing.
        shebang : str
            Shebang line for Python scripts.
        tm_xd_dict : dict
            Dictionary mapping 'Telemac2d' and 'Telemac3d' to their respective script names.
        bal_iteration : int
            Bayesian Active Learning iteration number based on max_runs.
        num_run : int
            Simulation number; iteratively updated based on collocation points.
        tm_results_filename : str
            File path for storing output data from Telemac simulations.

        Note
        ----
        The attributes specific to Telemac are listed above. For attributes inherited from the `HydroSimulations` class, please refer to its documentation.
        """
        super().__init__(*args, **kwargs)
        # Initialize subclass-specific attributes
        self.friction_file = friction_file
        self.tm_xd = tm_xd
        self.gaia_steering_file = gaia_steering_file
        self.gaia_results_filename_base = gaia_results_filename_base
        self.fortran_file = fortran_file
        self.results_filename_base = results_filename_base
        self.python_shebang = python_shebang
        self.tm_cas = "{}{}{}".format(self.model_dir, os.sep, self.control_file)
        self.fr_tbl = "{}{}{}".format(self.model_dir, os.sep, self.friction_file)
        if self.gaia_steering_file is not None:
            self.gaia_cas = "{}{}{}".format(self.model_dir, os.sep, self.gaia_steering_file)
        else:
            self.gaia_cas = None
        if self.fortran_file is not None:
            self.fortran_file = os.path.join(self.model_dir, "user_fortran", self.fortran_file)
        else:
            self.fortran_file = None
        self.shebang = python_shebang
        self.tm_xd_dict = {
            "Telemac2d": "telemac2d.py ",
            "Telemac3d": "telemac3d.py ",
        }
        self.stdout = stdout
        # Initializes the BAL iteration number
        self.bal_iteration = int()
        # Initializes the simulation number
        self.num_run = int()
        # Initializes the file where the output data from simulations will be stored
        self.tm_results_filename = ''
        self.gaia_results_filename = ''

    def update_model_controls(
            self,
            collocation_point_values,
            calibration_parameters,
            auxiliary_file_path=None,
            gaia_file_path=None,
            simulation_id=0,
    ):
        """
        Modifies the .cas steering file for each of the Telemac runs according to the
        values of the collocation points and the calibration parameters. If a
        "FRICTION DATA FILE" is provided for Telemac simulations, it is possible to
        consider any zone as a calibration parameter. The parameters must start with
        the prefix "zone" and the number of the friction zone. The .tbl file will be
        modified for this purpose. This method is called every time it is required
        that the .cas or .tbl files are modified. It also modifies the gaia cas file. If
        the parameter starts with the prefix "gaia", the method will look for the parameter
        in the gaia cas file and update it with the new value. If the parameter starts with "f.",
        the method will look for it in the fortran file and update it with the new value.
        The rest of the parameters will be updated in the telemac cas file.

        Parameters
        ----------
        collocation_point_values : list
            Values for each of the calibration parameters.

        calibration_parameters : list
            Names of the calibration parameters.

        auxiliary_file_path : str, optional
            Path to the friction file (.tbl).

        gaia_file_path : str, optional
            Path to the GAIA steering file (.cas). If provided, GAIA calibration
            parameters will also be updated.

        simulation_id : int, optional
            Identifier of the current simulation. Used when generating or updating
            control files for multiple simulations. Default is 0.

        Returns
        -------
        None
            Modified control files (telemac.cas, gaia.cas, fortran file, and/or friction .tbl)
            for Telemac simulations.
        """

        # ============================================================
        # TELEMAC RESULT FILE NAMES
        # ============================================================
        self.tm_results_filename = (
            self.results_filename_base + '_' + str(simulation_id) + '.slf'
        )

        if self.tm_xd == 'Telemac3d':
            # Main 3D result file
            tm_result_keys = ['3D RESULT FILE']

            # Additional 2D result file generated by TELEMAC-3D
            # Example:
            #   results_3d.slf    -> results_3d_2d.slf
            self.tm_2d_results_filename_from_3d = self.tm_results_filename.replace(
                '.slf',
                '_2d.slf'
            )

            tm_result_keys.append('2D RESULT FILE')
            tm_result_values = [
                self.tm_results_filename,
                self.tm_2d_results_filename_from_3d
            ]

        else:
            # Standard TELEMAC-2D result file
            tm_result_keys = ['RESULTS FILE']
            tm_result_values = [self.tm_results_filename]

        # ============================================================
        # GAIA RESULT FILE NAME
        # ============================================================
        if self.gaia_cas is not None:
            self.gaia_results_filename = (
                self.gaia_results_filename_base + '_' + str(simulation_id) + '.slf'
            )

            params_with_results = (
                calibration_parameters
                + tm_result_keys
                + ['GAIA RESULTS FILE']
            )

            values_with_results = [
                round(val, 7) if isinstance(val, float) else val
                for val in (
                    collocation_point_values
                    + tm_result_values
                    + [self.gaia_results_filename]
                )
            ]

        else:
            params_with_results = calibration_parameters + tm_result_keys

            values_with_results = [
                round(val, 7) if isinstance(val, float) else val
                for val in collocation_point_values + tm_result_values
            ]

        logger.info(f'Results file name for this simulation: {self.tm_results_filename}')

        if self.tm_xd == 'Telemac3d':
            logger.info(
                f'2D results file name for this 3D simulation: '
                f'{self.tm_2d_results_filename_from_3d}'
            )

        friction_file_path = auxiliary_file_path

        try:
            for param, value in zip(params_with_results, values_with_results):

                if param.lower().startswith("zone"):
                    zone_identifier = param[4:]
                    self.tbl_creator(zone_identifier, value, friction_file_path)

                elif param.lower().startswith("vg_zone"):
                    # Extract the zone identifier and vegetation parameter number
                    parts = param.split("-")  # Assuming the format is like "vg_zoneXX_X"

                    if len(parts) == 2:
                        zone_identifier = parts[0][7:]  # Extracts the number after 'vg_zone'
                        veg_param_value = parts[1]  # Extracts the vegetation parameter number

                        self.tbl_creator(
                            zone_identifier,
                            value,
                            friction_file_path,
                            veg_param_number=veg_param_value,
                            veg_indicator=True
                        )

                elif param.lower().startswith("gaia"):
                    gaia_param_name = param[4:].strip()
                    parts_gaia_name = gaia_param_name.rsplit(" ", 1)

                    # Check if the last part is a number
                    if parts_gaia_name[-1].isdigit():
                        gaia_parameter_name = parts_gaia_name[0].strip()
                        class_number = int(parts_gaia_name[1])

                        self.gaia_string_for_classes = parse_classes_keyword(
                            self.gaia_cas,
                            gaia_parameter_name
                        )

                        gaia_string = update_gaia_class_line(
                            self.gaia_string_for_classes,
                            class_number - 1,
                            value
                        )

                        self.rewrite_steering_file(
                            gaia_parameter_name,
                            gaia_string,
                            steering_module="gaia"
                        )

                    else:
                        gaia_string = self.create_cas_string(gaia_param_name, value)

                        self.rewrite_steering_file(
                            gaia_param_name,
                            gaia_string,
                            steering_module="gaia"
                        )

                elif param.lower().startswith("f."):
                    fortran_param_name = param[2:].strip()
                    fortran_string = "      " + self.create_cas_string(
                        fortran_param_name,
                        value
                    )

                    self.rewrite_steering_file(
                        fortran_param_name,
                        fortran_string,
                        steering_module="fortran"
                    )

                else:
                    cas_string = self.create_cas_string(param, value)

                    self.rewrite_steering_file(
                        param,
                        cas_string,
                        steering_module="telemac"
                    )

        except Exception as e:
            logger_error.error(f'Error occurred during CAS creation: {e}')
            raise RuntimeError

    @staticmethod
    def create_cas_string(
            param_name,
            value
    ):
        """
        Create string names with new values to be used in Telemac2d / Gaia steering files

        Parameters
        ----------

        param_name: string
            Name of parameter to update
        value: int , float or string
            Value to be assigned to param_name

         Returns
        -------
            None
            Update parameter line for a steering file
        """
        if isinstance(value, (int, float, str)) or ':' in value:

            return param_name + " = " + str(value)
        else:
            try:
                return param_name + " = " + "; ".join(map(str, value))
            except Exception as error:
                logger_error.error(
                    "ERROR: could not generate cas-file string for {0} and value {1}:\n{2}".format(str(param_name),
                                                                                                   str(value),
                                                                                                   str(error)))

    def rewrite_steering_file(
            self,
            param_name,
            updated_string,
            steering_module="telemac"
    ):
        """Rewrite the ``.cas`` steering file with updated parameters.

        Parameters
        ----------
        param_name : str
            Name of the calibration parameter.
        updated_string : str
            Updated string written into the ``.cas`` file with the new value.
        steering_module : str, optional
            Steering module to rewrite: ``"telemac"`` (default) or ``"gaia"``.

        Returns
        -------
        int
            ``0`` on success, ``-1`` on error.
        """

        # check if telemac or gaia cas type
        if "telemac" in steering_module:
            steering_file_name = self.tm_cas
        elif "gaia" in steering_module:
            steering_file_name = self.gaia_cas
        elif "fortran" in steering_module:
            steering_file_name = self.fortran_file

        # save the variable of interest without unwanted spaces
        variable_interest = param_name.rstrip().lstrip()

        # open steering file with read permission and save a temporary copy
        if os.path.isfile(steering_file_name):
            cas_file = open(steering_file_name, "r")
        else:
            logger_error.error("ERROR: no such steering file:\n" + steering_file_name)
            return -1
        read_steering = cas_file.readlines()

        # if the updated_string has more than 72 characters, then divide it into two
        if len(updated_string) >= 72:
            position = updated_string.find("=") + 1
            updated_string = updated_string[:position].rstrip().lstrip() + "\n" + updated_string[
                                                                                  position:].rstrip().lstrip()

        # preprocess the steering file
        # if in a previous case, a line had more than 72 characters then it was split into 2
        # this loop cleans up all lines that start with a number
        temp = []
        for i, line in enumerate(read_steering):
            if not isinstance(line[0], int):
                temp.append(line)
            else:
                previous_line = read_steering[i - 1].split("=")[0].rstrip().lstrip()
                if previous_line != variable_interest:
                    temp.append(line)

        # loop through all lines of the temp cas file, until it finds the line with the parameter of interest
        # and substitute it with the new formatted line
        for i, line in enumerate(temp):
            line_value = line.split("=")[0].rstrip().lstrip()
            if line_value == variable_interest:
                temp[i] = updated_string + "\n"

        # rewrite and close the steering file
        cas_file = open(steering_file_name, "w")
        cas_file.writelines(temp)
        cas_file.close()
        return 0

    def run_single_simulation(
            self,
            control_file="tel.cas",
    ):
        """
        Runs a Telemac2D or Telemac3D simulation with one or more processors.
        The number of processors to use is defined by self.nproc.

        Parameters
        ----------
        control_file : str
            The name of the control file used to launch the simulation.
            Default is "tel.cas". This file should be located in the model directory.

        Returns
        -------
        None
            The method executes the model run using a launcher command.
        """
        logger.info("Running full complexity model " + str(self.num_run))

        start_time = datetime.now()

        if self.nproc <= 1:
            logger.info("* Sequential run (single processor)")
        else:
            logger.info(f"* Parallel run on {self.nproc} processors")

        # Determine the launcher command for Telemac
        if self.tm_xd in self.tm_xd_dict:
            tm_launcher = self.tm_xd_dict[self.tm_xd]
            telemac_command = f"{tm_launcher} {os.path.join(self.model_dir, control_file)} --ncsize={self.nproc}"
        else:
            raise ValueError(f"Invalid launcher: {self.tm_xd}. Expected one of {list(self.tm_xd_dict.keys())}.")

        logger.info(f"Executing control file: {control_file}")

        try:
            # Run the simulation command
            process = subprocess.Popen(telemac_command, cwd=self.model_dir, shell=True, env=os.environ)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"Simulation failed with error: {stderr}")
            else:
                logger.info("Simulation completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred while running the simulation: {e}")

        logger.info("TELEMAC simulation time: " + str(datetime.now() - start_time))

    def run_multiple_simulations(
            self,
            collocation_points=None,
            bal_new_set_parameters=None,
            bal_iteration=int(),
            complete_bal_mode=True,
            output_extraction="interpolated",
            output_extraction_time="last",
            n=40,
            validation=False,
            kill_process = True
    ):
        """
        Runs multiple Telemac2d or Telemac3d simulations with a set of collocation points and a new set of
        calibration parameters when BAL mode is chosen. The number of processors to use is defined by self.nproc in user_inputs.

        Parameters
        ----------
        collocation_points : array
            Numpy array of shape [No. init_runs x No. calibration parameters] which contains the initial
            collocation points (parameter combinations) for iterative Telemac runs. Default is None, and it
            is filled with values for the initial surrogate model phase. It remains None during the BAL phase.
        bal_new_set_parameters : array
            2D array of shape [1 x No. parameters] containing the new set of values after each BAL iteration.
        bal_iteration : int
            The number of the BAL iteration. Default is 0.
        complete_bal_mode : bool
            Default is True when the code accounts for initial runs, surrogate construction and BAL phase. False when
            only initial runs are required.
        validation : bool
            If `True`, the method runs a separate set of simulations for validation purposes.
        output_extraction : str
            The mode for extracting model outputs. Options are "nearest", "index" or "interpolated".
        output_extraction_time : str
            The time mode for extracting model outputs. Options are "last", "index", or "mean_last".
        n : int
            The number of last time steps to consider when `output_extraction_time` is set to "mean_last". Default is 40.
        validation: bool
            If `True`, the method runs a separate set of simulations for validation purposes, and saves the collocation points used for validation in a separate CSV file.
        kill_process: bool
            If `True`, the method will attempt to kill any remaining Telemac processes after running the simulations. This is useful when preventing to running BAL after the initial runs.


        Returns
        -------
         model_evaluations:array
                     2D array containing processed model outputs.
            Shape: `[num_runs, nloc * num_calibration_quantities]`, where:
            - `num_runs` is the total number of model evaluations, including both initial runs and Bayesian Active Learning iterations.
            - `nloc * num_calibration_quantities` represents the total number of outputs, with results interleaved in columns.

            Example: For two calibration quantities and two calibration locations:
            - Columns 1 and 2 correspond to the outputs (2 quantities) of the first calibration location.
            - Columns 3 and 4 correspond to the outputs of the second location, and so on.
        """
        calibration_parameters = self.calibration_parameters
        res_dir = self.calibration_folder
        restart_data_path = self.restart_data_folder
        fr_tbl = self.fr_tbl
        init_runs = self.init_runs
        logger.info(
            "* Running multiple Telemac simulations can take time -- check CPU acitivity...")
        start_time = datetime.now()
        if complete_bal_mode:
            # This part of the code runs the initial runs for initial surrogate.
            if bal_new_set_parameters is None:
                # Convert collocation_points to a numpy array if it is not already
                if not isinstance(collocation_points, np.ndarray):
                    collocation_points = np.array(collocation_points)

                # Ensure collocation_points is a 2D array
                if collocation_points.ndim == 1:
                    collocation_points = collocation_points[:, np.newaxis]
                # Convert collocation_points to a list for saving to CSV
                array_list = collocation_points.tolist()
                if validation:
                    with open(res_dir + os.sep + "/collocation-points-validation.csv", mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(calibration_parameters)
                        writer.writerows(array_list)  # Write the array data
                else:
                    quantities_str = '_'.join(self.calibration_quantities)
                    with open(res_dir + os.sep + f"collocation-points-{quantities_str}.csv", mode='w',
                              newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(calibration_parameters)
                        writer.writerows(array_list)  # Write the array data
                    with open(restart_data_path + os.sep + "initial-collocation-points.csv", mode='w',
                              newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(calibration_parameters)
                        writer.writerows(array_list)  # Write the array data

                collocation_points=collocation_points 

                for i in range(init_runs):
                    self.num_run = i + 1
                    collocation_point_sim_list = collocation_points[i].tolist()
                    logger.info(
                        f" Running  full complexity model # {self.num_run}  with collocation point : {collocation_point_sim_list} ")
                    self.update_model_controls(collocation_point_values=collocation_point_sim_list,
                                               calibration_parameters=calibration_parameters,
                                               auxiliary_file_path=fr_tbl,
                                               simulation_id=self.num_run)
                    self.run_single_simulation(self.control_file)
                    self.extract_data_point(self.tm_results_filename, self.calibration_pts_df, self.dict_output_name,
                                            self.extraction_quantities, self.num_run, self.model_dir,
                                            res_dir,output_extraction=output_extraction,output_extraction_time=output_extraction_time,n=n,compute_wall_law_diagnostics=False)
                    self.model_evaluations = self.output_processing(output_data_path=os.path.join(res_dir,
                                                                      f'{self.dict_output_name}-detailed.json'),
                                                                    delete_slf_files=self.delete_complex_outputs,
                                                                    validation=validation,
                                                                    save_extraction_outputs=True)
                    if self.num_run == self.init_runs:
                        self.model_evaluations = self.output_processing(output_data_path=os.path.join(res_dir,
                                                                                                      f'{self.dict_output_name}-detailed.json'),
                                                                        delete_slf_files=self.delete_complex_outputs,
                                                                        validation=validation,
                                                                        filter_outputs=True,
                                                                        save_extraction_outputs=True,
                                                                        run_range_filtering=(1, init_runs + 1))

                    logger.info("TELEMAC simulations time for initial runs: " + str(datetime.now() - start_time))
            # This part of the code runs BAL
            else:
                # Convert collocation_points to a numpy array if it is not already
                if not isinstance(collocation_points, np.ndarray):
                    collocation_points = np.array(collocation_points)

                # Ensure collocation_points is a 2D array
                if collocation_points.ndim == 1:
                    collocation_points = collocation_points[:, np.newaxis]

                self.bal_iteration = bal_iteration
                self.num_run = bal_iteration + init_runs

                new_collocation_point=bal_new_set_parameters
                updated_collocation_points = np.vstack((collocation_points, new_collocation_point))
                collocation_point_sim_list = updated_collocation_points[-1].tolist()
                array_list = updated_collocation_points.tolist()
                logger.info(
                    f" Running  full complexity model after BAL # {self.bal_iteration} with collocation point : {collocation_point_sim_list} ")
                if validation:
                    update_collocation_pts_file(res_dir + "/collocation-points-validation.csv",
                                                new_collocation_point=array_list)
                else:
                    quantities_str = '_'.join(self.calibration_quantities)
                    with open(res_dir + os.sep + f"collocation-points-{quantities_str}.csv", mode='w',
                              newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(calibration_parameters)
                        writer.writerows(array_list)

                self.update_model_controls(collocation_point_values=collocation_point_sim_list,
                                           calibration_parameters=calibration_parameters,
                                           auxiliary_file_path=fr_tbl,
                                           simulation_id=self.num_run)
                self.run_single_simulation(self.control_file)
                output_name_calibration = f'{self.dict_output_name}{"_".join(self.calibration_quantities)}'
                self.extract_data_point(self.tm_results_filename, self.calibration_pts_df, self.dict_output_name,
                                        self.extraction_quantities, self.num_run, self.model_dir,
                                        res_dir,output_extraction=output_extraction,output_extraction_time=output_extraction_time,n=n)
                # In this first output processing, ALL the extraction quantities are saved as .csv file in the calibration folder.
                self.output_processing(output_data_path=os.path.join(res_dir,f'{self.dict_output_name}-detailed.json'),
                                                                delete_slf_files=self.delete_complex_outputs,
                                                                validation=validation,
                                                                save_extraction_outputs=True, # When True, it saves ALL model outputs of ALL extraction quantities at ALL points as a .csv file
                                                                extraction_mode=True)
                # This part extracts the calibration quantities from the detailed dictionary as a numpy array for BAL and creates a new filtered dictionary.
                self.model_evaluations = self.output_processing(output_data_path=os.path.join(res_dir,
                                                                                              f'{self.dict_output_name}-detailed.json'),
                                                                delete_slf_files=self.delete_complex_outputs,
                                                                validation=validation,
                                                                filter_outputs=True,
                                                                save_extraction_outputs=False,
                                                                run_range_filtering=(1, init_runs +bal_iteration))
                quantities_str = '_'.join(self.calibration_quantities)
                self.output_processing(output_data_path=os.path.join(res_dir,f'{quantities_str}-detailed.json'),
                                                                delete_slf_files=self.delete_complex_outputs,
                                                                validation=validation,
                                                                save_extraction_outputs=False,
                                                                calibration_mode=True)
                logger.info("TELEMAC simulations time for initial runs: " + str(datetime.now() - start_time))

        # This part of the code only runs iterative runs without performing BAL
        else:
            if collocation_points is not None:
                # Ensure collocation_points is a 2D NumPy array
                if not isinstance(collocation_points, np.ndarray):
                    collocation_points = np.array(collocation_points)

                if collocation_points.ndim == 1:
                    if collocation_points.size == 1:
                        collocation_points = collocation_points[:, np.newaxis]
                    else:
                        collocation_points = collocation_points.reshape(1, -1)

                # Convert to list for CSV writing
                array_list = collocation_points.tolist()

                if validation:
                    # Validation case — always write validation CSV
                    with open(os.path.join(restart_data_path, "collocation-points-validation.csv"), mode='w',
                              newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(calibration_parameters)
                        writer.writerows(array_list)

                else:
                    quantities_str = '_'.join(self.calibration_quantities)

                    if self.user_collocation_points is not None:
                        # USER case — create a separate CSV for user collocation points
                        self.calibration_folder = self.restart_data_folder
                        self.dict_output_name = "user-" + self.dict_output_name

                        user_csv_path = os.path.join(
                            res_dir, f"user-collocation-points-{quantities_str}.csv"
                        )

                        with open(user_csv_path, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(calibration_parameters)
                            writer.writerows(array_list)

                        # Do NOT create or overwrite the default collocation-points.csv here

                    else:
                        # DEFAULT case — create standard collocation CSVs
                        default_csv_path = os.path.join(
                            res_dir, f"collocation-points-{quantities_str}.csv"
                        )
                        with open(default_csv_path, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(calibration_parameters)
                            writer.writerows(array_list)

                        # Also create the initial-collocation-points.csv
                        init_csv_path = os.path.join(
                            restart_data_path, "initial-collocation-points.csv"
                        )
                        with open(init_csv_path, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(calibration_parameters)
                            writer.writerows(array_list)

                # Keep reference if needed later
                collocation_points = collocation_points

                for i in range(init_runs):
                    self.num_run = i + 1
                    collocation_point_sim_list = collocation_points[i].tolist()
                    logger.info(
                        f" Running  full complexity model # {self.num_run}  with collocation point : {collocation_point_sim_list} ")
                    self.update_model_controls(collocation_point_values=collocation_point_sim_list,
                                               calibration_parameters=calibration_parameters,
                                               auxiliary_file_path=fr_tbl,
                                               simulation_id=self.num_run)
                    self.run_single_simulation(self.control_file)
                    self.extract_data_point(self.tm_results_filename, self.calibration_pts_df, self.dict_output_name,
                                            self.extraction_quantities, self.num_run, self.model_dir,
                                            self.calibration_folder, self.validation, self.user_param_values,
                                            output_extraction=output_extraction,
                                            output_extraction_time=output_extraction_time,
                                            n=n )
                    if validation:
                        output_data_path = os.path.join(restart_data_path, 'model-results-validation.json')
                    else:
                        output_data_path = os.path.join(self.calibration_folder,
                                                        f'{self.dict_output_name}-detailed.json')

                    self.model_evaluations = self.output_processing(output_data_path=output_data_path,
                                                                    delete_slf_files=self.delete_complex_outputs,
                                                                    validation=validation,
                                                                    save_extraction_outputs=True,  # This option True saves ALL model outputs of ALL required quantities at ALL points as a .csv file
                                                                    calibration_mode=True,
                                                                    extraction_mode = True, #This option True extracts the data from the dictionary and populates the array with the values from the dictionary for the parameters in extraction_quantities
                                                                    )
                    logger.info("TELEMAC simulations time for initial runs: " + str(datetime.now() - start_time))
                if kill_process:
                    return self.model_evaluations

        return self.model_evaluations

    def output_processing(
            self,
            output_data_path='',
            calibration_quantities='',
            delete_slf_files=False,
            validation=False,
            save_extraction_outputs = False,
            filter_outputs=False,
            run_range_filtering=None,
            extraction_mode = False,
            calibration_mode = False,
    ):
        """
        Processes model output data from a JSON file into a 2D array format for Bayesian calibration
        and saves the results to a CSV file.

        This method reads a JSON file specified by `output_data_path`, extracts and processes the model
        outputs, and saves them in a CSV file format suitable for Bayesian calibration.

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
            [number of total runs, number of calibration points x number of quantities], where 'number of quantities'
            represents the calibration quantities processed, and 'number of total runs' is the sum
            of initial runs and Bayesian active learning iterations. The columns are intercalated to store the quantities outputs.
            This array is also saved to a CSV file in the specified directory.
        """

        with open(output_data_path, "r") as file:
            output_data = json.load(file)

        # Get quantities from the first entry dynamically
        extraction_quantities = self.extraction_quantities  # Extract quantities from the first run of the first calibration point
        calibration_quantities = self.calibration_quantities
        n_calibration_pts = self.nloc
        n_total_runs = self.init_runs + self.bal_iteration
        # Number of quantities per location for calibration purposes
        num_quantities_calibration = len(calibration_quantities)
        num_quantities_extraction = len(extraction_quantities)
        # Initialize a 2D NumPy array with zeros
        model_results_calibration = np.zeros((num_quantities_calibration * n_total_runs, n_calibration_pts))
        model_results_extraction = np.zeros((num_quantities_extraction * n_total_runs, n_calibration_pts))
        # Filtering
        if filter_outputs:
            quantities_str = '_'.join(calibration_quantities)
            filtered_output_data_extraction = filter_model_outputs(data_dict=output_data,
                                               quantities=extraction_quantities,
                                               run_range_filtering=run_range_filtering)
            filtered_output_data_calibration = filter_model_outputs(data_dict=output_data,
                                               quantities=calibration_quantities,
                                               run_range_filtering=run_range_filtering)
            if validation:
                pass
            else:
                with open(os.path.join(self.calibration_folder,f'{self.dict_output_name}-detailed.json'), 'w') as json_file:
                    json.dump(filtered_output_data_extraction, json_file, indent=4)
                with open(os.path.join(self.calibration_folder, f'{quantities_str}-detailed.json'),
                          'w') as json_file:
                    json.dump(filtered_output_data_calibration, json_file, indent=4)
            for i, (key, values) in enumerate(filtered_output_data_extraction.items()):  # Iterate over calibration points
                for j, value_set in enumerate(values):  # Iterate over runs
                    for k, quantity in enumerate(calibration_quantities):  # Calibrate quantities
                        if quantity in value_set:
                            model_results_calibration[k * n_total_runs + j, i] = value_set[quantity]
                        else:
                            raise ValueError(f"Quantity '{quantity}' not found in data for calibration point '{key}'.")
            model_results_calibration = rearrange_array(model_results_calibration, num_quantities_calibration)
            model_results_calibration=model_results_calibration[~np.all(model_results_calibration == 0, axis=1)]
        else:
            if extraction_mode: # This mode processes (extracts data from dictionary and populate the array with the values from the dictionary )
                                # the detailed dictionary data for ALL model parameters (extraction parameters).
                for i, (key, values) in enumerate(output_data.items()):  # Iterate over calibration points
                    for j, value_set in enumerate(values):  # Iterate over runs
                        for k, quantity in enumerate(extraction_quantities):  # Extract quantities
                            model_results_extraction[k * n_total_runs + j, i] = value_set[quantity]
            if calibration_mode: # This mode processes (extracts data from dictionary and transforms into numpy arrays )
                                # the detailed dictionary data for calibration parameters.
                for i, (key, values) in enumerate(output_data.items()):
                    for j, value_set in enumerate(values):  # Iterate over runs
                        for k, quantity in enumerate(calibration_quantities):  # Calibrate quantities
                            if quantity in value_set:
                                model_results_calibration[k * n_total_runs + j, i] = value_set[quantity]
                            else:
                                raise ValueError(f"Quantity '{quantity}' not found in data for calibration point '{key}'.")
            if not extraction_mode and not calibration_mode: # This mode processes (extracts data from dictionary and transforms into numpy arrays )
                                                            # the detailed dictionary data for the calibration parameters and ALL model outputs.
                for i, (key, values) in enumerate(output_data.items()):  # Iterate over calibration points
                    for j, value_set in enumerate(values):  # Iterate over runs
                        for k, quantity in enumerate(extraction_quantities):  # Extract quantities
                            model_results_extraction[k * n_total_runs + j, i] = value_set[quantity]

                    for j, value_set in enumerate(values):  # Iterate over runs
                        for k, quantity in enumerate(calibration_quantities):  # Calibrate quantities
                            if quantity in value_set:
                                model_results_calibration[k * n_total_runs + j, i] = value_set[quantity]

            if num_quantities_calibration == 1:
                column_headers_calibration = []
                for i in range(1, n_calibration_pts + 1):  # Calibration point indices
                    for quantity in calibration_quantities:
                        column_headers_calibration.append(f'PT{i}_{quantity}')

                if validation:
                    np.savetxt(
                        os.path.join(self.restart_data_folder, 'model-results-validation.csv'),
                        model_results_calibration,
                        delimiter=',',
                        fmt='%.8f',
                        header=','.join(column_headers_calibration),
                    )
                else:
                    quantities_str = '_'.join(calibration_quantities)
                    if self.user_param_values:
                        extraction_csv_file_name = f"user-model-results-calibration-{quantities_str}.csv"
                    else:
                        extraction_csv_file_name = f"model-results-calibration-{quantities_str}.csv"
                    np.savetxt(
                        os.path.join(self.calibration_folder, extraction_csv_file_name),
                        model_results_calibration,
                        delimiter=',',
                        fmt='%.8f',
                        header=','.join(column_headers_calibration),
                    )

                if save_extraction_outputs and extraction_mode:
                    model_results_extraction = rearrange_array(model_results_extraction, num_quantities_extraction)
                    if self.user_param_values:
                        extraction_csv_file_name = 'user-model-results-extraction.csv'
                    else:
                        extraction_csv_file_name = 'model-results-extraction.csv'
                    column_headers_extraction = []
                    for i in range(1, n_calibration_pts + 1):  # Calibration point indices
                        for quantity in extraction_quantities:
                            column_headers_extraction.append(f'PT{i}_{quantity}')
                    np.savetxt(
                        os.path.join(self.calibration_folder, extraction_csv_file_name),
                        model_results_extraction,
                        delimiter=',',
                        fmt='%.8f',
                        header=','.join(column_headers_extraction),
                    )

            else:
                model_results_extraction = rearrange_array(model_results_extraction,num_quantities_extraction)
                model_results_calibration = rearrange_array(model_results_calibration,num_quantities_calibration)
                num_columns_calibration = model_results_calibration.shape[1]
                num_columns_extraction = model_results_extraction.shape[1]
                column_headers_calibration = []
                column_headers_extraction = []
                for i in range(1, num_columns_calibration // num_quantities_calibration + 1):  # Loop through each set of columns
                    for quantity in calibration_quantities:  # Loop through each calibration parameter
                        column_headers_calibration.append(f'{i}_{quantity}')
                for i in range(1, num_columns_extraction // num_quantities_extraction + 1):  # Loop through each set of columns
                    for quantity in extraction_quantities:  # Loop through each calibration parameter
                        column_headers_extraction.append(f'{i}_{quantity}')
                if validation:
                    np.savetxt(
                        os.path.join(self.restart_data_folder, 'model-results-validation.csv'),
                        model_results_extraction,
                        delimiter=',',
                        fmt='%.8f',
                        header=','.join(column_headers_extraction),
                    )
                else:
                    quantities_str = '_'.join(calibration_quantities)
                    if self.user_param_values:
                        extraction_csv_file_name = f"user-model-results-calibration-{quantities_str}.csv"
                    else:
                        extraction_csv_file_name = f"model-results-calibration-{quantities_str}.csv"
                    np.savetxt(
                        os.path.join(self.calibration_folder, extraction_csv_file_name),
                        model_results_calibration,
                        delimiter=',',
                        fmt='%.8f',
                        header=','.join(column_headers_calibration),
                    )
                if save_extraction_outputs and extraction_mode:
                    if validation:
                        pass
                    else:
                        if self.user_param_values:
                            extraction_csv_file_name = 'user-model-results-extraction.csv'
                        else:
                            extraction_csv_file_name = 'model-results-extraction.csv'
                        column_headers_extraction = []
                        for i in range(1, n_calibration_pts + 1):  # Calibration point indices
                            for quantity in extraction_quantities:
                                column_headers_extraction.append(f'PT{i}_{quantity}')
                        np.savetxt(
                            os.path.join(self.calibration_folder, extraction_csv_file_name),
                            model_results_extraction,
                            delimiter=',',
                            fmt='%.8f',
                            header=','.join(column_headers_extraction),
                        )

            if delete_slf_files:
                delete_slf(self.calibration_folder)
        return model_results_calibration

    def extract_data_point(
            self,
            input_file,
            calibration_pts_df,
            output_name,
            extraction_quantity,
            simulation_number,
            model_directory,
            results_folder_directory,
            validation=False,
            user_param_values=False,
            output_extraction="interpolated",  # "nearest": nearest node; "interpolated": IDW interpolation
            k=3,
            output_extraction_time="last",  # "last", "index", "mean_last"
            time_index=0,
            n=5,
            compute_wall_law_diagnostics=False
    ):
        """
        Extract model results at specified calibration or validation points from
        TELEMAC and/or GAIA SELAFIN result files.

        The method supports extraction of scalar variables,
        vertical layer selection based on measurement height, inverse-distance
        interpolation, and optional wall-law diagnostics.
        Extracted values are written to JSON files and result files are moved to
        the designated results directory.

        Parameters
        ----------
        input_file : str
            Name of the TELEMAC result file (.slf) to extract data from.

        calibration_pts_df : pandas.DataFrame
            DataFrame containing extraction locations. The first column must
            contain point identifiers. The following columns are expected to be:

            - column 1: x-coordinate
            - column 2: y-coordinate
            - column 3: vertical measurement offset (z)

        output_name : str
            Base name used for generated JSON output files.

        extraction_quantity : list of str
            Quantities to extract from the model results. Variables may originate
            from TELEMAC or GAIA according to the configuration mapping
            ``classification_tm_gaia_dict``.

        simulation_number : int
            Current simulation number within the calibration workflow.

        model_directory : str
            Directory containing TELEMAC and GAIA result files.

        results_folder_directory : str
            Directory where extracted results and moved result files are stored.

        validation : bool, optional
            If True, extracted values are treated as validation results and are
            written to validation-specific JSON files. Default is False.

        user_param_values : bool, optional
            Flag controlling restart-data generation. Default is False.

        output_extraction : {"nearest", "interpolated"}, optional
            Spatial extraction method.

            - ``"nearest"``: use the closest model node.
            - ``"interpolated"``: perform inverse-distance-weighted interpolation
              using the k nearest nodes.

            Default is ``"interpolated"``.

        k : int, optional
            Number of nearest nodes used for interpolation when
            ``output_extraction="interpolated"``. Ignored when using nearest-node
            extraction. Default is 3.

        output_extraction_time : {"last", "index", "mean_last"}, optional
            Temporal aggregation mode applied to the extracted time series.

            - ``"last"``: use the final time step.
            - ``"index"``: use the time step specified by ``time_index``.
            - ``"mean_last"``: average the last ``n`` time steps.

            Default is ``"last"``.

        time_index : int, optional
            Time-step index used when
            ``output_extraction_time="index"``.
            Default is 0.

        n : int, optional
            Number of final time steps used when
            ``output_extraction_time="mean_last"``.
            Default is 5.

        compute_wall_law_diagnostics : bool, optional
            If True, compute wall-law diagnostic quantities from TELEMAC 3D
            results and the generated 2D result file. Diagnostics include
            friction velocity, y-plus values, bottom friction parameters,
            near-bed velocity information, and the complete modeled vertical
            velocity profile. Default is False.

        Returns
        -------
        None
            Results are written to JSON files and model result files are moved
            to the results directory.

        Notes
        -----
        - If ``"3D VELOCITY MAGNITUDE"`` is requested, it is computed from
          ``VELOCITY U``, ``VELOCITY V``, and ``VELOCITY W``.
        - For 3D simulations, the vertical layer closest to the measurement
          elevation is automatically selected using ``ELEVATION Z``.
        - Wall-law diagnostics require at least two vertical planes
          (``NPLAN >= 2``).
        """
        self.tm_results_filename = input_file

        classification_tm_gaia_dict = config_telemac.classification_tm_gaia_dict

        # ============================================================
        # WALL-LAW CONSTANTS FROM CONFIG
        # ============================================================
        kappa = config_telemac.von_Karman_constant
        nikuradse_log_factor = config_telemac.nikuradse_log_factor
        kinematic_viscosity = config_telemac.kinematic_viscosity_water

        bottom_friction_2d_variable_candidates = (
            config_telemac.slf_2d_variables_from_3d
        )

        # ============================================================
        # DERIVED TELEMAC QUANTITY: 3D VELOCITY MAGNITUDE
        # ============================================================
        velocity_magnitude_quantity = "3D VELOCITY MAGNITUDE"

        velocity_components = [
            "VELOCITY U",
            "VELOCITY V",
            "VELOCITY W"
        ]

        compute_3d_velocity_magnitude = (
                velocity_magnitude_quantity in extraction_quantity
        )

        # ============================================================
        # WALL-LAW DIAGNOSTIC QUANTITIES
        # ============================================================
        # Since the modeled vertical profile is diagnostic information,
        # it is also excluded from the normal calibration-output dictionary.
        wall_law_required_3d_components = [
            "VELOCITY U",
            "VELOCITY V",
            "VELOCITY W",
            "ELEVATION Z"
        ]

        wall_law_diagnostic_quantities = {
            "FRICTION VELOCITY",
            "Y PLUS",
            "DZ PLANE 1 2",
            "U PLANE 2",
            "BOTTOM FRICTION",
            "BOTTOM FRICTION SOURCE FILE",
            "BOTTOM FRICTION VARIABLE",
            "VELOCITY PROFILE"
        }

        # If these are accidentally included in extraction_quantity,
        # remove them from the calibration/model-output dictionary.
        # They are diagnostics and will be saved separately.
        if any(q in extraction_quantity for q in wall_law_diagnostic_quantities):
            compute_wall_law_diagnostics = True

        calibration_extraction_quantity = [
            q for q in extraction_quantity
            if q not in wall_law_diagnostic_quantities
        ]

        # ============================================================
        # CLASSIFY REQUESTED QUANTITIES
        # ============================================================
        telemac_quantities = [
            q for q in calibration_extraction_quantity
            if classification_tm_gaia_dict.get(q) == "telemac"
        ]

        gaia_quantities = [
            q for q in calibration_extraction_quantity
            if classification_tm_gaia_dict.get(q) == "gaia"
        ]

        internal_telemac_helpers = set()

        # If 3D velocity magnitude is requested, internally add U, V, W.
        # These are helper variables. They will not be saved unless explicitly requested.
        if compute_3d_velocity_magnitude:
            for comp in velocity_components:
                if comp not in telemac_quantities:
                    telemac_quantities.append(comp)
                    internal_telemac_helpers.add(comp)

        # If wall-law diagnostics are requested, internally add the 3D variables needed.
        # Bottom friction itself is NOT read from the 3D SLF.
        # It is read from the generated 2D SLF.
        #
        # VELOCITY W is included here because the wall-law diagnostics JSON now also
        # receives the full modeled vertical velocity profile.
        if compute_wall_law_diagnostics:
            for comp in wall_law_required_3d_components:
                if comp not in telemac_quantities:
                    telemac_quantities.append(comp)
                    internal_telemac_helpers.add(comp)

        # ============================================================
        # FILE PATHS
        # ============================================================
        slf_files = {
            "telemac": os.path.join(model_directory, self.tm_results_filename)
        }

        if self.gaia_cas is not None:
            slf_files["gaia"] = os.path.join(
                model_directory,
                self.gaia_results_filename
            )

        json_path = os.path.join(
            results_folder_directory,
            f"{output_name}.json"
        )

        json_path_detailed = os.path.join(
            results_folder_directory,
            f"{output_name}-detailed.json"
        )

        json_path_restart_data = os.path.join(
            self.restart_data_folder,
            "initial-model-outputs.json"
        )

        if validation:
            json_path_wall_law_diagnostics = os.path.join(
                self.restart_data_folder,
                "model-results-validation-wall-law-diagnostics.json"
            )
        else:
            json_path_wall_law_diagnostics = os.path.join(
                results_folder_directory,
                f"{output_name}-wall-law-diagnostics.json"
            )

        keys = calibration_pts_df.iloc[:, 0].tolist()

        differentiated_dict = {}
        wall_law_diagnostics_dict = {}

        logger.info(
            f"Extracting from {input_file} using quantities: {extraction_quantity}"
        )

        # ============================================================
        # PRECOMPUTE MODELS
        # ============================================================
        model_cache = {}

        for model_source, quantities in zip(
                ["telemac", "gaia"],
                [telemac_quantities, gaia_quantities]
        ):

            if not quantities:
                continue

            if model_source not in slf_files:
                continue

            slf = ppSELAFIN(slf_files[model_source])
            slf.readHeader()
            slf.readTimes()

            variables = [' '.join(v.split()) for v in slf.getVarNames()]
            var_index = {v: i for i, v in enumerate(variables)}

            # --------------------------------------------------------
            # CHECK AVAILABLE VARIABLES
            # --------------------------------------------------------
            # "3D VELOCITY MAGNITUDE" is derived, so it is not expected
            # to exist directly inside the SLF file.
            missing_quantities = [
                q for q in quantities
                if q not in var_index and q != velocity_magnitude_quantity
            ]

            if missing_quantities:
                raise ValueError(
                    f"The following quantities were requested for {model_source.upper()} "
                    f"but are not available in the SLF file: {missing_quantities}. "
                    f"Available variables are: {variables}"
                )

            # Check that U, V, W exist if 3D velocity magnitude is requested.
            if model_source == "telemac" and compute_3d_velocity_magnitude:
                missing_components = [
                    comp for comp in velocity_components
                    if comp not in var_index
                ]

                if missing_components:
                    raise ValueError(
                        "Cannot compute 3D VELOCITY MAGNITUDE because the following "
                        f"velocity components are missing in the TELEMAC SLF file: "
                        f"{missing_components}. Available variables are: {variables}"
                    )

            NVAR = len(variables)
            NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = slf.getMesh()
            NPLAN = slf.getNPLAN()

            # Check wall-law requirements after NPLAN is known.
            if model_source == "telemac" and compute_wall_law_diagnostics:

                missing_wall_law_components = [
                    comp for comp in wall_law_required_3d_components
                    if comp not in var_index
                ]

                if missing_wall_law_components:
                    raise ValueError(
                        "Cannot compute wall-law diagnostics and velocity profile "
                        "because these 3D SLF variables are missing: "
                        f"{missing_wall_law_components}. "
                        f"Available variables are: {variables}"
                    )

                if NPLAN < 2:
                    raise ValueError(
                        "Cannot compute wall-law diagnostics because the 3D result "
                        f"has NPLAN={NPLAN}. At least 2 vertical planes are required."
                    )

            x2d = x[:len(x) // NPLAN]
            y2d = y[:len(x) // NPLAN]

            tree = spatial.cKDTree(np.column_stack((x2d, y2d)))
            step = NPOIN // NPLAN

            model_cache[model_source] = {
                "slf": slf,
                "variables": variables,
                "var_index": var_index,
                "NVAR": NVAR,
                "NPLAN": NPLAN,
                "step": step,
                "tree": tree,
                "x2d": x2d,
                "y2d": y2d,
                "quantities": quantities
            }

        # ============================================================
        # LOAD GENERATED 2D SLF FOR WALL-LAW DIAGNOSTICS
        # ============================================================
        if compute_wall_law_diagnostics:
            slf_2d_data = self._load_generated_2d_slf_for_bottom_friction(
                model_directory=model_directory,
                bottom_friction_2d_variable_candidates=bottom_friction_2d_variable_candidates
            )
        else:
            slf_2d_data = None

        # ============================================================
        # MAIN LOOP
        # ============================================================
        for key_index, key in enumerate(keys):

            xu = calibration_pts_df.iloc[key_index, 1]
            yu = calibration_pts_df.iloc[key_index, 2]
            zu = calibration_pts_df.iloc[key_index, 3]

            differentiated_values = {}
            wall_law_diagnostic_values = {}

            for model_source in ["telemac", "gaia"]:

                if model_source not in model_cache:
                    continue

                data = model_cache[model_source]

                slf = data["slf"]
                tree = data["tree"]
                var_index = data["var_index"]
                NPLAN = data["NPLAN"]
                step = data["step"]
                x2d = data["x2d"]
                y2d = data["y2d"]
                quantities = data["quantities"]

                if not quantities:
                    continue

                # --------------------------------------------------------
                # NODE SELECTION
                # --------------------------------------------------------
                k_use = 1 if output_extraction == "nearest" else k

                d, idx = tree.query((xu, yu), k=k_use)

                if k_use == 1:
                    idx_base = np.array([idx])
                    idx_coord = np.array([[x2d[idx], y2d[idx]]])
                else:
                    idx_base = np.array(idx)
                    idx_coord = np.column_stack((x2d[idx], y2d[idx]))

                logger.info(
                    f"[{model_source.upper()}] Point {key} ({xu:.3f},{yu:.3f})"
                )

                # --------------------------------------------------------
                # WALL-LAW DIAGNOSTICS + FULL MODELED VELOCITY PROFILE
                # --------------------------------------------------------
                # Computed from:
                #   - 3D SLF: plane 1 and plane 2
                #   - generated 2D SLF: bottom friction / Nikuradse ks
                #
                # Saved separately in the wall-law diagnostics JSON.
                # Not saved in differentiated_dict.
                if model_source == "telemac" and compute_wall_law_diagnostics:
                    wall_law_diagnostic_values = (
                        self._compute_wall_law_from_3d_plane2_and_2d_bottom_friction(
                            slf_3d=slf,
                            var_index_3d=var_index,
                            idx_base_3d=idx_base,
                            idx_coord_3d=idx_coord,
                            step_3d=step,
                            k_use=k_use,
                            target_xy=(xu, yu),
                            key=key,
                            slf_2d_data=slf_2d_data,
                            output_extraction_time=output_extraction_time,
                            time_index=time_index,
                            n=n,
                            kappa=kappa,
                            nikuradse_log_factor=nikuradse_log_factor,
                            kinematic_viscosity=kinematic_viscosity
                        )
                    )

                    wall_law_diagnostic_values["VELOCITY PROFILE"] = (
                        self._vertical_velocity_profile(
                            slf=slf,
                            var_index=var_index,
                            idx_base=idx_base,
                            idx_coord=idx_coord,
                            step=step,
                            k_use=k_use,
                            NPLAN=NPLAN,
                            NVAR=data["NVAR"],
                            target_xy=(xu, yu),
                            output_extraction_time=output_extraction_time,
                            time_index=time_index,
                            n=n
                        )
                    )

                # --------------------------------------------------------
                # VERTICAL PROFILE FOR NORMAL EXTRACTION
                # --------------------------------------------------------
                # This block only selects the closest TELEMAC vertical layer
                # to the requested measurement height zu.
                # The full velocity profile is already saved above, but only
                # inside wall_law_diagnostic_values.
                if NPLAN == 1:
                    p = 0

                    logger.info(
                        f"[{model_source.upper()}] Point {key} � single layer model"
                    )

                else:
                    if "ELEVATION Z" not in var_index:
                        raise ValueError(
                            f"'ELEVATION Z' is required to select the vertical layer, "
                            f"but it is missing in the {model_source.upper()} SLF file."
                        )

                    z_index = var_index["ELEVATION Z"]
                    base_idx_for_z = idx_base[0]

                    z_profile = np.zeros(NPLAN)

                    for p_i in range(NPLAN):
                        node_idx = base_idx_for_z + p_i * step

                        slf.readVariablesAtNode(node_idx)
                        all_times_outputs = slf.getVarValuesAtNode()

                        results_z = self._apply_time_mode(
                            all_times_outputs=all_times_outputs,
                            output_extraction_time=output_extraction_time,
                            time_index=time_index,
                            n=n
                        )

                        z_profile[p_i] = results_z[z_index]

                    z_target = z_profile[0] + zu
                    p = np.argmin(np.abs(z_profile - z_target))

                    logger.info(
                        f"[{model_source.upper()}] Point {key} � selected layer "
                        f"p={p + 1}/{NPLAN} (zu={zu:.3f})"
                    )

                # --------------------------------------------------------
                # VALUE EXTRACTION
                # --------------------------------------------------------
                results_all = np.zeros((k_use, data["NVAR"]))

                for j in range(k_use):
                    node_idx = idx_base[j] + p * step

                    logger.debug(
                        f"[{model_source.upper()}] extracting node={node_idx}"
                    )

                    slf.readVariablesAtNode(node_idx)
                    all_times_outputs = slf.getVarValuesAtNode()

                    vals = self._apply_time_mode(
                        all_times_outputs=all_times_outputs,
                        output_extraction_time=output_extraction_time,
                        time_index=time_index,
                        n=n
                    )

                    results_all[j, :] = vals

                # --------------------------------------------------------
                # INTERPOLATION
                # --------------------------------------------------------
                if k_use != 1:
                    results = interpolate_values(
                        idx_coord,
                        results_all,
                        (xu, yu)
                    )
                else:
                    results = results_all[0]

                # --------------------------------------------------------
                # STORE NORMAL RESULTS
                # --------------------------------------------------------
                for q in quantities:

                    # Derived variable: compute internally from U, V, W.
                    if q == velocity_magnitude_quantity:

                        u = results[var_index["VELOCITY U"]]
                        v = results[var_index["VELOCITY V"]]
                        w = results[var_index["VELOCITY W"]]

                        differentiated_values[q] = float(
                            np.sqrt(u ** 2 + v ** 2 + w ** 2)
                        )

                    # Internal helper components:
                    # skip them if they were not explicitly requested by the user.
                    elif (
                            q in internal_telemac_helpers
                            and q not in calibration_extraction_quantity
                    ):

                        continue

                    # Normal SLF variable extraction.
                    else:

                        differentiated_values[q] = results[var_index[q]]

            differentiated_dict[key] = differentiated_values

            if compute_wall_law_diagnostics:
                wall_law_diagnostics_dict[key] = wall_law_diagnostic_values

        # ============================================================
        # JSON HANDLING
        # ============================================================
        if simulation_number == 1:

            if os.path.exists(json_path):
                os.rename(
                    json_path,
                    json_path.replace(".json", "_old.json")
                )

            if os.path.exists(json_path_detailed):
                os.rename(
                    json_path_detailed,
                    json_path_detailed.replace(".json", "_old.json")
                )

            if (
                    compute_wall_law_diagnostics
                    and os.path.exists(json_path_wall_law_diagnostics)
            ):
                os.rename(
                    json_path_wall_law_diagnostics,
                    json_path_wall_law_diagnostics.replace(".json", "_old.json")
                )

        if validation:
            json_target = os.path.join(
                self.restart_data_folder,
                "model-results-validation.json"
            )
        else:
            json_target = json_path_detailed

        update_json_file(
            json_path=json_target,
            modeled_values_dict=differentiated_dict,
            detailed_dict=True
        )

        # Wall-law diagnostics are saved separately.
        # They are not calibration targets.
        #
        # The full modeled velocity profile is also saved here, and only here.
        if compute_wall_law_diagnostics:
            update_json_file(
                json_path=json_path_wall_law_diagnostics,
                modeled_values_dict=wall_law_diagnostics_dict,
                detailed_dict=True
            )

        if (
                simulation_number == self.init_runs
                and not validation
                and not user_param_values
        ):
            update_json_file(
                json_path=json_path_detailed,
                modeled_values_dict=differentiated_dict,
                detailed_dict=True,
                save_dict=True,
                saving_path=json_path_restart_data
            )

        try:
            shutil.move(
                os.path.join(model_directory, self.tm_results_filename),
                results_folder_directory
            )
            shutil.move(
                os.path.join(model_directory, self.tm_2d_results_filename_from_3d),
                results_folder_directory
            )

            # Move generated 2D result file if it exists.
            if compute_wall_law_diagnostics:

                if (
                        hasattr(self, "tm_2d_results_filename")
                        and self.tm_2d_results_filename_from_3d is not None
                ):
                    tm_2d_results_filename = self.tm_2d_results_filename_from_3d
                else:
                    tm_2d_results_filename = self._get_2d_result_filename_from_3d(
                        self.tm_results_filename
                    )

                tm_2d_result_path = os.path.join(
                    model_directory,
                    tm_2d_results_filename
                )

                if os.path.exists(tm_2d_result_path):
                    shutil.move(
                        tm_2d_result_path,
                        results_folder_directory
                    )

            if self.gaia_cas is not None:
                shutil.move(
                    os.path.join(model_directory, self.gaia_results_filename),
                    results_folder_directory
                )

        except Exception as error:
            print(
                "ERROR: could not move results file to "
                + self.res_dir
                + "\nREASON:\n"
                + str(error)
            )

    @staticmethod
    def tbl_creator(
            zone_identifier,
            val,
            friction_file_path,
            veg_param_number=None,
            veg_indicator=False,
    ):
        """
        Modifies the FRICTION DATA FILE (.tbl) for Telemac simulations based on the specified zone,
        value, and optional vegetation parameters. This method updates the friction values in the table
        for different zones as part of the calibration process and also the friction parameters for a previous selected
        vegetation friction rule.

        Parameters
        ----------
        zone_identifier : str
            Identifier for the friction zone to be updated in the friction table.
        val : str
            The new friction value to be set for the specified zone.
        friction_file_path : str
            The file path to the existing friction file (.tbl) that will be modified.
        veg_param_number : str, optional
            The vegetation parameter number associated with the zone, if applicable.
            Default is None, indicating no vegetation parameter is to be updated.
        veg_indicator : bool, optional
            Indicator whether vegetation parameters should be modified in the friction file.
            Default is False, which means only friction values are updated.

        Returns
        -------
        None
            The function updates the friction file in place and does not return any value.
        """

        with open(friction_file_path, 'r') as file:
            file_lines = file.readlines()
        updated_lines = []
        for line in file_lines:
            line_list = list(filter(None, line.split()))
            if line_list and line_list[0].startswith('*'):
                updated_lines.append(line)
                continue
            if line_list[0] == zone_identifier:
                if veg_indicator==False:
                    if len(line_list) > 2 and line_list[2] != 'NULL':
                        try:
                            line_list[2] = str(val)
                        except ValueError:
                            pass
                else:
                    param_column_index=4+int(veg_param_number)
                    if len(line_list) > 2 and line_list[2] != 'NULL':
                        try:
                            line_list[param_column_index] = str(val)
                        except ValueError:
                            pass

                updated_zone_line = '\t'.join(line_list)
                updated_lines.append(updated_zone_line + '\n')
            else:
                updated_lines.append(line)

        with open(friction_file_path, 'w') as file:
            file.writelines(updated_lines)
    # ============================================================
    # TIME HELPER FOR TELEMAC EXTRACTION
    # ============================================================
    def _apply_time_mode(
            self,
            all_times_outputs,
            output_extraction_time="last",
            time_index=0,
            n=5
    ):
        if output_extraction_time == "last":
            return all_times_outputs[-1]

        elif output_extraction_time == "index":
            return all_times_outputs[time_index]

        elif output_extraction_time == "mean_last":
            return np.mean(all_times_outputs[-n:], axis=0)

        else:
            raise ValueError(
                "Invalid output_extraction_time. "
                "Use 'last', 'index', or 'mean_last'."
            )

    # ============================================================
    # GENERAL HELPERS FOR WALL-LAW DIAGNOSTICS
    # ============================================================
    def _get_2d_result_filename_from_3d(self, input_file_3d):
        """
        Converts:
            result.slf -> result_2d.slf
        """
        root, ext = os.path.splitext(input_file_3d)
        return f"{root}_2d{ext}"

    def _normalize_var_name(self, name):
        return " ".join(str(name).split()).upper()

    def _interpolate_scalar(self, idx_coord, values, target_xy):
        values = np.asarray(values, dtype=float)

        if len(values) == 1:
            return float(values[0])

        interpolated = interpolate_values(
            idx_coord,
            values.reshape(-1, 1),
            target_xy
        )

        return float(np.asarray(interpolated).ravel()[0])

    def _vertical_velocity_profile(
            self,
            slf,
            var_index,
            idx_base,
            idx_coord,
            step,
            k_use,
            NPLAN,
            NVAR,
            target_xy,
            output_extraction_time="last",
            time_index=0,
            n=5
    ):
        """
        Extracts the full modeled vertical velocity profile at one horizontal point.

        This is intended for the wall-law diagnostics JSON only.
        One profile entry is saved per TELEMAC vertical plane/layer.
        """

        required_vars = [
            "ELEVATION Z",
            "VELOCITY U",
            "VELOCITY V",
            "VELOCITY W"
        ]

        missing_vars = [
            var for var in required_vars
            if var not in var_index
        ]

        if missing_vars:
            raise ValueError(
                "Cannot extract modeled vertical velocity profile because these "
                f"variables are missing: {missing_vars}. "
                f"Available variables are: {list(var_index.keys())}"
            )

        z_index = var_index["ELEVATION Z"]
        u_index = var_index["VELOCITY U"]
        v_index = var_index["VELOCITY V"]
        w_index = var_index["VELOCITY W"]

        velocity_profile = []
        bed_z = None

        for p_i in range(NPLAN):

            results_all = np.zeros((k_use, NVAR))

            for j in range(k_use):
                node_idx = idx_base[j] + p_i * step

                slf.readVariablesAtNode(node_idx)
                all_times_outputs = slf.getVarValuesAtNode()

                vals = self._apply_time_mode(
                    all_times_outputs=all_times_outputs,
                    output_extraction_time=output_extraction_time,
                    time_index=time_index,
                    n=n
                )

                results_all[j, :] = vals

            if k_use != 1:
                results = interpolate_values(
                    idx_coord,
                    results_all,
                    target_xy
                )
            else:
                results = results_all[0]

            z_abs = float(results[z_index])

            if bed_z is None:
                bed_z = z_abs

            z_above_bed = float(z_abs - bed_z)

            u = float(results[u_index])
            v = float(results[v_index])
            w = float(results[w_index])

            velocity_horizontal = float(np.sqrt(u ** 2 + v ** 2))
            velocity_3d = float(np.sqrt(u ** 2 + v ** 2 + w ** 2))

            velocity_profile.append({
                "LAYER": int(p_i + 1),
                "Z": z_abs,
                "Z ABOVE BED": z_above_bed,
                "VELOCITY U": u,
                "VELOCITY V": v,
                "VELOCITY W": w,
                "HORIZONTAL VELOCITY MAGNITUDE": velocity_horizontal,
                "3D VELOCITY MAGNITUDE": velocity_3d
            })

        return velocity_profile
    # ============================================================
    # GENERATED 2D SLF HELPERS
    # ============================================================
    def _find_2d_bottom_friction_variable(
            self,
            variables_2d,
            bottom_friction_2d_variable_candidates=None
    ):
        """
        Finds the bottom friction / Nikuradse ks variable in the generated 2D SLF.
        """

        if bottom_friction_2d_variable_candidates is None:
            bottom_friction_2d_variable_candidates = config_telemac.slf_2d_variables_from_3d

        normalized_var_lookup = {
            self._normalize_var_name(var_name): var_name
            for var_name in variables_2d
        }

        for candidate in bottom_friction_2d_variable_candidates:
            candidate_clean = self._normalize_var_name(candidate)

            if candidate_clean in normalized_var_lookup:
                return normalized_var_lookup[candidate_clean]

        raise ValueError(
            "Could not find a bottom-friction / Nikuradse variable in the "
            "generated 2D SLF file. "
            f"Tried these names: {bottom_friction_2d_variable_candidates}. "
            f"Available 2D SLF variables are: {variables_2d}"
        )

    def _load_generated_2d_slf_for_bottom_friction(
            self,
            model_directory,
            bottom_friction_2d_variable_candidates=None
    ):
        """
        Loads the 2D result file generated from the TELEMAC-3D simulation.

        Expected default:
            3D file: result.slf
            2D file: result_2d.slf

        If self.tm_2d_results_filename exists, it is used instead.
        """

        if (
            hasattr(self, "tm_2d_results_filename")
            and self.tm_2d_results_filename_from_3d is not None
        ):
            input_file_2d = self.tm_2d_results_filename_from_3d
        else:
            input_file_2d = self._get_2d_result_filename_from_3d(
                self.tm_results_filename
            )

        slf_2d_path = os.path.join(model_directory, input_file_2d)

        if not os.path.exists(slf_2d_path):
            raise FileNotFoundError(
                "The generated 2D result file required for wall-law diagnostics "
                f"was not found: {slf_2d_path}. "
                "Expected the same name as the 3D SLF with '_2d' before '.slf', "
                "unless self.tm_2d_results_filename is explicitly defined."
            )

        slf_2d = ppSELAFIN(slf_2d_path)
        slf_2d.readHeader()
        slf_2d.readTimes()

        variables_2d = [' '.join(v.split()) for v in slf_2d.getVarNames()]
        var_index_2d = {v: i for i, v in enumerate(variables_2d)}

        bottom_friction_variable = self._find_2d_bottom_friction_variable(
            variables_2d=variables_2d,
            bottom_friction_2d_variable_candidates=bottom_friction_2d_variable_candidates
        )

        NELEM2D, NPOIN2D, NDP2D, IKLE2D, IPOBO2D, x2d_slf, y2d_slf = slf_2d.getMesh()

        tree_2d = spatial.cKDTree(np.column_stack((x2d_slf, y2d_slf)))

        logger.info(
            f"Using generated 2D SLF for wall-law diagnostics: {slf_2d_path}"
        )
        logger.info(
            f"Using bottom-friction variable from generated 2D SLF: "
            f"{bottom_friction_variable}"
        )

        return {
            "slf": slf_2d,
            "path": slf_2d_path,
            "filename": input_file_2d,
            "variables": variables_2d,
            "var_index": var_index_2d,
            "bottom_friction_variable": bottom_friction_variable,
            "bottom_friction_index": var_index_2d[bottom_friction_variable],
            "x": x2d_slf,
            "y": y2d_slf,
            "tree": tree_2d
        }

    def _extract_bottom_friction_from_2d_slf(
            self,
            slf_2d_data,
            target_xy,
            output_extraction_time="last",
            time_index=0,
            n=5
    ):
        """
        Extracts bottom friction / Nikuradse ks from the generated 2D SLF
        at a horizontal coordinate.

        Nearest-node extraction is used because the generated 2D SLF should
        share the same horizontal mesh as the 3D SLF.
        """

        slf_2d = slf_2d_data["slf"]
        tree_2d = slf_2d_data["tree"]
        friction_index = slf_2d_data["bottom_friction_index"]

        d2, idx2 = tree_2d.query(target_xy, k=1)
        node_2d = int(idx2)

        slf_2d.readVariablesAtNode(node_2d)
        all_times_outputs_2d = slf_2d.getVarValuesAtNode()

        vals_2d = self._apply_time_mode(
            all_times_outputs=all_times_outputs_2d,
            output_extraction_time=output_extraction_time,
            time_index=time_index,
            n=n
        )

        bottom_friction = float(vals_2d[friction_index])

        if bottom_friction <= 0.0:
            raise ValueError(
                f"Invalid bottom friction / Nikuradse ks extracted from 2D SLF "
                f"at coordinate {target_xy}: value={bottom_friction}. "
                "It must be > 0."
            )

        return bottom_friction

    # ============================================================
    # WALL-LAW DIAGNOSTICS FROM 3D PLANE 2 + 2D BOTTOM FRICTION
    # ============================================================
    def _compute_wall_law_from_3d_plane2_and_2d_bottom_friction(
            self,
            slf_3d,
            var_index_3d,
            idx_base_3d,
            idx_coord_3d,
            step_3d,
            k_use,
            target_xy,
            key,
            slf_2d_data,
            output_extraction_time="last",
            time_index=0,
            n=5,
            kappa=None,
            nikuradse_log_factor=None,
            kinematic_viscosity=None
    ):
        """
        Computes TELEMAC-style friction velocity and y+.

        From 3D SLF:
            - VELOCITY U at plane 2
            - VELOCITY V at plane 2
            - ELEVATION Z at plane 1 and plane 2

        From generated 2D SLF:
            - bottom friction / Nikuradse ks
        """

        if kappa is None:
            kappa = config_telemac.von_Karman_constant

        if nikuradse_log_factor is None:
            nikuradse_log_factor = config_telemac.nikuradse_log_factor

        if kinematic_viscosity is None:
            kinematic_viscosity = config_telemac.kinematic_viscosity_water

        if kinematic_viscosity <= 0.0:
            raise ValueError(
                f"Cannot compute Y PLUS because "
                f"kinematic_viscosity={kinematic_viscosity}. It must be > 0."
            )

        z_index = var_index_3d["ELEVATION Z"]
        u_index = var_index_3d["VELOCITY U"]
        v_index = var_index_3d["VELOCITY V"]

        friction_velocity_values = np.zeros(k_use)
        y_plus_values = np.zeros(k_use)
        dz_values = np.zeros(k_use)
        u_plane2_values = np.zeros(k_use)
        bottom_friction_values = np.zeros(k_use)

        for j in range(k_use):

            node_plane1 = idx_base_3d[j]
            node_plane2 = idx_base_3d[j] + step_3d

            xy_neighbor = (
                float(idx_coord_3d[j, 0]),
                float(idx_coord_3d[j, 1])
            )

            # ----------------------------------------------------
            # 2D SLF: bottom friction / Nikuradse ks
            # ----------------------------------------------------
            bottom_friction = self._extract_bottom_friction_from_2d_slf(
                slf_2d_data=slf_2d_data,
                target_xy=xy_neighbor,
                output_extraction_time=output_extraction_time,
                time_index=time_index,
                n=n
            )

            # ----------------------------------------------------
            # 3D SLF plane 1: bed elevation
            # ----------------------------------------------------
            slf_3d.readVariablesAtNode(node_plane1)
            all_times_outputs_plane1 = slf_3d.getVarValuesAtNode()

            vals_plane1 = self._apply_time_mode(
                all_times_outputs=all_times_outputs_plane1,
                output_extraction_time=output_extraction_time,
                time_index=time_index,
                n=n
            )

            z_plane1 = vals_plane1[z_index]

            # ----------------------------------------------------
            # 3D SLF plane 2: near-bed velocity and elevation
            # ----------------------------------------------------
            slf_3d.readVariablesAtNode(node_plane2)
            all_times_outputs_plane2 = slf_3d.getVarValuesAtNode()

            vals_plane2 = self._apply_time_mode(
                all_times_outputs=all_times_outputs_plane2,
                output_extraction_time=output_extraction_time,
                time_index=time_index,
                n=n
            )

            z_plane2 = vals_plane2[z_index]

            u_plane2 = vals_plane2[u_index]
            v_plane2 = vals_plane2[v_index]

            dz = float(z_plane2 - z_plane1)

            if dz <= 0.0:
                raise ValueError(
                    f"Cannot compute wall-law diagnostics for point {key}: "
                    f"Dz = Z_plane2 - Z_plane1 = {dz}. It must be > 0."
                )

            u_horizontal_plane2 = float(
                np.sqrt(u_plane2**2 + v_plane2**2)
            )

            log_argument = float(
                nikuradse_log_factor * dz / bottom_friction
            )

            if log_argument <= 1.0:
                raise ValueError(
                    f"Cannot compute wall-law diagnostics reliably for point {key}: "
                    f"log argument = {log_argument:.6f}. "
                    f"Computed as {nikuradse_log_factor} * Dz / bottom_friction, "
                    f"with Dz={dz:.6e} m and bottom_friction={bottom_friction:.6e} m. "
                    "The logarithm denominator must be positive. "
                    "Check whether the 2D variable is really Nikuradse ks in meters."
                )

            friction_velocity = float(
                u_horizontal_plane2 * kappa / np.log(log_argument)
            )

            y_plus = float(
                friction_velocity * dz / kinematic_viscosity
            )

            friction_velocity_values[j] = friction_velocity
            y_plus_values[j] = y_plus
            dz_values[j] = dz
            u_plane2_values[j] = u_horizontal_plane2
            bottom_friction_values[j] = bottom_friction

        friction_velocity_interp = self._interpolate_scalar(
            idx_coord_3d,
            friction_velocity_values,
            target_xy
        )

        y_plus_interp = self._interpolate_scalar(
            idx_coord_3d,
            y_plus_values,
            target_xy
        )

        dz_interp = self._interpolate_scalar(
            idx_coord_3d,
            dz_values,
            target_xy
        )

        u_plane2_interp = self._interpolate_scalar(
            idx_coord_3d,
            u_plane2_values,
            target_xy
        )

        bottom_friction_interp = self._interpolate_scalar(
            idx_coord_3d,
            bottom_friction_values,
            target_xy
        )

        return {
            "FRICTION VELOCITY": friction_velocity_interp,
            "Y PLUS": y_plus_interp,
            "DZ PLANE 1 2": dz_interp,
            "U PLANE 2": u_plane2_interp,
            "BOTTOM FRICTION": bottom_friction_interp,
            "BOTTOM FRICTION SOURCE FILE": slf_2d_data["filename"],
            "BOTTOM FRICTION VARIABLE": slf_2d_data["bottom_friction_variable"]
        }

    @staticmethod
    def check_tm_inputs(user_inputs):
        # Helper function to check if a path exists and print success message
        def path_exists(path, path_name):
            if os.path.exists(path):
                print(f"{path_name} exists: {path}")
            else:
                raise ValueError(f"{path_name} does not exist: {path}")
            if not isinstance(path, str):
                raise TypeError(f"{path_name} should be a string")

        # Extract values from dictionary
        control_file = user_inputs['control_file_name']
        friction_file = user_inputs['friction_file']
        tm_solver = user_inputs['Telemac_solver']
        model_simulation_path = user_inputs['model_simulation_path']
        results_folder_path = user_inputs['results_folder_path']
        calib_pts_csv_file = user_inputs['calib_pts_file_path']
        n_cpus = user_inputs['n_cpus']
        init_runs = user_inputs['init_runs']
        calibration_parameters = user_inputs['calibration_parameters']
        param_values = user_inputs['param_values']
        calibration_quantities = user_inputs['calibration_quantities']
        dict_output_name = user_inputs['dict_output_name']
        results_filename_base = user_inputs['results_filename_base']
        parameter_sampling_method = user_inputs['parameter_sampling_method']
        n_max_tp = user_inputs['n_max_tp']
        n_samples = user_inputs['n_samples']
        mc_samples = user_inputs['mc_samples']
        mc_exploration = user_inputs['mc_exploration']
        eval_steps = user_inputs['eval_steps']

        # Check model_simulation_path
        path_exists(model_simulation_path, "model_simulation_path")

        # Check control_file
        if os.path.exists(os.path.join(model_simulation_path, control_file)):
            print(f"Control file exists: {control_file}")
        else:
            raise ValueError(f"Control file does not exist: {control_file}")
        if not isinstance(os.path.join(model_simulation_path, control_file), str):
            raise TypeError("control_file should be a string")

        # Check telemac_solver
        if not isinstance(tm_solver, str):
            raise TypeError("Telemac_solver should be a string")
        if tm_solver not in ["1", "2"]:
            raise ValueError("Telemac_solver should be '1' (Telemac 2D) or '2' (Telemac 3D)")

        # Check results_folder_path
        path_exists(results_folder_path, "results_folder_path")

        # Check calib_pts_csv_file
        path_exists(calib_pts_csv_file, "calib_pts_csv_file")
        if not calib_pts_csv_file.endswith('.csv'):
            raise ValueError("calib_pts_csv_file should be a CSV file")

        # Check n_cpus
        if not isinstance(n_cpus, int):
            raise TypeError("CPUs should be an integer")

        # Check init_runs
        if not isinstance(init_runs, int):
            raise TypeError("init_runs should be an integer")

        # Check calibration_parameters and param_ranges
        if not isinstance(calibration_parameters, list):
            raise TypeError("calibration_parameters should be a list")
        if not isinstance(param_values, list):
            raise TypeError("param_ranges should be a list")
        if len(calibration_parameters) != len(param_values):
            raise ValueError("calibration_parameters and param_ranges must have the same length")

        for param in calibration_parameters:
            if not isinstance(param, str):
                raise TypeError("Each calibration parameter should be a string")

        # If all checks pass, print success message
        print("Calibration parameters and parameter ranges have been validated successfully.")

        # Check friction_file conditionally
        if any(param.lower().startswith('zone') for param in calibration_parameters):
            if isinstance(friction_file, str):
                print(f"Friction file is a valid string: {friction_file}")

                # Check if the friction file exists
                if os.path.exists(os.path.join(model_simulation_path, friction_file)):
                    print(f"Friction file exists: {friction_file}")
                else:
                    raise ValueError(f"Friction file does not exist: {friction_file}")
            else:
                raise TypeError("friction_file should be a string")

        # Check calibration_quantities
        if not isinstance(calibration_quantities, list):
            raise TypeError("calibration_quantities should be a list of strings")
        if len(calibration_quantities) > 2:
            raise ValueError("calibration_quantities can have a maximum of 2 quantities")
        for quantity in calibration_quantities:
            if not isinstance(quantity, str):
                raise TypeError("Each calibration quantity should be a string")
        if calibration_quantities:
            print(f"Calibration quantities are valid: {calibration_quantities}")

        # Check dict_output_name
        if not isinstance(dict_output_name, str):
            raise TypeError("dict_output_name should be a string")
        print(f"Dictionary output name is given as: {dict_output_name}.json")

        # Check results_filename_base
        if not isinstance(results_filename_base, str):
            raise TypeError("results_filename_base should be a string")
        print(f"Base Telemac results file name is gives as: {results_filename_base}")

        # Check parameter_sampling_method
        if not isinstance(parameter_sampling_method, str):
            raise TypeError("parameter_sampling_method should be a string")
        print(f"Parameter sampling method is selected as: {parameter_sampling_method}")

        # Check n_max_tp
        if not isinstance(n_max_tp, int):
            raise TypeError("n_max_tp should be an integer")
        if n_max_tp <= init_runs:
            raise ValueError("n_max_tp must be greater than init_runs")
        print(f"n_max_tp is valid: {n_max_tp}")

        # Check n_samples
        if not isinstance(n_samples, int):
            raise TypeError("n_samples should be an integer")
        print(f"n_samples is valid: {n_samples}")

        # Check mc_samples
        if not isinstance(mc_samples, int):
            raise TypeError("mc_samples should be an integer")
        if mc_samples > n_samples:
            raise ValueError("mc_samples must be less than or equal to n_samples")
        print(f"mc_samples is valid: {mc_samples}")

        # Check mc_exploration
        if not isinstance(mc_exploration, int):
            raise TypeError("mc_exploration should be an integer")
        if mc_exploration > mc_samples:
            raise ValueError("mc_exploration must be less than or equal to mc_samples")
        print(f"mc_exploration is valid: {mc_exploration}")

        # Check eval_steps
        if not isinstance(eval_steps, int):
            raise TypeError("eval_steps should be an integer")
        print(f"Surrogate evaluation steps is valid: {eval_steps}")

        print("All inputs are valid")

    def __call__(self, *args, **kwargs):
        self.run_single_simulation()

