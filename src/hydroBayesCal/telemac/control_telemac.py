#
# # coding: utf-8
# """
# Functional core for controlling Telemac simulations for coupling with the Surrogate-Assisted Bayesian inversion technique.
#
# Authors: Andres Heredia, Sebastian Schwindt
# """

# # TODO: Do you still use the global parameters from conf_telemac? -> this not anymore # Not anymore
# # from config_telemac import *
# # TODO: is pputils still relevant? # This is relevant to the code.

import sys
import os
import io, stat, shutil
import subprocess
from scipy import spatial
import numpy as np
import pandas as _pd
import csv
import json
import pdb

from src.hydroBayesCal.hysim import HydroSimulations
from src.hydroBayesCal.function_pool import *  # provides os, subprocess, logging

try:
    from telapy.api.t2d import Telemac2d
    from telapy.api.t3d import Telemac3d
    from telapy.tools.driven_utils import mpirun_cmd
    from data_manip.extraction.telemac_file import TelemacFile
except ImportError as e:
    print("%s\n\nERROR: load (source) pysource.X.sh Telemac before running HydroBayesCal.telemac" % e)

try:
    from mpi4py import MPI
except ImportError as e:
    print(e)

# from hydroBayesCal.function_pool import *  # provides os, subprocess, logging
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN


class TelemacModel(HydroSimulations):
    def __init__(
            self,
            friction_file ="",
            tm_xd="",
            gaia_steering_file=None,
            results_filename_base="",
            stdout=6,
            python_shebang="#!/usr/bin/env python3",
            *args,
            **kwargs
    ):
        """
        Constructor for the TelemacModel Class. Instantiating can take some seconds, so try to
        be efficient in creating objects of this class (i.e., avoid re-creating a new TelemacModel in long loops).

        Parameters
        ----------
        model_dir : str
            Directory (path) of the Telemac model (should NOT end on "/" or "\\") - not the software directory.
        res_dir : str
            Directory (path) of the folder where a subfolder called "auto-saved-results" will be created to store all the results files.
        control_file : str
            Name of the steering file to be used (should end on ".cas"); do not include the directory.
        friction_file : str
            Name of the friction file to be used .tbl ; do not include the directory.
        calibration_parameters : list
            Telemac model parameters considered for model calibration (up to 4 possible).
        calibration_pts_file_path : str
            Complete path of the .csv file containing the description, coordinates (x, y) of the measurement points
            and the measured values of the selected calibration quantities at those points.
        calibration_quantities : list
            Model outputs (quantities) to be extracted from Telemac .slf output files for calibration purposes (up to 4 possible).
        tm_xd : str
            Either 'Telemac2d' or 'Telemac3d'.
        n_processors : int
            Number of processors to use (>1 corresponds to parallelization); default is None (use .cas definition).
        gaia_steering_file : str
            Name of a Gaia steering file.
        init_runs : int
            Number of initial runs for initial surrogate construction.
        dict_output_name : str
            Name of the external .json file containing the model outputs of the calibration quantities for all model runs (initial surrogate phase + BAL phase).
        results_file_name_base : str
            Name of the results file to be iteratively changed in the .cas file.
        stdout : int
            Standard output (default=6 [console]; if 666 => file 'fort.666').
        python_shebang : str
            Header line for Python files the code writes for parallel processing (default="#!/usr/bin/env python3\n").
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.

        Attributes
        ----------

        TODO: tidy up docstrings. this should only Telemac-specific attributes as the rest is explained with the superclass (HydroSimulations)

        model_dir : str
            Directory of the Telemac model, where .cas file is located.
        res_dir : str
            Directory to store the results. A subfolder called "auto-saved-results" will be created.
        calibration_parameters : list
            Calibration parameters.
        init_runs : int
            Number of initial runs for initial surrogate construction.
        calibration_quantities : list
            Calibration quantities (model outputs) for model calibration.
        calibration_pts_df : pandas.DataFrame
            DataFrame containing the calibration points and measurements.
        tm_cas : str
            Full path to the Telemac steering file .cas.
        fr_tbl : str
            Full path to the friction file .tbl.
        dict_output_name : str
            Name of the external .json file where model outputs are going to be saved.
        results_file_name_base : str
            Base name for the results file .slf.
        gaia_steering_file : str or None
            Name of the Gaia steering file if provided; otherwise, None.
        nproc : int
            Number of processors to use for parallelization during Telemac runs.
        comm : MPI.Comm
            MPI communicator for parallel processing.
        results : None
            Placeholder for results loaded through `self.load_results()`.
        shebang : str
            Shebang line for Python scripts.
        tm_xd : str
            Either 'Telemac2d' or 'Telemac3d'.
        tm_xd_dict : dict
            Dictionary mapping 'Telemac2d' and 'Telemac3d' to their respective script names.
        stdout : int
            Standard output setting.
        bal_iteration : int
            Bayesian Active Learning iteration number.
        num_run : int
            Simulation number. Iteratively changing according the collocation points.
        """
        # TODO: I started to separate parameters that every HydroSimulation inheritance should have
        # TODO: from those that are specific to Telemac. Please continue and merge. - done accordingly
        super().__init__(*args, **kwargs)
        # Initialize subclass-specific attributes
        self.friction_file = friction_file
        self.tm_xd = tm_xd
        self.gaia_steering_file = gaia_steering_file
        self.results_filename_base = results_filename_base
        self.stdout = stdout
        self.python_shebang = python_shebang
        self.tm_cas = "{}{}{}".format(self.model_dir, os.sep, self.control_file)
        self.fr_tbl = "{}{}{}".format(self.model_dir, os.sep, self.friction_file)
        self.gaia_steering_file = gaia_steering_file

        self.comm = MPI.Comm(comm=MPI.COMM_WORLD)
        self.results = None  # will hold results loaded through self.load_results()
        self.shebang = python_shebang
        if tm_xd == '1':
            self.tm_xd = 'Telemac2d'
        elif tm_xd == '2':
            self.tm_xd = 'Telemac3d'
        self.tm_xd_dict = {
            "Telemac2d": "telemac2d.py ",
            "Telemac3d": "telemac3d.py ",
        }
        self.stdout = stdout
        # Initializes the BAL iteration number
        self.bal_iteration = int()
        # Initializes the simulation number
        self.num_run=int()
        # Initializes the file where the output data from simulations will be stored
        self.tm_results_filename = ''

        #self.output_data=None
        #self.case_loaded = False

    def update_model_controls(
            self,
            collocation_point_values,
            calibration_parameters,
            auxiliary_file_path,
            simulation_id = 0,
    ):
        """
        Modifies the .cas steering file for each of the Telemac runs according to the values of the collocation points and the
        calibration parameters. If a "FRICTION DATA FILE" is provided for Telemac simulations, it is possible to consider any zone
        as a calibration parameter. The parameters must start with the prefix "zone" and the number of the friction zone. The .tbl will be
        modified for this purpose. This method is called every time it is required that the .cas or .tbl are modified.

        Parameters
        ----------
        collocation_point_values : list
            Values for each of the calibration parameters.
        calibration_parameters : list
            Names of the calibration parameters.
        tm_results_filename : str
            Name of the results file for the Telemac simulation.
        auxiliary_file_path : str
            Path to the friction file .tbl.

        Returns
        -------
        None
        """
        self.tm_results_filename = self.results_filename_base + '_' + str(simulation_id) + '.slf'
        params_with_results = calibration_parameters + ['RESULTS FILE']
        values_with_results = collocation_point_values + [self.tm_results_filename]
        logger.info(f'Results file name for this simulation: {self.tm_results_filename}')
        friction_file_path = auxiliary_file_path
        try:
            for param, value in zip(params_with_results, values_with_results):
                if param.lower().startswith("zone"):
                    zone_identifier = param[4:]
                    self.tbl_creator(zone_identifier, value, friction_file_path)
                else:
                    cas_string = self.create_cas_string(param, value)
                    self.rewrite_steering_file(param, cas_string, steering_module="telemac")
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
                logger_error.error("ERROR: could not generate cas-file string for {0} and value {1}:\n{2}".format(str(param_name), str(value), str(error)))

    def rewrite_steering_file(
            self,
            param_name,
            updated_string,
            steering_module="telemac"
    ):
        """
        Rewrites the *.cas steering file with new (updated) parameters

        Parameters
        ----------
            param_name: string
                Name of the calibration parameter

            updated_string: string
                Updated string to be replaced in .cas file with the new value.
            steering_module: string
                By default Telemac
        Returns
        ----------
            int: 0 corresponds to success.
            int: -1 points to an error.

        """

        # check if telemac or gaia cas type
        if "telemac" in steering_module:
            steering_file_name = self.tm_cas
        else:
            steering_file_name = self.gaia_cas

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

    # def cmd2str(self, keyword):
    #     """
    #     Convert a keyword into Python code for writing a Python script
    #     used by self.mpirun(filename). Required for parallel runs.
    #     Routine modified from telemac/scripts/python3/telapy/tools/study_t2d_driven.py
    #
    #
    #     Parameters
    #     ----------
    #     keyword: string
    #         Keyword to convert into Python lines
    #
    #     Returns
    #     ----------
    #     string.encode(): string
    #
    #     """
    #
    #     # instantiate string object for consistency
    #     string = ""
    #     # basically assume that Telemac2d should be called; otherwise overwrite with Telemac3d
    #     telemac_import_str = "from telapy.api.t2d import Telemac2d\n"
    #     telemac_object_str = "tXd = Telemac2d('"
    #     if "3d" in self.tm_xd.lower():
    #         telemac_import_str = "from telapy.api.t3d import Telemac3d\n"
    #         telemac_object_str = "tXd = Telemac3d('"
    #
    #     if keyword == "header":
    #         string = (self.shebang + "\n"
    #                   "# this script was auto-generated by HyBayesCal and can be deleted\n"
    #                   "import sys\n"
    #                   "sys.path.append('"+self.model_dir+"')\n" +
    #                   telemac_import_str)
    #     elif keyword == "commworld":
    #         string = ("try:\n" +
    #                   "    from mpi4py import MPI\n" +
    #                   "    comm = MPI.COMM_WORLD\n" +
    #                   "except:\n" +
    #                   "    comm = None\n")
    #     elif keyword == "create_simple_case":
    #         string = (telemac_object_str + self.tm_cas +
    #                   "', " +
    #                   "comm=comm, " +
    #                   "stdout=" + str(self.stdout) + ")\n")
    #     elif keyword == "create_usr_fortran_case":
    #         string = (telemac_object_str + self.tm_cas +
    #                   "', " +
    #                   "user_fortran='" + self.test_case.user_fortran + "', " +
    #                   "comm=comm, " +
    #                   "stdout=" + str(self.stdout) + ")\n")
    #     elif keyword == "barrier":
    #         string = "comm.Barrier()\n"
    #     elif keyword == "setcase":
    #         string = "tXd.set_case()\n"
    #     elif keyword == "init":
    #         string = "tXd.init_state_default()\n"
    #     elif keyword == "run":
    #         string = "tXd.run_all_time_steps()\n"
    #     elif keyword == "finalize":
    #         string = "tXd.finalize()\n"
    #     elif keyword == "del":
    #         string = "del(tXd)\n"
    #     elif keyword == "resultsfile":
    #         string = "tXd.set('MODEL.RESULTFILE', '" + \
    #                  self.tm_results_filename + "')\n"
    #     elif keyword == "newline":
    #         string = "\n"
    #     if len(string) < 1:
    #         print("WARNING: empty argument written to run_launcher.py. This will likely cause and error.")
    #     return string.encode()

    # def create_launcher_pyscript(self, filename):
    #     """
    #     Creates a Python file for running Telemac in a Terminal (required for parallel runs)
    #     Routine modified from telemac/scripts/python3/telapy/tools/study_t2d_driven.py
    #
    #     Parameters
    #     ----------
    #     filename: string
    #         Name of the Python file for running it with MPI in Terminal
    #
    #     Returns
    #     ----------
    #     None
    #         Creation of Python pyscript
    #
    #     """
    #     with io.FileIO(filename, "w") as file:
    #         file.write(self.cmd2str("header"))
    #         file.write(self.cmd2str("newline"))
    #         file.write(self.cmd2str("commworld"))
    #         file.write(self.cmd2str("newline"))
    #         file.write(self.cmd2str("create_simple_case"))  # change this when using a usr fortran file
    #         if self.nproc > 1:
    #             file.write(self.cmd2str("barrier"))
    #         file.write(self.cmd2str("setcase"))
    #         file.write(self.cmd2str("resultsfile"))
    #         file.write(self.cmd2str("init"))
    #         file.write(self.cmd2str("newline"))
    #         if self.nproc > 1:
    #             file.write(self.cmd2str("barrier"))
    #         file.write(self.cmd2str("run"))
    #         if self.nproc > 1:
    #             file.write(self.cmd2str("barrier"))
    #         file.write(self.cmd2str("newline"))
    #         file.write(self.cmd2str("finalize"))
    #         if self.nproc > 1:
    #             file.write(self.cmd2str("barrier"))
    #         file.write(self.cmd2str("del"))
    #     file.close()
    #     os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)

    # def mpirun(self, filename):
    #     """
    #     Launch a Python script called 'filename' in parallel
    #     Routine modified from telemac/scripts/python3/telapy/tools/study_t2d_driven.py
    #
    #     Parameters
    #     ----------
    #     filename: string
    #         Python file name for MPI execution
    #
    #     Returns
    #     ----------
    #     None
    #         Creation of Python pyscript
    #
    #     """
    #     cmd = mpirun_cmd()
    #     cmd = cmd.replace("<ncsize>", str(self.nproc))
    #     cmd = cmd.replace("<exename>", filename)
    #     # cmd = cmd + " 1> mpi.out 2> mpi.err"
    #
    #     _, return_code = self.call_tm_shell(cmd)
    #     if return_code != 0:
    #         raise Exception("\nERROR IN PARALLEL RUN COMMAND: {} \n"
    #                         " PROGRAM STOP.\nCheck shebang, model_dir, and cas file.".format(cmd))

    def run_single_simulation(
            self,
            control_file="tel.cas",
            load_results=True
    ):
        """
        Runs a Telemac2D or Telemac3D simulation with one or more processors.
        The number of processors to use is defined by self.nproc.

        Parameters
        ----------
        filename : str, optional
            Name for a Python file that will be automatically created to control the simulation.
            Default is "run_launcher.py".
        load_results : bool, optional
            Whether to load results from the results.slf file. Default is True.

        Returns
        -------
        None
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

        # Load results if required
        if load_results:
            self.extract_data_point(self.tm_results_filename,
                                    self.calibration_pts_df,
                                    self.dict_output_name,
                                    self.calibration_quantities,
                                    self.num_run,
                                    self.model_dir,
                                    self.asr_dir)
    def run_multiple_simulations(
            self,
            collocation_points=None,
            bal_new_set_parameters=None,
            bal_iteration=int(),
            complete_bal_mode=True,
    ):
        """
        Runs multiple Telemac2d or Telemac3d simulations with a set of collocation points and a new set of
        calibration parameters when BAL mode is chosen. The number of processors to use is defined by self.nproc in user_inputs.

        TODO: Make sure this also runs outside the BAL context

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

        Returns
        -------
        model_evaluations: array
            Model evaluations as array with shape:
            For 1. quantity : [No. runs x No. loc]
            For 2.quantities: [No. runs x 2* No. loc] -> 1 column for each quantity
        """
        calibration_parameters = self.calibration_parameters
        res_dir =self.asr_dir
        fr_tbl=self.fr_tbl
        init_runs=self.init_runs

        logger.info(
            "* Running multiple Telemac simulations can take time -- check CPU acitivity...")
        start_time = datetime.now()
        if complete_bal_mode:
            # This part of the code runs the initial runs for initial surrogate.
            if collocation_points is not None:
                # Convert collocation_points to a numpy array if it is not already
                if not isinstance(collocation_points, np.ndarray):
                    collocation_points = np.array(collocation_points)

                # Ensure collocation_points is a 2D array
                if collocation_points.ndim == 1:
                    collocation_points = collocation_points[:, np.newaxis]

                # Convert collocation_points to a list for saving to CSV
                array_list = collocation_points.tolist()
                with open(res_dir + os.sep + "/collocation_points.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(calibration_parameters)
                    writer.writerows(array_list)  # Write the array data
                for i in range(init_runs):
                    self.num_run = i + 1
                    collocation_point_sim_list=collocation_points[i].tolist()
                    logger.info(f" Running  full complexity model # {self.num_run}  with collocation point : {collocation_point_sim_list} ")
                    self.update_model_controls(collocation_point_values=collocation_point_sim_list,
                                               calibration_parameters=calibration_parameters,
                                               auxiliary_file_path=fr_tbl,
                                               simulation_id=self.num_run)
                    self.run_single_simulation(self.control_file)
                    self.model_evaluations=self.output_processing(os.path.join(res_dir,
                                     f'{self.dict_output_name}.json'))
                    logger.info("TELEMAC simulations time for initial runs: " + str(datetime.now() - start_time))
            # This part of the code runs BAL
            else:
                self.bal_iteration = bal_iteration
                self.num_run = bal_iteration+init_runs
                if bal_new_set_parameters is not None:
                    collocation_point_sim_list= bal_new_set_parameters.tolist()[0]
                    logger.info(f" Running  full complexity model after BAL # {self.bal_iteration} with collocation point : {collocation_point_sim_list} ")
                    # TODO: the following needs to be re-integrated
                    update_collocation_pts_file(res_dir + "/collocation_points.csv", new_collocation_point=collocation_point_sim_list)
                    self.update_model_controls(collocation_point_values=collocation_point_sim_list,
                                               calibration_parameters=calibration_parameters,
                                               auxiliary_file_path=fr_tbl,
                                               simulation_id=self.num_run)
                    self.run_single_simulation(self.control_file)
                    self.model_evaluations=self.output_processing(os.path.join(res_dir,
                                     f'{self.dict_output_name}.json'))
                    logger.info("TELEMAC simulations time after Bayesian Active Learning: " + str(datetime.now() - start_time))
                else:
                    logger.error("BAL_new_set_parameters is None. Please provide valid parameters.")
                    raise ValueError("BAL_new_set_parameters is None. Please provide valid parameters.")

        # This part of the code only runs iterative runs without performing BAL
        else:
            if collocation_points is not None:
                # Convert collocation_points to a numpy array if it is not already
                if not isinstance(collocation_points, np.ndarray):
                    collocation_points = np.array(collocation_points)

                # Ensure collocation_points is a 2D array
                if collocation_points.ndim == 1:
                    if collocation_points.size == 1:
                        # If there's only one element, convert it to a column vector
                        collocation_points = collocation_points[:, np.newaxis]
                    else:
                        # If there are multiple elements, reshape it to a horizontal 2D array
                        collocation_points = collocation_points.reshape(1, -1)

                # Convert collocation_points to a list for saving to CSV
                array_list = collocation_points.tolist()
                with open(res_dir + "/collocation_points.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(calibration_parameters)
                    writer.writerows(array_list)
                for i in range(init_runs):
                    self.num_run = i + 1
                    collocation_point_sim_list=collocation_points[i].tolist()
                    logger.info(f" Running  full complexity model # {self.num_run}  with collocation point : {collocation_point_sim_list} ")
                    self.update_model_controls(collocation_point_values=collocation_point_sim_list,
                                               calibration_parameters=calibration_parameters,
                                               auxiliary_file_path=fr_tbl,
                                               simulation_id=self.num_run)
                    self.run_single_simulation(self.control_file)
                    self.model_evaluations=self.output_processing(os.path.join(res_dir,
                                     f'{self.dict_output_name}.json'))
                    logger.info("TELEMAC simulations time for initial runs: " + str(datetime.now() - start_time))

        return self.model_evaluations

    # def call_tm_shell(self, cmd):
    #     """
    #     Run Telemac in a Terminal in the model directory
    #
    #     Parameters
    #     ----------
    #     cmd: string
    #         Command to run.
    #     Returns
    #     ----------
    #     stdout: Standard output
    #
    #
    #     """
    #     # do not use stdout=subprocess.PIPE because the simulation progress will not be shown otherwise
    #     # print(cmd)
    #     # print(self.control_file)
    #     # print(self.tm_xd)
    #     # print(self.tm_xd_dict)
    #     if self.tm_xd in self.tm_xd_dict:
    #         # Retrieve the value corresponding to the key
    #         tm_launcher = self.tm_xd_dict[self.tm_xd]
    #         telemac_command = f"{tm_launcher} {os.path.join(self.model_dir,self.control_file)} --ncsize={self.nproc}"
    #     else:
    #         raise ValueError(f"Invalid launcher: {self.tm_xd}.")
    #     print(telemac_command)
    #     process = subprocess.Popen(telemac_command, cwd=r""+self.model_dir, shell=True, env=os.environ)
    #
    #     #process = subprocess.Popen(cmd, cwd=r""+self.model_dir, shell=True, env=os.environ)
    #     stdout, stderr = process.communicate()
    #     del stderr
    #     return stdout, process.returncode

    def output_processing(
            self,
            output_data_path=''
    ):
        """
        Extract data from a JSON file containing model outputs to 2D array ready to use in Bayesian calibration and saves
        the results to a CSV file.

        This method reads a JSON file specified by `output_data_path`, processes the model outputs,
        and saves the results in a CSV file.

        Parameters
        ----------
        output_data_path : str
            Path to the JSON file containing the model outputs. The JSON file should be structured
            such that its keys correspond to calibration points, and its values are lists of model
            output values for each run and quantity.

        Returns
        -------
        model_results : numpy.ndarray
            A 2D array containing the processed model outputs. The shape of the array is
            [No. of quantities x No. of total runs, No. of calibration points], where 'No. of quantities'
            is the number of calibration quantities being processed, and 'No. of total runs' is the sum
            of initial runs and Bayesian active learning iterations. The array is also saved to a CSV file
            in the specified directory.
        """

        with open(output_data_path, "r") as file:
            output_data = json.load(file)

        n_calibration_pts = self.nloc
        n_total_runs = self.init_runs+self.bal_iteration

        # Number of quantities per location
        num_quantities = self.num_quantities

        # Initialize a 2D NumPy array with zeros
        model_results = np.zeros((num_quantities * n_total_runs, n_calibration_pts))

        # Populate the array with the values from the dictionary
        for i, key in enumerate(output_data.keys()):
            values = output_data[key]
            for j, value_set in enumerate(values):
                for k in range(num_quantities):
                    model_results[k * n_total_runs + j, i] = value_set[k]
        if self.num_quantities == 1:
            headers = [f'{i + 1}' for i in range(n_calibration_pts)]
            np.savetxt(
                os.path.join(self.asr_dir,'model_results.csv'),
                model_results,
                delimiter=',',
                fmt='%.8f',
                header=','.join(headers),
            )
        else:
            model_results=rearrange_array(model_results)
            num_columns=model_results.shape[1]
            headers = []
            for i in range(1, num_columns//2 + 1):
                headers.append(f'{i}_{self.calibration_quantities[0]}')
                headers.append(f'{i}_{self.calibration_quantities[1]}')
            np.savetxt(
                os.path.join(self.asr_dir,'model_results.csv'),
                model_results,
                delimiter=',',
                fmt='%.8f',
                header=','.join(headers),
            )


        # if not complete_bal_mode:
        #     exit()

        return model_results

    def extract_data_point(
            self,
            input_slf_file,
            calibration_pts_df,
            output_name,
            extraction_quantity,
            simulation_number,
            model_directory,
            results_folder_directory
    ):
        """
        Extracts the model outputs (i.e., calibration quantities) from the slf_file.slf using the points located
        in a .csv  with the x,y coordinates of the measurement points. The function extracts the model output from
        the closest node in the mesh to the x,y measurement coordinate.
        This method is typically called for each model run, storing outputs sequentially in 2 dictionaries
        saved in a .json file. That means if the surrogate model needs 'n' runs of the complex numerical model, this
        function is called 'n' times and the dictionary will store the model outputs for the 'n' simulations.

        Parameters
        ----------
        input_slf_file : str
            Name of the SELAFIN (.slf) file containing the model outputs.
        calibration_pts_df : pd.DataFrame
            DataFrame containing calibration points. It should include the point descriptions (e.g., "P1"),
            x and y coordinates of the measurement points, and the measured value/error for the calibration quantities.
            The expected columns are: 'Point Name', 'X', 'Y', 'Measured Value', 'Measured Error'.
            If 2 calibration quantities are required the expected columns should be: 'Point Name', 'X', 'Y', 'Measured Value 1', 'Measured Error 1', 'Measured Value 2', 'Measured Error 2'.
        output_name : str
            Base name for the JSON file that will store the extracted model outputs. Two files are generated:
            a standard results file (`<output_name>.json`) and a detailed results file (`<output_name>_detailed.json`).
        extraction_quantity : list of str
            List of variables (calibration quantities) to be extracted from the SELAFIN file. Please check before extraction the variable name from the .slf file.
        simulation_number : int
            The current simulation number. This is used to manage the output file, ensuring that data is appended
            correctly across multiple simulations.
        model_directory : str
            Path to the directory containing the SELAFIN file.
        results_folder_directory : str
            Path to the directory where the output JSON files will be saved.

        Returns
        -------
        None
            The method saves the extracted model outputs to two JSON files in the specified results directory.
            If this is the first simulation run, any existing JSON files with the same name will be deleted.
            After each run, the extracted results are appended to the JSON files.The detailed results file stores
            the calibration quantities as a nested dictionary with the variable name and the point description.

        """
        # TODO: (1) why is this global defined inside a class method? - not defined as global variable
        # TODO: (2) this method needs refactoring. It is too long.
        calibration_quantities = extraction_quantity
        input_file = os.path.join(model_directory, input_slf_file)
        json_path = os.path.join(results_folder_directory, f"{output_name}.json")
        json_path_detailed = os.path.join(results_folder_directory, f"{output_name}_detailed.json")
        keys = list(calibration_pts_df.iloc[:, 0])
        modeled_values_dict = {}
        logger.info(
            f'Extracting {calibration_quantities} from results file {input_slf_file} \n')

        for key, h in zip(keys,range(len(calibration_pts_df))):
            xu = calibration_pts_df.iloc[h,1]
            yu = calibration_pts_df.iloc[h,2]

            # reads the *.slf file
            slf = ppSELAFIN(input_file)
            slf.readHeader()
            slf.readTimes()

            # gets times of the selafin file, and the variable names
            #times = slf.getTimes()
            variables = slf.getVarNames()
            units = slf.getVarUnits()

            NVAR = len(variables)

            # to remove duplicate spaces from variables and units
            for i in range(NVAR):
                variables[i] = ' '.join(variables[i].split())
                units[i] = ' '.join(units[i].split())

            common_indices = []

            # Iterate over the secondary list
            for value in calibration_quantities:
                # Find the index of the value in the original list
                index = variables.index(value)
                # Add the index to the common_indices list
                common_indices.append(index)

            # gets some of the mesh properties from the *.slf file
            NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = slf.getMesh()

            # determine if the *.slf file is 2d or 3d by reading how many planes it has
            NPLAN = slf.getNPLAN()
            #fout.write('The file has ' + str(NPLAN) + ' planes' + '\n')

            # store just the x and y coords
            x2d = x[0:int(len(x) / NPLAN)]
            y2d = y[0:int(len(x) / NPLAN)]

            # create a KDTree object
            source = np.column_stack((x2d, y2d))
            tree = spatial.cKDTree(source)

            # find the index of the node the user is seeking
            d, idx = tree.query((xu, yu), k=1)
            # print('*** Extraction performed at the closest node to the input coordinate!: ' + str(x[idx]) + ' ' + str(y[idx]) + '\n')
            # now we need this index for all planes
            idx_all = np.zeros(NPLAN, dtype=np.int32)
            # the first plane
            idx_all[0] = idx

            # start at second plane and go to the end
            for i in range(1, NPLAN, 1):
                idx_all[i] = idx_all[i - 1] + (NPOIN / NPLAN)

            # extract results for every plane (if there are multiple planes that is)
            for p in range(NPLAN):
                slf.readVariablesAtNode(idx_all[p])

                # Extracts the results at all times steps for ALL model variables. The time steps for the model are
                # stored in the variable 'times'
                results = slf.getVarValuesAtNode()

                # Extracts the results at the last time step for ALL model variables.
                # If the extraction of the calibration variable is required at a different time step,
                # change the code at this point
                #-------------------------------------------------------------------
                results_calibration = results[-1]
                #-------------------------------------------------------------------
                # Initializes an empty list to store values (calibration qunatities) for every key (point description) for the
                # current simulation
                modeled_values_dict[key] = []
                # Iterate over the common indices
                for index in common_indices:
                    # Extract value from the last row based on the index
                    value = results_calibration[index]
                    # Append the value to the list for the current key
                    modeled_values_dict[key].append(value)

            # New dictionary that stores the values of the calibration quantities for each calibration point. Extra alternative for the
            # Above-mentioned dictionary.
            differentiated_dict = {}

            # Iterate over the keys and values of the original dictionary
            for key, values in modeled_values_dict.items():
                # Create a dictionary to store the differentiated values for the current key
                differentiated_values = {}
                # Iterate over the titles and corresponding values
                for title, value in zip(calibration_quantities, values):
                    # Add the title and corresponding value to the dictionary
                    differentiated_values[title] = value
                # Add the differentiated values for the current key to the new dictionary
                differentiated_dict[key] = differentiated_values

        if simulation_number == 1:
            try:
                # Removes the output_file.json when starting a new run of the code
                os.remove(json_path)
                try:
                    os.remove(json_path_detailed)
                except FileNotFoundError:
                    print("No detailed result file found. Creating a new file.")
            except FileNotFoundError:
                print("No nested result file found. Creating a new file.")
        # Updating json files for every run
        update_json_file(json_path = json_path,modeled_values_dict=modeled_values_dict)
        update_json_file(json_path=json_path_detailed,modeled_values_dict=differentiated_dict,detailed_dict=True)

        try:
            if os.path.exists(os.path.join(results_folder_directory ,input_slf_file)):
                # Remove the existing destination file
                os.remove(os.path.join(results_folder_directory,input_slf_file))
            shutil.move(os.path.join(model_directory,input_slf_file),results_folder_directory)
        except Exception as error:
            print("ERROR: could not move results file to " + self.res_dir + "\nREASON:\n" + error)

    @staticmethod
    def tbl_creator(
            zone_identifier,
            val,
            friction_file_path
        ):
        """
         Modifies the FRICTION DATA FILE (.tbl)  for Telemac simulations based on the specified zone
         and value. This method is used to update the friction values in the table for different zones
         as part of the calibration process.

         Parameters
         ----------
         zone_identifier : int
             Identifier for the friction zone to be updated in the friction table.
         val : float
             The new friction value to be set for the specified zone.
         friction_file_path : str
             The file path to the existing friction file (.tbl) that will be modified.

         Returns
         -------
         None
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
                if len(line_list) > 2 and line_list[2] != 'NULL':
                    try:
                        line_list[2] = str(val)
                    except ValueError:
                        pass
                updated_zone_line = '\t'.join(line_list)
                updated_lines.append(updated_zone_line + '\n')
            else:
                updated_lines.append(line)

        with open(friction_file_path, 'w') as file:
            file.writelines(updated_lines)

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
        telemac_solver = user_inputs['Telemac_solver']
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
        if not isinstance(telemac_solver, str):
            raise TypeError("Telemac_solver should be a string")
        if telemac_solver not in ["1", "2"]:
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
        self.run_simulation()
