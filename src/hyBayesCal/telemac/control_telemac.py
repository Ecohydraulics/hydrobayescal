
# coding: utf-8
"""
Functional core for coupling Telemac with the Surrogate-Assisted Bayesian inversion technique.
"""
import os, io, stat,sys,re,shutil
import subprocess
from scipy import spatial
from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as _pd
import pdb
import json
try:
    from telapy.api.t2d import Telemac2d
    from telapy.api.t3d import Telemac3d
    from telapy.tools.driven_utils import mpirun_cmd
    from data_manip.extraction.telemac_file import TelemacFile
except ImportError as e:
    print("%s\n\nERROR: load (source) pysource.X.sh Telemac before running HyBayesCal.telemac" % e)
    exit()

# attention relative import usage according to docs/codedocs.rst
from src.hyBayesCal.telemac.config_telemac import * # provides os and sys
from src.hyBayesCal.telemac.global_config import *
import numpy as _np
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN
try:
    from mpi4py import MPI
except ImportError as e:
    logging.warning("Could not import mpi4py")
    print(e)

# get package scripts
from src.hyBayesCal.function_pool import *  # provides os, subprocess, logging
from src.hyBayesCal.model_structure.control_full_complexity import FullComplexityModel
from src.hyBayesCal.doepy.doe_control import DesignOfExperiment
#pdb.set_trace()

class TelemacModel(FullComplexityModel):
    def __init__(
            self,
            model_dir="",
            control_file="tm.cas",
            calibration_parameters=None,
            calibration_values_ranges=None,
            calibration_pts_file_path=None,
            calibration_quantities=None,
            tm_xd="Telemac2d",
            n_processors=None,
            parameter_sampling_method=None,
            dict_output_name="",
            results_file_name_base="",
            gaia_steering_file=None,
            load_case=False,
            stdout=6,
            python_shebang="#!/usr/bin/env python3",
            *args,
            **kwargs
    ):
        """
        Constructor for the TelemacModel Class. Instantiating can take some seconds, so try to
        be efficient in creating objects of this class (i.e., avoid re-creating a new TelemacModel in long loops)

        :param str model_dir: directory (path) of the Telemac model (should NOT end on "/" or "\\") - not the software
        :param list calibration_parameters: List of Telemac model parameters to be modified for model calibration. (up to 4 possible)
        :param list calibration_values_ranges: List of the selected ranges for each of the calibration parameters.
        :param list calibration_quantities: Model outputs (quantities) to be extracted from Telemac .slf output files for calibration purposes. (up to 4 possible)
        :param str calibration_pts_file_path: Complete path of the .csv file containing the coordinates x,y of the measurement points
                                                and the measured values of the selected calibration quantity/(ies).
        :param str control_file: name of the steering file to be used (should end on ".cas"); do not include directory
        :param str gaia_steering_file: name of a gaia steering file (optional)
        :param int n_processors: number of processors to use (>1 corresponds to parallelization); default is None (use cas definition)
        :param str parameter_sampling_method: Sampling method for the selected calibration parameters. Two options: 1) MIN - equal interval - MAX
                                            or 2)    MIN - random - MAX,

        :param str tm_xd: either 'Telemac2d' or 'Telemac3d'
        :param bool load_case: True loads the control file as Telemac case upon instantiation (default: True) - recommended for reading results
        :param int stdout: standard output (default=6 [console];  if 666 => file 'fort.666')
        :param str python_shebang: header line for python files the code writes for parallel processing
                                        (default="#!/usr/bin/env python3\n" for telling Debian-Linux to run with python)
        :param args:
        :param kwargs:
        """
        FullComplexityModel.__init__(self, model_dir=model_dir)

        self.num_run=int(sys.argv[1])
        self.calibration_parameters=calibration_parameters
        self.parameter_sampling_method=parameter_sampling_method
        self.init_runs=init_runs
        self.calibration_values_ranges = calibration_values_ranges
        self.calibration_quantities=calibration_quantities
        self.calibration_pts_df=_pd.read_csv(calibration_pts_file_path)
        self.doe = DesignOfExperiment()
        self.tm_cas = "{}{}{}".format(self.model_dir, os.sep, control_file)
        self.dict_output_name=dict_output_name
        self.results_file_name_base=results_file_name_base
        self.tm_results_filename = self.results_file_name_base + '_' + str(self.num_run) + '.slf'
        self.gaia_steering_file=gaia_steering_file

        print("Running full complexity model " + str(self.num_run))
        self.nproc = n_processors
        self.comm = MPI.Comm(comm=MPI.COMM_WORLD)
        self.results = None  # will hold results loaded through self.load_results()
        self.shebang = python_shebang

        self.tm_xd = tm_xd
        self.tm_xd_dict = {
            "Telemac2d": "telemac2d.py ",
            "Telemac3d": "telemac3d.py ",
        }

        self.stdout = stdout
        self.case = None
        self.case_loaded = False

        #self.loading_case=load_case    Not necessary if the idea is only running the model. It makles sense to load the case if
        # I want to run get_results_filename or load_results

        # ----------- Modifies the base .cas file of the Telemac simulation in each iteration with the NEW calibration parameters.
        # ----------- Until now it is possible to modify only the calibration parameters that were indicated in the global_config.py
        # ----------- However, the idea would be also to modify the roughness file of Telemac .tbl according to the roughness zones in the .brf. file (To be implemented)
        #self.get_results_filename()
        #self.cas_file_creation()
        #---------------------------------------------------------------------------------------------------------------------------

    def cas_creation(self):
        """
        Modifies the .cas steering file for each of the initial Telemac runs according to the parameter sampling
        method based on DoE (Design of Experiments). For the very first run, the calibration values are created and stored as data frame and saved
        in a .csv file called initial-run-parameters.csv. From the second run, this .csv file is read and the code extracts the next calibration parameter combinations.
        After the .cas file has been modified, it is loaded.
        :return : None
        """
        if self.num_run == 1:
            df_doe_calibration_values, self.calibration_values_list = self.parameter_sampling(
                self.calibration_parameters, self.calibration_values_ranges, self.parameter_sampling_method,
                self.init_runs)
            print(
                "The calibration values for the selected number of full complexity runs are: \n " + df_doe_calibration_values.to_string())
        else:
            df_doe_calibration_values = _pd.read_csv(self.model_dir + "/initial-runs-parameters.csv", sep=',',
                                                     index_col=0)
            print(
                "The calibration values for the selected number of full complexity runs are: \n " + df_doe_calibration_values.to_string())
            # Iterate over each row and extract calibration values as a list
            self.parameter_values_dict = {}
            for index, row in df_doe_calibration_values.iterrows():
                self.parameter_values_dict[index] = row.values.tolist()
            self.calibration_values_list = df_doe_calibration_values.loc['PC' + str(self.num_run)].tolist()

            ## In this point I need to add code that reads the new added parameter combination after Bayesian Rejection samplig?? has been done to initia_run_parameters.csv.
            ## This new set of parameters are assigned to self.calibration_values_list which is needed to modify the .cas file.
            ## This step has to be added for Bayesian calibration.

            if self.gaia_steering_file:
                print("* received gaia steering file: " + self.gaia_steering_file)
                self.gaia_cas = "{}{}{}".format(self.model_dir, os.sep, self.gaia_steering_file)
                self.gaia_results_file = "{}{}{}".format(self.res_dir, os.sep,
                                                         str("resIDX-" + self.gaia_steering_file.strip(".cas") + ".slf"))
            else:
                self.gaia_cas = None
                self.gaia_results_file = None

        self.calibration_parameters.append('RESULTS FILE')
        self.calibration_values_list.append(self.tm_results_filename)
        print('Results file name for this simulation:' + self.tm_results_filename)

        # These are the lines of code that modify the .cas file depending on the calibration_parameters[list] and calibration_values_list
        # The values_list is a list that contains the calibration parameters values for each of the parameters in calibration_parameters.
        # From this point we can create a new set of parameters to modify the .cas file for learning the surrogate model.

        for param, val in zip(self.calibration_parameters, self.calibration_values_list):
            cas_string = self.create_cas_string(param, val)
            self.rewrite_steering_file(param, cas_string, steering_module="telemac")

        #if self.loading_case:
        #    self.load_case()
    @staticmethod
    def create_cas_string(param_name, value):
        """
        Create string names with new values to be used in Telemac2d / Gaia steering files

        :param str param_name: name of parameter to update
        :param float or sequence value: new values for the parameter
        :return str: update parameter line for a steering file
        """
        if isinstance(value, (int, float, str)) or ':' in value:

            return param_name + " = " + str(value)
        else:
            try:
                return param_name + " = " + "; ".join(map(str, value))
            except Exception as error:
                print("ERROR: could not generate cas-file string for {0} and value {1}:\n{2}".format(str(param_name), str(value), str(error)))
    def move_slf_results(self):
        pass

    def load_case(self, reset_state=True):
        """Load Telemac case file and check its consistency.

        Parameters
        ----------
        reset_state (bool): use to activate case.init_state_default(); default is ``False``. Only set to ``True`` for
            running Telemac through the Python API. Otherwise, results cannot be loaded.

        Returns
        -------

        """

        print("* switching to model directory (if needed, cd back to TelemacModel.supervisor_dir)")
        os.chdir(self.model_dir)

        print("* loading {} case...".format(str(self.tm_xd)))
        if "telemac2d" in self.tm_xd.lower():
            self.case = Telemac2d(self.tm_cas, lang=2, comm=self.comm, stdout=self.stdout)
        elif "telemac3d" in self.tm_xd.lower():
            self.case = Telemac3d(self.tm_cas, lang=2, comm=self.comm, stdout=self.stdout)
        else:
            print("ERROR: only Telemac2d/3d available, not {}.".format(str(self.tm_xd)))
            return -1
        self.comm.Barrier()

        print("* setting and initializing case...")
        self.case.set_case()
        self.comm.Barrier()

        if reset_state:
            #pdb.set_trace()
            self.case.init_state_default()

        self.case_loaded = True
        print("* successfully activated TELEMAC case: " + str(self.tm_cas))
        return 0

    def close_case(self):
        """Close and delete case."""
        pdb.set_trace()
        if self.case_loaded:
            try:
                self.case.finalize()
                print(self.case)
                del self.case
                print(self.case)
            except Exception as error:
                print("ERROR: could not close case:\n   " + str(error))
        self.case_loaded = False

    def reload_case(self):
        """Iterative runs require first to close the current run."""
        # close and delete case
        self.close_case()
        # load with new specs
        self.load_case()

    def get_results_filename(self):
        """Routine is called with the __init__ and carefully written so that it can be called
        externally any time, too."""
        try:
            #pdb.set_trace()

            # The aim of the following 2-line code is to extract the RESULTFILE nem from the .cas file and assign it as the
            # self.results_file_name_base, but it cannot be done since it needs a loaded case first.

            self.tm_results_filename = self.case.get("MODEL.RESULTFILE")
            # self.results_file_name_base=re.split(r'[._]', self.results_file_name_base)
            # self.results_file_name_base=self.results_file_name_base[0]
            #pdb.set_trace()
            #print(self.results_file_name_base)
        except Exception as err:
            print("ERROR: could not retrieve results filename. Is the case loaded?\n\nTraceback:\n{}".format(str(err)))

    def load_results(self):
        """Load simulation results stored in TelemacModel.tm_results_filename

        Cannot work if case.init_default_state() was applied before.

        :return int: 0 corresponds to success; -1 points to an error
        """
        print("* opening results file: " + self.tm_results_filename)
        if not os.path.isfile(self.tm_results_filename):
            self.get_results_filename()
        print("* retrieving boundary file: " + self.tm_results_filename)
        boundary_file = os.path.join(self.model_dir, self.case.get("MODEL.BCFILE"))
        print("* loading results with boundary file " + boundary_file)
        try:
            os.chdir(self.model_dir)  # make sure to work in the model dir
            self.results = TelemacFile(self.tm_results_filename, bnd_file=boundary_file)
        except Exception as error:
            print("ERROR: could not load results. Did you use TelemacModel.load_case(reset_state=True)?\n" + str(error))
            return -1

        # to see more case variables that can be self.case.get()-ed, type print(self.case.variables)
        # examples to access liquid boundary equilibrium
        try:
            liq_bnd_info = self.results.get_liq_bnd_info()
            print("Liquid BC info:\n" + str(liq_bnd_info))
        except Exception as error:
            print("WARNING: Could not load case liquid boundary info because of:\n   " + str(error))
        return 0

    def update_model_controls(self,**kwargs):#,
            #new_parameter_values,
            #simulation_id=0,
    #):
        """In TELEMAC language: update the steering file
        Update the Telemac and Gaia steering files specifically for Bayesian calibration.

        :param dict new_parameter_values: provide a new parameter value for every calibration parameter
                    * keys correspond to Telemac or Gaia keywords in the steering file
                    * values are either scalar or list-like numpy arrays
        :param int simulation_id: optionally set an identifier for a simulation (default is 0)
        :return int:
        """

        # move existing results to auto-saved-results sub-folder
        try:
            print(self.tm_results_filename)
            print(self.res_dir)
            if os.path.exists(os.path.join(self.res_dir,self.tm_results_filename)):
                # Remove the existing destination file
                os.remove(os.path.join(self.res_dir,self.tm_results_filename))
            shutil.move(os.path.join(self.model_dir,self.tm_results_filename),self.res_dir)
        except Exception as error:
            print("ERROR: could not move results file to " + self.res_dir + "\nREASON:\n" + error)
        return -1

        # update telemac calibration pars
        # for par, has_more in lookahead(self.calibration_parameters["telemac"].keys()):
        #     self.calibration_parameters["telemac"][par]["current value"] = new_parameter_values[par]
        #     updated_string = self.create_cas_string(par, new_parameter_values[par])
        #     self.rewrite_steering_file(par, updated_string, self.tm_cas)
        #     if not has_more:
        #         updated_string = "RESULTS FILE" + " = " + self.tm_results_filename.replace(".slf", f"{simulation_id:03d}" + ".slf")
        #         self.rewrite_steering_file("RESULTS FILE", updated_string, self.tm_cas)
        #
        # # update gaia calibration pars - this intentionally does not iterate through self.calibration_parameters
        # for par, has_more in lookahead(self.calibration_parameters["gaia"].keys()):
        #     self.calibration_parameters["gaia"][par]["current value"] = new_parameter_values[par]
        #     updated_string = self.create_cas_string(par, new_parameter_values[par])
        #     self.rewrite_steering_file(par, updated_string, self.gaia_cas)
        #     if not has_more:
        #         updated_string = "RESULTS FILE" + " = " + self.gaia_results_file.replace(".slf", f"{simulation_id:03d}" + ".slf")
        #         self.rewrite_steering_file("RESULTS FILE", updated_string, self.gaia_cas)

        #return 0

    def rewrite_steering_file(self, param_name, updated_string, steering_module="telemac"):
        """
        Rewrite the *.cas steering file with new (updated) parameters

        :param str param_name: name of parameter to rewrite
        :param str updated_string: new values for parameter
        :param str steering_module: either 'telemac' (default) or 'gaia'
        :return None:
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
            print("ERROR: no such steering file:\n" + steering_file_name)
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

    def cmd2str(self, keyword):
        """Convert a keyword into Python code for writing a Python script
        used by self.mpirun(filename). Required for parallel runs.
        Routine modified from telemac/scripts/python3/telapy/tools/study_t2d_driven.py

        :param (str) keyword: keyword to convert into Python lines
        """
        # instantiate string object for consistency
        string = ""
        # basically assume that Telemac2d should be called; otherwise overwrite with Telemac3d
        telemac_import_str = "from telapy.api.t2d import Telemac2d\n"
        telemac_object_str = "tXd = Telemac2d('"
        if "3d" in self.tm_xd.lower():
            telemac_import_str = "from telapy.api.t3d import Telemac3d\n"
            telemac_object_str = "tXd = Telemac3d('"

        if keyword == "header":
            string = (self.shebang + "\n"
                      "# this script was auto-generated by HyBayesCal and can be deleted\n"
                      "import sys\n"
                      "sys.path.append('"+self.model_dir+"')\n" +
                      telemac_import_str)
        elif keyword == "commworld":
            string = ("try:\n" +
                      "    from mpi4py import MPI\n" +
                      "    comm = MPI.COMM_WORLD\n" +
                      "except:\n" +
                      "    comm = None\n")
        elif keyword == "create_simple_case":
            string = (telemac_object_str + self.tm_cas +
                      "', " +
                      "comm=comm, " +
                      "stdout=" + str(self.stdout) + ")\n")
        elif keyword == "create_usr_fortran_case":
            string = (telemac_object_str + self.tm_cas +
                      "', " +
                      "user_fortran='" + self.test_case.user_fortran + "', " +
                      "comm=comm, " +
                      "stdout=" + str(self.stdout) + ")\n")
        elif keyword == "barrier":
            string = "comm.Barrier()\n"
        elif keyword == "setcase":
            string = "tXd.set_case()\n"
        elif keyword == "init":
            string = "tXd.init_state_default()\n"
        elif keyword == "run":
            string = "tXd.run_all_time_steps()\n"
        elif keyword == "finalize":
            string = "tXd.finalize()\n"
        elif keyword == "del":
            string = "del(tXd)\n"
        elif keyword == "resultsfile":
            string = "tXd.set('MODEL.RESULTFILE', '" + \
                     self.tm_results_filename + "')\n"
        elif keyword == "newline":
            string = "\n"
        if len(string) < 1:
            print("WARNING: empty argument written to run_launcher.py. This will likely cause and error.")
        return string.encode()

    def create_launcher_pyscript(self, filename):
        """Create a Python file for running Telemac in a Terminal (required for parallel runs)
        Routine modified from telemac/scripts/python3/telapy/tools/study_t2d_driven.py

        :param (str) filename: name of the Python file for running it with MPI in Terminal
        """
        with io.FileIO(filename, "w") as file:
            file.write(self.cmd2str("header"))
            file.write(self.cmd2str("newline"))
            file.write(self.cmd2str("commworld"))
            file.write(self.cmd2str("newline"))
            file.write(self.cmd2str("create_simple_case"))  # change this when using a usr fortran file
            if self.nproc > 1:
                file.write(self.cmd2str("barrier"))
            file.write(self.cmd2str("setcase"))
            file.write(self.cmd2str("resultsfile"))
            file.write(self.cmd2str("init"))
            file.write(self.cmd2str("newline"))
            if self.nproc > 1:
                file.write(self.cmd2str("barrier"))
            file.write(self.cmd2str("run"))
            if self.nproc > 1:
                file.write(self.cmd2str("barrier"))
            file.write(self.cmd2str("newline"))
            file.write(self.cmd2str("finalize"))
            if self.nproc > 1:
                file.write(self.cmd2str("barrier"))
            file.write(self.cmd2str("del"))
        file.close()
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)

    def mpirun(self, filename):
        """Launch a Python script called 'filename' in parallel
        Routine modified from telemac/scripts/python3/telapy/tools/study_t2d_driven.py

        :param (str) filename: Python file name for MPI execution
        """
        cmd = mpirun_cmd()
        cmd = cmd.replace("<ncsize>", str(self.nproc))
        cmd = cmd.replace("<exename>", filename)
        # cmd = cmd + " 1> mpi.out 2> mpi.err"

        _, return_code = self.call_tm_shell(cmd)
        if return_code != 0:
            raise Exception("\nERROR IN PARALLEL RUN COMMAND: {} \n"
                            " PROGRAM STOP.\nCheck shebang, model_dir, and cas file.".format(cmd))

    def run_simulation(self, filename="run_launcher.py", load_results=False):
        """ Run a Telemac2d or Telemac3d simulation with one or more processors
        The number of processors to use is defined by self.nproc.

        :param (str) filename: optional name for a Python file that will be automatically
                        created to control the simulation
        :param (bool) load_results: default value of False; it True: load parameters of the results.slf file
        """

        start_time = datetime.now()
        filename = os.path.join(self.model_dir, filename)

        if self.nproc <= 1:
            print("* sequential run (single processor)")
        else:
            print("* parallel run on {} processors".format(self.nproc))
        self.create_launcher_pyscript(filename)
        try:
            self.mpirun(filename)
        except Exception as exception:
            print(exception)
        self.comm.Barrier()
        print("TELEMAC simulation time: " + str(datetime.now() - start_time))

        if load_results:
            self.load_results()

        #self.extract_data_point(self.tm_results_filename,self.calibration_pts_df,self.dict_output_name)


    def call_tm_shell(self, cmd):
        """ Run Telemac in a Terminal in the model directory

        :param (str) cmd:  command to run
        """
        logging.info("* running {}\n -- patience (Telemac simulations can take time) -- check CPU acitivity...".format(cmd))

        # do not use stdout=subprocess.PIPE because the simulation progress will not be shown otherwise
        process = subprocess.Popen(cmd, cwd=r""+self.model_dir, shell=True, env=os.environ)
        stdout, stderr = process.communicate()
        del stderr
        return stdout, process.returncode

    def rename_selafin(self, old_name=".slf", new_name=".slf"):
        """
        Merged parallel computation meshes (gretel subroutine) does not add correct file endings.
        This function adds the correct file ending to the file name.

        :param str old_name: original file name
        :param str new_name: new file name
        :return: None
        :rtype: None
        """

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
        else:
            print("WARNING: SELAFIN file %s does not exist" % old_name)

    def output_processing(self):
        self.extract_data_point(self.tm_results_filename,self.calibration_pts_df,self.dict_output_name)
        self.update_model_controls()
    def parameter_sampling(self,calibration_parameters,calibration_values,parameter_sampling_method,total_number_of_samples):
        """
        This functions creates (equally separated or ramdom) values for the selected calibration parameters.
        Creates 'n' rows corresponding to the number of initial runs and 'p' columns corresponding to the number of calibration parameters

        :param list calibration_parameters: Names of the selected calibration parameters
        :param list calibration_values: Ranges of selection for each of the calibration parameters
        :param str parameter_sampling_method: 1) 'MIN - equal interval - MAX' or 2) 'MIN - random - MAX'
        :return: A dataframe containing the values for each of the initial runs (rows) and calibration parameters (columns).
        :return: A list with the corresponding PC parameter combination values for the PCth run.
        """
        global sampling_method
        try:
            if parameter_sampling_method=='1':
                sampling_method ='MIN - equal interval - MAX'
            elif parameter_sampling_method=='2':
                sampling_method ='MIN - random - MAX'
        except subprocess.CalledProcessError as e:
            print(f"nor sampling method selected for calibration parameters: {e}")
        #pdb.set_trace()
        calib_par_value_dict = {}
        for param, range_ in zip(calibration_parameters, calibration_values):
            calib_par_value_dict[param] = {'name': param, 'bounds': range_}

        # currently only equal or random sampling enabled through doepy.doe.control
        # this will be IMPROVED in a future release to full DoE methods (see doepy.scripts)
        self.doe.generate_multi_parameter_space(
            parameter_dict=calib_par_value_dict,
            method=sampling_method,
            total_number_of_samples=total_number_of_samples
        )

        self.doe.df_parameter_spaces.to_csv(self.model_dir + "/initial-runs-parameters.csv")

        return self.doe.df_parameter_spaces,self.doe.df_parameter_spaces.loc['PC'+str(self.num_run)].tolist()

    def extract_data_point(self, input_slf_file, calibration_pts_df, dict_output_name):
        """
        This function extracts the (calibration quantities) model outputs from the input_slf_file.slf using the points located
         in a .csv  with the x,y coordinates of the measurement points. The function extracts the model output from the closest node
        in the mesh to the x,y measurement coordinate.

        :param str input_slf_file Name of the slf file containing the model outputs.
        :param df calibration_pts_df: Data frame with the name (description) of the measurement node , x, y and the measured value of the calibration quantity.
        :param str output_name: Name of the .json file containing a dictionary with the model outputs and calibration points description.         :return: A dataframe containing the values for each of the initial runs (rows) and calibration parameters (columns).
        :return: Dictionary .json containing the model outputs (selected calibration quantities) of the calibration points.
        """
        #calibration_pts_df=_pd.read_csv(calibration_pts_df)

        global differentiated_dict
        input_file = os.path.join(self.model_dir, input_slf_file)
        json_path = os.path.join(self.model_dir, f"{dict_output_name}.json")
        json_path_detailed = os.path.join(self.model_dir, f"{dict_output_name}_detailed.json")#f"{json_name}_{self.num_run}.json"
        keys = list(calibration_pts_df.iloc[:, 0])
        modeled_values_dict = {}
        print('Extracting values from results file ' + str(self.tm_results_filename) + '\n')
        print('Extracting calibration quantities ' + str(self.calibration_quantities)+ '\n')

        for key,h in zip(keys,range(len(calibration_pts_df))):
            xu = calibration_pts_df.iloc[h,1]
            yu = calibration_pts_df.iloc[h,2]

            # reads the *.slf file
            slf = ppSELAFIN(input_file)
            slf.readHeader()
            slf.readTimes()

            # get times of the selafin file, and the variable names
            #times = slf.getTimes()
            variables = slf.getVarNames()
            units = slf.getVarUnits()

            NVAR = len(variables)

            # to remove duplicate spaces from variables and units
            for i in range(NVAR):
                variables[i] = ' '.join(variables[i].split())
                units[i] = ' '.join(units[i].split())
            # print(variables)

            common_indices = []

            # Iterate over the secondary list
            for value in self.calibration_quantities:
                # Find the index of the value in the original list
                index = variables.index(value)
                # Add the index to the common_indices list
                common_indices.append(index)
            # print(common_indices)
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

            # print('Extracting values for simulation number '+ str(self.num_run)+ '\n')
            # print('Extracted calibration quantities '+ str(self.calibration_quantities) + ' for real calibration point coordinate: ' + str(key) + ' ' +str(xu) + ' ' + str(yu) + '\n')
            # print('*** Extraction performed at the closest node to the input coordinate!: ' + str(x[idx]) + ' ' + str(y[idx]) + '\n')

            # now we need this index for all planes
            idx_all = np.zeros(NPLAN, dtype=np.int32)

            # the first plane
            idx_all[0] = idx

            # start at second plane and go to the end
            for i in range(1, NPLAN, 1):
                idx_all[i] = idx_all[i - 1] + (NPOIN / NPLAN)

            ########################################################################
            # extract results for every plane (if there are multiple planes that is)
            for p in range(NPLAN):
                slf.readVariablesAtNode(idx_all[p])

                # Extracts the results at all times steps for ALL model variables. The time steps for the model are
                # stored in the variable 'times'
                results = slf.getVarValuesAtNode()

                # Extracts the results at the last time step for ALL model variables.
                #-------------------------------------------------------------------
                results_calibration = results[-1]
                #-------------------------------------------------------------------
                #print(results)
                #print(results_calibration)

                # Initialize an empty list to store values (calibration qunatities) for every key (point description) for the
                # current simulation
                modeled_values_dict[key] = []
                # Iterate over the common indices
                for index in common_indices:
                    # Extract value from the last row based on the index
                    value = results_calibration[index]
                    # Append the value to the list for the current key
                    modeled_values_dict[key].append(value)
            # print(modeled_values_dict)


            # New dictionary that stores the values of the calibration quantities for each calibration point. Extra alternative for the
            # Above-mentioned dictionary.
            differentiated_dict = {}

            # Iterate over the keys and values of the original dictionary
            for key, values in modeled_values_dict.items():
                # Create a dictionary to store the differentiated values for the current key
                differentiated_values = {}
                # Iterate over the titles and corresponding values
                for title, value in zip(self.calibration_quantities, values):
                    # Add the title and corresponding value to the dictionary
                    differentiated_values[title] = value
                # Add the differentiated values for the current key to the new dictionary
                differentiated_dict[key] = differentiated_values

            # print(differentiated_dict)


        if self.num_run == 1:
            try:
                # Removes the output_file.json when starting a new run of the code
                os.remove(json_path)
                try:
                    os.remove(json_path_detailed)
                except FileNotFoundError:
                    print("No detailed result file found. Creating a new file.")
            except FileNotFoundError:
                print("No nested result file found. Creating a new file.")

        if os.path.exists(json_path):
            # File exists, so open it for writing
            #pdb.set_trace()
            with open(json_path, "r") as file:
                original_data = json.load(file)
                for key, value in modeled_values_dict.items():
                     if key in original_data:
                        original_data[key].append(value)
                     else:
                        original_data[key] = [value]
                with open(json_path, 'w') as file:
                    json.dump(original_data, file,indent=4)

        else:
        # Save the updated JSON file
            #pdb.set_trace()
            with open(json_path, "w") as file:
                for key in modeled_values_dict:
                    # Convert the existing list into a nested list with a single element
                    modeled_values_dict[key] = [modeled_values_dict[key]]
                json.dump(modeled_values_dict, file,indent=4)

        if os.path.exists(json_path_detailed):
            # File exists, so open it for writing
            #pdb.set_trace()
            with open(json_path_detailed, "r") as file:
                original_data_detailed = json.load(file)
                for key, new_values in differentiated_dict.items():
                    if key in original_data_detailed:
                        # If the key exists in original_data_detailed, convert the existing value to a list
                        # and append the new values to that list
                        if isinstance(original_data_detailed[key], list):
                            original_data_detailed[key].append(new_values)
                        else:
                            original_data_detailed[key] = [original_data_detailed[key], new_values]
                    else:
                        # If the key does not exist, create a new list containing the new values
                        original_data_detailed[key] = [new_values]

                with open(json_path_detailed, 'w') as file:
                    json.dump(original_data_detailed, file,indent=4)
        else:
        # Save the updated JSON file
            #pdb.set_trace()
            with open(json_path_detailed, "w") as file:
                for key in differentiated_dict:
                    # Convert the existing list into a nested list with a single element
                    differentiated_dict[key] = differentiated_dict[key]
                json.dump(differentiated_dict, file,indent=4)
def run_telemac():
    my_simulation=TelemacModel(
        model_dir=cas_file_simulation_path,
        control_file=cas_file_name,
        calibration_parameters=calib_parameter_list,
        calibration_values_ranges=parameter_ranges_list,
        calibration_pts_file_path=calib_pts_file_path,
        calibration_quantities=calib_quantity_list,
        tm_xd='Telemac2d',
        n_processors=NCPS,
        parameter_sampling_method=parameter_sampling_method,
        dict_output_name=dict_output_name,
        results_file_name_base=results_file_name_base
        )
    my_simulation.cas_creation()
    my_simulation.run_simulation()
    my_simulation.output_processing()
    #my_object.extract_data_point('r2d-donau_1.slf',calib_pts_file_path)
if __name__ == "__main__":
    run_telemac()
