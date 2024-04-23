
# coding: utf-8
"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with Telemac.
"""
import os, io, stat,sys
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
import shutil
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
            calibration_parameters=None,
            calibration_values_ranges=None,
            calibration_quantities=None,
            calibration_pts_file_path=None,
            control_file="tm.cas",
            results_file_name_base='result_file',
            gaia_steering_file=None,
            n_processors=None,
            parameter_sampling_method=None,
            tm_xd="Telemac2d",
            load_case=True, # True
            stdout=6,
            python_shebang="#!/usr/bin/env python3",
            *args,
            **kwargs
    ):
        """
        Constructor for the TelemacModel Class. Instantiating can take some seconds, so try to
        be efficient in creating objects of this class (i.e., avoid re-creating a new TelemacModel in long loops)

        :param str model_dir: directory (path) of the Telemac model (should NOT end on "/" or "\\") - not the software
        :param list calibration_parameters: Telemac model parameters to be modified for model calibration. (up to 4 possible)
        :param list calibration_values_ranges: List of ranges for each of the calibration parameters.
        :param list calibration_quantities: Model outputs (quantities) to be extracted from Telemac .slf output files for calibration purposes. (up to 4 possible)
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
        self.calibration_parameters=calibration_parameters  #
        self.parameter_sampling_method=parameter_sampling_method
        self.init_runs=init_runs
        self.calibration_values_ranges = calibration_values_ranges
        self.calibration_quantities=calibration_quantities
        self.calibration_pts_df=_pd.read_csv(calibration_pts_file_path)
        #self.slf_input_file = slf_input_file
        self.doe = DesignOfExperiment()
        self.tm_cas = "{}{}{}".format(self.model_dir, os.sep, control_file)
        self.tm_results_filename = ""
        self.num_run=int(sys.argv[1])
        print(self.num_run)
        #pdb.set_trace()
        if self.num_run == 1:
            self.calibration_values_list=self.parameter_sampling(self.calibration_parameters,self.calibration_values_ranges,self.parameter_sampling_method,self.init_runs)
        else:
            df = _pd.read_csv(self.model_dir + "/initial-run-parameters-all.csv", sep=',', index_col=0)  # Assuming the first column is the index
            print(df)
            # Iterate over each row and extract values as a list
            self.parameter_values_dict = {}
            for index, row in df.iterrows():
                self.parameter_values_dict[index] = row.values.tolist()
            self.calibration_values_list=df.loc['PC'+str(self.num_run)].tolist()

        if gaia_steering_file:
            print("* received gaia steering file: " + gaia_steering_file)
            self.gaia_cas = "{}{}{}".format(self.model_dir, os.sep, gaia_steering_file)
            self.gaia_results_file = "{}{}{}".format(self.res_dir, os.sep,
                                                     str("resIDX-" + gaia_steering_file.strip(".cas") + ".slf"))
        else:
            self.gaia_cas = None
            self.gaia_results_file = None
        self.tm_results_filename = results_file_name_base + '-' + str(self.num_run) +'.slf'
        self.calibration_parameters.append('RESULTS FILE')
        self.calibration_values_list.append(self.tm_results_filename)
        for param, val in zip(self.calibration_parameters, self.calibration_values_list):
            cas_string = self.create_cas_string(param, val)
            self.rewrite_steering_file(param, cas_string, steering_module="telemac")
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
        if load_case:
            self.load_case()
        #pdb.set_trace()
        # self.get_results_filename()  # required for Telemac runs through stdout
        print(self.tm_results_filename)
        # pdb.set_trace()
        # self.load_results()
        #self.calibration_parameters = False

        #self.extract_data_point(self.tm_results_filename, 800, 790, 'output_file.csv')
        # if calibration_parameters:
        #     self.set_calibration_parameters(calibration_parameters)
        #self.create_cas_string(calibration_parameters,calibration_values)
    def set_calibration_parameters(self, list_of_value_names):
        # DELETE METHOD?
        # value corresponds to a list of parameter names -- REALLY needed?!
        self.calibration_parameters = {"telemac": {}, "gaia": {}}
        for par in list_of_value_names:
            if par in TM2D_PARAMETERS.iloc[:, 0].values:
                self.calibration_parameters["telemac"].update({par: {"current value": _np.nan}})
                continue
            if par in GAIA_PARAMETERS:
                self.calibration_parameters["gaia"].update({par: {"current value": _np.nan}})

    @staticmethod
    def create_cas_string(param_name, value):
        """
        Create string names with new values to be used in Telemac2d / Gaia steering files

        :param str param_name: name of parameter to update
        :param float or sequence value: new values for the parameter
        :return str: update parameter line for a steering file
        """
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
            return param_name + " = " + str(value)
        else:
            try:
                return param_name + " = " + "; ".join(map(str, value))
            except Exception as error:
                print("ERROR: could not generate cas-file string for {0} and value {1}:\n{2}".format(str(param_name), str(value), str(error)))

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
            #pdb.set_trace()
            print(self.case)
            self.case = Telemac2d(self.tm_cas, lang=2, comm=self.comm, stdout=self.stdout)
            print(self.case)
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
            self.tm_results_filename = self.case.get("MODEL.RESULTFILE")
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

    def update_model_controls(
            self,
            new_parameter_values,
            simulation_id=0,
    ):
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
            shutil.move(self.tm_results_filename, os.path.join(self.res_dir, self.tm_results_filename.split(os.sep)[-1]))
        except Exception as error:
            print("ERROR: could not move results file to " + self.res_dir + "\nREASON:\n" + error)
            return -1

        # update telemac calibration pars
        for par, has_more in lookahead(self.calibration_parameters["telemac"].keys()):
            self.calibration_parameters["telemac"][par]["current value"] = new_parameter_values[par]
            updated_string = self.create_cas_string(par, new_parameter_values[par])
            self.rewrite_steering_file(par, updated_string, self.tm_cas)
            if not has_more:
                updated_string = "RESULTS FILE" + " = " + self.tm_results_filename.replace(".slf", f"{simulation_id:03d}" + ".slf")
                self.rewrite_steering_file("RESULTS FILE", updated_string, self.tm_cas)

        # update gaia calibration pars - this intentionally does not iterate through self.calibration_parameters
        for par, has_more in lookahead(self.calibration_parameters["gaia"].keys()):
            self.calibration_parameters["gaia"][par]["current value"] = new_parameter_values[par]
            updated_string = self.create_cas_string(par, new_parameter_values[par])
            self.rewrite_steering_file(par, updated_string, self.gaia_cas)
            if not has_more:
                updated_string = "RESULTS FILE" + " = " + self.gaia_results_file.replace(".slf", f"{simulation_id:03d}" + ".slf")
                self.rewrite_steering_file("RESULTS FILE", updated_string, self.gaia_cas)

        return 0

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

        self.extract_data_point(self.tm_results_filename,self.calibration_pts_df)
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

    def get_variable_value(
            self,
            slf_file_name=".slf",
            calibration_par="",
            specific_nodes=None,
            save_name=None
    ):
        """
        Retrieve values of parameters (simulation parameters to calibrate)

        :param str slf_file_name: name of a SELAFIN *.slf file
        :param str calibration_par: name of calibration variable of interest
        :param list or numpy.array specific_nodes: enable to only get values of specific nodes of interest
        :param str save_name: name of a txt file where variable values should be written to
        :return:
        """

        # read SELAFIN file

        slf = ppSELAFIN(slf_file_name)
        slf.readHeader()
        slf.readTimes()

        ## FROM TELEMAC notebooks/telemac2d:
        #help(self.case.get_node)  # gets the nearest node number of an slf file

        # get the printout times
        times = slf.getTimes()
        # read variables names
        variable_names = slf.getVarNames()
        # remove unnecessary spaces from variables_names
        variable_names = [v.strip() for v in variable_names]
        # get position of the value of interest
        index_variable_interest = variable_names.index(calibration_par)

        # read the variables values in the last time step
        slf.readVariables(len(times) - 1)
        # get values (for each node) for the variable of interest at the last time step
        modeled_results = slf.getVarValues()[index_variable_interest, :]
        format_modeled_results = _np.zeros((len(modeled_results), 2))
        format_modeled_results[:, 0] = _np.arange(1, len(modeled_results) + 1, 1)
        format_modeled_results[:, 1] = modeled_results

        # get specific values of the model results associated with certain nodes number
        # to just compare selected nodes; requires that specific_nodes kwarg is defined
        if specific_nodes is not None:
            format_modeled_results = format_modeled_results[specific_nodes[:, 0].astype(int) - 1, :]

        if len(save_name) != 0:
            _np.savetxt(save_name, format_modeled_results, delimiter="	",
                        fmt=["%1.0f", "%1.3f"])

        # return the value of the variable of interest at mesh nodes (all or specific_nodes of interest)
        return format_modeled_results

    def parameter_sampling(self,calibration_parameters,calibration_values,parameter_sampling_method,total_number_of_samples):

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
        self.doe.df_parameter_spaces.to_csv(
            self.model_dir + "{}parameter-file.csv".format(os.sep),
            sep=";",
            index=True,
            header=False
        )
        self.doe.df_parameter_spaces.to_csv(self.model_dir + "/initial-run-parameters-all.csv")
        print(self.doe.df_parameter_spaces.loc['PC1'].tolist())

        return self.doe.df_parameter_spaces.loc['PC1'].tolist()

    #def extract_data_point(self,input_slf_file,calib_pts_file_path,json_name='json_file'):
    def extract_data_point(self, input_slf_file, calibration_pts_df, json_name='json_file'):
        #calibration_pts_df=_pd.read_csv(calib_pts_file_path)
        input_file = os.path.join(self.model_dir, input_slf_file)
        json_path = os.path.join(self.model_dir, f"{json_name}.json") #f"{json_name}_{self.num_run}.json"
        keys = list(calibration_pts_df.iloc[:, 0])
        #keys= [simulation + f"_{self.num_run}" for simulation in keys]
        modeled_values_dict = {}
        for key,h in zip(keys,range(len(calibration_pts_df))):
            #modeled_values_dict = {}
            #input_file = os.path.join(self.model_dir, input_slf_file)
            xu = calibration_pts_df.iloc[h,1]
            #print(xu)
            yu = calibration_pts_df.iloc[h,2]
            #print(yu)

            #output_file = os.path.join(self.model_dir, f"{output_file}_{self.num_run}.txt")

            # reads the *.slf file
            slf = ppSELAFIN(input_file)
            slf.readHeader()
            slf.readTimes()

            # get times of the selafin file, and the variable names
            times = slf.getTimes()
            variables = slf.getVarNames()
            units = slf.getVarUnits()
            # get the start date from the result file
            # this is a numpy array of [yyyy mm dd hh mm ss]
            # date = slf.getDATE()
            # year = date[0]
            # month = date[1]
            # day = date[2]
            # hour = date[3]
            # minute = date[4]
            # second = date[5]
            #
            # # use the date info from the above array to construct a python datetime
            # # object, in order to display day/time
            # try:
            #     pydate = datetime(year, month, day, hour, minute, second)
            # except:
            #     print('Date in file invalid. Printing default date in output file.')
            #     pydate = datetime(1997, 8, 29, 2, 15, 0)
            #
            # # this is the time step in seconds, as read from the file
            # # assumes the time steps are regular
            # if (len(times) > 1):
            #     pydelta = times[1] - times[0]
            # else:
            #     pydelta = 0.0

            # number of variables
            NVAR = len(variables)

            # to remove duplicate spaces from variables and units
            for i in range(NVAR):
                variables[i] = ' '.join(variables[i].split())
                units[i] = ' '.join(units[i].split())
            #print(variables)
            #variables = list(set(variables) & set(self.calibration_quantities))
            print(variables)
            print(self.calibration_quantities)

            common_indices = []

            # Iterate over the secondary list
            for value in self.calibration_quantities:
                # Find the index of the value in the original list
                index = variables.index(value)
                # Add the index to the common_indices list
                common_indices.append(index)
            print(common_indices)
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

            # print the node location to the output file
            print('Extraction performed at: ' + str(x[idx]) + ' ' + str(y[idx]) + '\n')
            print('Note this is the closest node to the input coordinate!' + '\n')

            # now we need this index for all planes
            idx_all = np.zeros(NPLAN, dtype=np.int32)

            # the first plane
            idx_all[0] = idx

            # start at second plane and go to the end
            for i in range(1, NPLAN, 1):
                idx_all[i] = idx_all[i - 1] + (NPOIN / NPLAN)

            ########################################################################
            # extract results for every plane (if there are multiple planes that is)
            #modeled_values_dict = {}
            for p in range(NPLAN):
                slf.readVariablesAtNode(idx_all[p])
                results = slf.getVarValuesAtNode()
                results_calibration = results[-1]
                print(results)
                print(results_calibration)
                    # Initialize an empty list to store values for this key
                modeled_values_dict[key] = []
                # Iterate over the common indices
                for index in common_indices:
                    # Extract value from the last row based on the index
                    value = results_calibration[index]
                    # Append the value to the list for the current key
                    modeled_values_dict[key].append(value)
            print(modeled_values_dict)
            # New dictionary to store the differentiated values
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

            print(differentiated_dict)

        # Define the condition for file deletion
        if self.num_run == 1:
            try:
                os.remove(json_path)
                print("File deleted successfully!")
            except FileNotFoundError:
                print("No result file found. Creating a new file.")
            # Set your condition here

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

def run_telemac():
    my_object=TelemacModel(
        model_dir=cas_file_simulation_path,
        control_file=cas_file_name,
        calibration_parameters=calib_parameter_list,
        calibration_values_ranges=parameter_ranges_list,
        calibration_pts_file_path=calib_pts_file_path,
        calibration_quantities=calib_quantity_list,
        results_file_name_base=results_file_name_base,
        tm_xd='Telemac2d',
        n_processors=NCPS,
        parameter_sampling_method=parameter_sampling_method
        )
    #my_object.set_calibration_parameters()
    my_object.run_simulation()
    #my_object.extract_data_point('r2d-weirs-2.slf', calib_pts_file_path)
if __name__ == "__main__":
    run_telemac()
