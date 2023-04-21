# coding: utf-8
"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with Telemac.
"""
import os, io, stat
import sys
import subprocess

try:
    from telapy.api.t2d import Telemac2d
    from telapy.api.t3d import Telemac3d
    from telapy.tools.driven_utils import mpirun_cmd
    from data_manip.extraction.telemac_file import TelemacFile
except ImportError as e:
    print("%s\n\nERROR: load (source) pysource.X.sh Telemac before running HyBayesCal.telemac" % e)
    exit()

# attention relative import usage according to docs/codedocs.rst
from .config_telemac import *  # provides os and sys
import shutil
import numpy as _np
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN

try:
    from mpi4py import MPI
except ImportError as e:
    logging.warning("Could not import mpi4py")
    print(e)

# get package script
from function_pool import *  # provides os, subprocess, logging
from model_structure.control_full_complexity import FullComplexityModel


class TelemacModel(FullComplexityModel):
    def __init__(
            self,
            model_dir="",
            calibration_parameters=None,
            control_file="tm.cas",
            gaia_steering_file=None,
            n_processors=None,
            slf_input_file=".slf",
            tm_xd="Telemac2d",
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
        :param list calibration_parameters: computationally optional, but in the framework of Bayesian calibration,
                    this argument must be provided
        :param str control_file: name of the steering file to be used (should end on ".cas"); do not include directory
        :param str gaia_steering_file: name of a gaia steering file (optional)
        :param int n_processors: number of processors to use (>1 corresponds to parallelization); default is None (use cas definition)
        :param str slf_input_file: name of the SLF input file (without directory, file has to be located in model_dir)
        :param str tm_xd: either 'Telemac2d' or 'Telemac3d'
        :param bool load_case: True loads the control file as Telemac case upon instantiation (default: False)
        :param int stdout: standard output (default=6 [console];  if 666 => file 'fort.666')
        :param str python_shebang: header line for python files the code writes for parallel processing
                                        (default="#!/usr/bin/env python3\n" for telling Debian-Linux to run with python)
        :param args:
        :param kwargs:
        """
        FullComplexityModel.__init__(self, model_dir=model_dir)

        self.slf_input_file = slf_input_file
        self.tm_cas = "{}{}{}".format(self.model_dir, os.sep, control_file)
        self.tm_results_filename = "{}{}{}".format(self.res_dir, os.sep, str("resIDX-" + control_file.strip(".cas") + ".slf"))
        if gaia_steering_file:
            print("* received gaia steering file: " + gaia_steering_file)
            self.gaia_cas = "{}{}{}".format(self.model_dir, os.sep, gaia_steering_file)
            self.gaia_results_file = "{}{}{}".format(self.res_dir, os.sep,
                                                     str("resIDX-" + gaia_steering_file.strip(".cas") + ".slf"))
        else:
            self.gaia_cas = None
            self.gaia_results_file = None
        self.nproc = n_processors
        self.comm = MPI.Comm(comm=MPI.COMM_WORLD)
        self.results = None  # will hold results loaded through self.load_results()
        self.shebang = python_shebang

        self.tm_xd = tm_xd
        self.tm_xd_dict = {
            "Telemac2d": "telemac2d.py ",
            "Telemac3d": "telemac3d.py ",
        }

        self.case = None
        self.case_loaded = False
        if load_case:
            self.load_case()

        self.stdout = stdout

        self.calibration_parameters = False
        if calibration_parameters:
            self.set_calibration_parameters("calibration_parameters", calibration_parameters)

    def set_calibration_parameters(self, name, value):
        # value corresponds to a list of parameters
        self.calibration_parameters = {"telemac": {}, "gaia": {}}
        for par in value:
            if par in TM2D_PARAMETERS:
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
            except Exception as e:
                print("ERROR: could not generate cas-file string for {0} and value {1}:\n{2}".format(str(param_name), str(value), str(e)))

    def load_case(self):
        """Load Telemac case file and check its consistency."""

        # secure directories
        init_dir = os.getcwd()
        os.chdir(self.model_dir)

        # instantiate case
        if "telemac2d" in self.tm_xd.lower():
            self.case = Telemac2d(self.tm_cas, lang=2, comm=self.comm, stdout=self.stdout)
        elif "telemac3d" in self.tm_xd.lower():
            self.case = Telemac3d(self.tm_cas, lang=2, comm=self.comm)
        else:
            logging.warning("Other solvers than Telemac2d/3d not available.")
            return -1
        if self.nproc > 1:
            self.comm.Barrier()

        # set and initialize case
        self.case.set_case()
        self.case.init_state_default()
        self.case_loaded = True
        os.chdir(init_dir)
        logging.info(" * successfully activated TELEMAC case: " + str(self.tm_cas))
        return 0

    def load_results(self):
        """Load simulation results stored in TelemacModel.tm_results_filename"""

        if not self.case_loaded:
            self.load_case()
        self.results = TelemacFile(self.tm_results_filename, bnd_file=self.case.get("MODEL.BCFILE"))
        # to see more case variables that can be self.case.get()-ed, type print(self.case.variables)

        # examples to access liquid boundary equilibrium
        liq_bnd_info = self.results.get_liq_bnd_info()
        print("Liquid BC info:\n" + str(liq_bnd_info))

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
        :return:
        """

        # update telemac calibration pars
        for par, has_more in lookahead(self.calibration_parameters["telemac"].keys()):
            self.calibration_parameters["telemac"][par]["current value"] = new_parameter_values[par]
            updated_string = self.create_cas_string(par, new_parameter_values[par])
            self.rewrite_steering_file(par, updated_string, self.tm_cas)
            if not has_more:
                updated_string = "RESULTS FILE" + " = " + self.tm_results_filename.replace("IDX", f"{simulation_id:03d}")
                self.rewrite_steering_file("RESULTS FILE", updated_string, self.tm_cas)

        # update gaia calibration pars - this intentionally does not iterate through self.calibration_parameters
        for par, has_more in lookahead(self.calibration_parameters["gaia"].keys()):
            self.calibration_parameters["gaia"][par]["current value"] = new_parameter_values[par]
            updated_string = self.create_cas_string(par, new_parameter_values[par])
            self.rewrite_steering_file(par, updated_string, self.gaia_cas)
            if not has_more:
                updated_string = "RESULTS FILE" + " = " + self.gaia_results_file.replace("IDX", f"{simulation_id:03d}")
                self.rewrite_steering_file("RESULTS FILE", updated_string, self.gaia_cas)

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

    def run_simulation(self, filename="run_launcher.py", load_results=True):
        """ Run a Telemac2d or Telemac3d simulation with one or more processors
        The number of processors to use is defined by self.nproc.

        :param (str) filename: optional name for a Python file that will be automatically
                        created to control the simulation
        :param (bool) load_results: default value of True re-assigns TelemacModel.results = TelemacFile("results.slf")
        """
        start_time = datetime.now()

        filename = os.path.join(self.model_dir, filename)
        self.create_launcher_pyscript(filename)

        if self.nproc == 1:
            print("* sequential run (single processor)")
            if not self.case_loaded:
                # every case must be initiated before it running
                self.load_case()
            self.case.run_all_time_steps()
        else:
            print("* parallel run on {} processors".format(self.nproc))
            try:
                self.mpirun(filename)
            except Exception as exception:
                print(exception)
        print("TELEMAC simulation time: " + str(datetime.now() - start_time))
        if load_results:
            self.load_results()

    def call_tm_shell(self, cmd):
        """ Run Telemac in a Terminal in the model directory

        :param (str) cmd:  command to run
        """
        logging.info("* running {}\n -- patience (Telemac simulations can take time) -- check CPU acitivity...".format(cmd))
        process = subprocess.Popen(cmd, cwd=r""+self.model_dir, shell=True, stdin=None,
                                   stdout=subprocess.PIPE, env=os.environ)
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
        help(my_case.get_node)  # gets the nearest node number of an slf file

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

    def __call__(self, *args, **kwargs):
        """
        Call method forwards to self.run_telemac()

        :param args:
        :param kwargs:
        :return:
        """
        self.run_simulation()
