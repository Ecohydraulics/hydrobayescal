"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with Telemac.
"""

from .config_telemac import *  # provides os and sys
import shutil
import numpy as _np
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN

# get package script
from .function_pool import *
from .model_structure.control_full_complexity import FullComplexityModel


class TelemacModel(FullComplexityModel):
    def __init__(
            self,
            model_dir="",
            calibration_parameters=None,
            control_file="tm.cas",
            gaia_steering_file=None,
            n_processors=1,
            slf_input_file=".slf",
            tm_xd="Telemac2d",
            *args,
            **kwargs
    ):
        """
        Constructor for the TelemacModel Class. Instantiating can take some seconds, so try to
        be efficient in creating objects of this class (i.e., avoid re-creating a new TelemacModel in long loops)

        :param str model_dir: directory (path) of the Telemac model (should NOT end on "/" or "\\")
        :param list calibration_parameters: computationally optional, but in the framework of Bayesian calibration,
                    this argument must be provided
        :param str control_file: name of the steering file to be used (should end on ".cas"); do not include directory
        :param str gaia_steering_file: name of a gaia steering file (optional)
        :param int n_processors: number of processors to use (>1 corresponds to parallelization); default is 1
        :param str slf_input_file: name of the SLF input file (without directory, file has to be located in model_dir)
        :param str tm_xd: either 'Telemac2d' or 'Telemac3d'
        :param args:
        :param kwargs:
        """
        FullComplexityModel.__init__(self, model_dir=model_dir)

        self.slf_input_file = slf_input_file
        self.tm_cas = "{}{}{}".format(self.model_dir, os.sep, control_file)
        self.tm_results_file = "{}{}{}".format(self.res_dir, os.sep, str("resIDX-" + control_file.strip(".cas") + ".slf"))
        if gaia_steering_file:
            print("* received gaia steering file: " + gaia_steering_file)
            self.gaia_cas = "{}{}{}".format(self.model_dir, os.sep, gaia_steering_file)
            self.gaia_results_file = "{}{}{}".format(self.res_dir, os.sep,
                                                     str("resIDX-" + gaia_steering_file.strip(".cas") + ".slf"))
        else:
            self.gaia_cas = None
            self.gaia_results_file = None
        self.nproc = n_processors

        self.tm_xd = tm_xd
        self.tm_xd_dict = {
            "Telemac2d": "telemac2d.py ",
            "Telemac3d": "telemac3d.py ",
        }

        self.__calibration_parameters = False
        if calibration_parameters:
            self.__setattr__("calibration_parameters", calibration_parameters)

    def __setattr__(self, name, value):
        if name == "calibration_parameters":
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

    def update_model_controls(
            self,
            new_parameter_values,
            simulation_id=0,
    ):
        """ In TELEMAC language: update the steering file
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
                updated_string = "RESULTS FILE" + " = " + self.tm_results_file.replace("IDX", f"{simulation_id:03d}")
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

    def run_simulation(self):
        """
        Run a Telemac simulation

        .. note::
            Generic function name to enable other simulation software in future implementations

        :return None:
        """
        start_time = datetime.now()
        if self.nproc > 1:
            call_subroutine(self.tm_xd_dict[self.tm_xd] + self.tm_cas + " --ncsize=" + str(self.nproc))
        else:
            call_subroutine(self.tm_xd_dict[self.tm_xd] + self.tm_cas)
        print("TELEMAC simulation time: " + str(datetime.now() - start_time))

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
