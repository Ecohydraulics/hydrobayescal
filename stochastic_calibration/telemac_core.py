"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with Telemac.
"""

import os as _os
import shutil
import numpy as _np
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN
from basic_functions import *


class TelemacModel:
    def __init__(
            self,
            model_dir="",
            calibration_parameters=None,
            steering_file="tm.cas",
            gaia_steering_file=None,
            n_processors=1,
            *args,
            **kwargs
    ):
        """
        Constructor for the TelemacModel Class. Instantiating can take some seconds, so try to
        be efficient in creating objects of this class (i.e., avoid re-creating a new TelemacModel in long loops)

        :param str model_dir: directory (path) of the Telemac model (should NOT end on "/" or "\\")
        :param list calibration_parameters: computationally optional, but in the framework of Bayesian calibration,
                    this argument must be provided
        :param str steering_file: name of the steering file to be used (should end on ".cas"); do not include directory
        :param str gaia_steering_file: name of a gaia steering file (optional)
        :param int n_processors: number of processors to use (>1 corresponds to parallelization); default is 1
        :param args:
        :param kwargs:
        """
        self.model_dir = _os.path.abspath(model_dir)
        self.tm_cas = "{}{}{}".format(self.model_dir, _os.sep, steering_file)
        if gaia_steering_file:
            print("* received gaia steering file: " + gaia_steering_file)
            self.gaia_cas = "{}{}{}".format(self.model_dir, _os.sep, gaia_steering_file)
        else:
            self.gaia_cas = None
        self.nproc = n_processors

        self.__calibration_parameters = False
        if calibration_parameters:
            self.__setattr__("calibration_parameters", calibration_parameters)

    def __setattr__(self, name, value):
        if name == "calibration_parameters":
            # value corresponds to a list of parameters
            self.calibration_parameters = {"telemac": [], "gaia": []}
            for par in value:
                if par in TM2D_PARAMETERS:
                    self.calibration_parameters["telemac"].append(par)
                    continue
                if par in GAIA_PARAMETERS:
                    self.calibration_parameters["gaia"].append(par)

    def update_steering_file(
            self,
            prior_distribution,
            parameters_name,
            initial_diameters,
            auxiliary_names,
            result_name_gaia,
            result_name_telemac,
            n_simulation
    ):
        """
        Update the Telemac and Gaia steering files specifically for Bayesian calibration.

        :param np.array prior_distribution: e.g., shear stress in N/m2, denisty in kg/m3, settling velocity in m/s
        :param list parameters_name: list of strings describing parameter names
        :param list initial_diameters: floats of diameters
        :param list auxiliary_names: strings of auxiliary parameter names
        :param str result_name_gaia: file name of gaia results SLF
        :param str result_name_telemac: file name of telemac results SLF
        :param int n_simulation:
        :return:
        """
        # update telemac calibration pars
        for par in self.calibration_parameters["telemac"]:
            pass
        # update gaia calibration pars
        for par in self.calibration_parameters["gaia"]:
            pass
        # Update deposition stress
        updated_values = _np.round(_np.ones(4) * prior_distribution[0], decimals=3)
        updated_string = create_cas_string(parameters_name[0], updated_values)
        self.rewrite_steering_file(parameters_name[0], updated_string, self.gaia_cas)

        # Update erosion stress
        updated_values = _np.round(_np.ones(2) * prior_distribution[1], decimals=3)
        updated_string = create_cas_string(parameters_name[1], updated_values)
        self.rewrite_steering_file(parameters_name[1], updated_string, self.gaia_cas)

        # Update density
        updated_values = _np.round(_np.ones(2) * prior_distribution[2], decimals=0)
        updated_string = create_cas_string(parameters_name[2], updated_values)
        self.rewrite_steering_file(parameters_name[2], updated_string, self.gaia_cas)

        # Update settling velocity
        new_diameters = initial_diameters * prior_distribution[3]
        settling_velocity = calculate_settling_velocity(new_diameters[1:])
        updated_values = "; ".join(map("{:.3E}".format, settling_velocity))
        updated_values = "-9; " + updated_values.replace("E-0", "E-")
        updated_string = parameters_name[3] + " = " + updated_values
        self.rewrite_steering_file(parameters_name[3], updated_string, self.gaia_cas)

        # Update other variables
        new_diameters[0] = initial_diameters[0]  # the first non-cohesive diameter stays the same
        updated_values = "; ".join(map("{:.3E}".format, new_diameters))
        updated_values = updated_values.replace("E-0", "E-")
        updated_string = auxiliary_names[0] + " = " + updated_values
        self.rewrite_steering_file(auxiliary_names[0], updated_string, self.gaia_cas)

        # Update result file name gaia
        updated_string = "RESULTS FILE" + "=" + result_name_gaia + str(n_simulation) + ".slf"
        self.rewrite_steering_file("RESULTS FILE", updated_string, self.gaia_cas)
        # Update result file name telemac
        updated_string = "RESULTS FILE" + "=" + result_name_telemac + str(n_simulation) + ".slf"
        self.rewrite_steering_file("RESULTS FILE", updated_string, self.tm_cas)

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
        if _os.path.isfile(steering_file_name):
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

    def run_simulation(self):
        """
        Run a Telemac simulation

        .. note::
            Generic function name to enable other simulation software in future implementations

        :return None:
        """
        start_time = datetime.now()
        call_subroutine("telemac2d.py " + self.tm_cas + " --ncsize=" + str(self.nproc))
        print("TELEMAC simulation time: " + str(datetime.now() - start_time))

    def run_gretel(self, telemac_file_name="SLF?", number_processors=1, sim_folder=""):
        """
        Launch Gretel for multi-core processing

        :param str telemac_file_name: name of a telemac file (SLF?)
        :param int number_processors: number of processors for parallelization
        :param str sim_folder: directory where the Telemac simulation lives
        :return:
        """
        # save original working directory
        original_directory = _os.getcwd()

        # access folder with results
        try:
            subfolders = [f.name for f in _os.scandir(original_directory) if f.is_dir()]
        except AttributeError:
            print("WARNING: No folders found in %s - skipping Gretel (parallel processing)" % original_directory)
            return -1
        try:
            simulation_index_list = [i for i, s in enumerate(subfolders) if telemac_file_name in s]
            simulation_index = simulation_index_list[0]
            original_name = subfolders[simulation_index]
            _os.rename(original_name, sim_folder)
            simulation_path = "./" + sim_folder
            _os.chdir(simulation_path)
        except Exception as e:
            print("WARNING: pre-processing for Gretel failed - skipping Gretel:\n" + str(e))
            return -1

        # run gretel code
        bash_base = "gretel.py --geo-file=T2DGEO --res-file="
        bash_ending = " --ncsize=" + str(number_processors) + " --bnd-file=T2DCLI"

        call_subroutine(str(bash_base + "GAIRES" + bash_ending))  # merge gaia files
        call_subroutine(str(bash_base + "T2DRES" + bash_ending))  # merge telemac files

        # copy result files into original folder
        shutil.copy("GAIRES", original_directory)
        shutil.copy("T2DRES", original_directory)
        _os.chdir(original_directory)

    def rename_selafin(self, old_name=".slf", new_name=".slf"):
        """
        Merged parallel computation meshes (gretel subroutine) does not add correct file endings.
        This function adds the correct file ending to the file name.

        :param str old_name: original file name
        :param str new_name: new file name
        :return: None
        :rtype: None
        """

        if _os.path.exists(old_name):
            _os.rename(old_name, new_name)
        else:
            print("WARNING: SELAFIN file %s does not exist" % old_name)

    def get_variable_value(
            self,
            file_name=".slf",
            calibration_variable="",
            specific_nodes=None,
            save_name=None
    ):
        """
        Retrieve values of parameters (simulation parameters to calibrate)

        :param str file_name: name of a SELAFIN *.slf file
        :param str calibration_variable: name of calibration variable of interest
        :param list or numpy.array specific_nodes: enable to only get values of specific nodes of interest
        :param str save_name: name of a txt file where variable values should be written to
        :return:
        """

        # read SELAFIN file
        slf = ppSELAFIN(file_name)
        slf.readHeader()
        slf.readTimes()

        # get the printout times
        times = slf.getTimes()
        # read variables names
        variable_names = slf.getVarNames()
        # remove unnecessary spaces from variables_names
        variable_names = [v.strip() for v in variable_names]
        # get position of the value of interest
        index_variable_interest = variable_names.index(calibration_variable)

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
