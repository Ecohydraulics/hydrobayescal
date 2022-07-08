"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with Telemac.
"""

import os as _os
import subprocess
import shutil
import numpy as _np
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN
from basic_functions import *


class TelemacModel:
    def __init__(self, *args, **kwargs):
        pass

    def update_steering_file(self, prior_distribution, parameters_name, initial_diameters,
                             auxiliary_names, gaia_name, telemac_name,
                             result_name_gaia, result_name_telemac, n_simulation):
        """
        Update the Telemac2d and Gaia steering files

        :param list prior_distribution: dep. stress in N/m2, eros. stress in N/m2, denisty in kg/m3, settling velocity in m/s
        :param list parameters_name: list of strings describing parameter names
        :param list initial_diameters: floats of diameters
        :param list auxiliary_names: strings of auxiliary parameter names
        :param str gaia_name: file name of Gaia cas
        :param str telemac_name: file name of Telemac2d cas
        :param str result_name_gaia: file name of gaia results SLF
        :param str result_name_telemac: file name of telemac results SLF
        :param int n_simulation:
        :return:
        """

        # Update deposition stress
        updated_values = _np.round(_np.ones(4) * prior_distribution[0], decimals=3)
        updated_string = create_string(parameters_name[0], updated_values)
        self.rewrite_steering_file(parameters_name[0], updated_string, gaia_name)

        # Update erosion stress
        updated_values = _np.round(_np.ones(2) * prior_distribution[1], decimals=3)
        updated_string = create_string(parameters_name[1], updated_values)
        self.rewrite_steering_file(parameters_name[1], updated_string, gaia_name)

        # Update density
        updated_values = _np.round(_np.ones(2) * prior_distribution[2], decimals=0)
        updated_string = create_string(parameters_name[2], updated_values)
        self.rewrite_steering_file(parameters_name[2], updated_string, gaia_name)

        # Update settling velocity
        new_diameters = initial_diameters * prior_distribution[3]
        settling_velocity = calculate_settling_velocity(new_diameters[1:])
        updated_values = "; ".join(map("{:.3E}".format, settling_velocity))
        updated_values = "-9; " + updated_values.replace("E-0", "E-")
        updated_string = parameters_name[3] + " = " + updated_values
        self.rewrite_steering_file(parameters_name[3], updated_string, gaia_name)

        # Update other variables
        new_diameters[0] = initial_diameters[0]  # the first non-cohesive diameter stays the same
        updated_values = "; ".join(map("{:.3E}".format, new_diameters))
        updated_values = updated_values.replace("E-0", "E-")
        updated_string = auxiliary_names[0] + " = " + updated_values
        self.rewrite_steering_file(auxiliary_names[0], updated_string, gaia_name)

        # Update result file name gaia
        updated_string = "RESULTS FILE" + "=" + result_name_gaia + str(n_simulation) + ".slf"
        self.rewrite_steering_file("RESULTS FILE", updated_string, gaia_name)
        # Update result file name telemac
        updated_string = "RESULTS FILE" + "=" + result_name_telemac + str(n_simulation) + ".slf"
        self.rewrite_steering_file("RESULTS FILE", updated_string, telemac_name)

    def rewrite_steering_file(self, param_name, updated_string, directory):
        """
        Rewrite the steering file with new parameters

        :param str param_name: name of parameter to rewrite
        :param str updated_string: new values for parameter
        :param str directory: directory where the steering file is tp be saved
        :return None:
        """

        # Save the variable of interest without unwanted spaces
        variable_interest = param_name.rstrip().lstrip()

        # Open the steering file with read permission and save a temporary copy
        gaia_file = open(directory, "r")
        read_steering = gaia_file.readlines()

        # If the updated_string have more 72 characters, then it divides it in two
        if len(updated_string) >= 72:
            position = updated_string.find("=") + 1
            updated_string = updated_string[:position].rstrip().lstrip() + "\n" + updated_string[position:].rstrip().lstrip()

        # Preprocess the steering file. If in a previous case, a line had more than 72 characters then it was split in 2,
        # so this loop clean all the lines that start with a number
        temp = []
        for i, line in enumerate(read_steering):
            if not isinstance(line[0], int):
                temp.append(line)
            else:
                previous_line = read_steering[i-1].split("=")[0].rstrip().lstrip()
                if previous_line != variable_interest:
                    temp.append(line)

        # Loop through all the lines of the temp file, until it finds the line with the parameter we are interested in,
        # and substitute it with the new formatted line
        for i, line in enumerate(temp):
            line_value = line.split("=")[0].rstrip().lstrip()
            if line_value == variable_interest:
                temp[i] = updated_string + "\n"

        # Rewrite and close the steering file
        gaia_file = open(directory, "w")
        gaia_file.writelines(temp)
        gaia_file.close()




    def run_telemac(self, telemac_file_name, number_processors=1):
        """
        Run a Telemac simulation

        :param str telemac_file_name: name of the steering file for the simulation (must end on .cas)
        :param int number_processors: number of processor to use for parallelization (default: 1
        :return None:
        """
        start_time = datetime.now()
        bash_cmd = "telemac2d.py " + telemac_file_name + " --ncsize=" + str(number_processors)
        process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Telemac simulation finished")
        print("Simulation time= " + str(datetime.now() - start_time))


    def run_gretel(self, telemac_file_name, number_processors, folder_rename):
        # Save original working directory
        original_directory = _os.getcwd()

        # Access folder with results
        subfolders = [f.name for f in _os.scandir(original_directory) if f.is_dir()]
        simulation_index = [i for i, s in enumerate(subfolders) if telemac_file_name in s]
        simulation_index = simulation_index[0]
        original_name = subfolders[simulation_index]
        _os.rename(original_name, folder_rename)
        simulation_path = "./"+ folder_rename
        _os.chdir(simulation_path)

        # Run gretel code
        bash_cmd = "gretel.py --geo-file=T2DGEO --res-file=GAIRES --ncsize="+number_processors+" --bnd-file=T2DCLI"
        process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Finish Gretel for GAIA")

        bash_cmd = "gretel.py --geo-file=T2DGEO --res-file=T2DRES --ncsize="+number_processors+" --bnd-file=T2DCLI"
        process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Finish Gretel for Telemac")

        # Copy result files in original folder
        shutil.copy("GAIRES", original_directory)
        shutil.copy("T2DRES", original_directory)
        _os.chdir(original_directory)

    def rename_selafin(self, original_name, new_name):
        """
        Merged parallel computation meshes (gretel subroutine) does not add correct file endings.
        This funciton adds the correct file ending to the file name.

        :param str original_name: original file name
        :param str new_name: new file name
        :return: None
        :rtype: None
        """

        if _os.path.exists(original_name):
            _os.rename(original_name, new_name)
        else:
            print("File not found")

    def get_variable_value(self, file_name, calibration_variable, specific_nodes=None, save_name=""):
        """
        Retrieve values of parameters

        :param str file_name:
        :param calibration_variable:
        :param specific_nodes:
        :param str save_name:
        :return:
        """

        # Read the SELEFIN file
        slf = ppSELAFIN(file_name)
        slf.readHeader()
        slf.readTimes()

        # Get the printout times
        times = slf.getTimes()

        # Read the variables names
        variables_names = slf.getVarNames()
        # Removed unnecessary spaces from variables_names
        variables_names = [v.strip() for v in variables_names]
        # Get the position of the value of interest
        index_variable_interest = variables_names.index(calibration_variable)

        # Read the variables values in the last time step
        slf.readVariables(len(times) - 1)

        # Get the values (for each node) for the variable of interest in the last time step
        modelled_results = slf.getVarValues()[index_variable_interest, :]
        format_modelled_results = _np.zeros((len(modelled_results), 2))
        format_modelled_results[:, 0] = _np.arange(1, len(modelled_results) + 1, 1)
        format_modelled_results[:, 1] = modelled_results

        # Get specific values of the model results associated in certain nodes number, in case the user want to use just
        # some nodes for the comparison. This part only runs if the user specify the parameter specific_nodes. Otherwise
        # this part is ommited and all the nodes of the model mesh are returned
        if specific_nodes is not None:
            format_modelled_results = format_modelled_results[specific_nodes[:, 0].astype(int) - 1, :]

        if len(save_name) != 0:
            _np.savetxt(save_name, format_modelled_results, delimiter="	",
                        fmt=["%1.0f", "%1.3f"])

        # Return the value of the variable of interest
        return format_modelled_results


