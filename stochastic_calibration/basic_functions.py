"""fundamental Python functions"""
from config import *
import numpy as _np
import subprocess


def append_new_line(file_name, text_to_append):
    """
    Add new line to steering file

    :param str file_name: path and name of the file to which the line should be appended
    :param str text_to_append: text of the line to append
    :return None:
    """
    # Open the file in append & read mode ("a+")
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append "\n"
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def call_subroutine(bash_command):
    """
    Call a Terminal process with a bash command through subprocess.Popen

    :param str bash_command: terminal process to call
    :return int: 0 (success) or -1 (error - read output message)
    """

    print("* calling %s " % bash_command)
    try:
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("* finished ")
        return 0
    except Exception as e:
        print("WARNING: command failed:\n" + str(e))
        return -1


def calculate_settling_velocity(diameters):
    """
    Calculate particle settling velocity as a function of diameter, densities of water and
    sediment, and kinematic viscosity

    :param np.array diameters: floats of sediment diameter in meters
    :return np.array settling_vevlocity: settling velocities in m/s for every diameter in the diameters list
    """
    settling_velocity = _np.zeros(diameters.shape[0])
    s = SED_DENSITY / WATER_DENSITY
    for i, d in enumerate(diameters):
        if d <= 0.0001:
            settling_velocity[i] = (s - 1) * GRAVITY * d ** 2 / (18 * KINEMATIC_VISCOSITY)
        elif 0.0001 < d < 0.001:
            settling_velocity[i] = 10 * KINEMATIC_VISCOSITY / d * (_np.sqrt(1 + 0.01 * (s-1) * GRAVITY * d**3 / KINEMATIC_VISCOSITY**2) - 1)
        else:
            settling_velocity[i] = 1.1 * _np.sqrt((s - 1) * GRAVITY * d)
    return settling_velocity


def create_cas_string(param_name, values):
    """
    Create string names with new values to be used in Telemac2d / Gaia steering files

    :param str param_name: name of parameter to update
    :param list values: new values for the parameter
    :return str: update parameter line for a steering file
    """
    return param_name + " = " + "; ".join(map(str, values))


def str2seq(list_like_string, separator=",", return_type="tuple"):
    """Convert a list-like string into a tuple or list based on a separator such as comma or semi-column

    :param str list_like_string: string to convert
    :param str separator: separator to use
    :param str return_type: defines if a list or tuple is returned (default: tuple)
    :return: list or tuple
    """
    seq = []
    for number in list_like_string.split(separator):
        try:
            seq.append(float(number))
        except ValueError:
            print("WARNING: Could not interpret user parameter value range definition (%s)" % number)
            print("         This Warning will probably cause an ERROR later in the script.")
    if "tuple" in return_type:
        return tuple(seq)
    else:
        return seq


def log_actions(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        for handler in logging.getLogger("stochastic_calibration").handlers:
            handler.close()
            logging.getLogger("stochastic_calibration").removeHandler(handler)
        for handler in logging.getLogger("warnings").handlers:
            handler.close()
            logging.getLogger("warnings").removeHandler(handler)
        for handler in logging.getLogger("errors").handlers:
            handler.close()
            logging.getLogger("errors").removeHandler(handler)
        print("Check the logfiles: logfile.log, warnings.log, and errors.log.")
    return wrapper
