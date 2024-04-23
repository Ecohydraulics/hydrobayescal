"""Function pool for usage at different package levels"""
import subprocess, os, logging
import numpy as _np
import pandas as _pd

from config_logging import logger, logger_warn, logger_error
from model_structure.config_physics import *


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


def call_process(bash_command, environment=None):
    """
    Call a Terminal process with a bash command through subprocess.Popen

    :param str bash_command: terminal process to call
    :param environment: run process in a specific environment (e.g.
    :return int: 0 (success) or -1 (error - read output message)
    """

    print("* CALLING SUBROUTINE: %s " % bash_command)
    try:
        # process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        if environment:
            # res = subprocess.run(bash_command, capture_output=True, shell=True, env=environment)
            process = subprocess.Popen(bash_command, stdout=subprocess.PIPE,
                                       shell=True, stdin=None, env=environment)
            output, error = process.communicate()
        else:
            res = subprocess.run(bash_command, capture_output=True, shell=True)
            print(res.stdout.decode())
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


def concatenate_csv_pts(file_directory, *args):
    """Concatenate a csv-files with lists of XYZ points into one CSV file that is saved to the same directory where the
    first CSV file name provided lives. The merged CSV file name starts with merged_ and also ends with the name
    of the first CSV file name provided.

    :param file_directory: os.path of the directory where the CSV files live, and which must NOT end on '/' or '\\'
    :param args: string or list of csv files (only names) containing comma-seperated XYZ coordinates without header
    :return pandas.DataFrame: merged points
    """
    point_file_names = []
    # receive arguments (i.e. csv point file names)
    for arg in args:
        if type(arg) is str:
            point_file_names.append(file_directory + os.sep + arg)
        if type(arg) is list:
            [point_file_names.append(file_directory + os.sep + e) for e in arg]

    # read csv files
    point_data = []
    for file_name in point_file_names:
        if os.path.isfile(file_name):
            point_data.append(_pd.read_csv(file_name, names=["X", "Y", "Z"]))
        else:
            print("WARNING: Points CSV file does not exist: %s" % file_name)

    # concatenate frames
    merged_pts = _pd.concat(point_data)

    # save concatenated points to a CSV file
    merged_pts.to_csv(
        # make sure to identify platform-independent separators
        file_directory + "merged-" + str(point_file_names[0]).split("/")[-1].split("\\")[-1],
        header=False,
        index=False
    )

    return merged_pts


def lookahead(iterable):
    """Pass through all values of an iterable, augmented by the information if there are more values to come
    after the current one (True), or if it is the last value (False).

    Source: Ferdinand Beyer (2015) on https://stackoverflow.com/questions/1630320/what-is-the-pythonic-way-to-detect-the-last-element-in-a-for-loop
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False


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
        for handler in logging.getLogger("HyBayesCal").handlers:
            handler.close()
            logging.getLogger("HyBayesCal").removeHandler(handler)
        for handler in logging.getLogger("warnings").handlers:
            handler.close()
            logging.getLogger("warnings").removeHandler(handler)
        for handler in logging.getLogger("errors").handlers:
            handler.close()
            logging.getLogger("errors").removeHandler(handler)
        print("Check the logfiles: logfile.log, warnings.log, and errors.log.")
    return wrapper
