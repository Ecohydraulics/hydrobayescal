"""Function pool for usage at different package levels"""
import subprocess, os, sys, logging
import numpy as _np
import pandas as _pd
import csv
import pickle
import h5py
import json
import glob
import shutil

sys.path.insert(0, os.path.abspath('..'))

from src.hydroBayesCal.utils.config_logging import *
# TODO: re-instate config_physics - Done
from src.hydroBayesCal.utils.config_physics import *


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
            settling_velocity[i] = 10 * KINEMATIC_VISCOSITY / d * (
                        _np.sqrt(1 + 0.01 * (s - 1) * GRAVITY * d ** 3 / KINEMATIC_VISCOSITY ** 2) - 1)
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
    """
    TODO: this is the logging wrapper!
    :param func:
    :return:
    """

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


def update_collocation_pts_file(
        file_path,
        new_collocation_point
):
    """
    Append a new row to a CSV file.

    :param file_path: Path to the CSV file.
    :param new_collocation_point: List of values to be added as a new row.
    """
    # Ensure new_set_parameters is a list
    if not isinstance(new_collocation_point, list):
        raise ValueError("new_set_parameters must be a list")

    # Create the new line with the PC description and the new parameters
    new_line = new_collocation_point

    # Open the file in append mode and write the new row
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_line)


def save_data(file_path, data):
    """
    Save NumPy array data to a file based on the file extension in the file path.

    :param file_path: Path to the file where data should be saved.
    :param data: NumPy array data to be saved.
    """
    try:
        # Determine file format based on file extension
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.npy':
            _np.save(file_path, data)

        elif file_extension in ['.pkl', '.pickle']:
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)

        elif file_extension == '.json':
            # Convert NumPy array to list for JSON serialization
            data_list = data.tolist()
            with open(file_path, 'w') as file:
                json.dump(data_list, file, indent=4)

        elif file_extension == '.csv':
            if isinstance(data, _np.ndarray):
                _np.savetxt(file_path, data, delimiter=',', fmt='%.8f')
            else:
                raise ValueError("For CSV format, data should be a NumPy array.")

        elif file_extension in ['.xlsx', '.xls']:
            df = _pd.DataFrame(data)
            df.to_excel(file_path, index=False)

        elif file_extension in ['.h5', '.hdf5']:
            with h5py.File(file_path, 'w') as file:
                file.create_dataset('dataset', data=data)

        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported formats are 'npy', 'pickle', 'json', 'csv', 'excel', 'h5py'.")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def save_np_data(data, path, name):
    """
    Save NumPy array data to a file based on the provided file name and extension.

    :param data: NumPy array data to be saved.
    :param path: Directory path where the file should be saved.
    :param name: Name of the file including the extension (e.g., 'data.csv', 'data.json').
    """
    try:
        # Determine the full file path
        file_path = os.path.join(path, name)
        # Extract the file extension from the name
        file_extension = os.path.splitext(name)[1].lower()

        if file_extension == '.npy':
            _np.save(file_path, data)

        elif file_extension in ['.pkl', '.pickle']:
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)

        elif file_extension == '.json':
            # Convert NumPy array to list for JSON serialization
            data_list = data.tolist()
            with open(file_path, 'w') as file:
                json.dump(data_list, file, indent=4)

        elif file_extension == '.csv':
            if isinstance(data, _np.ndarray):
                _np.savetxt(file_path, data, delimiter=',', fmt='%.8f')
            else:
                raise ValueError("For CSV format, data should be a NumPy array.")

        elif file_extension in ['.xlsx', '.xls']:
            df = _pd.DataFrame(data)
            df.to_excel(file_path, index=False)

        elif file_extension in ['.h5', '.hdf5']:
            with h5py.File(file_path, 'w') as file:
                file.create_dataset('dataset', data=data)

        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported formats are 'npy', 'pickle', 'json', 'csv', 'excel', 'h5py'.")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def rearrange_array(data):
    """
    Rearrange a NumPy array such that velocity and water depth data are interleaved by columns.

    :param data: A NumPy array of shape (2n, m) where n is the number of velocity/water depth data points.
    :return: A NumPy array with interleaved columns of velocity and water depth data.
    """
    # Determine the number of rows and columns
    num_rows, num_columns = data.shape

    # Ensure the number of rows is even
    if num_rows % 2 != 0:
        raise ValueError("The number of rows should be even, corresponding to velocity and water depth data.")

    # Split the data into velocity and water depth parts
    velocity_data = data[:num_rows // 2, :]
    water_depth_data = data[num_rows // 2:, :]

    # Initialize an empty array to store the rearranged data
    rearranged_data = _np.empty((num_rows // 2, num_columns * 2))

    # Interleave velocity and water depth data
    for i in range(num_columns):
        rearranged_data[:, 2 * i] = velocity_data[:, i]  # Velocity data
        rearranged_data[:, 2 * i + 1] = water_depth_data[:, i]  # Water depth data

    return rearranged_data




#----------------------------------------------
def update_json_file(json_path, modeled_values_dict, detailed_dict = False):
    """
    Updates the JSON file at `json_path` with data from `modeled_values_dict`.

    If the file exists, it appends new values to the existing data.
    If the file does not exist, it creates a new file with the initial data.

    Parameters
    ----------
    json_path: str
        The path to the JSON file to be updated or created.
    modeled_values_dict: dict
        A dictionary with data to be added or updated in the JSON file.
    """

    if os.path.exists(json_path):
        # File exists, so open it for writing
        with open(json_path, "r") as file:
            output_data = json.load(file)
            for key, value in modeled_values_dict.items():
                if key in output_data:
                    if detailed_dict:
                        if isinstance(output_data[key], list):
                            output_data[key].append(value)
                        else:
                            output_data[key] = [output_data[key], value]
                    else:
                        output_data[key].append(value)
                else:
                    output_data[key] = [value]
            with open(json_path, 'w') as file:
                json.dump(output_data, file, indent=4)

    else:
        # Save the updated JSON file
        with open(json_path, "w") as file:
            for key in modeled_values_dict:
                # Convert the existing list into a nested list with a single element
                modeled_values_dict[key] = [modeled_values_dict[key]]
            json.dump(modeled_values_dict, file, indent=4)

def delete_slf(folder_path):
    """
    Deletes all files with the .slf extension in the specified folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder where the .slf files will be deleted.

    Returns
    -------
    None
    """
    # Get all files with .slf extension in the specified folder
    slf_files = glob.glob(os.path.join(folder_path, '*.slf'))

    # Delete each .slf file
    for file_path in slf_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def filter_and_save_rows_by_column_avg(dataframe, arr, threshold, output_path):
    """
    Filters the rows of a DataFrame based on the indices of columns from a NumPy array
    where the column average exceeds the specified threshold and saves the filtered rows to a CSV file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing location data, with x and y coordinates in the 2nd and 3rd columns.
    arr : numpy.ndarray
        Input 2D array with shape (N, loc), where N is the number of runs and loc is the number of locations.
    threshold : float
        The threshold value. Columns whose average exceeds this value will be used to filter out rows.
    output_path : str
        The file path where the filtered DataFrame will be saved as a CSV file.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only the rows that correspond to the columns with high averages.
    """
    # Compute the column-wise average from the NumPy array
    column_averages = _np.mean(arr, axis=0)

    # Get the indices of columns whose average exceeds the threshold
    high_avg_column_indices = _np.where(column_averages > threshold)[0]

    # Use these indices to filter out the corresponding rows in the DataFrame
    filtered_dataframe = dataframe.iloc[high_avg_column_indices]

    # Save the filtered DataFrame as a CSV file
    filtered_dataframe.to_csv(output_path, index=False, header=True)

    return filtered_dataframe