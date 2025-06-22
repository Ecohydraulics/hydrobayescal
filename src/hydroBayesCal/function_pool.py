"""Function pool for usage at different package levels"""
import subprocess, os, sys, logging
import numpy as _np
import pandas as _pd
import csv
import pickle
import h5py
import json
import glob
import matplotlib.tri as mtri
from numpy import linspace, dtype


import pdb
import shutil

sys.path.insert(0, os.path.abspath('..'))

from src.hydroBayesCal.utils.config_logging import *
# TODO: re-instate config_physics - Done
from src.hydroBayesCal.utils.config_physics import *
from src.hydroBayesCal.telemac.pputils.ppmodules.selafin_io_pp import *


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
        new_collocation_point,
        mode="update"
):
    """
    Append a new row to a CSV file or create a new file depending on the mode.

    :param file_path: Path to the CSV file.
    :param new_collocation_point: List of values to be added as a new row.
    :param mode: Mode to determine whether to 'update' (append) or 'generate' (overwrite) the file.
    """
    # Ensure new_collocation_point is a list
    if not isinstance(new_collocation_point, list):
        raise ValueError("new_collocation_point must be a list")

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in the appropriate mode: append for 'update', write for 'generate'
    if mode == "generate" or not file_exists:
        # Open file for writing (overwrite if exists)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_collocation_point)  # Assuming the first row is the header
    elif mode == "update":
        # Open the file in append mode to add the new row
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_collocation_point)
    else:
        raise ValueError("Invalid mode. Use 'update' or 'generate'.")


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


def rearrange_array(data, num_quantities):
    """
    Rearrange a NumPy array such that data from multiple quantities is interleaved by columns.

    :param data: A NumPy array of shape (num_quantities * n, m) where n is the number of data points per quantity.
    :param num_quantities: An integer indicating the number of quantities (e.g., velocity, water depth, etc.).
    :return: A NumPy array with interleaved columns for all quantities.
    """
    # Determine the number of rows and columns
    num_rows, num_columns = data.shape

    # Ensure the number of rows is divisible by num_quantities
    if num_rows % num_quantities != 0:
        raise ValueError(
            f"The number of rows ({num_rows}) should be divisible by the number of quantities ({num_quantities})."
        )

    # Calculate the number of data points per quantity
    n = num_rows // num_quantities

    # Split the data into parts for each quantity
    quantity_data = [data[i * n:(i + 1) * n, :] for i in range(num_quantities)]

    # Initialize an empty array to store the rearranged data
    rearranged_data = _np.empty((n, num_columns * num_quantities))

    # Interleave the data columns for all quantities
    for i in range(num_columns):
        for j in range(num_quantities):
            rearranged_data[:, i * num_quantities + j] = quantity_data[j][:, i]
    return rearranged_data



#----------------------------------------------
def update_json_file(json_path, modeled_values_dict=None, detailed_dict=False, save_dict=False, saving_path=None):
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
    detailed_dict: bool, optional
        Whether to handle the data as nested lists for detailed structures.
    save_dict: bool, optional
        If True, saves the entire `output_data` to the `saving_path`.
    saving_path: str, optional
        The path to save the final JSON file when `save_dict` is True.
        If not provided, defaults to `json_path`.
    """
    if save_dict is False:
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
    else:
        if os.path.exists(json_path):
            # File exists, so open it for reading
            with open(json_path, "r") as file:
                output_data = json.load(file)

            # Save the data to the specified saving path
            if saving_path:
                with open(saving_path, "w") as file:
                    json.dump(output_data, file, indent=4)
        else:
            print(f"File at {json_path} does not exist. Cannot save to {saving_path}.")
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
    path = folder_path
    # Get all files with .slf extension in the specified folder
    slf_files = glob.glob(os.path.join(path, '*.slf'))
    # Delete each .slf file
    for file_path in slf_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def filter_model_outputs(data_dict, quantities, run_range_filtering=None):
    """
    Filters the data from the model outputs dictionary based on desired quantities
    and optionally limits the runs included to a specific range.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model outputs with points as keys and lists of run outputs as values.
    quantities : list of str
        List of quantities to extract from the model outputs.
    run_range : tuple of int, optional
        Range of runs to include (start, end). If None, includes all runs.
        The range is inclusive of the start index and exclusive of the end index.

    Returns
    -------
    dict
        Filtered dictionary containing only the selected quantities and runs within the specified range.
    """
    filtered_data = {}
    for point, runs in data_dict.items():
        # Extract the specified range of runs
        if run_range_filtering is not None:
            start, end = run_range_filtering
            runs = runs[start - 1:end]

        # Filter quantities from each run
        filtered_runs_data = []
        for run in runs:
            filtered_run = {quantity: run[quantity] for quantity in quantities if quantity in run}
            filtered_runs_data.append(filtered_run)

        filtered_data[point] = filtered_runs_data
    # pdb.set_trace()
    # all_values = {}
    # for point_name, data in filtered_data.items():
    #     # Extract inner list of dictionaries (removing one level of nesting)
    #     all_values[point_name] = data[0]

    return filtered_data




def interpolate_values(coords, values, point):
    """
    Interpolates values at a given point using Inverse Distance Weighting.

    Parameters:
        coords (np.ndarray): Coordinates of the triangle's vertices, shape (3, 2),
                             where each row is [X, Y] for a vertex.
        values (np.ndarray): Values at each vertex for each variable, shape (3, num_variables).
        point (tuple): Coordinates of the point where interpolation is desired, (px, py).

    Returns:
        np.ndarray: Interpolated values at the given point for each variable, shape (num_variables,).
    """
    px, py = point

    # Calculate distances from the point to each vertex
    distances = _np.linalg.norm(coords - _np.array([px, py]), axis=1)

    # Avoid division by zero for vertices that might be at the same location as the point
    distances[distances == 0] = _np.finfo(float).eps  # Small value to prevent division by zero

    # Calculate inverse distance weights
    weights = 1 / distances
    weights /= _np.sum(weights)  # Normalize weights to sum to 1

    # Interpolate each variable using weighted sum
    interpolated_values = _np.dot(weights, values)  # Dot product for weighted sum

    return interpolated_values.flatten()  # Return as a 1D array
def rasterize(saving_folder, slf_file_name, desired_variables, spacing):
    # Define the full path for the SLF file
    slf_file = os.path.join(saving_folder, slf_file_name)
    print("The input file being probed: " + slf_file)

    # constructor for pp_SELAFIN class
    slf = ppSELAFIN(slf_file)
    slf.readHeader()
    slf.readTimes()
    times = slf.getTimes()
    vnames = slf.getVarNames()
    vunits = slf.getVarUnits()
    ftype, fsize = slf.getPrecision()
    nplan = slf.getNPLAN()
    NELEM = slf.getNELEM()
    NPOIN = slf.getNPOIN()
    IPOBO = slf.getIPOBO()
    DATE = slf.getDATE()

    if nplan > 1:
        slf_type = '3d'
    else:
        slf_type = '2d'

    if ftype == 'f' and fsize == 4:
        precision = 'single'
    elif ftype == 'd' and fsize == 8:
        precision = 'double'
    else:
        precision = 'unknown'

    # prints variable names
    print('Precision: ' + precision)
    print('File type: ' + slf_type)
    if slf_type == '3d':
        print('Number of planes: ' + str(nplan))
    print('Number of elements: ' + str(NELEM))
    print('Number of nodes: ' + str(NPOIN))
    print('Date: ' + str(DATE[0]) + '-' + str(DATE[1]).zfill(2) + '-'
          + str(DATE[2]).zfill(2) + ' ' + str(DATE[3]).zfill(2) + ':'
          + str(DATE[4]).zfill(2) + ':' + str(DATE[5]).zfill(2))
    print(' ')
    print('#################################')
    print('Variables in ' + slf_file + ' are: ')
    print('---------------------------------')
    print('     v     variable        unit')
    print('---------------------------------')

    desired_var_indices = []  # Initialize to a default value, e.g., -1, to indicate that no variable has been found yet.

    for var in desired_variables:
        for i in range(len(vnames)):
            print(f"    {i} --> {vnames[i]} [{vunits[i].strip()}]")

            # Strip spaces from vnames[i] and check if it matches the current desired variable
            if vnames[i].strip() == var:  # Checks if the stripped vnames[i] matches the current desired_var
                desired_var_indices.append(i)  # Add the index to the list if there's a match

    # Print out the desired variable indices in the order of desired_var
    print("Desired variable indices in order:", desired_var_indices)
    print(' ')
    print('#################################')

    # prints times
    records = np.arange(len(times))
    nrecs = len(times)

    if len(times) < 2:
        print("    t    time (s)")
        print('---------------------------------')
        print(str(records[0]) + " -->" + str("{:10.1f}".format(times[0])))
    elif len(times) < 3:
        print("    t    time (s)")
        print('---------------------------------')
        print(str(records[0]) + " -->" + str("{:10.1f}".format(times[0])))
        print(str(records[1]) + " -->" + str("{:10.1f}".format(times[1])))
    elif len(times) < 4:
        print("    t    time (s)")
        print('---------------------------------')
        print(str(records[0]) + " -->" + str("{:10.1f}".format(times[0])))
        print(str(records[1]) + " -->" + str("{:10.1f}".format(times[1])))
        print(str(records[2]) + " -->" + str("{:10.1f}".format(times[2])))
    else:
        print("t        time (s)")
        print('---------------------------------')
        print(str(records[0]) + " -->" + str("{:10.1f}".format(times[0])))
        print(str(records[1]) + " -->" + str("{:10.1f}".format(times[1])))
        print(str(records[2]) + " -->" + str("{:10.1f}".format(times[2])))
        print(str(records[3]) + " -->" + str("{:10.1f}".format(times[3])))
        print('     ......')
        print(str(records[nrecs - 1]) + "-->" + str("{:10.1f}".format(times[nrecs - 1])))

    print('#################################')
    last_time_step_index = nrecs - 1

    # index number of grided output variable
    t = last_time_step_index  # index of time record of the output to use in griding (integer, 0 to n)

    # Read the header of the selafin result file and get geometry and
    # variable names and units
    slf.readVariables(t)

    # gets some of the mesh properties from the *.slf file
    NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = slf.getMesh()

    # the IKLE array starts at element 1, but matplotlib needs it to start
    # at zero
    IKLE[:,:] = IKLE[:,:] - 1

    # these are the results for all variables, for time step t
    master_results = slf.getVarValues()

    # creates a triangulation grid using matplotlib function Triangulation
    triang = mtri.Triangulation(x, y, IKLE)

    # determine the spacing of the regular grid
    range_in_x = x.max() - x.min()
    range_in_y = y.max() - y.min()

    max_range = max(range_in_x, range_in_y)

    # first index is integer divider, second is remainder
    num_x_pts = divmod(range_in_x, spacing)
    num_y_pts = divmod(range_in_y, spacing)

    print("Size of output matrix is : " + str(int(num_x_pts[0])) + " x " + str(int(num_y_pts[0])))
    print("Grid resolution is : " + str(spacing) + " m")

    # creates the regular grid
    xreg, yreg = np.meshgrid(np.linspace(x.min(), x.max(), int(num_x_pts[0])),
                             np.linspace(y.min(), y.max(), int(num_y_pts[0])))
    x_regs = xreg[1, :]
    y_regs = yreg[:, 1]

    raster_data = {}
    # to interpolate to a reg grid
    for var_index in desired_var_indices:
        interpolator = mtri.LinearTriInterpolator(triang, master_results[var_index])
        z = interpolator(xreg, yreg)

        print("Shape of array z: " + str(z.shape[0]))
        print("Shape of arrays xreg and yreg: " + str(x_regs.shape) + " " + str(y_regs.shape))

        where_are_NaNs = np.isnan(z)
        z[where_are_NaNs] = -999.0

        # open the output *.asc file, and write the header info
        header_str = "NCOLS " + str(z.shape[1]) + "\n"
        header_str = header_str + "NROWS " + str(z.shape[0]) + "\n"
        header_str = header_str + "XLLCORNER " + str(x_regs[0]) + "\n"
        header_str = header_str + "YLLCORNER " + str(y_regs[0]) + "\n"
        header_str = header_str + "CELLSIZE " + str(spacing) + "\n"
        header_str = header_str + "NODATA_VALUE " + str(-999.00) + "\n"

        # Save the output raster to a file
        output_file = os.path.join(saving_folder, f"interpolated_{vnames[var_index]}.asc")
        np.savetxt(output_file, np.flipud(z), fmt='%10.3f', header=header_str, comments='', delimiter='')

        raster_data[vnames[var_index].strip()] = {
            "data": np.flipud(z),
            "x_regs": x_regs,
            "y_regs": y_regs,
            "spacing": spacing
        }
    return raster_data


def classify_mu(raster_data, classification, output_folder, output_filename):
    """
    Classify the morphological units (MU) based on velocity and depth and save as a raster file.

    Parameters:
    raster_data (dict): Dictionary containing 'velocity' and 'depth' raster data as numpy arrays.
    classification (dict): Dictionary of classification criteria for different MUs.
    output_folder (str): Folder path where the output file will be saved.
    output_filename (str): The filename for the output raster file (without extension).

    Returns:
    None: The function will save the classified MU raster as an ASCII file in the output folder.
    """
    # Extract velocity and depth data from the raster_data dictionary
    velocity_data = raster_data['SCALAR VELOCITY']['data']  # Access 'data' if the raster data is a masked array
    depth_data = raster_data['WATER DEPTH']['data']

    # Ensure that both velocity and depth are numpy arrays, not dictionaries
    assert isinstance(velocity_data, np.ndarray), f"Expected numpy array for velocity_data, but got {type(velocity_data)}"
    assert isinstance(depth_data, np.ndarray), f"Expected numpy array for depth_data, but got {type(depth_data)}"

    # Create an empty array for the output MU classification
    mu_raster = np.zeros_like(velocity_data, dtype=int)  # Initialize with zeros (no classification)

    # Iterate through each classification and assign values to the raster
    for mu, criteria in classification.items():
        velocity_range = criteria['velocity_range']
        depth_range = criteria['depth_range']
        velocity_masked = velocity_data.mask
        depth_masked = depth_data.mask
        # Classify cells where velocity and depth are within the specified ranges
        mu_mask = (velocity_data >= velocity_range[0]) & (velocity_data <= velocity_range[1]) & \
                  (depth_data >= depth_range[0]) & (depth_data <= depth_range[1])
        mu_mask &= ~velocity_masked & ~depth_masked
        # Assign the classification index (or any desired value) for the morphological unit
        mu_raster[mu_mask] = mu  # Use the classification index directly from the dictionary (e.g., 0, 1, 2...)

    # Save the classified MU raster as an ASCII file
    header_str = f"NCOLS {mu_raster.shape[1]}\n"
    header_str += f"NROWS {mu_raster.shape[0]}\n"
    header_str += f"XLLCORNER {raster_data['SCALAR VELOCITY']['x_regs'].min()}\n"  # Assuming min x-coordinate for the corner
    header_str += f"YLLCORNER {raster_data['WATER DEPTH']['y_regs'].min()}\n"  # Assuming min y-coordinate for the corner
    header_str += f"CELLSIZE {raster_data['WATER DEPTH']['spacing']}\n"  # You can adjust the cell size depending on your data
    header_str += "NODATA_VALUE -999\n"

    # Construct output file path
    output_file = os.path.join(output_folder, f"{output_filename}.asc")

    # Save the classified MU raster as an ASCII file
    np.savetxt(output_file, mu_raster, fmt='%d', header=header_str, comments='', delimiter=' ')

    print(f"Classified MU raster saved to {output_file}")

def parse_classes_keyword(file_path, keyword):
    with open(file_path, 'r') as file:
        for line in file:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue

            # Check for ":" or "="
            for symbol in [":", "="]:
                if symbol in stripped:
                    key_part = stripped.split(symbol, 1)[0].strip()
                    if keyword in key_part:
                        return stripped  #

    return None  # If not found
def update_gaia_class_line(line, index, new_value):
    # Split at ":" or "="
    if ':' in line:
        key, values = line.split(':', 1)
        separator = ':'
    elif '=' in line:
        key, values = line.split('=', 1)
        separator = '='
    else:
        raise ValueError("Line must contain ':' or '='")

    # Split values and remove extra spaces
    values_list = [v.strip() for v in values.strip().split(';')]

    # Check if index is valid
    if index < 0 or index >= len(values_list):
        raise IndexError("Index out of range")

    # Update the value at the specified index
    values_list[index] = f"{new_value:.3f}"  # Format with 3 decimals if you want

    # Reconstruct the line
    updated_line = f"{key.strip()} {separator} {';'.join(values_list)}"
    return updated_line

def classify_parameters_tm_gaia(elements, classification_dict):
    telemac_vars = []
    gaia_vars = []

    for element in elements:
        source = classification_dict.get(element)
        if source == "telemac":
            telemac_vars.append(element)
        elif source == "gaia":
            gaia_vars.append(element)
        else:
            raise ValueError(f"Element '{element}' not recognized in CLASSIFICATION_DICT.")

    return telemac_vars, gaia_vars