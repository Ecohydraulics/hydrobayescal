"""This is a use case example for the automated stochastic calibration
with sample data from the https://hydro-informatics.com database."""

import os as _os
import os.path
import subprocess
import shutil
import numpy as _np
import pandas as _pd
from datetime import datetime
from stochastic_calibration.pputils.ppmodules.selafin_io_pp import ppSELAFIN
from stochastic_calibration.pputils import shp2csv


# load calibration package
import stochastic_calibration as sc


def concatenate_csv_pts(file_directory, *args):
    """Concatenate a csv-files with lists of XYZ points into one CSV file that is saved to the same directory where the
    first CSV file name provided lives. The merged CSV file name starts with merged_ and also ends with the name
    of the first CSV file name provided.

    :param file_directory: string of the directory where the CSV files live, and which must end on '/' or '\\'
    :param args: string or list of csv files (only names) containing comma-seperated XYZ coordinates without header
    :return pandas.DataFrame: merged points
    """
    point_file_names = []
    # receive arguments (i.e. csv point file names)
    for arg in args:
        if type(arg) is str:
            point_file_names.append(file_directory + "/" + arg)
        if type(arg) is list:
            [point_file_names.append(file_directory + "/" + e) for e in arg]

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
        file_directory + "merged-" + str(point_file_names[0]).split("/")[-1].split("\\")[-1],
        header=False,
        index=False
    )

    return merged_pts


# MAIN --------------------------
file_dir = "/home/public/test-data/example/epsg5678/"
point_files = [
    "boundary-pts.csv",
    "internal-line-pts.csv",
    "embedded-nodes.csv",
]
# concatenate_csv_pts(file_dir, point_files)
calibration_variable = "WATER DEPTH"
save_name = "/home/public/test-data/node-output.txt"

boundary_shp = "boundary.shp"
breaklines_shp = "breaklines.shp"

shp2csv.shp2csv(file_dir + boundary_shp, "boundary.csv")
shp2csv.shp2csv(file_dir + boundary_shp, "breaklines.csv")
