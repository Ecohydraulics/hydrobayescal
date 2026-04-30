"""
Load and analyze multiple Vectrino ASCII files stored in a directory.

The folder must contain matching .ntk.dat and .ntk.hdr files.

Workflow:
    1. Find all Vectrino ASCII datasets in data_directory
    2. Read each .ntk.dat file
    3. Read the transformation matrix from .ntk.hdr
    4. Apply beam-to-XYZ transformation if needed
       OR pass through already-XYZ data
    5. Save processed data as CSV
"""

import os
from get_ascii_data import read_ascii_file
from transformation import get_transformation_matrix, apply_transformation
from plot_velocities import plot_instantaneous_velocities
from compute_tke import compute_tke_data


# START USER INPUT -----------------------------------------------------------------------------------------------------

data_directory = "test/"

# END USER INPUT -------------------------------------------------------------------------------------------------------


# get all unique dataset names in the directory
unique_names = []

for filename in os.listdir(data_directory):
    if filename.endswith(".ntk.dat") or filename.endswith(".ntk.hdr"):
        base_name = filename.split(".ntk")[0]

        if base_name not in unique_names:
            unique_names.append(base_name)


# save the current working directory and temporarily switch to the data directory
original_directory = os.getcwd()
os.chdir(data_directory)


for ascii_file in unique_names:
    print("* processing " + str(ascii_file) + " ...")

    dat_file = ascii_file + ".ntk.dat"
    hdr_file = ascii_file + ".ntk.hdr"

    if not os.path.exists(dat_file):
        print("   - WARNING: missing " + dat_file + ". Skipping.")
        continue

    if not os.path.exists(hdr_file):
        print("   - WARNING: missing " + hdr_file + ". Skipping.")
        continue

    vectrino_data = read_ascii_file(ascii_file)

    M = get_transformation_matrix(ascii_file, scaling_factor=4096)

    vectrino_data = apply_transformation(
        vectrino_data,
        transformation_matrix=M,
        relevant_point_ids=(0,)
    )
    output_file = ascii_file + ".csv"
    vectrino_data.to_csv(output_file, index=False)

    print("   - saved " + output_file)

    plot_instantaneous_velocities(
        vectrino_data,
        output_name=ascii_file,
        output_directory="plots",
        averaging_window=0.05,
        show_plot=False
    )
    tke_data = compute_tke_data(
        vectrino_data,
        output_name=ascii_file,
        output_directory="tke",
        averaging_window=0.5,
        save_csv=True
    )
    print("   - saved " + output_file)


# go back to the original working directory
os.chdir(original_directory)
