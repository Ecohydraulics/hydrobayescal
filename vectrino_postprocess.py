"""
Main script for post-processing Nortek Vectrino Profiler ASCII files.
The script reads .ntk.dat and .ntk.hdr files detailed in variables case_names, gets velocities in XYZ coordinates, and saves the processed data as CSV files.
It also generates plots of instantaneous velocities and computes binned TKE tables.

Author: Andres Heredia

"""

import sys
import os
import time
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, "src")
hydroBayesCal_path = os.path.join(src_path, "hydroBayesCal")

sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# ======================================================================
# LOCAL VECTRINO POST-PROCESSING IMPORTS
# ======================================================================

from src.hydroBayesCal.utils.VectrinoPostproc.get_ascii_data import read_ascii_file
from src.hydroBayesCal.utils.VectrinoPostproc.transformation import get_transformation_matrix, apply_transformation
from src.hydroBayesCal.utils.VectrinoPostproc.plot_velocities import plot_instantaneous_velocities
from src.hydroBayesCal.utils.VectrinoPostproc.compute_tke import compute_tke_data

# ======================================================================
# USER INPUT
# ======================================================================

data_directory = "/media/andres/Windows/Users/andre/Documents/PhD/2026/cylinder_model/raw-data-vectrinoApril2026/ADV_pier_Andres/raw_ascii"
results_directory = "/media/andres/Windows/Users/andre/Documents/PhD/2026/cylinder_model/raw-data-vectrinoApril2026/ValidationQ25ls_50ls_postprocessed"
# Base names WITHOUT ".ntk.dat" or ".ntk.hdr"
case_names = [
    #Calibration points at different locations (Q=50 l/s)
    "xyz_-05_2cm_CL",
    "xyz_-05_4cm_CL",
    "xyz_-05_6cm_CL",
    "xyz_-05_8cm_CL",
    "xyz_-05_10cm_CL",
    "xyz_+05_2cm_CL",
    "xyz_+05_4cm_CL",
    "xyz_+05_6cm_CL",
    "xyz_+05_8cm_CL",
    "xyz_+05_10cm_CL",
    "xyz_+05_2cm_offCL20",
    "xyz_+05_4cm_offCL20",
    "xyz_+05_6cm_offCL20",
    "xyz_+05_8cm_offCL20",
    "xyz_+05_10cm_offCL20",
    "xyz_+1_2cm_CL",
    "xyz_+1_4cm_CL",
    "xyz_+1_6cm_CL",
    "xyz_+1_8cm_CL",
    "xyz_+1_10cm_CL",
    "xyz_+1_2cm_offCL20",
    "xyz_+1_4cm_offCL20",
    "xyz_+1_6cm_offCL20",
    "xyz_+1_8cm_offCL20",
    "xyz_+1_10cm_offCL20",
    "xyz_+2_2cm_CL",
    "xyz_+2_4cm_CL",
    "xyz_+2_6cm_CL",
    "xyz_+2_8cm_CL",
    "xyz_+2_10cm_CL",
    "xyz_+2_2cm_offCL20",
    "xyz_+2_4cm_offCL20",
    "xyz_+2_6cm_offCL20",
    "xyz_+2_8cm_offCL20",
    "xyz_+2_10cm_offCL20",
    #Validation points at different points (Q=50 l/s)
    # "xyz_+4m_2cm_offCL20left",
    # "xyz_+4m_4cm_offCL20left",
    # "xyz_+4m_6cm_offCL20left",
    # "xyz_+4m_8cm_offCL20left",
    # "xyz_+4m_10cm_offCl20left",
    # "xyz_+55m_2cm_CL",
    # "xyz_+55m_4cm_CL",
    # "xyz_+55m_6cm_CL",
    # "xyz_+55m_8cm_CL",
    # "xyz_+55m_10cm_CL",
    # "xyz_+55m_2cm_offCL35left",
    # "xyz_+55m_4cm_offCL35left",
    # "xyz_+55m_6cm_offCL35left",
    # "xyz_+55m_8cm_offCL35left",
    # "xyz_+55m_10cm_offCL35left", 
    #Validation points with different discharge (Q=25 l/s,60 l/s)
    # "xyz_extraQ60_+25_2cm_CL",
    # "xyz_extraQ60_+25_4cm_CL",
    # "xyz_extraQ60_+25_6cm_CL",
    # "xyz_extraQ60_+25_8cm_CL",
    # "xyz_extraQ60_+25_10cm_CL", 
    # "xyz_extraQ25_+25_2cm_CL",
    # "xyz_extraQ25_+25_4cm_CL",
    # "xyz_extraQ25_+25_6cm_CL",
    # "xyz_extraQ25_+25_8cm_CL",
    # "xyz_extraQ25_+25_10cm_CL",  
]

relevant_point_ids = (0,)

velocity_plot_window = 0.05
tke_averaging_window = 0.5

make_velocity_plots = True
compute_tke = True


# ======================================================================
# MAIN PROCESSING
# ======================================================================

start_time = time.time()
os.makedirs(results_directory, exist_ok=True)
if not os.path.isdir(data_directory):
    raise FileNotFoundError(
        f"ERROR: data_directory does not exist: {data_directory}"
    )

if len(case_names) == 0:
    raise ValueError("ERROR: case_names is empty.")

summary_rows = []

original_directory = os.getcwd()
os.chdir(data_directory)

try:
    for case_name in case_names:

        file_base_names = [case_name]

        case_output_directory = os.path.join(results_directory, case_name)

        csv_output_directory = os.path.join(case_output_directory, "processed_csv")
        plot_output_directory = os.path.join(case_output_directory, "plots")
        tke_output_directory = os.path.join(case_output_directory, "tke")

        os.makedirs(csv_output_directory, exist_ok=True)
        os.makedirs(plot_output_directory, exist_ok=True)
        os.makedirs(tke_output_directory, exist_ok=True)

        print("============================================================")
        print(" Vectrino Profiler ASCII post-processing")
        print("============================================================")
        print(f" Data directory       : {data_directory}")
        print(f" Case name            : {case_name}")
        print(f" File base names      : {file_base_names}")
        print(f" Point IDs used       : {relevant_point_ids}")
        print(f" Plot averaging window: {velocity_plot_window} s")
        print(f" TKE averaging window : {tke_averaging_window} s")
        print(f" Case output directory: {case_output_directory}")
        print(f" CSV output directory : {csv_output_directory}")
        print(f" Plot output directory: {plot_output_directory}")
        print(f" TKE output directory : {tke_output_directory}")
        print("============================================================")

        for ascii_file in file_base_names:
            print(f"\n* processing {ascii_file} ...")

            dat_file = ascii_file + ".ntk.dat"
            hdr_file = ascii_file + ".ntk.hdr"

            if not os.path.exists(dat_file):
                print(f"   - WARNING: missing {dat_file}. Skipping.")
                continue

            if not os.path.exists(hdr_file):
                print(f"   - WARNING: missing {hdr_file}. Skipping.")
                continue

            vectrino_data = read_ascii_file(ascii_file)

            M = get_transformation_matrix(ascii_file, scaling_factor=4096)

            vectrino_data = apply_transformation(
                vectrino_data,
                transformation_matrix=M,
                relevant_point_ids=relevant_point_ids
            )

            output_file = os.path.join(
                csv_output_directory,
                ascii_file + ".csv"
            )

            vectrino_data.to_csv(output_file, index=False)
            print(f"   - saved processed velocity CSV: {output_file}")

            if make_velocity_plots:
                plot_instantaneous_velocities(
                    vectrino_data,
                    output_name=ascii_file,
                    output_directory=plot_output_directory,
                    averaging_window=velocity_plot_window,
                    show_plot=False
                )

            if compute_tke:
                tke_data = compute_tke_data(
                    vectrino_data,
                    output_name=ascii_file,
                    output_directory=tke_output_directory,
                    averaging_window=tke_averaging_window,
                    save_csv=True
                )

                print(f"   - TKE rows computed: {len(tke_data)}")

                # ------------------------------------------------------
                # Extract final MEAN row from TKE table
                # ------------------------------------------------------
                mean_row = tke_data[tke_data["time_start (s)"] == "MEAN"]

                if len(mean_row) == 1:
                    mean_row = mean_row.iloc[0]

                    summary_rows.append({
                        "case_name": case_name,
                        "file_name": ascii_file,
                        "n_samples_total": mean_row["n_samples"],

                        "u_mean (m/s)": mean_row["u_mean (m/s)"],
                        "v_mean (m/s)": mean_row["v_mean (m/s)"],
                        "w_mean (m/s)": mean_row["w_mean (m/s)"],

                        "u_var_mean (m2/s2)": mean_row["u_var (m2/s2)"],
                        "v_var_mean (m2/s2)": mean_row["v_var (m2/s2)"],
                        "w_var_mean (m2/s2)": mean_row["w_var (m2/s2)"],

                        "TKE_mean (m2/s2)": mean_row["TKE (m2/s2)"],
                    })

                else:
                    print(
                        f"   - WARNING: no unique MEAN row found in TKE table for {ascii_file}"
                    )

finally:
    os.chdir(original_directory)


# ======================================================================
# SAVE SUMMARY TABLE
# ======================================================================

if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)

    summary_output_file = os.path.join(
        results_directory,
        "summary_mean_velocities_and_tke.csv"
    )

    summary_df.to_csv(summary_output_file, index=False)

    print("\n============================================================")
    print(" Summary table saved")
    print(f" File: {summary_output_file}")
    print("============================================================")
else:
    print("\nWARNING: No summary rows were created.")


elapsed_time = time.time() - start_time

print("\n============================================================")
print(" Processing finished")
print(f" Elapsed time: {elapsed_time:.2f} s")
print("============================================================")
