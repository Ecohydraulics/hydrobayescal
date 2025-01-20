"""
Code that runs full complexity model (i.e., hydrodynamic models) of Telemac one time
with a desired set of parameters and extracts a .xyz file for water depth and scalar velocity (or any two variables).

Author: Andres Heredia Hidalgo
"""
import sys,os
import numpy as np
# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# Add the paths to sys.path
sys.path.insert(0, src_path)  # Prepend to prioritize over other paths
sys.path.insert(0, hydroBayesCal_path)

from src.hydroBayesCal.telemac.control_telemac import TelemacModel

full_complexity_model = TelemacModel(
    control_file="tel_ering_mu_initial.cas",
    model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/telemac_files/",
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/telemac_files",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/telemac_files/points_wet_area_MU.csv",
    n_cpus=4,
    init_runs=1,
    calibration_parameters=[
                            "zone3",
                            "zone4",
                            "zone5",
                            "zone7",
                            "zone11",
                            "zone12",
                            "zone13",
                            "zone14",
                            "zone15"],
    calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
    dict_output_name="wd-v-channel-points",
    # TelemacModel class parameters
    friction_file="friction_ering_MU.tbl",
    tm_xd="1",
    results_filename_base="initial_results",
    complete_bal_mode=False,
    only_bal_mode=False,
    check_inputs=False,
    delete_complex_outputs=True,
)
simulation_set = [[3,3,3,3,3,3,3,3,3]]
full_complexity_model.run_multiple_simulations(collocation_points=simulation_set,
                                               complete_bal_mode=False,)
model_outputs = full_complexity_model.model_evaluations
X_coord=full_complexity_model.calibration_pts_df["X"].values
Y_coord=full_complexity_model.calibration_pts_df["Y"].values
num_simulations, num_columns = model_outputs.shape
# Separate the columns for each quantity
# Take every second column starting from 0
water_depth = model_outputs[:, 0:num_columns:2]

# Take every second column starting from 1
velocity = model_outputs[:,1:num_columns:2]

water_depth = water_depth.reshape(full_complexity_model.nloc)  # or water_depth.flatten()
velocity = velocity.reshape(full_complexity_model.nloc)  # or velocity.flatten()
x_y_var=np.column_stack((X_coord, Y_coord, water_depth,velocity))
header = "X,Y,WATER DEPTH,VELOCITY"
np.savetxt(full_complexity_model.asr_dir + '/' + 'output.xyz', x_y_var, fmt='%.6f', delimiter=',', header=header, comments='')