"""
Script to extract model outputs from a TELEMAC result file (.slf) based on
specified EXTRACTION QUANTITIES. The extraction is performed at locations
defined in a calibration points file, which must be provided.
Author: Andrés Heredia Hidalgo
"""

import sys
import os


# Import own scripts
from hydroBayesCal.telemac.control_telemac import TelemacModel
#from hydroBayesCal.function_pool import *

# Define paths and other parameters
model_dir = '/home/IWS/hidalgo/Documents/HIC2026-MO-GPE/HICHydrodynamicsEring50it/auto-saved-results-HydroBayesCal/restart_data/' # the .slf file must be in this directory.
res_dir = '/home/IWS/hidalgo/Documents/HIC2026-MO-GPE/HICHydrodynamicsEring50it/auto-saved-results-HydroBayesCal/restart_data/' # a dictionary with the model outputs will be saved in this directory
calibration_pts_file_path = '/home/IWS/hidalgo/Documents/EringMO-GPECalibration/MU2026-AllRange/measurements-calibration-EringCalib-test.csv'
output_name='results_restart_1'
calibration_quantities= ['WATER DEPTH','SCALAR VELOCITY'] #['TURBULENT ENERG','DISSIPATION']
extraction_quantities= ['WATER DEPTH','SCALAR VELOCITY'] #['TURBULENT ENERG','DISSIPATION']
input_slf_file='results_restart_1.slf' # the .slf file must be in model_dir
# Initialize TelemacModel object
control_tm = TelemacModel(
    model_dir=model_dir,
    res_dir=res_dir,
    control_file='',
    friction_file='',
    calibration_parameters='',
    calibration_pts_file_path=calibration_pts_file_path,
    calibration_quantities=calibration_quantities,
    extraction_quantities=extraction_quantities,
    tm_xd='',
    n_processors=1,
    dict_output_name=output_name,
    results_file_name_base='',
    init_runs=1,
)

# Correctly define output_name and extraction_quantity
calibration_pts_df=control_tm.calibration_pts_df
output_name=control_tm.dict_output_name
calibration_quantities=control_tm.calibration_quantities
# Call extract_data_point method
control_tm.extract_data_point(input_file=input_slf_file, calibration_pts_df=calibration_pts_df,
                              output_name=output_name, extraction_quantity=extraction_quantities, simulation_number=1,
                              model_directory=model_dir, results_folder_directory=control_tm.calibration_folder,output_extraction_time="mean_last", time_index=100, n=80,compute_wall_law_diagnostics=False)
control_tm.output_processing(output_data_path=os.path.join(control_tm.calibration_folder,f'{output_name}-detailed.json'),calibration_quantities=control_tm.calibration_quantities,delete_slf_files=False)
