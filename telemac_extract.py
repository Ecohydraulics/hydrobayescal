"""
Script to extract model outputs from a TELEMAC result file (.slf) based on
specified EXTRACTION QUANTITIES. The extraction is performed at locations
defined in a calibration points file, which must be provided.
Author: Andrés Heredia Hidalgo
"""

import sys
import os

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# Import own scripts
from src.hydroBayesCal.telemac.control_telemac import TelemacModel
#from src.hydroBayesCal.function_pool import *

# Define paths and other parameters
model_dir = '/home/IWS/hidalgo/Documents/cylinderModel/convergence/0.01/' # the .slf file must be in this directory.
res_dir = '/home/IWS/hidalgo/Documents/cylinderModel/convergence/0.01/' # a dictionary with the model outputs will be saved in this directory
calibration_pts_file_path = '/home/IWS/hidalgo/Documents/cylinderModel/convergence/measurements-calibration.csv'
output_name='output_file_3d'
calibration_quantities= ['VELOCITY U'] #['TURBULENT ENERG','DISSIPATION']
extraction_quantities= ['VELOCITY U'] #['TURBULENT ENERG','DISSIPATION']
input_slf_file='3d-conv-0.01-3d.slf' # the .slf file must be in model_dir
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
                              model_directory=model_dir, results_folder_directory=control_tm.calibration_folder)
control_tm.output_processing(output_data_path=os.path.join(control_tm.calibration_folder,f'{output_name}-detailed.json'),calibration_quantities=control_tm.calibration_quantities,delete_slf_files=False)
