"""

"""
from control_telemac import TelemacModel

# Define paths and other parameters
model_dir = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/' # the .slf file must be in this directory.
res_dir = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/' # a dictionary with the model outputs will be saved in this directory
calibration_pts_file_path = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/calibration-pts_H.csv'
output_name='output_file_name_example'
calibration_quantities=['VELOCITY U']
# Initialize TelemacModel object
control_tm = TelemacModel(
    model_dir=model_dir,
    res_dir=res_dir,
    control_file='',
    friction_file='',
    calibration_parameters='',
    calibration_pts_file_path=calibration_pts_file_path,
    calibration_quantities=calibration_quantities,
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
control_tm.extract_data_point(input_file='R_donau_1.slf', calibration_pts_df=calibration_pts_df,
                              output_name=output_name, extraction_quantity=calibration_quantities, simulation_number=1,
                              model_directory=model_dir, results_folder_directory=res_dir)