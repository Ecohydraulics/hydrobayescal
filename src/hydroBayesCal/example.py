from telemac.control_telemac import TelemacModel

# Define paths and other parameters
model_dir = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/'
res_dir = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/'
calibration_pts_file_path = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/calibration-pts_example.csv'
output_name='output_file_name_example'
calibration_quantities=['MVELOC']
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

# Call extract_data_point method
control_tm.extract_data_point(input_slf_file='rmesh-friction.slf',
                              calibration_pts_file_path=calibration_pts_file_path,
                              output_name=output_name,
                              extraction_quantity=calibration_quantities,simulation_number=1)