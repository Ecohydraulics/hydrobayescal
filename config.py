
# Definition of output_excel_file_name saves in auto-saved-results folder
output_excel_file_name="simulation_outputs.xlsx"
#-----------------------------------------------------------------------------------------------------
# Definition of the results file name taken from .cas file (example: RESULTS FILE:r2d-donau.slf)
results_filename_base = "r2d_weirs"
#-----------------------------------------------------------------------------------------------------
# Definition of the lines in the .cas file (calibration parameters) which have to be modified based on the selection made in the user_input_parameter.xlsx file.
# Example: Taken from .cas file
# INITIAL ELEVATION:309.4D0
# ROUGHNESS COEFFICIENT OF BOUNDARIES:0.01
# TIME STEP                                   : 0.5
# Important:
# **** Maximum number of parameters to be changed in the .cas file: 4
# **** Names of the parameters used in this section MUST coincide with the selected names in user_input_parameter.xlsx file

cas_line_1= 'INITIAL ELEVATION                            = 1.35'
cas_line_2='VELOCITY DIFFUSIVITY                         = 1.'
cas_line_3=''
cas_line_4=''
#-----------------------------------------------------------------------------------------------------

# Definition of the line in the .cas file which gives the name of the result file (Example: RESULTS FILE:r2d-donau.slf)
cas_line_results_file="RESULTS FILE                    = r2d_weirs.slf"
#-----------------------------------------------------------------------------------------------------
