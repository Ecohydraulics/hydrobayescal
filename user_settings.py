"""
Global user-defined settings for Telemac full complexity simulations.

Author: Andres Heredia M.Sc.
"""
import os

HOMEDIR = os.path.abspath(os.getcwd())
# print(HOMEDIR)

# PHYSICS
GRAVITY = 9.81  # gravitational acceleration in m/s2
KINEMATIC_VISCOSITY = 10 ** -6  # kinematic viscosity in m2/s
WATER_DENSITY = 10. ** 3  # water density in m3/kg
SED_DENSITY = 2650  # sediment density in m3/kg

# 1.- Global full complexity model parameters----------------------------------
# -----------------------------------------------------------------------------

# 1.1.- Name of TELEMAC steering file (.cas)------------------------------------
control_file_name= 'tel_ering_restart.cas'

# 1.2.- TELEMAC solver---------------------------------------------------------
# Select from:
# 1) Telemac 2D
# 2) Telemac 3D

telemac_solver='1'

# 1.3.- Folder path where all the necessary Telemac simulation files (.cas ; .cli ; .tbl ; .slf) are located ---------------------------------------

model_simulation_path= '/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering_restart/'

# 1.4.- Results folder path (Folder path where you want to store the simulation outputs):
# Note: Inside this folder a subfolder called "auto-saved-results" will be created with the following files:
    # .slf (For all simulations)
    # collocation_points.csv
    # collocation_points.npy
    # model_results.npy
    # surrogate_xxx_.pickle (One for each surrogate evaluation)


# comment

results_folder_path = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering_restart/'

# 1.5.- Calibration points .csv file path (complete file path)------------------
#Example: /home/.../..../hydroBayesCal/Simulations/..../calibration-pts.csv
# The .csv file MUST be structured as follows:

# Header:         Point  |    X       |	     Y	     |       MEASUREMENT 1	     |   ERROR 1
# Content:          P1   |  71122.89  |	  9514.50    |     	    2.409	         |   0.0254

calib_pts_file_path= '/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering_restart/measurements_calibration_TKE.csv'

# 1.6.- Number of CPUs for Telemac Simulations----------------------------------------

n_cpus= 16

# 1.7.- Initial full-complexity model runs (init_runs) (Initial training points for surrogate model)

init_runs=4

# 1.8.- Assign calibration parameters-------------------------------------------------
# Notes:
# * MAXIMUM number of calibration parameters = 4
# * The calibration parameters MUST coincide with the Telemac KEYWORD in the .cas file.
#       Example: calib_parameter_1 = "LAW OF FRICTION ON LATERAL BOUNDARIES"
#                calib_parameter_2 = "INITIAL ELEVATION"
# * If you want to calibrate different roughness zones, the roughness zones description MUST be indicated in the .tbl file
#       -> The .tbl file name MUST be indicated in the friction file input
# * The calibration zone MUST contain the word zone,ZONE or Zone as a prefix in the calib_parameter field
#       Example: calib_parameter_1='zone99999100'        if the zone description is: 99999100

calib_parameter_1 = 'zone8'
calib_parameter_2 = 'zone9'
calib_parameter_3 = 'zone10'
calib_parameter_4 = 'zone11'
calib_parameter_5 = 'zone1'

# 1.9.- Friction file .tbl---------------------------------------------------------------

friction_file = 'friction_ering.tbl'

# 1.10.- Range of values for calibration parameters exploration--------------------------
# Notes:
#   * Assign a RANGE [min. value , max. value] according to the prior knowledge of the calibration parameters,
#   * Assign a RANGE [min. value , max. value] of each of the friction zones in the  FRICTION DATA FILE .tbl in case you want to calibrate individual calibration zones.
# to each of the zones.

param_range_1 = [0.045,0.085]
param_range_2 = [0.045,0.085]
param_range_3 = [0.045,0.085]
param_range_4 = [0.045,0.085]
param_range_5 = [0.045,0.085]

# 1.11.- Assign calibration quantities (Target quantities for model calibration)----------
# Notes:
#   * Select the calibration quantity obtained from the full complex numerical model for model calibration purposes.
#   * IMPORTANT: The relevant calibration quantities MUST be assigned as follows for output extraction. More can be added depending on the
#                model output variables.
#*** WATER DEPTH    : Water depth                   [M]
#*** VELOCITY U     : Velocity in X direction       [M/S]
#*** VELOCITY V     : Velocity in Y direction       [M/S]
#*** FREE SURFACE   : Z(Bottom) + Water Depth       [M]
#*** BOTTOM         : Z(Bottom)                     [M]
#*** SCALAR VELOCITY: Scalar velocity               [M/S]

## Calibration quantity 1(calib_target1)-----------------------------------------------------
calib_quantity_1 = 'TURBULENT ENERG'
## Calibration quantity 2(calib_target2)-----------------------------------------------------
calib_quantity_2 = ''
## Calibration quantity 3(calib_target3)-----------------------------------------------------
calib_quantity_3 = ''
## Calibration quantity 4(calib_target4)-----------------------------------------------------
calib_quantity_4 = ''

# 1.12.- Desired NAME of the external.json file containing the model outputs of the calibration quantities--------------------

dict_output_name = 'model_output_dict'

# 1.13.- Desired name of the RESULTS FILE .slf to be iteratively created after each simulation. -------------------------------------------------
# * The results file (.slf) will be stored inside the auto-saved-results folder.
results_filename_base = 'R_ering2m3'

# 2.- Global Bayesian Active Learning parameters (SURROGATE MODEL CREATION  AND CALIBRATION)

# 2.1.- Parameter sampling (Experimental design)---------------------------------------------------
#         Name of the sampling method for the experimental design. The following
#         sampling method are supported:
#
#         * random
#         * latin_hypercube
#         * sobol
#         * halton
#         * hammersley
#         * chebyshev(FT)
#         * grid(FT)
#         * user

parameter_sampling_method = 'latin_hypercube'

# 2.3.- Maximum number of model runs (training points) (At the end of the calibration process)
# Note:
#   * n_max_tp > init_runs
#   * For example: If you want 10 Bayesian Active Learning iterations after the first runs (init_runs = 10) the variable n_max_tp
#                   includes the init_runs. n_max_tp =  init_runs + BAL iterations = 20.

n_max_tp =6

# 2.4.- Number of prior samples (parameter combinations) from the selected ranges for surrogate model evaluation

n_samples=10000

# 2.5.- Number of samples (parameter combinations) selected from the prior samples for exploration during the Bayesian Active Learning

mc_samples=1000
# 2.6.- Number of Monte Carlo realizations from GPE surrogate model to calculate the Kullback–Leibler (KL) divergence "dkl"
# or Bayesian Model Evidence (bme) to select the new training point

mc_exploration = 200
#2.7.- GPE library
# Choose between these two GPE libraries:
# gpy: GPyTorch or skl: Scikit-Learn
gp_library = 'gpy'

# 2.8.- Evaluation steps
#       * Every how many iterations the code evaluates a surrogate
eval_steps=1

# 2.9.- Complete_bal_mode
# Note:
#       * By Default True: When after the initial runs a Bayesian Active Learning is performed (complete surrogate-assisted calibration process)
#                           This option MUST be selected if you choose to perform only BAL (only_bal_mode = True).
#       * False: If only the initial runs are required. The model outputs are stored as .json files
complete_bal_mode = True

# 2.10.- Only BAL mode
# Note:
#       * By Default False: This option executes either a complete surrogate-assisted calibration or only the initial runs (depending of what is indicated above.)
#       * True: When only the surrogate model construction and Bayesian Active Learning of preexisting model outputs
# #              at predefined collocation points is required. This can be run ONLY if a complete process (Complete_bal_mode_mode = True) has been performed.
only_bal_mode = False
### ------------------------------------------- END OF USER INPUT PARAMETERS ----------------------------------------------------------------

## Creation of LISTS containing the calibration parameters, parameter ranges and calibration quantities.
calibration_parameters = [param for param in [calib_parameter_1, calib_parameter_2, calib_parameter_3, calib_parameter_4, calib_parameter_5] if param != '']
parameter_ranges_list=[value for value in [param_range_1, param_range_2, param_range_3, param_range_4, param_range_5] if value != '']
calibration_quantities = [quant for quant in [calib_quantity_1, calib_quantity_2, calib_quantity_3, calib_quantity_4] if quant != '']
## Store variables in a dictionary called user_inputs_tm
user_inputs_tm = {
    'control_file_name': control_file_name,
    'friction_file': friction_file,
    'Telemac_solver': telemac_solver,
    'model_simulation_path': model_simulation_path,
    'results_folder_path': results_folder_path,
    'calib_pts_file_path':calib_pts_file_path,
    'n_cpus':n_cpus,
    'init_runs':init_runs,
    'calibration_parameters':calibration_parameters,
    'parameter_ranges_list':parameter_ranges_list,
    'calibration_quantities':calibration_quantities,
    'dict_output_name':dict_output_name,
    'results_filename_base':results_filename_base,
    'parameter_sampling_method':parameter_sampling_method,
    'n_max_tp':n_max_tp,
    'n_samples':n_samples,
    'mc_samples':mc_samples,
    'mc_exploration':mc_exploration,
    'gp_library':gp_library,
    'complete_bal_mode': complete_bal_mode,
    'only_bal_mode' : only_bal_mode,
    'eval_steps':eval_steps,
}
