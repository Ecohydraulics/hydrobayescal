import os

HOMEDIR = os.path.abspath(os.getcwd())
print(HOMEDIR)

# =====================================================
# ===========GLOBAL SIMULATION PARAMETERS==============
# =====================================================

# ===========Name of TELEMAC steering file (.cas)==============
cas_file_name= 't2d-donau.cas'

# ===========TELEMAC solver==============
# Select from:
# 1) Telemac 2D
# 2) Telemac 3D

Telemac_solver='1'

# ===========Steering file (.cas) simulation path==============
cas_file_simulation_path='/home/IWS/hidalgo/Documents/hybayescal/examples/donau/'

# =============Calibration points .csv file (file complete path) =============
calib_pts_file_path= '/home/IWS/hidalgo/Documents/hybayescal/examples/donau/Bal-data/calibration-pts_RED.csv'

# =====================Number of CPUs==========================
NCPS= 16

# ======Initial full-complexity model runs (init_runs)=========
init_runs=10

# ======Calibration parameters=========
calib_parameter_1='ROUGHNESS COEFFICIENT OF BOUNDARIES'
calib_parameter_2='INITIAL ELEVATION'
calib_parameter_3='IMPLICITATION FOR VELOCITY'
calib_parameter_4=''


# ======Range of calibration parameters=========
param_range_1=[0.01,0.080]
param_range_2=[305.4,312.5]
param_range_3=[0.5,0.7]
param_range_4=''


## Creation of LISTS containing the calibration parameters and parameter ranges of each of the calibration parameters.
calib_parameter_list = [param for param in [calib_parameter_1, calib_parameter_2, calib_parameter_3, calib_parameter_4] if param != '']
parameter_ranges_list=[value for value in [param_range_1, param_range_2, param_range_3, param_range_4] if value != '']

# ===================Parameter sampling =======================
# Select your method for parameter sampling
# 1) MIN - equal interval - MAX
# 2)    MIN - random - MAX

parameter_sampling_method='2'

# ===================Calibration quantities====================
# Select your method for parameter sampling
# 1) DEPTH
# 2)    MIN - random - MAX
# 3)    MIN - random - MAX
# IMPORTANT: The relevant calibration quantities MUST be assigned as follows for output extraction:
#*** WATER DEPTH    : Water depth                   [M]
#*** VELOCITY U     : Velocity in X direction       [M/S]
#*** VELOCITY V     : Velocity in Y direction       [M/S]
#*** FREE SURFACE   : Z(Bottom) + Water Depth       [M]
#*** BOTTOM         : Z(Bottom)                     [M]


## Calibration quantity 1(calib_target1)
calib_quantity_1='VELOCITY U'
## Calibration quantity 2(calib_target2)
calib_quantity_2=''
## Calibration quantity 3(calib_target3)
calib_quantity_3=''
## Calibration quantity 4(calib_target4)
calib_quantity_4=''

## Creation of LISTS containing the calibration quantities
calib_quantity_list = [quant for quant in [calib_quantity_1, calib_quantity_2, calib_quantity_3, calib_quantity_4] if quant != '']

# =============Desired name of the external.json file containing the model outputs of the calibration quantities =============
dict_output_name='model_output_dict'

#============= Desired name of the RESULTS FILE to be iteratively changed in the .cas file =============
results_file_name_base = 'TelemacResults'
