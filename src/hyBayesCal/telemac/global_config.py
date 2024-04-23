

# =====================================================
# ===========GLOBAL SIMULATION PARAMETERS==============
# =====================================================

# ===========Name of TELEMAC steering file (.cas)==============
cas_file_name= 't2d-donau.cas'

# ===========Steering file (.cas) simulation path==============
cas_file_simulation_path='/home/amintvm/modeling/hybayescal/examples/donau/'

# =====================Number of CPUs==========================
NCPS= 2

# ======Initial full-complexity model runs (init_runs)=========
init_runs=3

# ======Calibration parameters=========
calib_parameter_1='ROUGHNESS COEFFICIENT OF BOUNDARIES'
calib_parameter_2='INITIAL ELEVATION'
calib_parameter_3=''
calib_parameter_4=''


# ======Range of calibration parameters=========
param_range_1=[0.009,0.030]
param_range_2=[305.4,312.5]
param_range_3=''
param_range_4=''



calib_parameter_list = [param for param in [calib_parameter_1, calib_parameter_2, calib_parameter_3, calib_parameter_4] if param != '']
parameter_ranges_list=[value for value in [param_range_1, param_range_2, param_range_3, param_range_4] if value != '']

# ===================Parameter sampling =======================
# Select your method for parameter sampling
# 1) MIN - equal interval - MAX
# 2)    MIN - random - MAX

parameter_sampling_method='1'

# ===================Calibration quantities====================
# Select your method for parameter sampling
# 1) DEPTH
# 2)    MIN - random - MAX
# 3)    MIN - random - MAX

## Calibration quantity 1(calib_target1)
calib_quantity_1='WATER DEPTH'

## Calibration quantity 2(calib_target2)
calib_quantity_2='VELOCITY U'
## Calibration quantity 3(calib_target3)
calib_quantity_3='VELOCITY V'
## Calibration quantity 4(calib_target4)
calib_quantity_4=''

calib_quantity_list = [quant for quant in [calib_quantity_1, calib_quantity_2, calib_quantity_3, calib_quantity_4] if quant != '']

# =============Name of the Base Results file .slf =============
results_file_name_base='r2d-donau'

# =============Calibration points .csv file (file complete path) =============

calib_pts_file_path= '/home/amintvm/modeling/hybayescal/examples/donau/meas_data1.csv'