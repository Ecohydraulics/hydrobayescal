"""
Global user-defined settings for Telemac full complexity simulations.

Author: Andres Heredia M.Sc.
"""
import os

HOMEDIR = os.path.abspath(os.getcwd())
# print(HOMEDIR)

# GLOBAL SIMULATION PARAMETERS (COMPLEX MODEL)

# ===========Name of TELEMAC steering file (.cas)==============
cas_file_name= 't2d-donau.cas'

# ===========TELEMAC solver==============
# Select from:
# 1) Telemac 2D
# 2) Telemac 3D

Telemac_solver='1'

# ===========Steering file (.cas) folder path==============
cas_file_simulation_path='/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/'

# ===========Results folder path (Folder where you want to save the following files: )==============
# .slf (For all simulations)
# collocation_points.csv
# collocation_points.npy
# model_results.npy
#
#
results_folder_path='/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/'

# =============Calibration points .csv file (complete file path) =============
#Example: /home/.../..../hydroBayesCal/Simulations/..../calibration-pts.csv
# The .csv file MUST be structured as follows:
# Header:         Point  |    X       |	     Y	     |       MEASUREMENT 1	     |   ERROR 1
# Content:          P1   |  71122.89  |	  9514.50    |     	    2.409	         |   0.0254

calib_pts_file_path= '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/Bal-data/calibration-pts_H.csv'

# =====================Number of CPUs for Telemac Simulations==========================
n_cpus= 16

# ======Initial full-complexity model runs (init_runs) (Initial training points for surrogate model)=========
init_runs=4

# ======Calibration parameters=========
# Note: The calibration parameters MUST coincide with the way they are written in the .cas file
calib_parameter_1='ROUGHNESS COEFFICIENT OF BOUNDARIES'
calib_parameter_2='INITIAL ELEVATION'
calib_parameter_3=''
calib_parameter_4=''


# ======Range of values for calibration exploration =========
# According to the prior knowledge of the calibration parameters, assign a RANGE [min. value , max. value]
# for calibration purposes
param_range_1=[0.009,0.8]
param_range_2=[304,320]
param_range_3=''
param_range_4=''


## Creation of LISTS containing the calibration parameters and parameter ranges of each of the calibration parameters.
calib_parameter_list = [param for param in [calib_parameter_1, calib_parameter_2, calib_parameter_3, calib_parameter_4] if param != '']
parameter_ranges_list=[value for value in [param_range_1, param_range_2, param_range_3, param_range_4] if value != '']



# ===================Calibration quantities====================
# Select the calibration quantity obtained from the full complex numerical model.
# IMPORTANT: The relevant calibration quantities MUST be assigned as follows for output extraction:
#*** WATER DEPTH    : Water depth                   [M]
#*** VELOCITY U     : Velocity in X direction       [M/S]
#*** VELOCITY V     : Velocity in Y direction       [M/S]
#*** FREE SURFACE   : Z(Bottom) + Water Depth       [M]
#*** BOTTOM         : Z(Bottom)                     [M]
#*** SCALAR VELOCITY: Scalar velocity               [M/S]

## Calibration quantity 1(calib_target1)
calib_quantity_1='SCALAR VELOCITY'
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
results_file_name_base = 'R_donau'


# ===========GLOBAL PARAMETERS (SURROGATE MODEL AND CALIBRATION)==============
# =====================================================================

# ===================Parameter sampling (Experimental design)=======================
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

# ======Maximum number of training points (At the end of the calibration process)=========
# n_max_tp > init_runs

n_max_tp =6
# ======Prior samples (parameter combinations) from the selected ranges for surrogate model evaluation=========
n_samples=10000

# ======Samples (parameter combinations) for exploration during the Bayesian Active Learning=========

n_samples_exploration_BAL=2000

# ======GPE library=========
# Choose between these two GPE libraries:
# gpy: GPyTorch or skl: Scikit-Learn
gp_library = 'gpy'
