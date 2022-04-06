"""
Global variables
"""
import os as _os
import pandas as _pd

GRAVITY = 9.81  # gravitational acceleration in m/s2
KINEMATIC_VISCOSITY = 10 ** -6  # kinematic viscosity in m2/s
WATER_DENSITY = 10. ** 3  # water density in m3/kg
SED_DENSITY = 2650  # sediment density in m3/kg

# directories
SCRIPT_DIR = r"" + _os.path.dirname(__file__) + "/"
RESULTS_DIR = "../results"  # relative path for results
SIM_DIR = "../simulations"  # relative path for simulations

# get telemac and gaia control parameters to enable differentiated writing of steering files
GAIA_PARAMETERS = [p for p in _pd.read_csv(SCRIPT_DIR+"templates/parameters-gaia", header=None).to_dict()[0].values()]
TM2D_PARAMETERS = [p for p in _pd.read_csv(SCRIPT_DIR+"templates/parameters-telemac2d", header=None).to_dict()[0].values()]

# instantiate other global variables
global CALIB_PARAMETERS  # list for calibration optimization parameters
global CALIB_PTS  # numpy array to be loaded from calibration_points file
global IT_LIMIT  # int limit for Bayesian iterations
global MC_SAMPLES  # int for Monte Carlo samples
global N_CPUS  # int number of CPUs to use for Telemac models
global AL_SAMPLES  # int for no. of active learning sampling size
global AL_STRATEGY  # str for active learning strategy



