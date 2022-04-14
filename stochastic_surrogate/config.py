"""
Global constant and variable definitions
"""
import os as _os
import pandas as _pd
import numpy as _np

# physics
GRAVITY = 9.81  # gravitational acceleration in m/s2
KINEMATIC_VISCOSITY = 10 ** -6  # kinematic viscosity in m2/s
WATER_DENSITY = 10. ** 3  # water density in m3/kg
SED_DENSITY = 2650  # sediment density in m3/kg

# directories
SCRIPT_DIR = r"" + _os.path.dirname(__file__) + "/"
RESULTS_DIR = "../results"  # relative path for results
SIM_DIR = str()  # relative path for simulations

# get telemac and gaia control parameters to enable differentiated writing of steering files
GAIA_PARAMETERS = _pd.read_csv(SCRIPT_DIR+"templates/parameters-gaia.csv", names=["parameter", "type"])
TM2D_PARAMETERS = _pd.read_csv(SCRIPT_DIR+"templates/parameters-telemac2d.csv", names=["parameter", "type"])
TM_TRANSLATOR = {
    "TOPOGRAPHIC CHANGE": "BOTTOM",
    "DEPTH": "DEPTH",
    "VELOCITY": "VELOCITY",
}

RESULT_NAME_GAIA = "'res-gaia-PC"  # PC stands for parameter combination
RESULT_NAME_TM = "'res-tel-PC"  # PC stands for parameter combination


# define relevant data ranges in user-input.xlsx
TM_RANGE = "A6:B9"
AL_RANGE = "A13:B19"
PRIOR_DIR_RANGE = "A25:B28"
PRIOR_INDIR_RANGE = "A31:B32"
PRIOR_REC_RANGE = "A35:B35"

# global variables for user input
INPUT_XLSX_NAME = str()  # str of path to user-input.xlsx including filename
CALIB_PAR_SET = {}  # dict for direct calibration optimization parameters
CALIB_ID_PAR_SET = {}  # dict for indirect calibration parameters
CALIB_PTS = None  # numpy array to be loaded from calibration_points file
CALIB_TARGET = str()
IT_LIMIT = int()  # int limit for Bayesian iterations
MC_SAMPLES = int()  # int for Monte Carlo samples
N_CPUS = int()  # int number of CPUs to use for Telemac models
AL_SAMPLES = int()  # int for no. of active learning sampling size
AL_STRATEGY = str()  # str for active learning strategy
TM_CAS = str()
GAIA_CAS = str()

# global surrogate and active learning variables
BME = _np.zeros((IT_LIMIT, 1))
RE = _np.zeros((IT_LIMIT, 1))
al_BME = _np.zeros((AL_SAMPLES, 1))
al_RE = _np.zeros((AL_SAMPLES, 1))
