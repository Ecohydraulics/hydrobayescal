"""
Global constant and variable definitions
"""
import os as _os
import pandas as _pd

# physics
GRAVITY = 9.81  # gravitational acceleration in m/s2
KINEMATIC_VISCOSITY = 10 ** -6  # kinematic viscosity in m2/s
WATER_DENSITY = 10. ** 3  # water density in m3/kg
SED_DENSITY = 2650  # sediment density in m3/kg

# directories
SCRIPT_DIR = r"" + _os.path.dirname(__file__) + "/"

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
TM_RANGE = "A6:B10"
AL_RANGE = "A14:B21"
PRIOR_SCA_RANGE = "A27:B30"
PRIOR_VEC_RANGE = "A33:B35"
PRIOR_REC_RANGE = "A38:B39"
ZONAL_PAR_RANGE = "A42:A44"

# define recalculation parameters
RECALC_PARS = {
    "CLASSES SEDIMENT DENSITY": "CLASSES SETTLING VELOCITIES",
    "CLASSES SEDIMENT DIAMETERS": "CLASSES SETTLING VELOCITIES",
}



