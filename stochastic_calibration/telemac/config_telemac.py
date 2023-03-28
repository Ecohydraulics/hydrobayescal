"""
Global constant and variable definitions
"""
import pandas as _pd
from config_logging import *

# get telemac and gaia control parameters to enable differentiated writing of steering files
GAIA_PARAMETERS = _pd.read_csv(SCRIPT_DIR+"templates/parameters-gaia.csv", names=["parameter", "type"])
TM2D_PARAMETERS = _pd.read_csv(SCRIPT_DIR+"templates/parameters-telemac2d.csv", names=["parameter", "type"])
TM_TRANSLATOR = {
    "TOPOGRAPHIC CHANGE": "BOTTOM",
    "DEPTH": "DEPTH",
    "VELOCITY": "VELOCITY",
    "NONE": None
}

# define relevant data ranges in user-input.xlsx
TM_RANGE = "A6:B10"
AL_RANGE = "A14:B22"
MEASUREMENT_DATA_RANGE = "A23:B26"
PRIOR_SCA_RANGE = "A32:B35"
PRIOR_VEC_RANGE = "A38:B40"
PRIOR_REC_RANGE = "A43:B44"
ZONAL_PAR_RANGE = "A47:A49"

# define recalculation parameters
RECALC_PARS = {
    "CLASSES SEDIMENT DENSITY": "CLASSES SETTLING VELOCITIES",
    "CLASSES SEDIMENT DIAMETERS": "CLASSES SETTLING VELOCITIES",
}
