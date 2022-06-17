"""
Global constant and variable definitions
"""
import os as _os
import pandas as _pd
import logging

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
AL_RANGE = "A14:B22"
PRIOR_SCA_RANGE = "A28:B31"
PRIOR_VEC_RANGE = "A34:B36"
PRIOR_REC_RANGE = "A39:B40"
ZONAL_PAR_RANGE = "A43:A45"

# define recalculation parameters
RECALC_PARS = {
    "CLASSES SEDIMENT DENSITY": "CLASSES SETTLING VELOCITIES",
    "CLASSES SEDIMENT DIAMETERS": "CLASSES SETTLING VELOCITIES",
}

# setup logging
info_formatter = logging.Formatter("%(asctime)s - %(message)s")
warn_formatter = logging.Formatter("WARNING [%(asctime)s]: %(message)s")
error_formatter = logging.Formatter("ERROR [%(asctime)s]: %(message)s")
logger = logging.getLogger("stochastic_calibration")
logger.setLevel(logging.INFO)
logger_warn = logging.getLogger("warnings")
logger_warn.setLevel(logging.WARNING)
logger_error = logging.getLogger("errors")
logger_error.setLevel(logging.ERROR)
# create console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(info_formatter)
logger.addHandler(console_handler)
console_whandler = logging.StreamHandler()
console_whandler.setLevel(logging.WARNING)
console_whandler.setFormatter(warn_formatter)
logger_warn.addHandler(console_whandler)
console_ehandler = logging.StreamHandler()
console_ehandler.setLevel(logging.ERROR)
console_ehandler.setFormatter(error_formatter)
logger_error.addHandler(console_ehandler)
# create info file handler and set level to debug
info_handler = logging.FileHandler(SCRIPT_DIR + "logfile.log", "w")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(info_formatter)
logger.addHandler(info_handler)
# create warning file handler and set level to error
warn_handler = logging.FileHandler(SCRIPT_DIR + "warnings.log", "w")
warn_handler.setLevel(logging.WARNING)
warn_handler.setFormatter(warn_formatter)
logger_warn.addHandler(warn_handler)
# create error file handler and set level to error
err_handler = logging.FileHandler(SCRIPT_DIR + "errors.log", "w")
err_handler.setLevel(logging.ERROR)
err_handler.setFormatter(error_formatter)
logger_error.addHandler(err_handler)
