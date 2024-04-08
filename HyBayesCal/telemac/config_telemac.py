"""
Global constant and variable definitions
"""
import os, sys
import pandas as _pd

# get package directories
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..".replace("/", os.sep))))

# get telemac and gaia control parameters to enable differentiated writing of steering files
TM_TEMPLATE_DIR = os.path.abspath(os.path.join(__file__, "..")) + os.sep + "templates" + os.sep
#print(TM_TEMPLATE_DIR)
GAIA_PARAMETERS = _pd.read_csv(TM_TEMPLATE_DIR+"parameters-gaia.csv", names=["parameter", "type"])
#print(GAIA_PARAMETERS)
TM2D_PARAMETERS = _pd.read_csv(TM_TEMPLATE_DIR+"parameters-telemac2d.csv", names=["parameter", "type"])
#print(TM2D_PARAMETERS)

TM_TRANSLATOR = {
    "TOPOGRAPHIC CHANGE": "BOTTOM",
    "DEPTH": "WATERDEPTH",
    "VELOCITY U": "VELOCITYU",
    "VELOCITY V": "VELOCITYV",
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
this_dir = os.path.abspath(".")


activateTM_path = os.path.abspath(os.path.join(this_dir, "env-scripts", "activateTM.sh"))
results_filename_base = "r2d-donau"

cas_line_1= 'INITIAL ELEVATION:309.4D0'
cas_line_2='ROUGHNESS COEFFICIENT OF BOUNDARIES:0.01'
cas_line_3='TIME STEP                                   : 0.5'
cas_line_4=''

cas_line_results_file="RESULTS FILE:r2d-donau.slf"
