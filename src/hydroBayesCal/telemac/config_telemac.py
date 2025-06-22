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
# Dictionary mapping each variable name to its source model
classification_tm_gaia_dict = {
    "WATER DEPTH": "telemac",
    "SCALAR VELOCITY": "telemac",
    "FREE SURFACE": "telemac",
    "VELOCITY U": "telemac",
    "VELOCITY V": "telemac",
    "FROUDE NUMBER": "telemac",
    "TURBULENT ENERG": "telemac",
    "BOTTOM SHEAR STRESS": "telemac",

    "BED ELEVATION": "gaia",
    "CUMUL BED EVOL": "gaia",
    "SUSPENDED LOAD CONC.": "gaia",
    "BED LOAD": "gaia",
    "TOTAL SEDIMENT DISCHARGE": "gaia",
    "SEDIMENT DIAMETER": "gaia",
    "CRITICAL SHEAR STRESS": "gaia"
}