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
    "VELOCITY W": "telemac",
    "FROUDE NUMBER": "telemac",
    "TURBULENT ENERG": "telemac",
    "BOTTOM SHEAR STRESS": "telemac",
    "DISSIPATION": "telemac",
    "3D VELOCITY MAGNITUDE": "telemac",

    "BED ELEVATION": "gaia",
    "CUMUL BED EVOL": "gaia",
    "SUSPENDED LOAD CONC.": "gaia",
    "BED LOAD": "gaia",
    "TOTAL SEDIMENT DISCHARGE": "gaia",
    "SEDIMENT DIAMETER": "gaia",
    "CRITICAL SHEAR STRESS": "gaia"
}
# ============================================================
# DEFAULT PHYSICAL CONSTANTS FOR WALL-LAW DIAGNOSTICS
# ============================================================
# These constants are used to compute TELEMAC-style friction
# velocity and y+ from the near-bed velocity and vertical spacing.
#
# They can be overwritten inside the extraction function if needed.
# ============================================================

DEFAULT_VON_KARMAN_CONSTANT = 0.40
DEFAULT_NIKURADSE_LOG_FACTOR = 30.0
DEFAULT_KINEMATIC_VISCOSITY_WATER = 1.0e-6

# ============================================================
# 2D SLF VARIABLE NAMES FOR BOTTOM FRICTION / NIKURADSE ks
# ============================================================

GENERATED_2D_SLF_VARIABLES_FROM_3D = [
    "FRICTION COEFFICIENT",
    "BOTTOM FRICTION COEFFICIENT",
    "FRICTION COEFFICIENT FOR THE BOTTOM",
    "BOTTOM FRICTION",
    "ROUGHNESS COEFFICIENT",
    "NIKURADSE ROUGHNESS",
    "NIKURADSE KS"
]