"""
Global constant and variable definitions
"""
import os
import pandas as _pd

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
    "FRICTION VELOCI": "telemac",
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

von_Karman_constant = 0.40
nikuradse_log_factor = 30.0
kinematic_viscosity_water = 1.0e-6

# ============================================================
# 2D SLF VARIABLE NAMES FOR BOTTOM FRICTION / NIKURADSE ks
# ============================================================

slf_2d_variables_from_3d = [
    "FRICTION COEFFICIENT",
    "BOTTOM FRICTION COEFFICIENT",
    "FRICTION COEFFICIENT FOR THE BOTTOM",
    "BOTTOM FRICTION",
    "ROUGHNESS COEFFICIENT",
    "NIKURADSE ROUGHNESS",
    "NIKURADSE KS"
]