"""
Global variables
"""
import os as _os
import pandas as _pd

GRAVITY = 9.81  # gravitational acceleration in m/s2
KINEMATIC_VISCOSITY = 10 ** -6  # kinematic viscosity in m2/s
WATER_DENSITY = 10. ** 3  # water density in m3/kg
SED_DENSITY = 2650  # sediment density in m3/kg
SCRIPT_DIR = r"" + _os.path.dirname(__file__) + "/"

# get telemac and gaia control parameters to enable differentiated writing of steering files
GAIA_PARAMETERS = [p for p in _pd.read_csv(SCRIPT_DIR+"templates/parameters-gaia", header=None).to_dict()[0].values()]
TM2D_PARAMETERS = [p for p in _pd.read_csv(SCRIPT_DIR+"templates/parameters-telemac2d", header=None).to_dict()[0].values()]
