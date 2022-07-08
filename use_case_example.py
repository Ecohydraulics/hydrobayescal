"""This is a use case example for the automated stochastic calibration
with sample data from the https://hydro-informatics.com database."""

import os as _os
import subprocess
import shutil
import numpy as _np
from datetime import datetime
from pputils.ppmodules.selafin_io_pp import ppSELAFIN
from pputils import shp2csv


# load calibration package
import stochastic_calibration as sc

# USER INPUT --------------------------
file_name = "/home/public/test-data/steady2d-restart/r2dsteady.slf"
calibration_variable = "WATER DEPTH"
save_name = "/home/public/test-data/node-output.txt"

boundary_shp = "/home/public/test-data/example/boundary.shp"