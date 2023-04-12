"""This is a use case example for the automated stochastic calibration
with sample data from the https://hydro-informatics.com database."""

import os
import os.path
import subprocess
import shutil
import numpy as _np
import pandas as _pd
from datetime import datetime
# from HyBayesCal.pputils.ppmodules.selafin_io_pp import ppSELAFIN
# from HyBayesCal.pputils import shp2csv


# load calibration package
import HyBayesCal as sc





if __name__ == "__main__":
    dir1up = os.path.abspath(os.path.join(__file__, ".."))
    print("this is dir1up: " + dir1up)


    file_dir = "/home/public/test-data/example/epsg5678/".replace("/", os.sep)
    point_files = [
        "boundary-pts.csv",
        "internal-line-pts.csv",
        "embedded-nodes.csv",
    ]
    # concatenate_csv_pts(file_dir, point_files)
    calibration_variable = "WATER DEPTH"
    save_name = "/home/public/test-data/node-output.txt"

    boundary_shp = "boundary.shp"
    breaklines_shp = "breaklines.shp"

    # shp2csv.shp2csv(file_dir + boundary_shp, "boundary.csv")
    # shp2csv.shp2csv(file_dir + boundary_shp, "breaklines.csv")
