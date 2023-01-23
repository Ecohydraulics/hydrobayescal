"""
Purpose: Convert a 2dm mesh into adcirc file format. This script works only for triangular elements.
Author: Pat Prodanovic, Ph.D., P.Eng., modularized by Sebastian Schwindt
Date: Feb 20, 2016 / January 23, 2023
"""

from ppmodules.readMesh import *
from ppmodules.writeMesh import *


def twodm2adcirc(two_dm_file=".2dm", adcirc_file=".grd"):
    """ Function runs 2dm conversion to adcirc.grd file
    :param str two_dm_file: 2dm mesh file
    :param str adcirc_file: output adcirc mesh file
    :return int: 0 in case of success
    """

    # read 2dm file
    n, e, x, y, z, ikle = read2dm(two_dm_file)
    # write the adcirc file
    writeAdcirc(n, e, x, y, z, ikle, adcirc_file)

    return 0
