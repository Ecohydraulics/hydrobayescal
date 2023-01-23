"""
Convert an adcirc grid to TELEMAC SELAFIN (SLF) - formatted file
Author: Pat Prodanovic, Ph.D., P.Eng., modularized by Sebastian Schwindt
Date: Feb 15, 2016 / Jan 23, 2023
"""

from ppmodules.selafin_io_pp import *      # pp's SELAFIN io
from ppmodules.utilities import *          # getIPOBO_IKLE() method


def adcirc2sel(adcirc_file="mesh.grd", precision="single", slf_out_name="mesh.slf"):
    """ Function converts an adcirc mesh into a slf mesh with either single (default) or double precision

    :param str adcirc_file: name of the adcirc grd file to convert
    :param str precision: either 'single' or 'double' precision - default is 'single'
    :param str slf_out_name: name of the SELFAN slf file to create
    :return:
    """
    # reads mesh data using the get IPOBO_IKLE() method from utilities.py
    # the ikle and the ppIPOB are one-based
    n, e, x, y, z, ikle, ppIPOB = getIPOBO_IKLE(adcirc_file)

    # the above method generates a file called temp.cli, which we rename here
    cli_file = slf_out_name.split(".", 1)[0] + ".cli"
    os.rename("temp.cli", cli_file)

    # write the *.slf file
    if precision == "single":
        ftype = "f"
        fsize = 4
    elif precision == "double":
        ftype = "d"
        fsize = 8
    else:
        print("Precision unknown! Exiting!")
        sys.exit(0)

    # getIPOBO_IKLE() method provides these values
    NELEM = e
    NPOIN = n
    NDP = 3  # must be 3 for triangular elements
    IKLE = ikle
    IPOBO = ppIPOB

    slf = ppSELAFIN(slf_out_name)
    slf.setPrecision(ftype, fsize)
    slf.setTitle("created with pputils")
    slf.setVarNames(["BOTTOM          "])
    slf.setVarUnits(["M               "])
    slf.setIPARAM([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    slf.setMesh(NELEM, NPOIN, NDP, IKLE, IPOBO, x, y)
    slf.writeHeader()

    # if writing only 1 variable, must have numpy array of size (1,NPOIN)
    # for 2 variables, must have numpy array of size (2,NPOIN), and so on.
    zz = np.zeros((1, NPOIN))
    zz[0, :] = z

    slf.writeVariables(0.0, zz)
