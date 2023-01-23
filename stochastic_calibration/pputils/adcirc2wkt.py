"""
Convert adcirc grid to WKT csv file
Author: Pat Prodanovic, Ph.D., P.Eng., modularized by Sebastian Schwindt
Date: Feb 20, 2016 / July 22, 2022
"""
from ppmodules.readMesh import *


def adcirc2wkt(adcirc_file="out.grd", output_file="outWKT.csv"):
    """ Function takes a mesh in ADCIRC format and generates two *.csv files in WKT (well known text) format,
    one corresponding to elements (this is that will visualize the triangulation, and one corresponding to nodes
    (this one visualizes the node numbers). The WKT format is essentially equivalent to a shapefile format.

    :param str adcirc_file:  adcirc mesh file
    :param str output_file: generated *.csv WKT files for element polygons and nodes
    :return:
    """
    # create the element and node output files
    output_file_e = output_file.rsplit(".", 1)[0] + "_e.csv"
    output_file_n = output_file.rsplit(".", 1)[0] + "_n.csv"

    # to create the output file
    fout = open(output_file_e, "w")

    # read the adcirc file
    n, e, x, y, z, ikle = readAdcirc(adcirc_file)

    # write the header of the WKT file\
    fout.write("WKT,element" + "\n")

    for i in range(e):
        fout.write("'POLYGON ((")
        fout.write(str("{:.3f}".format(x[ikle[i, 0]])) + " " +
                   str("{:.3f}".format(y[ikle[i, 0]])) + " " +
                   str("{:.3f}".format(z[ikle[i, 0]])) + ", " +

                   str("{:.3f}".format(x[ikle[i, 1]])) + " " +
                   str("{:.3f}".format(y[ikle[i, 1]])) + " " +
                   str("{:.3f}".format(z[ikle[i, 1]])) + "," +

                   str("{:.3f}".format(x[ikle[i, 2]])) + " " +
                   str("{:.3f}".format(y[ikle[i, 2]])) + " " +
                   str("{:.3f}".format(z[ikle[i, 2]])) + ", " +

                   str("{:.3f}".format(x[ikle[i, 0]])) + " " +
                   str("{:.3f}".format(y[ikle[i, 0]])) + " " +
                   str("{:.3f}".format(z[ikle[i, 0]])) + "))'," + str(i + 1) + "\n")

    # to generate the node file
    fout_n = open(output_file_n, "w")

    # write the header of the WKT file\
    fout_n.write("WKT,node" + "\n")

    for i in range(n):
        fout_n.write("'POINT (")
        fout_n.write(str("{:.3f}".format(x[i])) + " " +
                     str("{:.3f}".format(y[i])) + " " +
                     str("{:.3f}".format(z[i])) + ")'," +
                     str(i + 1) + "\n")
