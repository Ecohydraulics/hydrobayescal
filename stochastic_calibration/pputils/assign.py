#!/usr/bin/env python3
#
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 assign.py                             # 
#                                                                       #
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng., modularized by Sebastian Schwindt
#
# Date: Oct 25, 2015 / July 22, 2022

from ppmodules.readMesh import *
from ppmodules.utilities import *
from progressbar import ProgressBar, Bar, Percentage, ETA


def assign(input_grd="out.grd", boundary_csv="boundary.csv", output_grd="out_friction.grd"):
    """ Function takes in mesh in ADCIRC format, and a set of closed boundaries in pputils CSV format, and creates
    another ADCIRC file with node z-values assigned with polygon attributes. This is useful for assigning friction
    zones through GIS delineation.

    :param str input_grd: Full path and name of an input adcirc mesh file (*.grd)
    :param str boundary_csv: Full path and name of an input boundary CSV file, where every polygon has an attribute value
                              to be assigned to the mesh.
    :param str output_grd: Full path and name of an output adcirc (*.grd) mesh file containing the attribute as the z
                              value
    :return None: creates a grd mesh with assigned (friction) values
    """
    # create the output file
    fout = open(output_grd, "w")

    # read the adcirc file
    n, e, x, y, z, ikle = readAdcirc(input_grd)

    # use numpy to read the boundary polygon file in pputils format
    poly_data = np.loadtxt(boundary_csv, delimiter=',', skiprows=0, unpack=True)

    # boundary data
    shapeid_poly = poly_data[0, :]
    x_poly = poly_data[1, :]
    y_poly = poly_data[2, :]
    attr_poly = poly_data[3, :]

    # round boundary nodes to three decimals
    x_poly = np.around(x_poly, decimals=3)
    y_poly = np.around(y_poly, decimals=3)

    # total number of nodes in the polygon file
    nodes = len(x_poly)

    # get the unique polygon ids
    polygon_ids = np.unique(shapeid_poly)

    # find out how many different polygons there are
    n_polygons = len(polygon_ids)

    # to get the attribute data for each polygon
    attribute_data = np.zeros(n_polygons)
    attr_count = -1

    # go through the polygons, and assign attribute_data -- WEAK INEFFICIENT CODE BLOCK!
    for i in range(nodes - 1):
        # print str(shapeid_poly[i]) + ' ' + str(shapeid_poly[i+1])
        if shapeid_poly[i] - shapeid_poly[i + 1] < 0:
            attr_count = attr_count + 1
            attribute_data[attr_count] = attr_poly[i]

    # manually assign the attribute_data for the last polygon
    attribute_data[n_polygons - 1] = attr_poly[nodes - 1]

    # default attribute
    default = 0.0

    # define the mesh attribute as the value read from the file
    f = z

    # for the progress bar
    w = [Percentage(), Bar(), ETA()]
    pbar = ProgressBar(widgets=w, maxval=n_polygons * n).start()
    count_bar = 0

    # loop over all polygons
    for i in range(n_polygons):
        # construct each polygon
        poly = []
        for j in range(nodes):
            if shapeid_poly[j] == polygon_ids[i]:
                poly.append((x_poly[j], y_poly[j]))

        # to loop over all nodes in the mesh (inneficient brute force code block!)
        for k in range(n):
            count_bar = count_bar + 1
            pbar.update(count_bar)
            poly_test = point_in_poly(x[k], y[k], poly)
            if poly_test == 'IN':
                f[k] = attribute_data[i]
            elif poly_test == 'OUT':
                f[k] = f[k]
            else:
                f[k] = -999

        # delete all elements in the poly list
        del poly[:]

    # finish the bar
    pbar.finish()

    # write the adcirc mesh file
    fout.write("ADCIRC" + "\n")
    # write the number of elements and number of nodes in the header file
    fout.write(str(e) + " " + str(n) + "\n")

    # write nodes
    for i in range(n):
        fout.write(str(i + 1) + " " + str("{:.3f}".format(x[i])) + " " +
                   str("{:.3f}".format(y[i])) + " " + str("{:.3f}".format(f[i])) + "\n")

    # write elements
    for i in range(e):
        fout.write(str(i + 1) + " 3 " + str(ikle[i, 0] + 1) + " " + str(ikle[i, 1] + 1) + " " +
                   str(ikle[i, 2] + 1) + "\n")
