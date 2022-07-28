#!/usr/bin/env python3
#
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 interp_from_pts.py                    # 
#                                                                       #
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng., modularized by Sebastian Schwindt
#
# Date: May 26, 2016 / July 28, 2022

from scipy import spatial  # scipy to get kdTree
from ppmodules.readMesh import *
from progressbar import ProgressBar, Bar, Percentage, ETA


def interp_from_pts(points_csv="points.csv", mesh_grd="mesh.grd", interp_mesh_grd="mesh_interp.grd", neighbors=1):
    """ Function takes a xyz CSV point file and a mesh file (in ADCIRC format), and interpolates the nodes of the mesh
    file from the points CSV file. It uses scipy's kdtree to assign the closest the xyz dataset to a mesh node.

    :param str points_csv: Full path and name of an xyz CSV point file, no headers, comma delimited
    :param str mesh_grd: Full path and name of a .grd mesh file whose nodes are to be interpolated
    :param str interp_mesh_grd: Full path and name of the resulting interpolated mesh
    :param int neighbors: The default of ``1`` uses the closest neighboring point only (good choice for dense XYZ point
                         clouds). The maximum number of neighbors for interpolating Z or M values on the mesh is ``10``.
    :return None: Creates a new mesh file with interpolated values.
    """
    # impose a limit on neighbors to be between 1 and 10
    if (neighbors < 1) or (neighbors > 10):
        print("ERROR: Number of neighbors must be between 1 and 10. Exiting program.")
        return -1

    print("reading input data...")
    # read the points file
    pts_data = np.loadtxt(points_csv, delimiter=',', skiprows=0, unpack=True)
    x = pts_data[0, :]
    y = pts_data[1, :]
    z = pts_data[2, :]

    # read the adcirc mesh file (_m is for mesh) in which the z values are all zeros
    m_n, m_e, m_x, m_y, m_z, m_ikle = readAdcirc(mesh_grd)

    print("constructing KDTree object...")
    # to create a KDTree object
    source = np.column_stack((x, y))
    tree = spatial.cKDTree(source)

    den = 0.0
    tmp_sum = 0.0

    print("interpolating...")
    w = [Percentage(), Bar(), ETA()]
    pbar = ProgressBar(widgets=w, maxval=m_n).start()
    for i in range(m_n):
        d, idx = tree.query((m_x[i], m_y[i]), k=neighbors)

        # calculate denominator
        if neighbors > 1:
            for j in range(neighbors):
                if d[j] < 1.0E-6:
                    d[j] = 1.0E-6
                den = den + (1.0 / (d[j] ** 2))
        else:
            if d < 1.0E-6:
                d = 1.0E-6
            den = den + (1.0 / (d ** 2))

        # calculate weights
        weights = (1.0 / d ** 2) / den

        # get interpolated value
        if neighbors > 1:
            for j in range(neighbors):
                tmp_sum = tmp_sum + weights[j] * z[idx[j]]
        else:
            tmp_sum = weights * z[idx]

        # assign value
        m_z[i] = tmp_sum

        # reset the denominator
        den = 0.0
        tmp_sum = 0.0

        pbar.update(i + 1)
    pbar.finish()

    print("writing results to file...")
    # create the output file (i.e., the interpolated mesh)
    fout = open(interp_mesh_grd, "w")

    # write the adcirc mesh file
    fout.write("ADCIRC" + "\n")
    # write the number of elements and number of nodes in the header file
    fout.write(str(m_e) + " " + str(m_n) + "\n")

    # write nodes
    for i in range(0, m_n):
        fout.write(str(i + 1) + " " + str("{:.3f}".format(m_x[i])) + " " +
                   str("{:.3f}".format(m_y[i])) + " " + str("{:.3f}".format(m_z[i])) + "\n")

    # write elements
    for i in range(0, m_e):
        fout.write(str(i + 1) + " 3 " + str(m_ikle[i, 0] + 1) + " " + str(m_ikle[i, 1] + 1) + " " +
                   str(m_ikle[i, 2] + 1) + "\n")
