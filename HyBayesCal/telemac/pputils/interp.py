#!/usr/bin/env python3
#
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 interp.py                             # 
#                                                                       #
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng., modularized by Sebastian Schwindt
# 
# Date: June 28, 2015 / July 28, 2022

import matplotlib.tri as mtri
from ppmodules.readMesh import *


def inter(tin_file="surface.tin", mesh_msh="mesh.grd", output_grd="mesh_interp.grd", interpolate_nans=True):
    """ Function takes a tin and a mesh file (both in ADCIRC format), and interpolates the nodes of the mesh file
    from the tin file.

    :param str tin_file: Full path and name of *.tin surface file
    :param str mesh_msh: Full path and name of *.grd mesh file
    :param str output_grd: Full path and name of output *.grd file with interpolated heights
    :param bool interpolate_nans: Optional argument to deactivate interpolation of mesh values where elevation info is
                        missing. The default is ``True``. If set to ``False``, no-elevation nodes will have Z=-999.0
    :return None: creates interpolated mesh file
    """
    # read the adcirc tin file
    t_n, t_e, t_x, t_y, t_z, t_ikle = readAdcirc(tin_file)

    # read the adcirc mesh file that has zero z values
    m_n, m_e, m_x, m_y, m_z, m_ikle = readAdcirc(mesh_msh)

    # create tin triangulation object using matplotlib
    tin = mtri.Triangulation(t_x, t_y, t_ikle)

    # run the triangulation
    interpolator = mtri.LinearTriInterpolator(tin, t_z)
    m_z = interpolator(m_x, m_y)

    # if a node is outside of the boundary of the domain, assign the value -999.0 for interpolation
    where_are_nans = np.isnan(m_z)
    m_z[where_are_nans] = -999.0

    # rather than keeping -999.0 as the mesh node value outside the tin, assign the elevation of the closest tin node
    if interpolate_nans:
        print("WARNING: Some nodes are outside of the TIN boundary ---")
        if np.sum(where_are_nans) > 0:
            print("          --- I will interpolate Z-values from closest nodes")
            for i in range(len(m_x)):
                if where_are_nans[i]:
                    xdist = np.subtract(t_x, m_x[i])
                    ydist = np.subtract(t_y, m_y[i])
                    dist = np.sqrt(np.power(xdist, 2.0) + np.power(ydist, 2.0))
                    minidx = np.argmin(dist)
                    m_z[i] = t_z[minidx]

    # create the output file
    fout = open(output_grd, "w")

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
