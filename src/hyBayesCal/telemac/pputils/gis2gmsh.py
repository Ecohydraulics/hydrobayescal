#!/usr/bin/env python3
#
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 gis2gmsh.py                           # 
#                                                                       #
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng., modified by Sebastian Schwindt for functionalization
# Date: June 29, 2015
# Modified: July 21, 2022
#
import numpy as np
from collections import OrderedDict


def gis2gmsh(nodes_csv, boundary_csv, out_file_name="out.geo", lines_csv=None, holes_csv=None, duplicates_flag=True):
    """ Takes CSV files of the geometry generated in QGIS (or any other gis or cad software) and produces geometry files
    for use with gmsh mesh generator.

    :param str nodes_csv: file listing all nodes (incl. embedded nodes if any). The nodes file contains of x,y,z
                        or x,y,z,size. The size parameter is an optional input, and is used by gmsh as an extra
                        parameter that forces an element size around a particular node. The nodes file must
                        be comma separated, and have no header lines.
    :param str boundary_csv: is file listing nodes of the outer boundary for the mesh. The boundary file can be generated
                      by snapping lines to the nodes from the nodes.csv file. The boundary file has the columns
                       shapeid,x,y of all lines in the file. The boundary has to be a closed line, where the first and
                       last nodes are identical. The shapeid is a integer, where the boundary is defined with a
                       distinct id (i.e., shapeid of 0).
    :param str out_file_name: is the name of the output file that should end on ``.geo`` (default is ``out.geo``) to be
                        a recognizable gmsh file format.
    :param str lines_csv: is an optional file listing nodes of constraint (internal) breaklines for the mesh.
                        The lines file can  consist of open or closed polylines. The columns are shapeid,x,y, where x
                        and y have to match nodes.csv. Every distinct line has to have an individual integer shapeid.
                        If the mesh should not have any constraint lines in the mesh, omit this argument (default: None).
    :param str holes_csv: is an optional file listing nodes of holes in the mesh. The holes file must consist of closed
                        polylines. The columns are shapeid,x,y, where x,y have to match nodes in nodes.csv.
                        Every distinct hole has to have an individual (integer) shapeid. If the mesh has no holes
                        (or islands), keep the default (None).
    :param bool duplicates_flag: optional boolen flag (default: ``True``) to ignore removal of duplicate nodes in
                        nodes.csv. By default, duplicate nodes are removed from nodes.csv.

    :return:
    """

    # find out if the nodes file is x,y,z or x,y,x,size
    with open(nodes_csv, 'r') as f:
        line = next(f)  # read 1 line
        n_attr = len(line.split(','))

    # to create the output file
    fout = open(out_file_name, "w")

    # use numpy to read the file
    # each column in the file is a row in data read by no.loadtxt method
    nodes_data = np.loadtxt(nodes_csv, delimiter=',', skiprows=0, unpack=True)
    boundary_data = np.loadtxt(boundary_csv, delimiter=',', skiprows=0, unpack=True)

    if lines_csv:
        lines_data = np.loadtxt(lines_csv, delimiter=',', skiprows=0, unpack=True)

    if holes_csv:
        holes_data = np.loadtxt(holes_csv, delimiter=',', skiprows=0, unpack=True)

    # master nodes in the file (from the nodes file)
    x = nodes_data[0, :]
    y = nodes_data[1, :]
    z = nodes_data[2, :]
    if n_attr == 4:
        size = nodes_data[3, :]
    else:
        size = np.zeros(len(x))

    # n is the number of nodes
    n = len(x)

    # creates node numbers from the nodes file
    node = np.zeros(n, dtype=np.int32)

    # to check for duplicate nodes
    # this piece of code uses OrderedDict to remove duplicate nodes
    # source "http://stackoverflow.com/questions/12698987"
    # ###################################################################
    tmp = OrderedDict()
    for point in zip(x, y, z, size):
        tmp.setdefault(point[:2], point)

    # in python 3 tmp.values() is a view object that needs to be
    # converted to a list
    mypoints = list(tmp.values())
    n_rev = len(mypoints)

    # replace x,y,z,size and n with their unique equivalents
    if duplicates_flag:
        for i in range(n_rev):
            x[i] = mypoints[i][0]
            y[i] = mypoints[i][1]
            z[i] = mypoints[i][2]
            size[i] = mypoints[i][3]
        n = n_rev

    # if node is part of boundary or lines, then it is not embedded
    is_node_emb = np.zeros(n, dtype=np.int32)
    for i in range(0, n):
        node[i] = i + 1
        is_node_emb[i] = 1

    # boundary data
    shapeid_bnd = boundary_data[0, :]
    x_bnd = boundary_data[1, :]
    y_bnd = boundary_data[2, :]
    # number of nodes in the boundary file
    n_bnd = len(x_bnd)

    # count lines from boundary lines
    count_bnd = 0

    # lines data
    if lines_csv:
        shapeid_lns = lines_data[0, :]
        x_lns = lines_data[1, :]
        y_lns = lines_data[2, :]
        # number of nodes in the lines file
        n_lns = len(x_lns)

    # holes data
    if holes_csv:
        shapeid_hls = holes_data[0, :]
        x_hls = holes_data[1, :]
        y_hls = holes_data[2, :]
        # number of nodes in the holes file
        n_hls = len(x_hls)

    count_lns = 0

    # writes the nodes in gmsh format
    for i in range(0, n):
        fout.write("Point(" + str(i + 1) + ") = {" + str("{:.3f}".format(x[i])) +
                   str(", ") + str("{:.3f}".format(y[i])) + str(", ") +
                   str("{:.3f}".format(z[i])) + str(", ") + str("{:.3f}".format(size[i])) +
                   str("};") + "\n")

    # BOUNDARY LINES
    # index of the minimum, for each boundary node
    minidx = np.zeros(n_bnd, dtype=np.int32)
    # distance between each boundary node and node in the main nodes file
    # initialize to 999
    xdist = np.add(np.zeros(n), 999.0)
    ydist = np.add(np.zeros(n), 999.0)
    dist = np.add(np.zeros(n), 999.0)

    for i in range(0, n_bnd):
        xdist = np.subtract(x, x_bnd[i])
        ydist = np.subtract(y, y_bnd[i])
        dist = np.sqrt(np.power(xdist, 2.0) + np.power(ydist, 2.0))

        # find the minimum of the dist array
        minidx[i] = np.argmin(dist)

        # fill in the is_node_emb array
        is_node_emb[minidx[i]] = 0

    # write the boundary in gmsh format
    for i in range(0, n_bnd - 1):
        if i == 0:
            fout.write("Line(" + str(i + 1) + str(") = {") + str(node[minidx[0]])
                       + str(", ") + str(node[minidx[1]]) + str("};") + "\n")
            count_bnd = count_bnd + 1
        else:
            fout.write("Line(" + str(i + 1) + str(") = {") + str(node[minidx[i]])
                       + str(", ") + str(node[minidx[i + 1]]) + str("};") + "\n")
            count_bnd = count_bnd + 1

    # the lines numbering continues from the boundary numbering
    count_lns = count_bnd + 1

    # CONSTRAINT LINES
    if lines_csv:
        # index for the minimum, for each lines node
        minidx_lns = np.zeros(n_lns, dtype=np.int32)
        # distance between each lines node and node in the master nodes file
        xdist_lns = np.add(np.zeros(n), 999.0)
        ydist_lns = np.add(np.zeros(n), 999.0)
        dist_lns = np.add(np.zeros(n), 999.0)
        for i in range(0, n_lns):
            xdist_lns = np.subtract(x, x_lns[i])
            ydist_lns = np.subtract(y, y_lns[i])
            dist_lns = np.sqrt(np.power(xdist_lns, 2.0) + np.power(ydist_lns, 2.0))

            # find the minimum of the dist array
            minidx_lns[i] = np.argmin(dist_lns)
            # fout.write(str(i) + " " + str(minidx_lns[i]) + "\n")

            # fill in the is_node_emb array
            is_node_emb[minidx_lns[i]] = 0

        cur_lns_shapeid = shapeid_lns[0]
        prev_lns_shapeid = shapeid_lns[0]

        # write the constraint lines
        for i in range(0, n_lns):
            if i > 0:
                cur_lns_shapeid = shapeid_lns[i]
                prev_lns_shapeid = shapeid_lns[i - 1]
                if cur_lns_shapeid - prev_lns_shapeid < 0.001:
                    # fout.write(str(cur_lns_shapeid) + " " + str(prev_lns_shapeid) + " ")
                    fout.write("Line(" + str(count_lns) + str(") = {") +
                               str(node[minidx_lns[i - 1]]) + str(", ") + str(node[minidx_lns[i]]) + str("};") + "\n")
                    count_lns = count_lns + 1

    # holes
    count_hls = count_lns + 1

    hole_nodes = list()
    if holes_csv:
        # index for the minimum, for each lines node
        minidx_hls = np.zeros(n_hls, dtype=np.int32)
        # distance between each lines node and node in the master nodes file
        xdist_hls = np.add(np.zeros(n), 999.0)
        ydist_hls = np.add(np.zeros(n), 999.0)
        dist_hls = np.add(np.zeros(n), 999.0)
        for i in range(0, n_hls):
            xdist_hls = np.subtract(x, x_hls[i])
            ydist_hls = np.subtract(y, y_hls[i])
            dist_hls = np.sqrt(np.power(xdist_hls, 2.0) + np.power(ydist_hls, 2.0))

            # find the minimum of the dist array
            minidx_hls[i] = np.argmin(dist_hls)
            # fout.write(str(i) + " " + str(minidx_hls[i]) + "\n")

            # fill in the is_node_emb array
            is_node_emb[minidx_hls[i]] = 0

        cur_hls_shapeid = shapeid_hls[0]
        prev_hls_shapeid = shapeid_hls[0]

        # write the constraint lines
        for i in range(0, n_hls):
            if (i > 0):
                cur_hls_shapeid = shapeid_hls[i]
                prev_hls_shapeid = shapeid_hls[i - 1]
                if (cur_hls_shapeid - prev_hls_shapeid < 0.001):
                    # fout.write(str(cur_hls_shapeid) + " " + str(prev_hls_shapeid) + " ")
                    hole_nodes.append(count_hls)
                    fout.write("Line(" + str(count_hls) + str(") = {") +
                               str(node[minidx_hls[i - 1]]) + str(", ") + str(node[minidx_hls[i]]) + str("};") + "\n")
                    count_hls = count_hls + 1

    n_holes = len(hole_nodes)

    # writes the line loop and the plane surface
    fout.write("Line Loop(1) = {1:" + str(count_bnd))
    if holes_csv:
        fout.write(', ')
        for i in range(n_holes - 1):
            fout.write(str(hole_nodes[i] * -1) + ", ")
        fout.write(str(-1 * hole_nodes[n_holes - 1]))
    fout.write(str("};") + "\n")

    fout.write("Physical Line(1) = {1:" + str(count_bnd))
    if holes_csv:
        fout.write(', ')
        for i in range(n_holes - 1):
            fout.write(str(hole_nodes[i] * -1) + ", ")
        fout.write(str(-1 * hole_nodes[n_holes - 1]))
    fout.write(str("};") + "\n")

    fout.write("Plane Surface(1) = {1};" + "\n")
    fout.write("Physical Surface(1) = {1};" + "\n")

    if lines_csv:
        # write the embedded lines
        # re-set the count_lns back to what it was before
        count_lns = count_bnd + 1
        for i in range(0, n_lns):
            if i > 0:
                cur_lns_shapeid = shapeid_lns[i]
                prev_lns_shapeid = shapeid_lns[i - 1]
                if cur_lns_shapeid - prev_lns_shapeid < 0.001:
                    # fout.write(str(cur_lns_shapeid) + " " + str(prev_lns_shapeid) + " ")
                    fout.write(str("Line {") + str(count_lns) + "} In Surface {1};" + "\n")
                    count_lns = count_lns + 1

    # if there are embedded nodes, write them to the file
    # embedded nodes should be used in tin applications, not in mesh generation
    for i in range(0, n):
        if is_node_emb[i] == 1:
            fout.write(str("Point {") + str(node[i]) + "} In Surface {1};" + "\n")

    # gmsh option to make sure the elements size extend from boundary
    # write zero when doing a TIN; write one when doing a mesh!
    # fout.write(str("Mesh.CharacteristicLengthExtendFromBoundary = 0;") + "\n")
