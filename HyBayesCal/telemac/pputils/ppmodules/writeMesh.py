"""
pputils functions for writing meshes, re-formatted and optimized for Python3
Original Author: Pad Prodanovic
Modularized by: Sebastian Schwindt
"""


def writeAdcirc(n, e, x, y, z, ikle, name):
    """
    Write an adcirc grd mesh assuming that the indices in the ikle array are zero-based.
    Recall: Telemac uses one-based ikle arrays.

    :param n:
    :param e:
    :param x:
    :param y:
    :param z:
    :param ikle:
    :param str name: name of the adcirc output file (must end on ".grd")
    :return: None
    """

    # write the output file where the name argument is the name of the output adcirc file
    fout = open(name, "w")

    # write the adcirc mesh file
    fout.write("ADCIRC" + "\n")
    # write the number of elements and number of nodes to the file header
    fout.write(str(e) + " " + str(n) + "\n")

    # write nodes
    for i in range(n):
        fout.write(str(i + 1) + " " + str("{:.3f}".format(x[i])) + " " +
                   str("{:.3f}".format(y[i])) + " " + str("{:.3f}".format(z[i])) + "\n")

    # write the elements
    # the readAdcirc function assigns the ikle starting at zero, so that is why we have to add 1
    for i in range(e):
        fout.write(str(i + 1) + " 3 " + str(ikle[i, 0] + 1) + " " + str(ikle[i, 1] + 1) +
                   " " + str(ikle[i, 2] + 1) + "\n")

    # close the fout file
    fout.close()

    return None


def write2dm(n, e, x, y, z, ikle, name):
    """
    Write an SMS 2dm mesh assuming that the indices in the ikle array are zero-based.
    Recall: Telemac uses one-based ikle arrays.

    :param n:
    :param e:
    :param x:
    :param y:
    :param z:
    :param ikle:
    :param str name: name of the 2dm output file (must end on ".2dm")
    :return: None
    """
    # start output file, where the name argument is the name of the output 2dm file
    fout = open(name, "w")

    # write the mesh file
    fout.write("MESH2D" + "\n")

    # write the elements, where the n,e,x,y,z,ikle are zero-based,
    # so we add 1 to make it 1 based
    for i in range(e):
        fout.write("E3T " + str(i + 1) + " " + str(ikle[i, 0] + 1) + " " + str(ikle[i, 1] + 1) +
                   " " + str(ikle[i, 2] + 1) + " 1" + "\n")

    # write nodes
    for i in range(n):
        fout.write("ND " + str(i + 1) + " " + str("{:.3f}".format(x[i])) + " " +
                   str("{:.3f}".format(y[i])) + " " + str("{:.3f}".format(z[i])) + "\n")

    # close the fout file
    fout.close()

    return None


def writeVTKscalar(n, e, x, y, z, ikle, fname, vname):
    """
    Write a VTK scalar file.

    :param n:
    :param e:
    :param x:
    :param y:
    :param z:
    :param ikle:
    :param str fname: name of the 2dm output file (must end on ".2dm")
    :param str vname: name of the scalar variable to write
    :return: None
    """
    # write the file output
    fout = open(fname, "w")

    # vtk header file
    fout.write("# vtk DataFile Version 3.0" + "\n")
    fout.write("Created with modularized pputils" + "\n")
    fout.write("ASCII" + "\n")
    fout.write("" + "\n")
    fout.write("DATASET UNSTRUCTURED_GRID" + "\n")
    fout.write("POINTS " + str(len(x)) + " float" + "\n")

    # write node coordinates
    for i in range(len(x)):
        fout.write(str("{:.3f}".format(x[i])) + " " +
                   str("{:.3f}".format(y[i])) + " " + str("{:.3f}".format(0.0)) +
                   "\n")

    # write node connectivity table
    fout.write("CELLS " + str(len(ikle)) + " " + str(len(ikle) * 4) + "\n")

    for i in range(len(ikle)):
        fout.write("3 " + str(ikle[i][0]) + " " + str(ikle[i][1]) + " " +
                   str(ikle[i][2]) + "\n")

    # write cell types
    fout.write("CELL_TYPES " + str(len(ikle)) + "\n")
    for i in range(len(ikle)):
        fout.write("5" + "\n")

    # write empty line
    fout.write("" + "\n")

    # write data
    fout.write("POINT_DATA " + str(len(x)) + "\n")

    # write the z as scalar data also
    fout.write("SCALARS " + vname + "\n")
    fout.write("float" + "\n")
    fout.write("LOOKUP_TABLE default" + "\n")
    for i in range(len(x)):
        fout.write(str("{:.3f}".format(z[i])) + "\n")

    fout.close()

    return None
