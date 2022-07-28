#!/usr/bin/env python3
#
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 shp2csv.py                            # 
#                                                                       #
# +!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng., functionalized by Sebastian Schwindt
# 
# Date: Nov 1, 2016
# Last Modified: July 22, 2022
from pyshp.shapefile import *


def shp2csv(shapefile, target_csv="converted-shp.csv"):
    """ Takes a shapefile of types POINT, POLYLINE, POLYGON,  POINTZ, POLYLINEZ, or POLYGONZ and converts it to
    pputils CSV file(s). The function creates a nodes.csv file if the input shapefile file contains lines (i.e.,
    is of type POLYLINE, POLYGON, or POLYGONZ). This script uses Joel Lawhead's pyshp, which is available on GitHub at
    https://github.com/GeospatialPython/pyshp

    :param str shapefile: full path to and name of a shapefile (e.g., '/home/project/boundary.shp')
    :param target_csv: name of the target CSV file (default is 'converted-shp.csv'), which is automatically stored in
                        the same directory where the source shapefile lives
    :return: None (saves CSV file)
    """
    working_dir = str(shapefile).strip(str(shapefile).split("/")[-1].split("\\")[-1])

    # create the output file
    fout = open(working_dir + target_csv, "w")

    # read the shapefile using pyshp reader
    sf = Reader(shapefile)

    # these are the type ids for shapefiles
    POINT = 1
    POLYLINE = 3
    POLYGON = 5

    POINTZ = 11
    POLYLINEZ = 13
    POLYGONZ = 15

    # print the type of the first object
    shape_type = sf.shape(0).shapeType

    if shape_type == 1:
        print('Type: ' + 'POINT')
    elif shape_type == 3:
        print('Type: ' + 'POLYLINE')
    elif shape_type == 5:
        print('Type: ' + 'POLYGON')
    elif shape_type == 11:
        print('Type: ' + 'POINTZ')
    elif shape_type == 13:
        print('Type: ' + 'POLYLINEZ')
    elif shape_type == 15:
        print('Type: ' + 'POLYGONZ')
    else:
        print('Unknown type. Exiting!')
        sys.exit(0)

    # write node files too in case of polygons or polylines
    if (shape_type == 3 or shape_type == 5 or shape_type == 13 or shape_type == 15):
        nodes_file = str(working_dir + target_csv).rsplit('.', 1)[0] + '_nodes.csv'
        fout2 = open(nodes_file, 'w')

    # these are the fields
    fields = sf.fields

    # ignore the first (the first one is a dummy)
    fields = fields[1:]

    # number of fields
    nf = len(fields)

    # the list of field names
    field_names = list()

    # assigns the fields to the list of field names
    for i in range(nf):
        field_names.append(fields[i][0])

    print('The shapefile has the following fields:')
    print(field_names)
    print(' ')

    # each shape has a value for each field name
    records = sf.records()

    # shapeid is initialized here
    shapeid = -1

    # set up an iterator (thanks to Joel Lawhead's email!)
    for s in sf.iterShapes():
        # this is just a counter
        shapeid = shapeid + 1

        # if the shapefile is of type POINTZ, field names are not written
        if shape_type == POINTZ:
            xyz = s.points[0]
            xyz.append(s.z[0])
            fout.write(str(xyz[0]) + ',' + str(xyz[1]) + ',' + str(xyz[2]) + '\n')

        elif shape_type == POINT:
            xyz = s.points[0]

            # write all fields for all records in the POINT file
            # write the coordinates
            fout.write(str(xyz[0]) + ',' + str(xyz[1]))

            if len(field_names) > 0:
                for i in range(len(field_names)):
                    fout.write(',' + str(records[shapeid][i]))
                    if i == int(len(field_names) - 1):
                        fout.write('\n')
            else:
                fout.write(',' + str(0.0) + '\n')

        if (shape_type == POLYLINE) or (shape_type == POLYGON):
            xyz = s.points

            # if the data has an attributes, write them in the output file
            # this is useful when processing contours with shapefiles

            for j in range(len(xyz)):
                fout.write(str(shapeid) + ',' + str(xyz[j][0]) + ',' + str(xyz[j][1]))

                # write the field values
                if len(field_names) > 0:
                    for i in range(len(field_names)):
                        fout.write(',' + str(records[shapeid][i]))
                        if i == int(len(field_names) - 1):
                            fout.write('\n')
                else:
                    # if there are not fields, write nothing
                    fout.write('\n')

            # to write the nodes file (fields are written in the nodes file)
            for j in range(len(xyz)):
                fout2.write(str(xyz[j][0]) + ',' + str(xyz[j][1]))
                if len(field_names) > 0:
                    for i in range(len(field_names)):
                        fout2.write(',' + str(records[shapeid][i]))
                        if i == int(len(field_names) - 1):
                            fout2.write('\n')
                else:
                    # if there are no fields, write 0.0
                    fout2.write(',' + str(0.0) + '\n')

        if (shape_type == POLYLINEZ) or (shape_type == POLYGONZ):
            xyz = s.points

            # polylineZ and polygonZ shapefiles are assumed not to have fields
            for j in range(len(xyz)):
                fout.write(str(shapeid) + ',' + str(xyz[j][0]) + ',' + \
                           str(xyz[j][1]) + ',' + str(s.z[j]) + '\n')

                fout2.write(str(xyz[j][0]) + ',' + str(xyz[j][1]) + ',' + \
                            str(s.z[j]) + '\n')
