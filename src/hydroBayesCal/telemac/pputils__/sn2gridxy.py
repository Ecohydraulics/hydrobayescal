#!/usr/bin/env python3
#
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                       sn2gridxy.py                               # 
#                                                                       #
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng.
# 
# Date: Jan 25, 2020
#
# Purpose: Takes in a river centerline cl (xy space), an xyz point file 
# (xy space) that is clipped to the river boundary, and a tin (sn space). 
# The tin is assumed to have been previously created post-processing
# xy2sn.py output). The output generated is a grid of the interpolated 
# points in xy space. This script assumes that the user first creates
# converts the survey points to from xy to sn space. Then, in a GIS
# program the user adds the tin boundary and tin breaklines (in sn space), 
# and ultimately create a tin in sn space. It is that tin in sn space 
# that is used to assign elevations to the grid (xy space) that is
# generated in this script.
#
# Note the centerline in this script must be identical to one used to generate
# the tin is sn space!!!
#
# Uses: Python 2 or 3, Matplotlib, Numpy, Scipy
#
# Example:
#
# python3 sn2gridxy_rev1.py -i ingrid_xy.csv -c cl_xy.csv -t tin_sn.csv -o outgrid_xy.csv
#
# where:
# -i --> input grid (as pputils points file)
# -c --> centerline in xy space
# -t --> tin in sn space (generated after processing xy2sn.py script)
# -o --> output grid with its points interpolated
#
# Modified: Jul 10, 2021
# This version is decorated with numba, which speeds up computations greatly.
# The inputs were changed, so there is no cropping with matplotlib anymore.
#
# Modified: Oct 10, 2021
# Added a simple progress bar (replaced progress bar that had an external
# dependency). Also decorated the computational functions with numba to 
# speed up the calulations. 
# 
# Modified: Jun 7, 2023
# Got rid of the dependency on external progress bar. Now everything
# works with internal methods.
#
# Modified: Jul 8, 2023
# Merged the original code to PPUTILS, and did some code refactoring. 
# Note that this script has the dependency on python3-numba.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os,sys
import numpy as np
import matplotlib.tri as mtri
from numba import jit
from ppmodules.utilities import progress   # import the simple progress bar
from ppmodules.readMesh import readAdcirc   # import the readAdcirc function

# computes the magnitude between two points
@jit(nopython=True)
def magnitude(x1,y1,x2,y2):
  xdist = x2 - x1
  ydist = y2 - y1
  return np.sqrt(np.power(xdist,2.0) + np.power(ydist,2.0))
    
# computes the shortest distance between a point (px,py) and a line
# contained between points (startx,starty) and (endx,endy)

# the method computes the shortest distance between a point and a line;
# the shortest distance is a line perpendicular from somewhere on
# the line segment to the point; in case the point tested is beyond the
# line segment, the method will just return the closest point on the segment
@jit(nopython=True)
def pp_intersect_point_to_line(px,py,startx,starty,endx,endy):

  # assume at the begining that the point px,py is on the
  # line segment (startx,starty,endx,endy)
  is_on_segment = True

  line_magnitude = magnitude(startx,starty,endx,endy)

  u = ((px - startx) * (endx - startx) +
       (py - starty) * (endy - starty)) / (line_magnitude ** 2)

  # closest point does not fall within the line segment, 
  # take the shorter distance to an endpoint
  if u < 0.00001 or u > 1:
    is_on_segment = False
    ix = magnitude(px,py, startx,starty)
    iy = magnitude(px,py, endx,endy)
    if ix > iy:
      return (endx,endy,is_on_segment)
    else:
      return (startx,starty,is_on_segment)
  else:
    is_on_segment = True
    ix = startx + u * (endx - startx)
    iy = starty + u * (endy - starty)
    
  return (ix,iy,is_on_segment)

# a simple method to check if a particular point is above or below a line
# the way I have it here, to the left is defined as +ve, and to the right
# is defined as -ve (as the observer is standing at the start of the segment
# and is looking towards the end of the segment)
@jit(nopython=True)
def is_point_above_segment(px,py,startx,starty,endx,endy):
  v1 = (endx-startx, endy-starty)
  v2 = (endx-px, endy-py)
  xp = v1[0]*v2[1] - v1[1]*v2[0]

  if xp > 0:
    return -1
  elif xp < 0:
    return 1
  else:
    return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I/O
if len(sys.argv) != 9 :
  print('Wrong number of Arguments, stopping now...')
  print('Usage:')
  print('python3 sn2gridxy_rev1.py -i ingrid_xy.csv -c cl_xy.csv -t tin_sn.csv -o outgrid_xy.csv')
  sys.exit()

print(' ')
print('Reading input data ...')

# these are the input/output file names read from the prompt
ingrid_xy_file = sys.argv[2]
cl_xy_file = sys.argv[4]
tin_sn_file = sys.argv[6]
grid_xy_file = sys.argv[8]

# output grid
fout_grid_xy = open(grid_xy_file, 'w')

# read the input files using numpy
cl_xy = np.loadtxt(cl_xy_file, delimiter=',',skiprows=0,unpack=True)
ingrid_xy = np.loadtxt(ingrid_xy_file, delimiter=',',skiprows=0,unpack=True)

# now we need to read the tin that is in sn space using the read_adcirc function
tin_n,tin_e,tin_x,tin_y,tin_z,tin_ikle = readAdcirc(tin_sn_file)

centerline_x = cl_xy[1,:]
centerline_y = cl_xy[2,:]

ingrid_x = ingrid_xy [0,:]
ingrid_y = ingrid_xy [1,:]

# ######################################################################
# transform centerline to s-n coordinate system

# n is the number of vertices on the centerline
n = len(centerline_x)

# coordinates of the centerline are in s-n system
centerline_s = np.zeros(n) 
centerline_n = np.zeros(n) # by definition

# start of the centerline_spline_s is zero by definition
centerline_s[0] = 0.0

# this is the total length of the centerline (i.e., the max s coord)
s_length = 0.0

# this is how many segments on the spline we have
n_segments = n-1

# individual segment lengths
centerline_segment_lengths = np.zeros(n_segments)

print(' ')
print('Converting centerline from x-y to s-n space ...')

# go though each coordinate on the centerline and compute segment lengths
for i in range(n_segments):
  xdist = centerline_x[i+1] - centerline_x[i]
  ydist = centerline_y[i+1] - centerline_y[i]

  dist = np.sqrt(np.power(xdist,2.0) + np.power(ydist,2.0))  
  s_length = s_length + dist
  centerline_segment_lengths[i] = s_length
  
  # it's i+1 because these are centerline_spline_s is a coordinates array
  centerline_s[i+1] = s_length

# ######################################################################
# now we read a grid of un-interpolated points (that are cropped to our 
# boundary

# to accomodate code pasting
xreg_1d_in = ingrid_x
yreg_1d_in = ingrid_y
zreg_1d_in = np.zeros(len(ingrid_x))

# ######################################################################
# now the task is to convert the grid to s-n space using only the points
# that are inside the boundary, as these are the only points that need
# to be assigned an elevation value

# the s-n coordinates of the grid
sreg_1d_in = np.zeros(len(xreg_1d_in))
nreg_1d_in = np.zeros(len(xreg_1d_in))

# this is the segment no. (index starts at zero) that is closest to the point
target_segment = 0

#w = [Percentage(), Bar(), ETA()]
#pbar = ProgressBar(widgets=w, maxval=len(zreg_1d_in)).start()

print(' ')
print('Converting input grid from x-y to s-n space ...')

for i in range(len(sreg_1d_in)):

  # to have smaller variable names
  ptx = xreg_1d_in[i] 
  pty = yreg_1d_in[i]

  # min dist set to a large number
  mindist = 1.0E6

  # mininum intesecting point
  int_ptx_min = 1.0E6
  int_pty_min = 1.0E6

  #print('point ' + str(i))
  #print('{:.3f}'.format(ptx) + ',' + '{:.3f}'.format(pty))

  # this is how I had it before
  for j in range(n_segments):
    
    # to have smaller variable names
    startx = centerline_x[j]
    starty = centerline_y[j]
    endx = centerline_x[j+1]
    endy = centerline_y[j+1]
  
    # find the closent distance from the point to the segment
    int_ptx, int_pty, is_on_segment = pp_intersect_point_to_line(
      ptx,pty,startx,starty,endx,endy)
  
    # now find the minimum distance from the ptx,pty and int_ptx, int_pty
    # whether it is on the segment or not
    
    tempdist = magnitude(ptx,pty,int_ptx,int_pty)
    if (tempdist < mindist):
      mindist = tempdist
      int_ptx_min = int_ptx
      int_pty_min = int_pty
      target_segment = j
  
  # find distance from the start of the target segment to the int_ptx_min,int_pty_min
  tempdist = magnitude(centerline_x[target_segment],
    centerline_y[target_segment], int_ptx_min,int_pty_min)

  # we can find the s,n coordinates of the survey points
  # for s-coordinates, it is the tempdist and the cummulative distance 
  sreg_1d_in[i] = tempdist + centerline_s[target_segment]

  # for n-coordinate, it is the distance between the point on the segment on the
  # centerline and the coordinate

  # this is the magnitude, it tells us nothing about the sign of the n-coord
  nreg_1d_in[i] = magnitude(ptx,pty,int_ptx_min,int_pty_min)

  # determine the sign of points
  mult = is_point_above_segment(ptx,pty,
    centerline_x[target_segment],centerline_y[target_segment],
    centerline_x[target_segment+1],centerline_y[target_segment+1])

  if (mult == 0):
    nreg_1d_in[i] = 0.0
  else:
    nreg_1d_in[i] = mult * nreg_1d_in[i]

  # reset these before next loop
  target_segment = 0
  mindist = 1.0E6
  progress(i, len(sreg_1d_in))

print('')

# ######################################################################
# now we can carry out the interpolations in the s-n space using the
# tin that is in the sn space

# create tin triangulation object using matplotlib
tin = mtri.Triangulation(tin_x, tin_y, tin_ikle)

# to perform the triangulation
interpolator = mtri.LinearTriInterpolator(tin, tin_z)
zreg_1d_in = interpolator(sreg_1d_in,nreg_1d_in)

# if the node is outside of the boundary of the domain, assign value -999.0
# as the interpolated node
where_are_NaNs = np.isnan(zreg_1d_in)
zreg_1d_in[where_are_NaNs] = -999.0

# ######################################################################
# now we are ready to output the outgrid

print(' ')
print('Writing output grid in x-y space ...')

# {:10.3f}'.format(ks[2])

for i in range(len(zreg_1d_in)):
  if (zreg_1d_in[i] > -999.0):
    fout_grid_xy.write('{:.3f}'.format(xreg_1d_in[i]) + ',' + '{:.3f}'.format(yreg_1d_in[i]) + ',' + '{:.3f}'.format(zreg_1d_in[i]) + '\n')
fout_grid_xy.close()

