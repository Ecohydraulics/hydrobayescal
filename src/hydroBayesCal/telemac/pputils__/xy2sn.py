#!/usr/bin/env python3
#
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                       xy2sn.py                                        # 
#                                                                       #
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng.
# 
# Date: Jan 25, 2020
#
# Purpose: Takes in surveyed points pts, a river centerline cl, and a boundary
# polygon bnd (representing the area for which interpolation is to take place)
# in xy space and converts the same to sn space. The script takes three inputs
# in xy space, and produces three outputs in sn space.
#
# Uses: Python 2 or 3, Matplotlib, Numpy, Scipy
#
# Example:
#
# python xy2sn.py -p1 pts_xy.csv -c1 cl_xy.csv -b1 bnd_xy.csv
#                 -p2 pts_sn.csv -c2 cl_sn.csv -b2 bnd_sn.csv
#
# where:
# -p1, p2 --> points in xy (p1) and sn (p2) space 
# -c1, c2 --> centerline in xy (c1) and sn (c2) space
# -b1, b2 --> boundary polygon in xy (b1) and sn (b2) space
#
# Modified: Oct 27, 2021
# Got rid of the dependency progressbar, and replaced it with a simpler
# progress bar that is one function only.
#
# Modified: Jun 7, 2023
# Got rid of the dependency on external progress bar. Now everything
# works with internal methods.
#
# Modified: Jul 8, 2023
# Merged the original code to PPUTILS, and did some code refactoring. 
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from ppmodules.utilities import progress   # import the simple progress bar

# computes the magnitude between two points
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
if len(sys.argv) != 13 :
  print('Wrong number of Arguments, stopping now...')
  print('Usage:')
  print('python xy2sn.py -p1 pts_xy.csv -c1 cl_xy.csv -b1 bnd_xy.csv')
  print('                -p2 pts_sn.csv -c2 cl_sn.csv -b2 bnd_sn.csv')
  sys.exit()

# these are the input files
pts_xy_file = sys.argv[2]
cl_xy_file = sys.argv[4]
bnd_xy_file = sys.argv[6]

# these are the output files in sn space
pts_sn_file = sys.argv[8]
cl_sn_file = sys.argv[10]
bnd_sn_file = sys.argv[12]

# create the actual outputs
fout_pts_sn = open(pts_sn_file, 'w')
fout_cl_sn = open(cl_sn_file, 'w')
fout_bnd_sn = open(bnd_sn_file, 'w')

# read the input files using numpy
pts_xy = np.loadtxt(pts_xy_file, delimiter=',',skiprows=0,unpack=True)
cl_xy = np.loadtxt(cl_xy_file, delimiter=',',skiprows=0,unpack=True)
bnd_xy = np.loadtxt(bnd_xy_file, delimiter=',',skiprows=0,unpack=True)

# store the data read into variables
points_x = pts_xy[0,:]
points_y = pts_xy[1,:]
points_z = pts_xy[2,:]

centerline_x = cl_xy[1,:]
centerline_y = cl_xy[2,:]

boundary_x = bnd_xy[1,:]
boundary_y = bnd_xy[2,:]

# ######################################################################
# to plot it in xy space

# let's verify by plotting
# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
# plt.gca().set_aspect('equal', adjustable='box')

# plt.plot(centerline_x,centerline_y, 'o-', label='Centerline')
# plt.plot(points_x, points_y, 'o', label='Points') 
# plt.plot(boundary_x,boundary_y, 'k-', label='Boundary')
# plt.show()
# 
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

  progress(i, n_segments-1)
print('')

# to write the centerline as a pputils lines file
for i in range(len(centerline_x)):
  fout_cl_sn.write('0,' + str(centerline_s[i]) + ',0.0' + '\n')
fout_cl_sn.close()

# ######################################################################
# now we have to convert the points to the n-s space

# number of points
n_points = len(points_x)

# the s-n coordinates of the points_cr
points_s = np.zeros(n_points)
points_n = np.zeros(n_points)

# this is the segment no. (index starts at zero) that is closest to 
# the point
target_segment = 0

print(' ')
print('Converting points from x-y to s-n space ...')

for i in range(n_points):

  # to have smaller variable names
  ptx = points_x[i] 
  pty = points_y[i]

  # min dist set to a large number
  mindist = 1.0E6

  # mininum intesecting point
  int_ptx_min = 1.0E6
  int_pty_min = 1.0E6

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
  points_s[i] = tempdist + centerline_s[target_segment]

  # for n-coordinate, it is the distance between the point on the segment
  # on the centerline and the coordinate

  # this is the magnitude, it tells us nothing about the sign of the n-coord
  points_n[i] = magnitude(ptx,pty,int_ptx_min,int_pty_min)

  # determine the sign of points
  mult = is_point_above_segment(ptx,pty,
    centerline_x[target_segment],centerline_y[target_segment],
    centerline_x[target_segment+1],centerline_y[target_segment+1])

  if (mult == 0):
    points_n[i] = 0.0
  else:
    points_n[i] = mult * points_n[i]

    # reset these before next loop
  target_segment = 0
  mindist = 1.0E6

  progress(i, n_points-1)
print('')

# to write the points as a pputils lines file
for i in range(len(points_x)):
  fout_pts_sn.write(str(points_s[i]) + ',' + str(points_n[i])
  + ',' + str(points_z[i]) + '\n')
fout_pts_sn.close()

# ######################################################################
# now we will transform the boundary to s-n space recycling the same
# code as above, except now we use the boundary points

# the s-n coordinates of the points_cr
boundary_s = np.zeros(len(boundary_x))
boundary_n = np.zeros(len(boundary_x))

# this is the segment no. (index starts at zero) that is closest 
# to the point
target_segment = 0

print(' ')
print('Converting boundary from x-y to s-n space ...')

for i in range(len(boundary_x)):

  # to have smaller variable names
  ptx = boundary_x[i] 
  pty = boundary_y[i]

  # min dist set to a large number
  mindist = 1.0E6

  # mininum intesecting point
  int_ptx_min = 1.0E6
  int_pty_min = 1.0E6

  #print('point ' + str(i))
  #print('{:.3f}'.format(ptx) + ',' + '{:.3f}'.format(pty))
  
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
  boundary_s[i] = tempdist + centerline_s[target_segment]

  # for n-coordinate, it is the distance between the point on the segment 
  # on the centerline and the coordinate

  # this is the magnitude, it tells us nothing about the sign of the n-coord
  boundary_n[i] = magnitude(ptx,pty,int_ptx_min,int_pty_min)

  # determine the sign of points
  mult = is_point_above_segment(ptx,pty,
    centerline_x[target_segment],centerline_y[target_segment],
    centerline_x[target_segment+1],centerline_y[target_segment+1])

  if (mult == 0):
    boundary_n[i] = 0.0
  else:
    boundary_n[i] = mult * boundary_n[i]

  # reset these before next loop
  target_segment = 0
  mindist = 1.0E6

  progress(i, len(boundary_x)-1)
print('')

# to write the boundary in sn space as a pputils file
for i in range(len(boundary_x)):
  fout_bnd_sn.write('0,' + str(boundary_s[i]) 
    + ',' + str(boundary_n[i]) + '\n')
fout_bnd_sn.close()

# ######################################################################
# to plot it in sn space

# let's verify by plotting
# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
# plt.gca().set_aspect('equal', adjustable='box')
# 
# plt.plot(centerline_s,centerline_n, 'o-', label='Centerline')
# plt.plot(points_s, points_n, 'o', label='Points') 
# plt.plot(boundary_s,boundary_n, 'k-', label='Boundary')
# plt.show()

# ######################################################################
# now that we have it all as pputils formatted files, let's write it as
# WKT files for easier loading in GIS

# I made this work when the paths are sources, and also when they are not

curdir = os.getcwd()
#
try:
  # this only works when the paths are sourced!
  pputils_path = os.environ['PPUTILS']
except:
  pputils_path = curdir
  
  # this is to maintain legacy support
  if (sys.version_info > (3, 0)):
    version = 3
    pystr = 'python3'
  elif (sys.version_info > (2, 7)):
    version = 2
    pystr = 'python'

# for the centerline file
wkt_cl_sn_file = cl_sn_file.rsplit('.',1)[0] + '_WKT.csv'
print('Writing centerline in sn space as a wkt file ...')
try:
  # this only works when the paths are sourced!
  pputils_path = os.environ['PPUTILS']
  subprocess.call(['breaklines2wkt.py', '-i', cl_sn_file,
    '-o', wkt_cl_sn_file])
except:
  subprocess.call([pystr, 'breaklines2wkt.py', '-i', cl_sn_file,
    '-o', wkt_cl_sn_file])

# for the boundary file
wkt_bnd_sn_file = bnd_sn_file.rsplit('.',1)[0] + '_WKT.csv'
print('Writing boundary in sn space as a wkt file ...')
try:
  # this only works when the paths are sourced!
  pputils_path = os.environ['PPUTILS']
  subprocess.call(['breaklines2wkt.py', '-i', bnd_sn_file,
    '-o', wkt_bnd_sn_file])
except:
  subprocess.call([pystr, 'breaklines2wkt.py', '-i', bnd_sn_file,
    '-o', wkt_bnd_sn_file])

# for the points file
wkt_pts_sn_file = pts_sn_file.rsplit('.',1)[0] + '_WKT.csv'
print('Writing points in sn space as a wkt file ...')
fout_wkt_pts_sn = open(wkt_pts_sn_file, 'w')

# write the header of the WKT file
fout_wkt_pts_sn.write('WKT,node' + '\n')

for i in range(len(points_s)):
  fout_wkt_pts_sn.write('"POINT (')
  fout_wkt_pts_sn.write(str('{:.3f}'.format(points_s[i] )) + ' ' + 
    str('{:.3f}'.format(points_n[i] )) + ' ' + 
    str('{:.3f}'.format(points_z[i] )) + ')",' + 
    str(i+1) + '\n')

# ######################################################################
  
print('All done!')
