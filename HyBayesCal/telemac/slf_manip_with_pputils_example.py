import numpy as np
import csv

#Activate PPUTILS
import sys
from os import path
sys.path.append(path.join( path.dirname(sys.argv[0]),'D:\\TELEMAC\\WIND_TXY_WRF\\pputils-master'))# put your pputils adress here
from ppmodules.selafin_io_pp import *

import pandas as pd

#reading the csv file containing rainfall values for all the 19 time steps",
df=pd.read_csv("data_t19.csv", header=None)
ar_19t= df.to_numpy().transpose()

# read *.slf mesh geo file
slf = ppSELAFIN('Raini_15_12.slf')#bts3d.slf result_2d-verao5.slf
slf.readHeader()
slf.readTimes()

# store into arrays
times = slf.getTimes()
vnames = slf.getVarNames()
vunits = slf.getVarUnits()
float_type,float_size = slf.getPrecision()
NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = slf.getMesh()

# read the variables for the time step 0
slf.readVariables(0)
results = slf.getVarValues()

# write *.slf file
slf2 = ppSELAFIN('file_to_be_written.slf')
slf2.setPrecision(float_type,float_size)
slf2.setTitle('created with pputils')
slf2.setVarNames(vnames)
slf2.setVarUnits(vunits)

slf2.setIPARAM([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
slf2.setMesh(NELEM, NPOIN, NDP, IKLE, IPOBO, x, y)
slf2.writeHeader()

vunits.clear()
vunits.append('mm')

times.clear()
times.extend(range(0, 19))

vnames.clear()
vnames.append('RAINI')

# read the variables for the time step 0
slf.readVariables(0)
results = slf.getVarValues()

timear19t=np.arange(0,19*3600,3600)#suposed 3600 s or 1hour time step for your data

for t in range(timear19t.shape[0]):
    rain=ar_19t[t]
    #vsel=griddata((lonx, laty), v, (x,y),method='nearest')
     
    slf2.writeVariables(timear19t[t], np.array([rain]))
    
slf2.close()
slf.close() 

# read results for checking
slf = ppSELAFIN('file_to_be_written.slf')#
slf.readHeader()
slf.readTimes()

# store into arrays
vnames = slf.getVarNames()
vunits = slf.getVarUnits()

# read the variables for the time step 0 to 5
for i in range(5):
    slf.readVariables(1)
    results = slf.getVarValues()
    print (results)
        
