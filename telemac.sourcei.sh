# This file is a template for a Linux environment file
# running "source pysource.openmpi.sh" will position all
# the necessary environment variables for telemac
# To adapt to your installation replace word <word> by their local value
###
### TELEMAC settings -----------------------------------------------------------
###
# Path to telemac root dir
export HOMETEL=/home/schwindt/Software/telemac/v8p4r0
# Adding python scripts to PATH
export PATH=$HOMETEL/scripts/python3:.:$PATH
# Configuration file
export SYSTELCFG=$HOMETEL/configs/systel.cis-mint.cfg
# Name of the configuration to use
export USETELCFG=debgfopenmpi
# Path to this file
export SOURCEFILE=$HOMETEL/configs/pysource.openmpi.sh
### Python
# To force python to flush its output
export PYTHONUNBUFFERED='true'
### API
export PYTHONPATH=$HOMETEL/scripts/python3:$PYTHONPATH
export LD_LIBRARY_PATH=$HOMETEL/builds/$USETELCFG/wrap_api/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOMETEL/builds/$USETELCFG/wrap_api/lib:$PYTHONPATH
###
### COMPILERS -----------------------------------------------------------
###
# Here are a few examples for external libraries
export SYSTEL=$HOMETEL/optionals
### MPI -----------------------------------------------------------
export MPIHOME=/usr/bin/mpifort.mpich
export PATH=/usr/lib/x86_64-linux-gnu/openmpi:$PATH
export LD_LIBRARY_PATH=$PATH/lib:$LD_LIBRARY_PATH
###
### EXTERNAL LIBRARIES -----------------------------------------------------------
###
### HDF5 -----------------------------------------------------------
export HDF5HOME=$SYSTEL/hdf5
export LD_LIBRARY_PATH=$HDF5HOME/lib:$LD_LIBRARY_PATH
export LD_RUN_PATH=$HDF5HOME/lib:$MEDHOME/lib:$LD_RUN_PATH
### MED  -----------------------------------------------------------
export MEDHOME=$SYSTEL/med-4.0.0
export LD_LIBRARY_PATH=$MEDHOME/lib:$LD_LIBRARY_PATH
export PATH=$MEDHOME/bin:$PATH
### MUMPS -------------------------------------------------------------
#export MUMPSHOME=$SYSTEL/LIBRARY/mumps/gnu
#export SCALAPACKHOME=$SYSTEL/LIBRARY/scalapack/gnu
#export BLACSHOME=$SYSTEL/LIBRARY/blacs/gnu
### METIS -------------------------------------------------------------
export METISHOME=$SYSTEL/metis-5.1.0/build/
export LD_LIBRARY_PATH=$METISHOME/lib:$LD_LIBRARY_PATH
### AED ---------------------------------------------------------------
# export AEDHOME=$SYSTEL/aed2
# export LD_LIBRARY_PATH=$AEDHOME/obj:$LD_LIBRARY_PATH
#
