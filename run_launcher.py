#!/usr/bin/env python3
# Class Telemac2d import
import sys
sys.path.append('/home/schwindt/Software/telemac/v8p4r0/examples/telemac2d/donau')
from telapy.api.t2d import Telemac2d

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    comm = None

t2d = Telemac2d('/home/schwindt/Software/telemac/v8p4r0/examples/telemac2d/donau/t2d_donau.cas', comm=comm, stdout=6)
comm.Barrier()

t2d.set_case()

t2d.set('MODEL.RESULTFILE', '/home/schwindt/Software/telemac/v8p4r0/examples/telemac2d/donau/auto-results/resIDX-t2d_donau.slf')

t2d.init_state_default()


comm.Barrier()

t2d.run_all_time_steps()

comm.Barrier()

t2d.finalize()

del(t2d)
