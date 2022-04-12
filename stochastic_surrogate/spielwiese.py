import os, sys
sys.path.append(os.path.abspath("")+"/stochastic_surrogate/")
from usr_defs import *

input_fn = SCRIPT_DIR + "user-input.xlsx"

#myd = read_wb_range(input_fn, PRIOR_DIR_RANGE)

x = str()

def modify():
  global x
  x = 'modified'

modify()

print(x)