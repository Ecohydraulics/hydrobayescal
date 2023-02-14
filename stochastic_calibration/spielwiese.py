import os, sys

import self as self

sys.path.append(os.path.abspath("")+"/stochastic_calibration/")
from usr_defs_TELEMAC import *
input_fn = SCRIPT_DIR + "user-input.xlsx"
input_calib = os.path.abspath("")+ "/calibration-points.csv"

# tm_par_df = read_wb_range(input_fn, TM_RANGE)
# tm_par_dict=dict(zip(tm_par_df[0].to_list(), tm_par_df[1].to_list()))
#
# temp = np.loadtxt(SCRIPT_DIR+ "/parameter_file_test.txt", dtype=str, delimiter=";")
# collocation_points = temp[:, 1:].astype(float)
# n_simulation = collocation_points.shape[0]

# global myx
# myx=1
#
# @log_actions
# def modify():
#     myx = 5
#     logger.info("info message")
#     logger_warn.warning("warn message")
#     logger_error.error("error message")
#
# modify()

#
# modify()
# print(myx)

# for par, bounds in indir_par_dict.items():
#   if not (("TELEMAC" or "GAIA") in par):
#     bounds_tuple = bounds
#     print(par)
#
#   print(bounds)
from functools import partial, wraps

class A:
    def __init__(self):
        #self.real_dec = partial(self.wrap_run, argument=self.run())
        pass

    def wrap_run(self, func):
        print("PRE")
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
        return wrapper
        #print("POST")

    @wraps(wrap_run)
    def run(self):
        print("Run A")
        return True



a = A()
a.run()
