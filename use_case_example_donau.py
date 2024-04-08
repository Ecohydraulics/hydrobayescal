"""Minimal use case for running a TelemacModel through HBC algorithms with
the default examples/telemac2d/donau example case. Used for developing the algorithm with
fictitious (imaginary) measurement data.
"""

import os
import os.path

# load calibration package
import HyBayesCal as hbc

# global constant parameters
#this_dir = os.path.abspath(".")

# global user parameters
input_worbook_name = "/home/amintvm/modeling/hybayescal/tm-user-input_parameters.xlsx"

def run_telemac():
    """Instantiate a TelemacModel object with the required parameters

    :return: None
    """
    tm_model = hbc.BalWithGPE(input_worbook_name)
    tm_model.run_initial_simulations()


if __name__ == "__main__":
    run_telemac()
