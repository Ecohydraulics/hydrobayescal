"""Minimal use case for running a TelemacModel through HBC algorithms with
the default examples/telemac2d/donau example case. Used for developing the algorithm with
fictitious (imaginary) measurement data.
"""

import os
import os.path

# load calibration package
import HyBayesCal as hbc
#from control_telemac import TelemacModel

# global user parameters
input_worbook_name = "/home/amintvm/modeling/hybayescal/tm-user-input_parameters.xlsx"
#number_cpu = 4

# PUT Telemac specific parameters defined in user_defs_stelemac


def run_telemac():
    """Instantiate a TelemacModel object with the required parameters
    :simulation_name str: path to simulation CAS file
    :return: None
    """
    tm_model = hbc.BalWithGPE(input_worbook_name)
    # tm_model = TelemacModel(
    #     model_dir=simulation_name,
    #     control_file=self.case_file,
    #     tm_xd=tm_xd,
    #     n_processors=N_CPUS
    # )
    tm_model.run_initial_simulations()
    #tm_model.full_model_calibration()


if __name__ == "__main__":
    run_telemac()
