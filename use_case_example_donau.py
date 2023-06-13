"""Minimal use case for running a TelemacModel through HBC algorithms with
the default examples/telemac2d/donau example case. Used for developing the algorithm with
fictitious (imaginary) measurement data.
"""

import os
import os.path

# load calibration package
import HyBayesCal as hbc

# global constant parameters
this_dir = os.path.abspath(os.path.join(__file__, ".."))

# global user parameters
tm_model_dir = "/home/schwindt/Software/telemac/v8p4r0/examples/telemac2d/donau"
cas_file = "t2d_donau.cas"


def run_telemac():
    """Instantiate a TelemacModel object with the minimally required parameters

    :return: None
    """
    tm_model = hbc.TelemacModel(
        model_dir=tm_model_dir,
        control_file=cas_file,
        tm_xd="Telemac2d",
        n_processors=4,
    )

    tm_model.run_simulation(load_results=True)


if __name__ == "__main__":
    run_telemac()
