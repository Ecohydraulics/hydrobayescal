"""Minimal use case for running a TelemacModel through HBC algorithms with
the default examples/telemac2d/donau example case. Used for developing the algorithm with
fictitious (imaginary) measurement data.
"""

import os
import os.path

# load calibration package
# import HyBayesCal as hbc
from hyBayesCal import TelemacModel

# global constant parameters
this_dir = os.path.abspath(".")

# global user parameters
tm_model_dir = os.path.join(this_dir, "examples/donau")
#cas_file = "t2d-donau-const-n-FV.cas"
cas_file = "t2d-donau-const-n.cas"


def run_telemac():
    """Instantiate a TelemacModel object with the minimally required parameters

    :return: None
    """
    tm_model = hbc.TelemacModel(
        model_dir=tm_model_dir,
        control_file=cas_file,
        tm_xd="Telemac2d",
        n_processors=8,
    )

    tm_model.run_simulation(load_results=False)


if __name__ == "__main__":
    run_telemac()
