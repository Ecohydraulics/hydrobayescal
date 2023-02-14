"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with any Model.
Any other model should use this template class and rewrite the minimum functions given in the
FullComplexityModel class, adapted for a particular model.

For example, control_TELEMAC.py contains the TelemacModel class that inherits from the here defined
FullComplexityModel class.
"""

import os as _os
import numpy as _np
from datetime import datetime
from basic_functions import *


class FullComplexityModel:
    def __init__(
            self,
            model_dir="",
            calibration_parameters=None,
            control_file="tm.cas",
            *args,
            **kwargs
    ):
        """
        Constructor for the FullComplexityModel Class. Instantiating can take some seconds, so try to
        be efficient in creating objects of this class (i.e., avoid re-creating a new FullComplexityModel in long loops)

        :param str model_dir: directory (path) of the Telemac model (should NOT end on "/" or "\\")
        :param list calibration_parameters: computationally optional, but in the framework of Bayesian calibration,
                    this argument must be provided
        :param str control_file: name of the model control file to be used (e.g., Telemac: cas file); do not include directory
        :param args:
        :param kwargs:
        """
        self.model_dir = _os.path.abspath(model_dir)
        self.control_file = control_file
        self.res_dir = self.model_dir + _os.sep + "auto-results"
        if not _os.path.exists(self.res_dir):
            _os.makedirs(self.res_dir)

        self.__calibration_parameters = False
        if calibration_parameters:
            self.__setattr__("calibration_parameters", calibration_parameters)

    def __setattr__(self, name, value):
        if name == "calibration_parameters":
            # value corresponds to a list of parameters
            self.calibration_parameters = {}
            for par in value:
                self.calibration_parameters.update({par: {"current value": _np.nan}})

    def update_model_controls(
            self,
            new_parameter_values,
            simulation_id=0,
    ):
        """
        Update the model control files specifically for Bayesian calibration.

        :param dict new_parameter_values: provide a new parameter value for every calibration parameter
                    * keys correspond to Telemac or Gaia keywords in the steering file
                    * values are either scalar or list-like numpy arrays
        :param int simulation_id: optionally set an identifier for a simulation (default is 0)
        :return:
        """

        pass

    def run_simulation(self):
        """
        Run a full-complexity model simulation

        :return None:
        """
        start_time = datetime.now()
        # implement call to run the model from command line, for example:
        # call_subroutine("openTeleFoam " + self.control_file)
        print("Full-complexity simulation time: " + str(datetime.now() - start_time))

    def __call__(self, *args, **kwargs):
        """
        Call method forwards to self.run_simulation()

        :param args:
        :param kwargs:
        :return:
        """
        self.run_simulation()
