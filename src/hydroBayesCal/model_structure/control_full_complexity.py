"""
Functional core for coupling the Surrogate-Assisted Bayesian inversion technique with any Model.
Any other model should use this template class and rewrite the minimum functions given in the
FullComplexityModel class, adapted for a particular model.

For example, control_TELEMAC.py contains the TelemacModel class that inherits from the here defined
FullComplexityModel class.
"""

import os as _os
from datetime import datetime


class FullComplexityModel:
    def __init__(
            self,
            model_dir="",
            res_dir="",
            calibration_parameters=None,
            control_file="control.file",
            init_runs=int(),
            *args,
            **kwargs
    ):
        """
          Constructor for the FullComplexityModel Class. Instantiating can take some seconds, so try to
          be efficient in creating objects of this class (i.e., avoid re-creating a new FullComplexityModel in long loops).

          Parameters
          ----------
          model_dir : str
              Directory (path) of the model files (should NOT end on "/" or "\\") - not the software directory.
          res_dir : str
              Directory of the folder where a subfolder called "auto-saved-results" will be created to store all the results files.
          calibration_parameters : list, optional
              Computationally optional, but in the framework of Bayesian calibration, this argument must be provided.
          control_file : str
              Name of the model control file to be used (e.g., Telemac: cas file); do not include the directory. Default is "control.file".
          init_runs : int, optional
              Initial runs of the full complexity model. Default is `int()`.


          Attributes
          ----------
          model_dir : str
              Full complexity model directory.
          control_file : str
              Name of the file that controls the full complexity model simulation (to be called from a terminal).
          collocation_file : str
              Name of the file for collocation points (model parameters combinations.).
          res_dir : str
            Directory of the folder where a subfolder called "auto-saved-results" will be created to store all the results files.
          init_runs : int
              Initial runs of the full complexity model.
          calibration_parameters : list
              Calibration parameters if provided; otherwise, False.
          supervisor_dir : str
              Directory of the code that is controlling the full complexity model.
          """
        self.model_dir = model_dir
        self.control_file = control_file
        self.collocation_file = "calibration-par-combinations.csv"
        self.res_dir = res_dir
        self.init_runs=init_runs
        if not _os.path.exists(res_dir + _os.sep + "auto-saved-results"):
            _os.makedirs(res_dir + _os.sep + "auto-saved-results")
        if not _os.path.exists(res_dir + _os.sep + "auto-saved-results" + _os.sep + "plots"):
            _os.makedirs(res_dir + _os.sep + "auto-saved-results" + _os.sep + "plots")
        self.calibration_parameters = False
        if calibration_parameters:
            self.set_calibration_parameters(calibration_parameters)
        self.supervisor_dir = _os.getcwd()  # preserve directory of code that is controlling the full complexity model

    def set_calibration_parameters(self, names):
        # value corresponds to a list of parameters
        self.calibration_parameters = names
        # for par in value:
        #     self.calibration_parameters.update({par: {"current value": _np.nan}})

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
        print("DUMMY CALL")
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
