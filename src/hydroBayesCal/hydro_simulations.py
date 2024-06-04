"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
# Import own scripts
import pdb
from telemac.control_telemac import TelemacModel
from model_structure.control_full_complexity import FullComplexityModel

from config_logging import *
setup_logging()
logger_info = logging.getLogger("HydroBayesCal")

class HydroSimulations(FullComplexityModel):
    def __init__(
            self,
            model_dir="",
            res_dir="",
            control_file="",
            calibration_parameters=None,
            bal_mode=True,
            n_max_tp=int(),
            user_inputs=None
    ):


        """
        Constructor for the HydroSimulations Class. Wraps functions for running Telemac and OpenFoam
        in the context of Bayesian Calibration using GPE.

        Attributes
        ____________

        :param dict user_inputs: : User input parameters.
        :param bal_mode: Default True: Runs Bayesian Active Learning ,
                            False: Only runs the initial collocation points.
        :param n_max_tp: Maximum number of model simulations including BAL iterations.
        """
        FullComplexityModel.__init__(self, model_dir=model_dir,
                                     res_dir=res_dir,
                                     calibration_parameters=calibration_parameters,
                                     control_file=control_file)
        self.user_inputs = user_inputs
        self.bal_mode = bal_mode
        self.n_max_tp=n_max_tp



    def tm_simulations(
            self,
            collocation_points=None,
            bal_iteration=int(),
            bal_new_set_parameters=None,
            ):
        control_tm = TelemacModel(
            model_dir=self.model_dir,
            res_dir=self.res_dir,
            control_file=self.control_file,
            friction_file=self.user_inputs['friction_file'],
            calibration_parameters=self.calibration_parameters,
            calibration_pts_file_path=self.user_inputs['calib_pts_file_path'],
            calibration_quantities=self.user_inputs['calib_quantity_list'],
            tm_xd=self.user_inputs['Telemac_solver'],
            n_processors=self.user_inputs['n_cpus'],
            dict_output_name=self.user_inputs['dict_output_name'],
            results_file_name_base=self.user_inputs['results_file_name_base'],
            init_runs=self.user_inputs['init_runs'],

        )
        control_tm.run_multiple_simulations(collocation_points, bal_new_set_parameters, bal_iteration,
                                            bal_mode=self.bal_mode)
        model_results = control_tm.output_processing(dict_output_name=self.user_inputs['dict_output_name'])

        return model_results

    def of_simulations(self):
        pass

    def run(self, collocation_points, bal_iteration, bal_new_set_parameters):
        if collocation_points is not None:
            logger_info.info("Running full complexity models with initial collocation points:")
            return self.tm_simulations(collocation_points=collocation_points, bal_iteration=bal_iteration)
        elif bal_new_set_parameters is not None:
            return self.tm_simulations(bal_iteration=bal_iteration, bal_new_set_parameters=bal_new_set_parameters)
        else:
            raise ValueError("Error: At least one of 'collocation_points' or 'bal_new_set_parameters' must be provided.")


if __name__ == "__main__":
    full_complexity_simulation = HydroSimulations()
    full_complexity_simulation.run()
