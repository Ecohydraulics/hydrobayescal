"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
# import own scripts
import pdb
from src.hyBayesCal.telemac.control_telemac import TelemacModel


class HydroSimulations:
    def __init__(
            self,
            user_inputs,
            bal_mode=True,
    ):


        """
        Constructor for the HydroSimulations Class. Wraps functions for running Telemac and OpenFoam
        in the context of Bayesian Calibration using GPE.

        Attributes
        ____________

        :param dict user_inputs: : Dictionary with the user input parameters
        :param bal_mode: Default True: Runs Bayesian Active Learning ,
                            False: Only runs the first initial collocation points.
        """

        self.user_inputs = user_inputs
        self.bal_mode = bal_mode

    def tm_simulations(
            self,
            collocation_points=None,
            bal_iteration=int(),
            bal_new_set_parameters=None,
            ):
        control_tm = TelemacModel(
            model_dir=self.user_inputs['cas_file_simulation_path'],
            res_dir=self.user_inputs['results_folder_path'],
            control_file=self.user_inputs['cas_file_name'],
            friction_file=self.user_inputs['friction_file'],
            friction_zones=self.user_inputs['friction_zones'],
            calibration_parameters=self.user_inputs['calib_parameter_list'],
            calibration_pts_file_path=self.user_inputs['calib_pts_file_path'],
            calibration_quantities=self.user_inputs['calib_quantity_list'],
            tm_xd=self.user_inputs['Telemac_solver'],
            n_processors=self.user_inputs['n_cpus'],
            dict_output_name=self.user_inputs['dict_output_name'],
            results_file_name_base=self.user_inputs['results_file_name_base'],
            init_runs=self.user_inputs['init_runs']
        )
        control_tm.run_multiple_simulations(collocation_points, bal_new_set_parameters, bal_iteration, bal_mode=self.bal_mode)
        model_results = control_tm.output_processing(dict_output_name=self.user_inputs['dict_output_name'])

        return model_results

    def of_simulations(self):
        pass

    def run(self, collocation_points, bal_iteration, bal_new_set_parameters):
        if collocation_points is not None:
            return self.tm_simulations(collocation_points=collocation_points, bal_iteration=bal_iteration)
        elif bal_new_set_parameters is not None:
            return self.tm_simulations(bal_iteration=bal_iteration, bal_new_set_parameters=bal_new_set_parameters)
        else:
            raise ValueError(
                "Error: At least one of 'collocation_points' or 'bal_new_set_parameters' must be provided.")


if __name__ == "__main__":
    full_complexity_simulation = HydroSimulations()
    full_complexity_simulation.run()
