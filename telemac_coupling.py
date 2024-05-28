"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
# import own scripts
from global_config import *
import pdb
from src.hyBayesCal.telemac.control_telemac import TelemacModel


class HydroSimulations:
    def __init__(self):
        self.output_name=dict_output_name
    def tm_simulations(
            self,
            collocation_points=None,
            bal_iteration=int(),
            bal_new_set_parameters=None,
            bal_mode=True,
    ):
        control_tm = TelemacModel(
            model_dir=cas_file_simulation_path,
            res_dir=results_folder_path,
            control_file=cas_file_name,
            calibration_parameters=calib_parameter_list,
            calibration_pts_file_path=calib_pts_file_path,
            calibration_quantities=calib_quantity_list,
            tm_xd=Telemac_solver,
            n_processors=n_cpus,
            parameter_sampling_method=parameter_sampling_method,
            dict_output_name=self.output_name,
            results_file_name_base=results_file_name_base,
            init_runs=init_runs,
        )
        control_tm.run_multiple_simulations(collocation_points, bal_new_set_parameters, bal_iteration, bal_mode)
        model_results = control_tm.output_processing(self.output_name)

        return model_results

    def of_simulations(self):
        pass

    def run(self, collocation_points, bal_iteration, bal_new_set_parameters,bal_mode):
        if collocation_points is not None:
            return self.tm_simulations(collocation_points=collocation_points, bal_iteration=bal_iteration,bal_mode=bal_mode)
        elif bal_new_set_parameters is not None:
            return self.tm_simulations(bal_iteration=bal_iteration, bal_new_set_parameters=bal_new_set_parameters, bal_mode=bal_mode)
        else:
            raise ValueError(
                "Error: At least one of 'collocation_points' or 'bal_new_set_parameters' must be provided.")


if __name__ == "__main__":
    full_complexity_simulation = HydroSimulations()
    full_complexity_simulation.run()
