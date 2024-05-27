"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and BAL.

Author: Andres Heredia M.Sc.

"""
# import own scripts
from global_config import *
import pdb
from src.hyBayesCal.telemac.control_telemac import TelemacModel


class HydroSimulations():
    def __init__(self):
        self.n_max_tp = n_max_tp
        self.init_runs = init_runs

    def run_TM_simulations(
            self,
            collocation_points=None,
            BAL_iteration=int(),
            BAL_new_set_parameters=None,
            BAL_mode=True
    ):
        TM_instance = TelemacModel(
            model_dir=cas_file_simulation_path,
            res_dir=results_folder_path,
            control_file=cas_file_name,
            calibration_parameters=calib_parameter_list,
            collocation_points=collocation_points,
            calibration_pts_file_path=calib_pts_file_path,
            calibration_quantities=calib_quantity_list,
            tm_xd=Telemac_solver,
            n_processors=n_cpus,
            parameter_sampling_method=parameter_sampling_method,
            dict_output_name=dict_output_name,
            results_file_name_base=results_file_name_base,
            init_runs=self.init_runs,
            # calibration_phase=calibration_phase,
            n_max_tp=self.n_max_tp,
            BAL_iteration=BAL_iteration,
            BAL_new_set_parameters=BAL_new_set_parameters
        )
        TM_instance.run_multiple_simulations(collocation_points,BAL_mode)
        model_results = TM_instance.output_processing()


        return model_results

    def run_OFoam_simulations(self):

        pass

    def run(self, collocation_points, BAL_iteration , BAL_new_set_parameters):
        if collocation_points is not None:
            return self.run_TM_simulations(collocation_points=collocation_points,
                                           BAL_iteration=BAL_iteration,
                                           )
        else:
            return self.run_TM_simulations(BAL_iteration=BAL_iteration,
                                           BAL_new_set_parameters=BAL_new_set_parameters)


if __name__ == "__main__":
    full_complexity_simulation = HydroSimulations()
    full_complexity_simulation.run()
