# import own scripts
from global_config import *
from src.hyBayesCal.telemac.control_telemac import TelemacModel


class TelemacSimulations():
    def __init__(self):
        self.init_runs = init_runs

    def run_multiple_simulations(self, runs):
        for i in range(runs):  # Use 'runs' instead of 'init_runs'
            simulation_num = i + 1
            TM_instance = TelemacModel(
                model_dir=cas_file_simulation_path,
                control_file=cas_file_name,
                calibration_parameters=calib_parameter_list,
                calibration_values_ranges=parameter_ranges_list,
                calibration_pts_file_path=calib_pts_file_path,
                calibration_quantities=calib_quantity_list,
                tm_xd=Telemac_solver,
                n_processors=NCPS,
                parameter_sampling_method=parameter_sampling_method,
                dict_output_name=dict_output_name,
                results_file_name_base=results_file_name_base,
                num_run=simulation_num,
                init_runs=runs
            )
            TM_instance()  # Call the TelemacModel instance like a function


if __name__ == "__main__":
    instance = TelemacSimulations()
    instance.run_multiple_simulations(init_runs)  # Pass 'init_runs' to the method

