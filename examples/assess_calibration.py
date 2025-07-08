import sys
import os
import numpy as np
import bayesvalidrox as bvr

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter

# Initialize full complexity model
full_complexity_model = TelemacModel(
            # Telemac parameters
            friction_file="friction_ering_MU_initial_NIKU.tbl",
            tm_xd="1",
            gaia_steering_file="gaia_ering_initial_NIKU.cas",
            # General hydrosimulation parameters
            results_filename_base="results2m3",
            control_file="tel_ering_initial_NIKU.cas",
            model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation-folder-telemac-gaia",
            res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
            calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
            n_cpus=16,
            init_runs=7,
            calibration_parameters=["gaiaCLASSES SHIELDS PARAMETERS 1",
                                    "gaiaCLASSES SHIELDS PARAMETERS 2",
                                    "gaiaCLASSES SHIELDS PARAMETERS 3",
                                    # "zone0",
                                    # "zone1",
                                    "zone2",
                                    "zone3",
                                    "zone4",
                                    "zone5",
                                    "zone6",
                                    # "zone7",
                                    "zone8",
                                    "zone9",
                                    # "zone10",
                                    # "zone11",
                                    # "zone12",
                                    "zone13"],
            param_values = [[0.047,0.070], # critical shields parameter class 1
                            [0.047, 0.070], # critical shields parameter class 2
                            [0.047, 0.070], # critical shields parameter class 3
                            [0.008, 0.6], # zone2 Pool
                            [0.008, 0.6], # zone3 Slackwater
                            [0.002, 0.6], # zone4 Glide
                            [0.002, 0.6], # zone5 Riffle
                            [0.040, 0.6], # zone6 Run
                            [0.002, 0.6], # zone8 Backwater
                            [0.040, 0.6], # zone9 Wake
                            [0.002, 2.8]], # zone 13 LW
            extraction_quantities = ["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V","CUMUL BED EVOL"],
            calibration_quantities=["WATER DEPTH","SCALAR VELOCITY","CUMUL BED EVOL"],
            dict_output_name="extraction-data",
            user_param_values=True,
            max_runs=7,
            complete_bal_mode=False,
            only_bal_mode=False,
            delete_complex_outputs=True,
            validation=False
            )
surrogate_to_analyze = 100
results_folder_path = full_complexity_model.asr_dir
restart_data_folder = full_complexity_model.restart_data_folder
plotter = BayesianPlotter(results_folder_path=results_folder_path)
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
n_loc = full_complexity_model.nloc
calibration_names = full_complexity_model.calibration_quantities
n_quantities = full_complexity_model.num_calibration_quantities
num_simulations = full_complexity_model.init_runs

collocation_points = full_complexity_model.user_collocation_points
coordinates = full_complexity_model.calibration_pts_df[["x", "y"]]
if n_quantities==1:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}.pkl")
else:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.multitask_selection}.pkl")
sm_predictions = (sm.predict_(input_sets=collocation_points,get_conf_int=True))
sm_outputs=sm_predictions["output"]

cm_outputs = full_complexity_model.output_processing(output_data_path=os.path.join(full_complexity_model.restart_data_folder,
                                                                                          f'user-extraction-data-detailed.json'),
                                                            delete_slf_files=full_complexity_model.delete_complex_outputs,
                                                            validation=full_complexity_model.validation,
                                                            filter_outputs=True,
                                                            save_extraction_outputs=True,
                                                            run_range_filtering=(1, full_complexity_model.init_runs))
# Split columns dynamically for each quantity
cm_outputs_split = {}
sm_outputs_split = {}
sm_upper_ci_split = {}
sm_lower_ci_split = {}
obs_split = {}
err_split = {}

for i in range(n_quantities):
    # Extract interleaved columns for the current quantity
    cm_outputs_split[f'cm_outputs_{i+1}'] = cm_outputs[:, i::n_quantities]
    sm_outputs_split[f'sm_outputs_{i+1}'] = sm_outputs[:, i::n_quantities]
    sm_upper_ci_split[f'sm_upper_ci_{i+1}'] = sm_predictions["upper_ci"][:, i::n_quantities]
    sm_lower_ci_split[f'sm_lower_ci_{i+1}'] = sm_predictions["lower_ci"][:, i::n_quantities]
    obs_split[f'obs_{i+1}'] = obs[:, i::n_quantities]
    err_split[f'err_{i+1}'] = err[i::n_quantities]

plotter.evaluate_calibration(cm_outputs_split,
            sm_outputs_split,
            sm_upper_ci_split,
            sm_lower_ci_split,
            obs_split,
            coordinates,
            model_names=[r"Deterministic: $K_{NKU} = 3 \times D_{50}$",
                                 "$K_{NKU} = Const$",
                                 "SO-GPE: h",
                                 r"MO-GPE: $h, \bar{U}, \Delta Z_{\mathrm{DEM}}$",
                                 r"MO-GPE: $h, \bar{U}$",
                                 r"SO-GPE: $\bar{U}$",
                                 r"SO-GPE: $\Delta Z_{\mathrm{DEM}}$"],
            quantity_names=calibration_names,)
# for i in range(n_quantities):
#     cm_output = cm_outputs_split[f'cm_outputs_{i+1}']
#     sm_output = sm_outputs_split[f'sm_outputs_{i+1}']
#     sm_upper_ci = sm_upper_ci_split[f'sm_upper_ci_{i+1}']
#     sm_lower_ci = sm_lower_ci_split[f'sm_lower_ci_{i+1}']
#     obs_quantity = obs_split[f'obs_{i+1}']
#     err_quantity = err_split[f'err_{i+1}']
#
#     # Plot comparisons for each quantity
#     # plotter.plot_validation_results(obs_quantity, sm_output.reshape(1, -1), cm_output.reshape(1, -1))
#
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name = full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-1, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-1, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-1, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-1, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
#
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name=full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-2, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-2, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-2, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-2, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
#
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name=full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-3, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-3, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-3, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-3, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
#
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name=full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-4, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-4, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-4, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-4, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name=full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-5, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-5, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-5, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-5, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name=full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-6, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-6, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-6, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-6, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
#     plotter.plot_model_outputs_vs_locations(
#         observed_values=obs_quantity,
#         quantity_name=full_complexity_model.calibration_quantities[i],
#         surrogate_outputs=sm_output[-7, :].reshape(1, -1),
#         complex_model_outputs=cm_output[-7, :].reshape(1, -1),
#         selected_locations=list(range(1,37)),
#         gpe_lower_ci=sm_lower_ci[-7, :].reshape(1, -1),
#         gpe_upper_ci=sm_upper_ci[-7, :].reshape(1, -1),
#         measurement_error=err_quantity,
#     )
    # plotter.plot_model_outputs_vs_locations(
    #     observed_values=obs_quantity,
    #     quantity_name=full_complexity_model.calibration_quantities[i],
    #     surrogate_outputs=sm_output[-5, :].reshape(1, -1),
    #     complex_model_outputs=cm_output[-5, :].reshape(1, -1),
    #     selected_locations=list(range(1,37)),
    #     gpe_lower_ci=sm_lower_ci[-5, :].reshape(1, -1),
    #     gpe_upper_ci=sm_upper_ci[-5, :].reshape(1, -1),
    #     measurement_error=err_quantity,
    # )
