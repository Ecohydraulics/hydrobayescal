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
            friction_file="friction_ering_MU.tbl",
            tm_xd="1",
            gaia_steering_file="",
            # General hydrosimulation parameters
            results_filename_base="results2m3",
            control_file="tel_ering_mu_restart.cas",
            model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/",
            res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
            calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
            n_cpus=8,
            init_runs=4,
            calibration_parameters=["zone11", "zone12", "zone13", "zone14", "zone15"],
            param_values = [[0.010, 0.1], [0.050, 0.79], [0.0020, 0.1], [0.002, 0.1], [0.050, 0.79]], # coarse-coarse -fine -fine -coarse
            extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V"],
            calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
            dict_output_name="extraction-data",
            user_param_values=True,
            max_runs=4,
            complete_bal_mode=False,
            only_bal_mode=False,
            delete_complex_outputs=True,
            validation=True
            )
surrogate_to_analyze = 170
results_folder_path = full_complexity_model.asr_dir
restart_data_folder = full_complexity_model.restart_data_folder
plotter = BayesianPlotter(results_folder_path=results_folder_path)
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
n_loc = full_complexity_model.nloc
n_quantities = full_complexity_model.num_calibration_quantities
collocation_points = full_complexity_model.user_collocation_points
if n_quantities==1:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.calibration_parameters}.pkl")
else:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.calibration_parameters}_{full_complexity_model.multitask_selection}.pkl")
sm_predictions = (sm.predict_(input_sets=collocation_points,get_conf_int=True))
sm_outputs=sm_predictions["output"]

full_complexity_model.run_multiple_simulations(collocation_points=collocation_points,
                                       complete_bal_mode=full_complexity_model.complete_bal_mode,
                                       validation=full_complexity_model.validation)
cm_outputs = full_complexity_model.model_evaluations
num_simulations, num_columns = sm_outputs.shape
columns_per_quantity = num_columns // n_quantities
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

# Combine outputs (if needed, interleaving columns)
cm_outputs_combined = np.hstack([cm_outputs_split[f'cm_outputs_{i+1}'] for i in range(n_quantities)])
sm_outputs_combined = np.hstack([sm_outputs_split[f'sm_outputs_{i+1}'] for i in range(n_quantities)])
sm_upper_ci_combined = np.hstack([sm_upper_ci_split[f'sm_upper_ci_{i+1}'] for i in range(n_quantities)])
sm_lower_ci_combined = np.hstack([sm_lower_ci_split[f'sm_lower_ci_{i+1}'] for i in range(n_quantities)])

# Plot results dynamically for each quantity
for i in range(n_quantities):
    cm_output = cm_outputs_split[f'cm_outputs_{i+1}']
    sm_output = sm_outputs_split[f'sm_outputs_{i+1}']
    sm_upper_ci = sm_upper_ci_split[f'sm_upper_ci_{i+1}']
    sm_lower_ci = sm_lower_ci_split[f'sm_lower_ci_{i+1}']
    obs_quantity = obs_split[f'obs_{i+1}']
    err_quantity = err_split[f'err_{i+1}']

    # Plot comparisons for each quantity
    # plotter.plot_validation_results(obs_quantity, sm_output.reshape(1, -1), cm_output.reshape(1, -1))

    plotter.plot_model_outputs_vs_locations(
        observed_values=obs_quantity,
        quantity_name = full_complexity_model.calibration_quantities[i],
        surrogate_outputs=sm_output[-1, :].reshape(1, -1),
        complex_model_outputs=cm_output[-1, :].reshape(1, -1),
        selected_locations=[24,25,21,27,29,31,1,13,9,8,6,14,3],
        gpe_lower_ci=sm_lower_ci[-1, :].reshape(1, -1),
        gpe_upper_ci=sm_upper_ci[-1, :].reshape(1, -1),
        measurement_error=err_quantity,
    )

    plotter.plot_model_outputs_vs_locations(
        observed_values=obs_quantity,
        quantity_name=full_complexity_model.calibration_quantities[i],
        surrogate_outputs=sm_output[-2, :].reshape(1, -1),
        complex_model_outputs=cm_output[-2, :].reshape(1, -1),
        selected_locations=[24,25,21,27,29,31,1,13,9,8,6,14,3],
        gpe_lower_ci=sm_lower_ci[-2, :].reshape(1, -1),
        gpe_upper_ci=sm_upper_ci[-2, :].reshape(1, -1),
        measurement_error=err_quantity,
    )

    plotter.plot_model_outputs_vs_locations(
        observed_values=obs_quantity,
        quantity_name=full_complexity_model.calibration_quantities[i],
        surrogate_outputs=sm_output[-3, :].reshape(1, -1),
        complex_model_outputs=cm_output[-3, :].reshape(1, -1),
        selected_locations=[24,25,21,27,29,31,1,13,9,8,6,14,3],
        gpe_lower_ci=sm_lower_ci[-3, :].reshape(1, -1),
        gpe_upper_ci=sm_upper_ci[-3, :].reshape(1, -1),
        measurement_error=err_quantity,
    )

    plotter.plot_model_outputs_vs_locations(
        observed_values=obs_quantity,
        quantity_name=full_complexity_model.calibration_quantities[i],
        surrogate_outputs=sm_output[-4, :].reshape(1, -1),
        complex_model_outputs=cm_output[-4, :].reshape(1, -1),
        selected_locations=[24,25,21,27,29,31,1,13,9,8,6,14,3],
        gpe_lower_ci=sm_lower_ci[-4, :].reshape(1, -1),
        gpe_upper_ci=sm_upper_ci[-4, :].reshape(1, -1),
        measurement_error=err_quantity,
    )