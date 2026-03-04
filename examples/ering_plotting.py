"""
Code that plots results for 1 quantity calibration.
Plots:
Histograms of uncertain parameters for the last iteration
BME evolution over iterations
Interpolation of BME over iterations. Plots regions with the highest BME
Plots of initial training points and selected during BAL.
Plots model comparison between surrogate model , complex model and measured data.
Model outputs at locations (sm, complex model, observed)

Author: Andres Heredia Hidalgo MSc
"""
import os
import sys
import numpy as np

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter

# Instance of Telemac Model for plotting results (calibration)
full_complexity_model = TelemacModel(
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/HICHydrodynamicsEring50it/",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
    init_runs=25,
    calibration_parameters=[r"$k_{s_{chn}}$",
        r"$k_{s_{back}}$",
        r"$k_{s_{wake}}$",
        r"$k_{s_{LW}}$"],
    param_values=[[0.002, 0.1],  # zone2 Channel
                  [0.002, 0.6],  # zone8 Backwater
                  [0.002, 0.6],  # zone9 Wake
                  [0.002, 1.5]],  # zone 13 LW
    calibration_quantities =["WATER DEPTH","SCALAR VELOCITY"],
    multitask_selection="variables",
    check_inputs=False,
)
results_folder_path = full_complexity_model.asr_dir
quantities_str = '_'.join(full_complexity_model.calibration_quantities)
plotter = BayesianPlotter(results_folder_path=results_folder_path,variable_name = quantities_str)
iterations_to_plot =50
surrogate_to_analyze =75
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
quantities_str = '_'.join(full_complexity_model.calibration_quantities)
n_loc = full_complexity_model.nloc
n_quantities=full_complexity_model.num_calibration_quantities
bayesian_data=full_complexity_model.read_data(full_complexity_model.calibration_folder,'BAL_dictionary.pkl')
collocation_points = full_complexity_model.read_data(full_complexity_model.calibration_folder,f"collocation-points-{quantities_str}.csv")
cm_outputs = full_complexity_model.read_data(full_complexity_model.calibration_folder,f"model-results-calibration-{quantities_str}.csv")
if n_quantities==1:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}.pkl")
else:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.multitask_selection}.pkl")
sm_predictions = (sm.predict_(input_sets=collocation_points,get_conf_int=True))
sm_outputs=sm_predictions["output"]
# Number of columns per quantity
num_simulations, num_columns = cm_outputs.shape
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
# Plot Bayesian results
# plotter.plot_combined_bal_3d(collocation_points = collocation_points,
#                   n_init_tp = full_complexity_model.init_runs,
#                   bayesian_dict = bayesian_data)
plotter.plot_bme_re(bayesian_dict=bayesian_data, num_bal_iterations=iterations_to_plot, plot_type='both')
plotter.plot_posterior_updates(
    posterior_arrays=bayesian_data['posterior'],
    parameter_names=full_complexity_model.calibration_parameters,
    prior=bayesian_data['prior'],
    param_values=full_complexity_model.param_values,
    iterations_to_plot=[iterations_to_plot],
    bins=20,
    plot_prior=True,
    parameter_units=['m','m','m','m'],
    # parameter_indices=[0,9,10,6,7,8]
    parameter_indices=[0,1,2,3]
)