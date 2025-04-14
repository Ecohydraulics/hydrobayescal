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
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
    init_runs=20,
    # calibration_parameters=["gaiaCLASSES SHIELDS PARAMETERS 1", "gaiaCLASSES SHIELDS PARAMETERS 2",
    #                         "gaiaCLASSES SHIELDS PARAMETERS 3", "gaiaCLASSES SHIELDS PARAMETERS 4",
    #                         "gaiaCLASSES SHIELDS PARAMETERS 5", "gaiaMPM COEFFICIENT", "zone2", "zone3", "zone4",
    #                         "zone5", "zone6", "zone8", "zone9", "zone13"],  # pool-slackwater-glide-riffle-run
    calibration_parameters=["gaiaCLASSES SHIELDS PARAMETERS 1",
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
                            "zone10",
                            # "zone11",
                            # "zone12",
                            "zone13"],  # pool-sla
    # param_values=[[0.0016, 0.79], [0.0016, 0.79], [0.0016, 0.79], [0.0016, 0.79], [0.0016, 0.79]],
    param_values=[[0.048, 0.070], [0.010, 0.79], [0.010, 0.79], [0.0010, 0.79], [0.0010, 0.79], [0.060, 0.79],
                  [0.0010, 0.79], [0.060, 0.79], [0.7, 2.4], [0.0010, 0.79]],  # coarse-coarse -fine -fine -coarse

    # param_values=[[0.048, 0.070],  # critical shields parameter class 1
    #               # [0.5,17.45], # zone0
    #               # [0.5,17.45], # zone 1
    #               [0.24, 17.45],  # zone 2
    #               [0.24, 17.45],  # zone 3
    #               [0.04, 31.86],  # zone 4
    #               [0.04, 31.86],  # zone 5
    #               [1.53, 17.45],  # zone 6
    #               # [0.16,3.60], # zone 7
    #               [0.04, 31.86],  # zone 8
    #               [2.79, 31.86],  # zone 9
    #               # [1.53, 32.80], # zone 10
    #               # [1.5, 98],# zone 11
    #               # [1.5, 98],  # zone 12
    #               [0.02, 17.45]],  # zone 13
    calibration_quantities=["SCALAR VELOCITY","WATER DEPTH"],
    multitask_selection="variables",
    check_inputs=False,
)
results_folder_path = full_complexity_model.asr_dir
quantities_str = '_'.join(full_complexity_model.calibration_quantities)
plotter = BayesianPlotter(results_folder_path=results_folder_path,variable_name = quantities_str)
iterations_to_plot =  15
surrogate_to_analyze = 35
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

    # Plot comparisons for each quantity
    # plotter.plot_validation_results(obs_quantity, sm_output.reshape(1, -1), cm_output.reshape(1, -1))
    plotter.plot_model_outputs_vs_locations(
        observed_values=obs_quantity,
        quantity_name=full_complexity_model.calibration_quantities[i],
        surrogate_outputs=sm_output[-1, :].reshape(1, -1),
        complex_model_outputs=cm_output[-1, :].reshape(1, -1),
        selected_locations=[24,25,21,28,31,1,11,7,6,14,3],
        gpe_lower_ci=sm_lower_ci[-1, :].reshape(1, -1),
        gpe_upper_ci=sm_upper_ci[-1, :].reshape(1, -1),
        measurement_error=err_quantity,
    )

# Plot Bayesian results
plotter.plot_bme_re(bayesian_dict=bayesian_data, num_bal_iterations=iterations_to_plot, plot_type='both')
plotter.plot_posterior_updates(
    posterior_arrays=bayesian_data['posterior'],
    parameter_names=full_complexity_model.calibration_parameters,
    prior=bayesian_data['prior'],
    param_values=full_complexity_model.param_values,
    iterations_to_plot=[iterations_to_plot],
    bins=10,
    plot_prior=True,
)
plotter.plot_prior_posterior_kde(bayesian_data=bayesian_data,parameter_names=full_complexity_model.calibration_parameters,iterations_to_plot=iterations_to_plot)
plotter.plot_posterior_iteration(posterior_samples=bayesian_data['posterior'][iterations_to_plot],
                               parameter_names=full_complexity_model.calibration_parameters,param_values = full_complexity_model.param_values)
#plotter.plot_bme_3d(collocation_points=collocation_points,param_ranges=full_complexity_model.param_values,param_names=full_complexity_model.calibration_parameters,bme_values=bayesian_data['BME'],param_indices=(1,4),extra_param_index=2,iteration_range=(60,140))
# if len(full_complexity_model.calibration_quantities)==2:
#     num_simulations, num_columns = cm_outputs.shape
#     # Separate the columns for each quantity
#     cm_outputs1 = cm_outputs[:,
#                              0:num_columns:2]  # Take every second column starting from 0
#     cm_outputs2 = cm_outputs[:,
#                               1:num_columns:2]  # Take every second column starting from 1
#     cm_outputs = np.hstack((cm_outputs1, cm_outputs2))
#     sm_outputs1 =sm_outputs[:,
#                              0:num_columns:2]
#     sm_outputs2 = sm_outputs[:,
#                              1:num_columns:2]
#     sm_upper_ci_1=sm_predictions["upper_ci"][:,
#                              0:num_columns:2]
#     sm_upper_ci_2=sm_predictions["upper_ci"][:,
#                              1:num_columns:2]
#     sm_lower_ci_1=sm_predictions["lower_ci"][:,
#                              0:num_columns:2]
#     sm_lower_ci_2=sm_predictions["lower_ci"][:,
#                              1:num_columns:2]
#     num_columns = sm_outputs.shape[1]
#     # Calculate the midpoint
#     midpoint = num_columns // 2  # Integer division to get the floor value
#     obs1 = obs[:, :midpoint]
#     err1 = err[:midpoint]
#     # Get the second half of the columns
#     obs2 = obs[:, midpoint:]
#     err2 = err[midpoint:]
#
# # surrogate_outputs = surrogate_outputs_dic["output"]
#
#
# plotter.plot_bme_re(bayesian_dict=bayesian_data,
#             num_bal_iterations=104,
#                     plot_type='both')
# # plotter.plot_combined_bal(collocation_points = collocation_points,
# #                   n_init_tp = full_complexity_model.init_runs,
# #                   bayesian_dict = bayesian_data)
# plotter.plot_posterior_updates(posterior_arrays = bayesian_data['posterior'],
#                        parameter_names = full_complexity_model.calibration_parameters,
#                        prior = bayesian_data['prior'],
#                         param_values = full_complexity_model.param_values,
#                         iterations_to_plot=[104],
#                         bins=20,
#                         plot_prior=True)
# if len(full_complexity_model.calibration_quantities)==1:
#     sm_upper_ci = sm_predictions["upper_ci"]
#     sm_lower_ci = sm_predictions["lower_ci"]
#     plotter.plot_model_comparisons(obs, sm_outputs[-1, :].reshape(1, -1), cm_outputs[-1, :].reshape(1, -1))
#     plotter.plot_model_outputs_vs_locations(observed_values=obs, surrogate_outputs=sm_outputs[-1, :].reshape(1, -1), complex_model_outputs=cm_outputs[-1, :].reshape(1, -1),gpe_lower_ci=sm_lower_ci[-1, :].reshape(1, -1),gpe_upper_ci=sm_upper_ci[-1, :].reshape(1, -1),measurement_error=err)
#
# if len(full_complexity_model.calibration_quantities)==2:
#
#     plotter.plot_model_comparisons(obs1, sm_outputs1[-1, :].reshape(1, -1), cm_outputs1[-1, :].reshape(1, -1))
#     plotter.plot_model_outputs_vs_locations(observed_values=obs1, surrogate_outputs=sm_outputs1[-1, :].reshape(1, -1), complex_model_outputs=cm_outputs1[-1, :].reshape(1, -1),gpe_lower_ci=sm_lower_ci_1[-1, :].reshape(1, -1), gpe_upper_ci=sm_upper_ci_1[-1, :].reshape(1, -1),measurement_error=err1)
#     #
#     plotter.plot_model_comparisons(obs2, sm_outputs2[-1, :].reshape(1, -1), cm_outputs2[-1, :].reshape(1, -1))
#     plotter.plot_model_outputs_vs_locations(observed_values=obs2, surrogate_outputs=sm_outputs2[-1, :].reshape(1, -1), complex_model_outputs=cm_outputs2[-1, :].reshape(1, -1),gpe_lower_ci=sm_lower_ci_2[-1, :].reshape(1, -1), gpe_upper_ci=sm_upper_ci_2[-1, :].reshape(1, -1),measurement_error=err2)
# plotter.plot_bme_3d(param_sets=collocation_points,
#                     param_ranges=full_complexity_model.param_values,
#                     param_names=full_complexity_model.calibration_parameters,
#                     bme_values=bayesian_data['BME'],
#                     param_indices=(4,8),
#                     grid_size=400,
#                     iteration_range=(15,40),
#                     plot_criteria="BME"
#                     )
# plotter.plot_bme_3d(param_sets=collocation_points,
#                     param_ranges=full_complexity_model.param_values,
#                     param_names=full_complexity_model.calibration_parameters,
#                     bme_values=bayesian_data['RE'],
#                     param_indices=(3, 6),
#                     grid_size=400,
#                     iteration_range=(30, 63),
#                     plot_criteria="RE"
#                     )
# plotter.plot_bme_comparison(param_sets=collocation_points,
#                     param_ranges=full_complexity_model.param_values,
#                     param_names=full_complexity_model.calibration_parameters,
#                     bme_values=bayesian_data['RE'],
#                     param_indices=(3, 4),
#                     total_iterations_range=(0,10),
#                     iterations_per_subplot=10,
#                     plot_criteria="RE"
#                     )