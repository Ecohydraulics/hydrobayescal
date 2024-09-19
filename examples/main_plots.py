"""
Code that plots results for 1 quantity calibration.
Plots:
Histograms of uncertain parameters for the last iteration
BME evolution over iterations
Interpolation of BME over iterations. Plots regions with highest BME
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

# Add the paths to sys.path
sys.path.insert(0, src_path)  # Prepend to prioritize over other paths
sys.path.insert(0, hydroBayesCal_path)

from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter

#Instance of Telemac Model for plotting results (calibration)
full_complexity_model = TelemacModel(
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurementsWDEPTH_filtered.csv",
    init_runs=30,
    calibration_parameters=["zone11", "zone9", "zone10", "zone1", "zone5", "zone7", "vg_zone7-par1",
                            "vg_zone7-par2", "vg_zone7-par3",
                            "ROUGHNESS COEFFICIENT OF BOUNDARIES", "VELOCITY DIFFUSIVITY"],
    param_values=[[0.01, 0.18], [0.002, 0.07], [0.01, 0.18], [0.002, 0.07], [0.15, 0.30], [0.02, 0.10], [0.7, 1.3],
                  [5, 6],
                  [0.3, 0.6], [0.013, 0.035], [0.000015, 0.0015]],
    calibration_quantities=["WATER DEPTH"],
    check_inputs=False,
)
results_folder_path = full_complexity_model.asr_dir
plotter = BayesianPlotter(results_folder_path=results_folder_path)

obs=full_complexity_model.observations
err=full_complexity_model.measurement_errors
n_loc=full_complexity_model.nloc
n_quantities=full_complexity_model.num_quantities
bayesian_data=full_complexity_model.read_data(results_folder_path,'BAL_dictionary.pkl')
collocation_points = full_complexity_model.read_data(results_folder_path,"collocation-points.csv")
cm_outputs = full_complexity_model.read_data(results_folder_path,"model-results.csv")
sm = full_complexity_model.read_data(results_folder_path, "surrogate-gpe/bal_dkl/gpr_gpy_TP150_bal_quantities1.pkl")
sm_predictions = (sm.predict_(input_sets=collocation_points))#,get_conf_int=True))
sm_outputs=sm_predictions["output"]
if len(full_complexity_model.calibration_quantities)==2:
    num_simulations, num_columns = cm_outputs.shape
    # Separate the columns for each quantity
    cm_outputs1 = cm_outputs[:,
                             0:num_columns:2]  # Take every second column starting from 0
    cm_outputs2 = cm_outputs[:,
                              1:num_columns:2]  # Take every second column starting from 1
    cm_outputs = np.hstack((cm_outputs1, cm_outputs2))
    num_columns = sm_outputs.shape[1]

    # Calculate the midpoint
    midpoint = num_columns // 2  # Integer division to get the floor value

    # Get the first half of the columns
    sm_outputs1 = sm_outputs[:, :midpoint]

    # Get the second half of the columns
    sm_outputs2 = sm_outputs[:, midpoint:]
    # Get the first half of the columns
    obs1 = obs[:, :midpoint]
    err1 = err[:midpoint]
    # Get the second half of the columns
    obs2 = obs[:, midpoint:]
    err2 = err[midpoint:]


# sm_upper_ci=sm_predictions["upper_ci"]
# sm_lower_ci=sm_predictions["lower_ci"]


# surrogate_outputs = surrogate_outputs_dic["output"]


plotter.plot_bme_re(bayesian_dict=bayesian_data,
            num_bal_iterations=120,
                    plot_type='both')
plotter.plot_combined_bal(collocation_points = collocation_points,
                  n_init_tp = full_complexity_model.init_runs,
                  bayesian_dict = bayesian_data)
plotter.plot_posterior_updates(posterior_arrays = bayesian_data['posterior'],
                       parameter_names = full_complexity_model.calibration_parameters,
                       prior = bayesian_data['prior'],
                        param_values = full_complexity_model.param_values,
                        iterations_to_plot=[0,120],
                        plot_prior=True)
plotter.plot_model_comparisons(obs, sm_outputs[-1, :].reshape(1, -1), cm_outputs[-1, :].reshape(1, -1))
plotter.plot_model_outputs_vs_locations(observed_values=obs, surrogate_outputs=sm_outputs[-1, :].reshape(1, -1), complex_model_outputs=cm_outputs[-1, :].reshape(1, -1),measurement_error=err)

# plotter.plot_model_comparisons(obs1, sm_outputs1[-1, :].reshape(1, -1), cm_outputs1[-1, :].reshape(1, -1))
# plotter.plot_model_outputs_vs_locations(observed_values=obs1, surrogate_outputs=sm_outputs1[-1, :].reshape(1, -1), complex_model_outputs=cm_outputs1[-1, :].reshape(1, -1),measurement_error=err1)
#
# plotter.plot_model_comparisons(obs2, sm_outputs2[-1, :].reshape(1, -1), cm_outputs2[-1, :].reshape(1, -1))
# plotter.plot_model_outputs_vs_locations(observed_values=obs2, surrogate_outputs=sm_outputs2[-1, :].reshape(1, -1), complex_model_outputs=cm_outputs2[-1, :].reshape(1, -1),measurement_error=err2)
plotter.plot_bme_3d(param_sets=collocation_points,
                    param_ranges=full_complexity_model.param_values,
                    param_names=full_complexity_model.calibration_parameters,
                    bme_values=bayesian_data['BME'],
                    param_indices=(1, 3),
                    grid_size=400,
                    iteration_range=(100, 120),
                    plot_criteria="BME"
                    )
# plotter.plot_bme_3d(param_sets=collocation_points,
#                     param_ranges=full_complexity_model.param_values,
#                     param_names=full_complexity_model.calibration_parameters,
#                     bme_values=bayesian_data['RE'],
#                     param_indices=(3, 6),
#                     grid_size=400,
#                     iteration_range=(30, 63),
#                     plot_criteria="RE"
#                     )
plotter.plot_bme_comparison(param_sets=collocation_points,
                    param_ranges=full_complexity_model.param_values,
                    param_names=full_complexity_model.calibration_parameters,
                    bme_values=bayesian_data['RE'],
                    param_indices=(3, 4),
                    total_iterations_range=(0,120),
                    iterations_per_subplot=10,
                    plot_criteria="RE"
                    )