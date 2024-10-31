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
from pathlib import Path
import sys,os
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
    model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/",
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/points_wet_area_MU.csv",
    #init_runs=1,
    calibration_parameters=["zone1", "zone2", "zone3", "zone4", "zone5", "zone6", "zone7",
                            "vg_zone7-par2", "vg_zone7-par3"],
    # param_values=[[0.01, 0.18], [0.002, 0.07], [0.01, 0.18], [0.002, 0.07], [0.15, 0.30], [0.02, 0.10], [0.7, 1.3],
    #               [5, 6],
    #               [0.3, 0.6], [0.013, 0.035], [0.000015, 0.0015]],
    dict_output_name="wd-v-channel-points",
    calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
    check_inputs=False,
)
# full_complexity_model.extract_data_point(input_slf_file="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/results2m3_1.slf",
#                                          calibration_pts_df=full_complexity_model.calibration_pts_df,
#                                          output_name=full_complexity_model.dict_output_name,
#                                          extraction_quantity=full_complexity_model.calibration_quantities,
#                                          simulation_number=1,
#                                          model_directory=full_complexity_model.model_dir,
#                                          results_folder_directory=full_complexity_model.asr_dir)
model_results=full_complexity_model.output_processing(os.path.join(full_complexity_model.asr_dir,f'{full_complexity_model.dict_output_name}.json'))
X_coord=full_complexity_model.calibration_pts_df["X"].values
Y_coord=full_complexity_model.calibration_pts_df["Y"].values
model_results_flat=model_results.flatten()
x_y_var=np.column_stack((X_coord, Y_coord, model_results_flat))
header = "X,Y,WATER DEPTH,VELOCITY"
np.savetxt(full_complexity_model.asr_dir + '/' + 'output.xyz', x_y_var, fmt='%.6f', delimiter=',', header=header, comments='')


print(model_results)



# results_folder_path = full_complexity_model.asr_dir
# plotter = BayesianPlotter(results_folder_path=results_folder_path)
#
# obs=full_complexity_model.observations
# err=full_complexity_model.measurement_errors
# n_loc=full_complexity_model.nloc
# n_quantities=full_complexity_model.num_quantities
# bayesian_data=full_complexity_model.read_data(results_folder_path,'BAL_dictionary.pkl')
# collocation_points = full_complexity_model.read_data(results_folder_path,"collocation-points.csv")
# cm_outputs = full_complexity_model.read_data(results_folder_path,"model-results.csv")
# sm = full_complexity_model.read_data(results_folder_path, "surrogate-gpe/bal_dkl/gpr_gpy_TP150_bal_quantities1.pkl")
# sm_predictions = (sm.predict_(input_sets=collocation_points))#,get_conf_int=True))
# sm_outputs=sm_predictions["output"]
# # sm_upper_ci=sm_predictions["upper_ci"]
# # sm_lower_ci=sm_predictions["lower_ci"]
#
#
# # surrogate_outputs = surrogate_outputs_dic["output"]
#
#
# plotter.plot_bme_re(bayesian_dict=bayesian_data,
#             num_bal_iterations=120,
#                     plot_type='both')
# plotter.plot_combined_bal(collocation_points = collocation_points,
#                   n_init_tp = full_complexity_model.init_runs,
#                   bayesian_dict = bayesian_data)
# plotter.plot_posterior_updates(posterior_arrays = bayesian_data['posterior'],
#                        parameter_names = full_complexity_model.calibration_parameters,
#                        prior = bayesian_data['prior'],
#                         param_values = full_complexity_model.param_values,
#                         iterations_to_plot=[120])
# # plotter.plot_model_comparisons(obs, sm_outputs[-1, :].reshape(1, -1), cm_outputs[-1, :].reshape(1, -1))
# # plotter.plot_model_outputs_vs_locations(obs, sm_outputs[-1, :].reshape(1, -1), cm_outputs[-1, :].reshape(1, -1), sm_upper_ci[-1, :].reshape(1, -1), sm_lower_ci[-1, :].reshape(1, -1), err)
# plotter.plot_bme_3d(param_sets=collocation_points,
#                     param_ranges=full_complexity_model.param_values,
#                     param_names=full_complexity_model.calibration_parameters,
#                     bme_values=bayesian_data['BME'],
#                     param_indices=(3, 6),
#                     grid_size=400,
#                     last_iterations=30,
#                     plot_criteria="BME"
#                     )
# plotter.plot_bme_3d(param_sets=collocation_points,
#                     param_ranges=full_complexity_model.param_values,
#                     param_names=full_complexity_model.calibration_parameters,
#                     bme_values=bayesian_data['RE'],
#                     param_indices=(3, 6),
#                     grid_size=400,
#                     last_iterations=30,
#                     plot_criteria="RE"
#                     )
