import sys
import os

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
    control_file="tel_ering_initial_NIKU.cas",
    model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation-folder-telemac-gaia",
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
    #n_cpus=16,
    init_runs=25, # Number oF samples for validation
    calibration_parameters=["gaiaCLASSES SHIELDS PARAMETERS 1",
                            "gaiaCLASSES SHIELDS PARAMETERS 2",
                            "gaiaCLASSES SHIELDS PARAMETERS 3",
                            # "zone0",
                            # "zone1",
                            "Pool",
                            "Slackwater",
                            "Glide",
                            "Riffle",
                            "Run",
                            # "zone7",
                            "Backwater",
                            "Wake",
                            # "zone10",
                            # "zone11",
                            # "zone12",
                            "LW"],
    # calibration_quantities=["WATER DEPTH"],
    # calibration_quantities = ["SCALAR VELOCITY"],
    calibration_quantities=["WATER DEPTH","SCALAR VELOCITY","CUMUL BED EVOL"],
    # calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
    extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V",
                           "CUMUL BED EVOL"],
    validation=True
)
surrogate_to_analyze = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]  # Train points to analyze
surrogates_to_evaluate = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

# Define the desired order manually (you can also automate if needed)
calibration_quantities = full_complexity_model.calibration_quantities
keep_calibration_quantities = False
if keep_calibration_quantities:
    quantities_to_validate = [calibration_quantities]
else:
    quantities_to_validate = [[q] for q in calibration_quantities] + [calibration_quantities]

# Load the surrogate model
results_folder_path = full_complexity_model.asr_dir
restart_data_folder = full_complexity_model.restart_data_folder
coordinates = full_complexity_model.calibration_pts_df[["x", "y"]]

plotter = BayesianPlotter(results_folder_path=results_folder_path)
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
n_loc = full_complexity_model.nloc
# Build the quantities_to_validate list

surrogate_metrics = {
    "TrainPoints": [],
    "Quantity": [],
    "SurrogateType": [],
    "MSE": [],
    "RMSE": [],
    "MAE": [],
    "Correlation": [],
    "CI": [],
    "metrics_per_location":[],
}

for train_points in surrogate_to_analyze:
    for quantity_group in quantities_to_validate:
        full_complexity_model.calibration_quantities = quantity_group
        n_quantities = len(quantity_group)

        # Construct surrogate filename
        if n_quantities == 1:
            surrogate_filename = f"surrogate-gpe/bal_dkl/gpr_gpy_TP{train_points}_bal_quantities_{quantity_group}.pkl"
        else:
            multitask_selection = "variables"
            # group_name = "_".join(quantity_group)
            surrogate_filename = f"surrogate-gpe/bal_dkl/gpr_gpy_TP{train_points}_bal_quantities_{quantity_group}_{multitask_selection}.pkl"

        # Load surrogate
        sm = full_complexity_model.read_data(results_folder_path, surrogate_filename)

        # Load validation sets
        validation_sets = full_complexity_model.read_data(restart_data_folder, "collocation-points-validation.csv")

        # Predict
        sm_predictions = sm.predict_(input_sets=validation_sets, get_conf_int=True)

        # Load complex model outputs
        cm_outputs = full_complexity_model.output_processing(output_data_path=os.path.join(full_complexity_model.restart_data_folder,
                                                                                                  f'collocation-points-validation.json'),
                                                                    validation=full_complexity_model.validation,
                                                                    filter_outputs=True,
                                                                    run_range_filtering=(1, full_complexity_model.init_runs))
        # Split outputs dynamically
        for idx, quantity_name in enumerate(quantity_group):
            cm_output = cm_outputs[:, idx::n_quantities]
            sm_output = sm_predictions["output"][:, idx::n_quantities]
            sm_upper_ci = sm_predictions["upper_ci"][:, idx::n_quantities]
            sm_lower_ci = sm_predictions["lower_ci"][:, idx::n_quantities]

            # Compute metrics
            overall_mse, overall_rmse, overall_mae, overall_corr, ci_range_evolution,locations_metrics, ci_range_evolution_location = plotter.compute_evolution_metrics(sm_output,
                                                                                                                         cm_output,
                                                                                                                         sm_upper_ci,
                                                                                                                         sm_lower_ci,
                                                                                                                         selected_locations=list(range(0,37)))


            # Save metrics
            surrogate_metrics["TrainPoints"].append(train_points)
            surrogate_metrics["Quantity"].append(quantity_name)
            surrogate_metrics["SurrogateType"].append("SO" if n_quantities == 1 else "MO")
            surrogate_metrics["MSE"].append(overall_mse)
            surrogate_metrics["RMSE"].append(overall_rmse)
            surrogate_metrics["MAE"].append(overall_mae)
            surrogate_metrics["Correlation"].append(overall_corr)
            surrogate_metrics["CI"].append(ci_range_evolution)
            if train_points in surrogates_to_evaluate:
                surrogate_type = "SO" if n_quantities == 1 else "MO"
                per_loc_data = {
                    "TrainPoints": train_points,
                    "Quantity": quantity_name,
                    "SurrogateType": [surrogate_type] * n_loc,
                    "LocationIdx": list(range(locations_metrics.shape[0])),
                    "MSE": locations_metrics[:, 0].tolist(),
                    "RMSE": locations_metrics[:, 1].tolist(),
                    "MAE": locations_metrics[:, 2].tolist(),
                    "Correlation": locations_metrics[:, 3].tolist(),
                    "CI": ci_range_evolution_location
                }
                surrogate_metrics["metrics_per_location"].append(per_loc_data)
plotter.plot_metric_comparison(surrogate_metrics, calibration_quantities, metrics=["RMSE","Correlation", "CI"])
plotter.location_metrics(surrogate_metrics,coordinates)
# plotter.location_metric_heatmap(surrogate_metrics)
# print(surrogate_metrics)
# for surrogate_to_analyze in surrogate_to_analyze:
#     # Load the surrogate model
#     results_folder_path = full_complexity_model.asr_dir
#     restart_data_folder = full_complexity_model.restart_data_folder
#     plotter = BayesianPlotter(results_folder_path=results_folder_path)
#     obs = full_complexity_model.observations
#     err = full_complexity_model.measurement_errors
#     n_loc = full_complexity_model.nloc
#     n_quantities = full_complexity_model.num_calibration_quantities
#
#     # Import last trained surrogate model
#     if n_quantities==1:
#         sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}.pkl")
#     else:
#         sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.multitask_selection}.pkl")
#     # Load validation sets from CSV
#     validation_sets = full_complexity_model.read_data(restart_data_folder, "collocation-points-validation.csv")
#
#     # Predict outputs using the surrogate model
#     sm_predictions = sm.predict_(input_sets=validation_sets, get_conf_int=True)
#
#     # Load complex model outputs from CSV
#     cm_outputs = full_complexity_model.output_processing(output_data_path=os.path.join(full_complexity_model.restart_data_folder,
#                                                                                               f'collocation-points-validation.json'),
#                                                                 validation=full_complexity_model.validation,
#                                                                 filter_outputs=True,
#                                                                 run_range_filtering=(1, full_complexity_model.init_runs))
#
#     # Number of columns per quantity
#     num_simulations, num_columns = cm_outputs.shape
#     columns_per_quantity = num_columns // n_quantities
#
#     # Split outputs dynamically for each quantity
#     cm_outputs_split = {}
#     sm_outputs_split = {}
#     sm_upper_ci_split = {}
#     sm_lower_ci_split = {}
#     obs_split = {}
#     err_split = {}
#
#     for i in range(n_quantities):
#         cm_outputs_split[f'cm_outputs_{i + 1}'] = cm_outputs[:, i::n_quantities]
#         sm_outputs_split[f'sm_outputs_{i + 1}'] = sm_predictions["output"][:, i::n_quantities]
#         sm_upper_ci_split[f'sm_upper_ci_{i + 1}'] = sm_predictions["upper_ci"][:, i::n_quantities]
#         sm_lower_ci_split[f'sm_lower_ci_{i + 1}'] = sm_predictions["lower_ci"][:, i::n_quantities]
#         obs_split[f'obs_{i + 1}'] = obs[:, i::n_quantities]
#         err_split[f'err_{i + 1}'] = err[i::n_quantities]
#
#     # Plot results dynamically for each quantity
#     for i in range(n_quantities):
#         cm_output = cm_outputs_split[f'cm_outputs_{i + 1}']
#         sm_output = sm_outputs_split[f'sm_outputs_{i + 1}']
#         sm_upper_ci = sm_upper_ci_split[f'sm_upper_ci_{i + 1}']
#         sm_lower_ci = sm_lower_ci_split[f'sm_lower_ci_{i + 1}']
#         obs_quantity = obs_split[f'obs_{i + 1}']
#         err_quantity = err_split[f'err_{i + 1}']
#
#         # plotter.plot_validation_results(obs_quantity, sm_output, cm_output,gpe_lower_ci=sm_lower_ci ,gpe_upper_ci=sm_upper_ci,
#         #                                 measurement_error=err_quantity, plot_ci=True,N=1)
#         overall_mse, overall_rmse, overall_mae, overall_corr,ci_range_evolution=plotter.compute_evolution_metrics(sm_output, cm_output,sm_upper_ci,sm_lower_ci,selected_locations=list(range(1,37)))
#         surrogate_metrics["TrainPoints"].append(surrogate_to_analyze)
#         surrogate_metrics["QuantityIndex"].append(i + 1)  # or use variable names if you have them
#         surrogate_metrics["MSE"].append(overall_mse)
#         surrogate_metrics["RMSE"].append(overall_rmse)
#         surrogate_metrics["MAE"].append(overall_mae)
#         surrogate_metrics["Correlation"].append(overall_corr)
#         surrogate_metrics["CI"].append(ci_range_evolution)

# print(surrogate_metrics)
# Optional: name the quantities if you have them
# quantity_names = full_complexity_model.calibration_quantities
#
# # Call the function
# final_table = plotter.analyze_surrogate_evolution(surrogate_metrics, quantity_names)
# print(final_table)

