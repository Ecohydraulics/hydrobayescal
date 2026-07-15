import os
import argparse
import importlib.util

from hydroBayesCal.telemac.control_telemac import TelemacModel
from hydroBayesCal.visualize import BayesianPlotter

def load_config(config_path):
    """
    Load configuration from Python file.

    Parameters
    ----------
    config_path : str
        Path to the Python configuration file

    Returns
    -------
    module
        Configuration module with all variables
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config
def main():
    # Initialize full complexity model
    # Initialize full complexity model
    parser = argparse.ArgumentParser(description="Assess calibraed deterministic models  against measurements.")
    parser.add_argument(
        '--config',
        type=str,
        default='config_Ering.py',
        help='Path to Python configuration file (default: config_Ering.py)'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    full_complexity_model = TelemacModel(
        res_dir=config.paths['res_dir'],
        calibration_pts_file_path=config.paths['calibration_pts_file_path'],
        init_runs=30, # Number oF samples for validation
        calibration_parameters=config.calibration['parameters'],
        calibration_quantities=config.calibration['calibration_quantities'],
        # calibration_quantities=["SCALAR VELOCITY", "WATER DEPTH", "CUMUL BED EVOL"],
        # calibration_quantities=["WATER DEPTH","SCALAR VELOCITY"],
        extraction_quantities=config.calibration['extraction_quantities'],
        validation=True
    )
    surrogate_to_analyze = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]  # Train points to analyze
    surrogates_to_evaluate = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]  # Train points to evaluate per location

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
                                                                                                      f'model-results-validation.json'),
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
    plotter.plot_metric_comparison(surrogate_metrics, calibration_quantities, metrics=["MAE"])
    plotter.location_metrics(surrogate_metrics,coordinates)



if __name__ == "__main__":
    main()
