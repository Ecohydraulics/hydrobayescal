import sys
import os
import time
import numpy as np

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
            model_dir="/home/IWS/hidalgo/Documents/EringMO-GPECalibration/MU2026-AllRange/simulation2026MU",
            res_dir="/home/IWS/hidalgo/Documents/EringMO-GPECalibration/MU2026-AllRange/",
            calibration_pts_file_path = "/home/IWS/hidalgo/Documents/EringMO-GPECalibration/MU2026-AllRange/measurements-calibration-EringCalib.csv",
            n_cpus=16,
            init_runs=7,
            calibration_parameters=["gaiaCLASSES SHIELDS PARAMETERS 1",
                                    "gaiaCLASSES SHIELDS PARAMETERS 2",
                                    "zone2", # Pool
                                    "zone3", # Slackwater
                                    "zone4", # Glide
                                    "zone5", # Riffle
                                    "zone6"], # Run
            param_values=[[0.047, 0.070],  # critical shields parameter class 1
                          [0.047, 0.070],  # critical shields parameter class 2
                          [0.002, 0.6],  # zone2
                          [0.002, 0.6],  # zone3
                          [0.002, 0.6],  # zone4
                          [0.002, 0.6],  # zone5
                          [0.002, 0.6]],  # zone6
            extraction_quantities = ["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V","CUMUL BED EVOL"],
            calibration_quantities=["WATER DEPTH", "SCALAR VELOCITY", "CUMUL BED EVOL"],
            dict_output_name="extraction-data",
            user_param_values=True,
            # max_runs=8,
            # complete_bal_mode=False,
            # only_bal_mode=False,
            # delete_complex_outputs=True,
            # validation=False
            )
surrogate_to_analyze = 30
results_folder_path = full_complexity_model.asr_dir
restart_data_folder = full_complexity_model.restart_data_folder
plotter = BayesianPlotter(results_folder_path=results_folder_path)
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
n_loc = full_complexity_model.nloc
calibration_names = full_complexity_model.calibration_quantities
n_quantities = full_complexity_model.num_calibration_quantities
num_simulations = full_complexity_model.init_runs

collocation_points = full_complexity_model.user_collocation_points # To be used for the surrogate model predictions.
coordinates = full_complexity_model.calibration_pts_df[["x", "y"]]
# The next block calls the metamodel to use for the predictions. The predictions are done in the collocation points.
# -------------------------------------------------------------------------
# Call the surrogate model
# -------------------------------------------------------------------------
# Call the surrogate model
#
# surrogate_type = "MO" uses the multi-output GPE.
# surrogate_type = "SO" loops through the three single-output GPEs and fills
#                  the same interleaved output matrix as the MO-GPE.
# -------------------------------------------------------------------------

surrogate_type = "MO"   # options: "MO" or "SO"

start_time = time.time()

if surrogate_type == "MO":

    if n_quantities == 1:
        sm = full_complexity_model.read_data(
            results_folder_path,
            f"surrogate-gpe/bal_dkl/"
            f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
            f"{full_complexity_model.calibration_quantities}.pkl"
        )
    else:
        sm = full_complexity_model.read_data(
            results_folder_path,
            f"surrogate-gpe/bal_dkl/"
            f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
            f"{full_complexity_model.calibration_quantities}_"
            f"{full_complexity_model.multitask_selection}.pkl"
        )

    sm_predictions = sm.predict_(
        input_sets=collocation_points,
        get_conf_int=True
    )

    sm_outputs = np.asarray(sm_predictions["output"])
    sm_upper_ci = np.asarray(sm_predictions["upper_ci"])
    sm_lower_ci = np.asarray(sm_predictions["lower_ci"])


elif surrogate_type == "SO":

    # Number of parameter sets/models to predict.
    n_models = collocation_points.shape[0]

    # Number of reproduction/calibration points.
    n_points = n_loc

    # Three calibration quantities:
    # ["WATER DEPTH", "SCALAR VELOCITY", "CUMUL BED EVOL"]
    surrogate_quantities = full_complexity_model.calibration_quantities

    # Empty matrices with same shape as MO-GPE output.
    # Example: 5 models, 37 points, 3 quantities -> (5, 111)
    sm_outputs = np.full((n_models, n_points * n_quantities), np.nan)
    sm_upper_ci = np.full((n_models, n_points * n_quantities), np.nan)
    sm_lower_ci = np.full((n_models, n_points * n_quantities), np.nan)

    for q_idx, quantity in enumerate(surrogate_quantities):

        # Read the corresponding single-output GPE.
        # Example filename:
        # gpr_gpy_TP100_bal_quantities_['WATER DEPTH'].pkl
        sm = full_complexity_model.read_data(
            results_folder_path,
            f"surrogate-gpe/bal_dkl/"
            f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
            f"{[quantity]}.pkl"
        )

        sm_predictions_q = sm.predict_(
            input_sets=collocation_points,
            get_conf_int=True
        )

        output_q = np.asarray(sm_predictions_q["output"])
        upper_q = np.asarray(sm_predictions_q["upper_ci"])
        lower_q = np.asarray(sm_predictions_q["lower_ci"])

        # Make sure each SO-GPE prediction has shape:
        # (n_models, n_points)
        if output_q.ndim == 1:
            output_q = output_q.reshape(n_models, n_points)

        if upper_q.ndim == 1:
            upper_q = upper_q.reshape(n_models, n_points)

        if lower_q.ndim == 1:
            lower_q = lower_q.reshape(n_models, n_points)

        # In case the surrogate returns shape (n_points, n_models), transpose.
        if output_q.shape == (n_points, n_models):
            output_q = output_q.T

        if upper_q.shape == (n_points, n_models):
            upper_q = upper_q.T

        if lower_q.shape == (n_points, n_models):
            lower_q = lower_q.T

        # Final safety check.
        if output_q.shape != (n_models, n_points):
            raise ValueError(
                f"Wrong output shape for SO-GPE quantity {quantity}. "
                f"Expected {(n_models, n_points)}, got {output_q.shape}."
            )

        if upper_q.shape != (n_models, n_points):
            raise ValueError(
                f"Wrong upper_ci shape for SO-GPE quantity {quantity}. "
                f"Expected {(n_models, n_points)}, got {upper_q.shape}."
            )

        if lower_q.shape != (n_models, n_points):
            raise ValueError(
                f"Wrong lower_ci shape for SO-GPE quantity {quantity}. "
                f"Expected {(n_models, n_points)}, got {lower_q.shape}."
            )

        # Fill the interleaved matrix:
        #
        # q_idx = 0 -> WATER DEPTH       -> columns 0, 3, 6, ...
        # q_idx = 1 -> SCALAR VELOCITY   -> columns 1, 4, 7, ...
        # q_idx = 2 -> CUMUL BED EVOL    -> columns 2, 5, 8, ...
        sm_outputs[:, q_idx::n_quantities] = output_q
        sm_upper_ci[:, q_idx::n_quantities] = upper_q
        sm_lower_ci[:, q_idx::n_quantities] = lower_q

    # Rebuild sm_predictions with the same keys used later in your code.
    sm_predictions = {
        "output": sm_outputs,
        "upper_ci": sm_upper_ci,
        "lower_ci": sm_lower_ci,
    }

else:
    raise ValueError(
        "surrogate_type must be either 'MO' or 'SO'."
    )

end_time = time.time()

print(
    f"{surrogate_type}-GPE surrogate model predictions took "
    f"{end_time - start_time:.2f} seconds."
)

print("sm_outputs shape:", sm_outputs.shape)
print("sm_upper_ci shape:", sm_predictions["upper_ci"].shape)
print("sm_lower_ci shape:", sm_predictions["lower_ci"].shape)

# -------------------------------------------------------------------------
# This line filters the outputs according to the calibration_quantities.
cm_outputs = full_complexity_model.output_processing(output_data_path=os.path.join(full_complexity_model.restart_data_folder,
                                                                                          f'user-extraction-data-detailed.json'),
                                                            validation=True, # Putting validation=True to avoid rewriting the full complex model outputs .json (extraction-data-detailed.json) in the original calibration-data according to calibration_quantities= ""..." folder.
                                                            filter_outputs=True, # Filters from the .json file output_data_path the outputs according to the calibration_quantities.
                                                            run_range_filtering=(1, full_complexity_model.init_runs)) # This is to filter the outputs according to the range of runs that we want to analyze accoring to init_runs.
# --------------------------------------------------------------------------

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

df_spatial,df_summary= plotter.evaluate_calibration(cm_outputs_split,
            sm_outputs_split,
            sm_upper_ci_split,
            sm_lower_ci_split,
            obs_split,
            coordinates,
            model_names=[
                                 r"MO-GPE: $h, \bar{U}, \delta_{z}$",
                                 r"MO-GPE: $h, \bar{U}$",
                                 r"SO-GPE: $h$",
                                 r"SO-GPE: $\bar{U}$",
                                 r"SO-GPE: $\delta_{z}$",
                                 r"Benchmark: $k_{s} = \mathrm{mean}$",
                                 r"Benchmark: $k_{s} = 3 \times d_{50}$"
                             ],
            quantity_names=calibration_names,
            plot_models=list(range(5)))
plotter.observed_vs_modeled_compare(df_spatial=df_spatial, df_summary=df_summary, model_ids=[6,7],
                                    quantity_names=[
                                        r"$h$",
                                        r"$\bar{U}$",
                                        r"$\delta_z$"
                                    ],
                                    points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                    points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37]
                                    )
plotter.surrogate_vs_deterministic_compare(df_spatial=df_spatial, df_summary=df_summary, model_ids=[6,7],
                                    quantity_names=[
                                        r"$h$",
                                        r"$\bar{U}$",
                                        r"$\delta_z$"
                                    ],
                                    points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                    points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37]
                                    )


# plotter.plot_residuals(
#         df_spatial,
#         df_summary,
#         model_ids = [1,2,3,4,5],
#         quantity_names = [
#                                         r"$h$",
#                                         r"$\bar{U}$",
#                                         r"$\delta_z$"
#                                     ],
#                                     points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
#                                     points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
#                                                     35, 36, 37])