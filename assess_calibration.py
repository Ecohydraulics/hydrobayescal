import sys
import os
import time

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
            model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation2026MU",
            res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU2026-AllRange",
            calibration_pts_file_path = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements-calibration.csv",
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
            calibration_quantities=["WATER DEPTH","SCALAR VELOCITY","CUMUL BED EVOL"],
            dict_output_name="extraction-data",
            user_param_values=True,
            # max_runs=8,
            # complete_bal_mode=False,
            # only_bal_mode=False,
            # delete_complex_outputs=True,
            # validation=False
            )
surrogate_to_analyze =100
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
if n_quantities==1:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}.pkl")
else:
    sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.multitask_selection}.pkl")
start_time = time.time()
sm_predictions = sm.predict_(input_sets=collocation_points, get_conf_int=True)
end_time = time.time()
print(f"Surrogate model predictions took {end_time - start_time:.2f} seconds.")
sm_outputs=sm_predictions["output"]
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
                                 r"MO-GPE: $h, \bar{U}, \Delta_{z}$",
                                 r"MO-GPE: $h, \bar{U}$",
                                 r"SO-GPE: $h$",
                                 r"SO-GPE: $\bar{U}$",
                                 r"SO-GPE: $\Delta_{z}$",
                                 r"Benchmark: $k_{s} = \mathrm{Const}$",
                                 r"Benchmark: $k_{s} = 3 \times d_{50}$"
                             ],
            quantity_names=calibration_names,)
plotter.observed_vs_modeled_compare(df_spatial=df_spatial, df_summary=df_summary, model_ids=[1,2,3,4,5,6,7],
                                    quantity_names=[
                                        r"$h$",
                                        r"$\bar{U}$",
                                        r"$\Delta_z$"
                                    ],
                                    points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                    points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37]
                                    )
plotter.surrogate_vs_deterministic_compare(df_spatial=df_spatial, df_summary=df_summary, model_ids=[1,2,3,4,5,6,7],
                                    quantity_names=[
                                        r"$h$",
                                        r"$\bar{U}$",
                                        r"$\Delta_z$"
                                    ],
                                    points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                    points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37]
                                    )


plotter.plot_residuals(
        df_spatial,
        df_summary,
        model_ids = [1,2,3,4,5,6,7],
        quantity_names = [
                                        r"$h$",
                                        r"$\bar{U}$",
                                        r"$\Delta_z$"
                                    ],
                                    points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                    points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37])