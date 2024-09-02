from pathlib import Path
import sys,os

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

#from user_settings import user_inputs_tm
from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter

full_complexity_model = TelemacModel(
    control_file="tel_ering_restart0.5.cas",
    model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/",
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements_VITESSE_WDEPTH_filtered.csv",
    n_cpus=4,
    init_runs=5,
    calibration_parameters=["zone11",
                            "zone9",
                            "zone8",
                            "zone10",
                            "zone1",
                            "zone4",
                            "zone5",
                            "zone7"],
    param_values=[[0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.03],
                  [0.15, 0.30],
                  [0.02, 0.10]],
    calibration_quantities=["WATER DEPTH"],
    dict_output_name="model-outputs_scalar-velocity_valid",
    parameter_sampling_method="user",
    max_runs=120,
    # TelemacModel class parameters
    friction_file="friction_ering.tbl",
    tm_xd="1",
    results_filename_base="Results_ering2m3",
    stdout=6,
    python_shebang="#!/usr/bin/env python3",
    complete_bal_mode=False,
    only_bal_mode=False,
    check_inputs=False
)
results_folder_path = full_complexity_model.asr_dir
plotter = BayesianPlotter(results_folder_path=results_folder_path)

obs=full_complexity_model.observations
err=full_complexity_model.measurement_errors
n_loc=full_complexity_model.nloc
n_quantities=full_complexity_model.num_quantities
bayesian_data=full_complexity_model.read_data(results_folder_path,'BAL_dictionary.pkl')
collocation_points = full_complexity_model.read_data(results_folder_path,"collocation_points.csv")
model_evaluations = full_complexity_model.read_data(results_folder_path,"model_results.csv")
surrogate_outputs = full_complexity_model.read_data(results_folder_path,"surrogate_output_iter_100.pkl")
#IMPORT LAST TRAINED SURROGATE
sm = full_complexity_model.read_data(results_folder_path,"surrogate-gpe/bal_dkl/gpr_gpy_TP120_bal_quantities1.pkl")
#GENERATE "N" RANDOM SAMPLES FOR MODEL PARAMETERS FROM EXP. DESIGN TO VALIDATE MY SURROGATE AND
# I SAVE THEM AS NPARRAY VALIDATION SET
sm.Exp_Design.sampling_method = 'random'
sm.Exp_Design.generate_ED(n_samples=5,max_pce_deg=1)
validation_sets = sm.Exp_Design.X
# PREDICT OUTPUTS USING MY SURROGATE WITH THE VALIDATION SET
validation_sets_metamodel_output = sm.predict_(input_sets=validation_sets, get_conf_int=True)
# RUN COMPLEX MODEL WITH THE SAME VALIDATION SET AND GET THE OUTPUTS
full_complexity_model.run_multiple_simulations(collocation_points=validation_sets,
                          complete_bal_mode=False
                          )
# validation_set_complexmodel_output = full_complexity_model.model_evaluations
#
# set=collocation_points[-1].reshape(1, -1)
# lower_ci = validation_sets_metamodel_output["lower_ci"]
# upper_ci = validation_sets_metamodel_output["upper_ci"]
# plotter.plot_model_comparisons(obs, validation_sets_metamodel_output["output"], model_evaluations[-70, :].reshape(1, -1))
# plotter.plot_model_outputs_vs_locations(obs, validation_sets_metamodel_output["output"], model_evaluations[-1, :].reshape(1, -1), upper_ci, lower_ci, err)
# #
# print(validation_sets_metamodel_output)
# # full_complexity_model.run(collocation_points=validation_sets,
# #                           complete_bal_mode=False
# #                           )
# --------------------------------------------------------------------------------------------------------------------



#
#
# from pathlib import Path
# import sys
#
# # Base directory
# base_dir = Path(__file__).resolve().parent.parent.parent
# src_path = base_dir / 'src'
# hydroBayesCal_path = src_path / 'hydroBayesCal'
# sys.path.append(str(src_path))
# sys.path.append(str(hydroBayesCal_path))
#
# from user_settings import user_inputs_tm
# from hydro_simulations import HydroSimulations
# from plots.plots import BayesianPlotter
#
# results_folder_path = user_inputs_tm['results_folder_path']
# auto_saved_results_folder = Path(results_folder_path) / 'auto-saved-results'
#
# pickle_path_bayesian_dict = str(Path(auto_saved_results_folder ) / 'BAL_dictionary.pkl')
# pickle_path_surrogate_outputs = str(Path(auto_saved_results_folder ) / 'surrogate_output_iter_10.pkl')
# model_evaluations_path = str(Path(auto_saved_results_folder ) / 'model_results.npy')
# collocation_points_path = str(Path(auto_saved_results_folder ) / 'colocation_points.npy')
# path_gpe_surrogate = str(Path(auto_saved_results_folder ) / 'surrogate_R_ering2m3_ndim_5_nout_34/bal_dkl/gpr_gpy_TP25_bal_quantities1.pickle')
#
#
# user_inputs_tm['dict_output_name'] = 'MODEL_OUTPUTS_VALID'
# user_inputs_tm['init_runs'] = 2
# validation_samples = user_inputs_tm['init_runs']
# full_complexity_model = HydroSimulations(user_inputs=user_inputs_tm)
# plotter = BayesianPlotter(results_folder_path=results_folder_path)
#
# bayesian_data = full_complexity_model.read_stored_data(pickle_path_bayesian_dict)
# collocation_points = full_complexity_model.read_stored_data(collocation_points_path)
# model_evaluations = full_complexity_model.read_stored_data(model_evaluations_path)
# surrogate_outputs = full_complexity_model.read_stored_data(pickle_path_surrogate_outputs)
# sm = full_complexity_model.read_stored_data(path_gpe_surrogate)
# sm.Exp_Design.sampling_method = 'random'
# sm.Exp_Design.generate_ED(n_samples=validation_samples,max_pce_deg=1)
# validation_sets = sm.Exp_Design.X
# validation_surrogate_output = sm.predict_(input_sets=validation_sets, get_conf_int=True)
# full_complexity_model.run(collocation_points=validation_sets,
#                           bal_iteration=0,
#                           bal_new_set_parameters=None,
#                           complete_bal_mode=False
#                           )
# validation_fullcomplex_outputs = full_complexity_model.model_evaluations

