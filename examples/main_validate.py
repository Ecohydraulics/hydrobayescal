import sys,os
import numpy as np
import bayesvalidrox as bvr
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
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurementsWDEPTH_filtered.csv",
    n_cpus=4,
    init_runs=1,
    calibration_parameters=["zone1", "zone2", "zone3", "zone4", "zone5","zone6","zone7",
                            "vg_zone7-par2", "vg_zone7-par3"],
    # param_values=[[0.01, 0.18],
    #               [0.01, 0.18],
    #               [0.01, 0.18],
    #               [0.01, 0.18],
    #               [0.01, 0.18],
    #               [0.01, 0.03],
    #               [0.15, 0.30],
    #               [0.02, 0.10]],
    calibration_quantities=["WATER DEPTH"],
    dict_output_name="model-outputs",
    # TelemacModel class parameters
    friction_file="friction_ering.tbl",
    tm_xd="1",
    results_filename_base="Results_ering2m3",
    # stdout=6,
    # python_shebang="#!/usr/bin/env python3",
    complete_bal_mode=False,
    only_bal_mode=False,
    check_inputs=False,
    delete_complex_outputs=True,
    validation=True,
)

results_folder_path = full_complexity_model.asr_dir
plotter = BayesianPlotter(results_folder_path=results_folder_path)
sm = full_complexity_model.read_data(results_folder_path, "surrogate-gpe/bal_dkl/gpr_gpy_TP100_bal_quantities2.pkl")
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
n_loc = full_complexity_model.nloc
n_quantities = full_complexity_model.num_quantities
def get_validation_data():

    #IMPORT LAST TRAINED SURROGATE
    #sm = full_complexity_model.read_data(results_folder_path,"surrogate-gpe/gpr_gpy_TP125_bal_quantities2.pkl")
    #GENERATE "N" RANDOM SAMPLES FOR MODEL PARAMETERS FROM EXP. DESIGN TO VALIDATE MY SURROGATE AND
    # I SAVE THEM AS NPARRAY VALIDATION SET
    sm.exp_design.sampling_method = 'user'
    # sm.exp_design.generate_ED(n_samples=10,max_pce_deg=1)
    sm.exp_design.X = [[0.066, 0.066, 0.012, 0.006, 0.22,
                          4.35,0.4]]
    validation_sets = sm.exp_design.X
    # validation_sets=full_complexity_model.read_data(results_folder_path,"collocation-points-validation.csv")
    # PREDICT OUTPUTS USING THE SURROGATE WITH THE VALIDATION SET
    validation_sets_metamodel_output = sm.predict_(input_sets=validation_sets)# , get_conf_int=True
    # RUN COMPLEX MODEL WITH THE SAME VALIDATION SET AND GET THE OUTPUTS
    full_complexity_model.run_multiple_simulations(collocation_points=validation_sets,
                              complete_bal_mode=False,validation=full_complexity_model.validation
                              )
    # validation_set_complexmodel_output = full_complexity_model.output_processing(results_folder_path + "/model-outputs-validation.json"
    #                                                                              ,False,False)
    # num_simulations, num_columns = validation_set_complexmodel_output.shape

    validation_set_complexmodel_output = full_complexity_model.model_evaluations
    num_simulations, num_columns = validation_set_complexmodel_output.shape
    # Separate the columns for each quantity
    first_quantity_columns = validation_set_complexmodel_output[:, 0:num_columns:2]  # Take every second column starting from 0
    second_quantity_columns = validation_set_complexmodel_output[:, 1:num_columns:2] # Take every second column starting from 1

    validation_set_complexmodel_output = np.hstack((first_quantity_columns, second_quantity_columns))

    # Concatenate the rearranged quantities horizontally
    return validation_set_complexmodel_output,validation_sets_metamodel_output


    # bayesian_data=full_complexity_model.read_data(results_folder_path,'BAL_dictionary.pkl')
    # collocation_points = full_complexity_model.read_data(results_folder_path,"collocation-points.csv")
    # model_evaluations = full_complexity_model.read_data(results_folder_path,"model-results.csv")
    #surrogate_outputs = full_complexity_model.read_data(results_folder_path,"surrogate_output_iter_15.pkl")

    #sm.Exp_Design.X=np.array([0.06,0.13,0.09,0.12,0.0105,0.018,0.24,0.048]).reshape(1, -1) # post calibration set
    #sm.Exp_Design.X=np.array([0.066,0.066,0.066,0.066,0.066,0.22,0.027,0.077]).reshape(1, -1) # pre calibration set

    # RUN COMPLEX MODEL WITH THE SAME VALIDATION SET AND GET THE OUTPUTS

#
# set=collocation_points[-1].reshape(1, -1)
# lower_ci = validation_sets_metamodel_output["lower_ci"]
# upper_ci = validation_sets_metamodel_output["upper_ci"]
# plotter.plot_model_comparisons(obs, validation_sets_metamodel_output["output"], validation_set_complexmodel_output[-1, :].reshape(1, -1))
# plotter.plot_model_outputs_vs_locations(obs, validation_sets_metamodel_output["output"], model_evaluations[-1, :].reshape(1, -1), upper_ci, lower_ci, err)
#plotter.plot_model_comparisons()

if __name__ == "__main__":
    complex_output,sm_output,=get_validation_data()
    mid_index_locations = sm_output["output"].shape[1] // 2
    sm_output1 = sm_output["output"][:, :mid_index_locations]
    sm_output2 = sm_output["output"][:, mid_index_locations:]
    complex_output1=complex_output[:, :mid_index_locations]
    complex_output2=complex_output[:, mid_index_locations:]
    # Get the first half of the columns
    obs1 = obs[:, :mid_index_locations]
    err1 = err[:mid_index_locations]
    # Get the second half of the columns
    obs2 = obs[:, mid_index_locations:]
    err2 = err[mid_index_locations:]
    # print(sm_output1)
    # print(sm_output2)
    # print(complex_output)

    # collocation_points_validation = full_complexity_model.read_data(results_folder_path,"collocation-points-validation.csv")
    # validation_sets_metamodel_output = sm.predict_(input_sets=collocation_points_validation, get_conf_int=True)
    # validation_set_complexmodel_output = full_complexity_model.output_processing(os.path.join(full_complexity_model.asr_dir,
    #                                                                              f'{full_complexity_model.dict_output_name}.json'))
    # plotter.plot_correlation(sm_output1,complex_output1,["WATER DEPTH"],n_loc_=n_loc,fig_title="water depth")
    # plotter.plot_correlation(sm_output2,complex_output2,["SCALAR VELOCITY"],n_loc_=n_loc,fig_title="velocity")

    plotter.plot_model_outputs_vs_locations(observed_values=obs1, surrogate_outputs=sm_output1[-1, :].reshape(1, -1), complex_model_outputs=complex_output1[-1, :].reshape(1, -1),measurement_error=err1)
    #
    plotter.plot_model_outputs_vs_locations(observed_values=obs2, surrogate_outputs=sm_output2[-1, :].reshape(1, -1), complex_model_outputs=complex_output2[-1, :].reshape(1, -1),measurement_error=err2)

