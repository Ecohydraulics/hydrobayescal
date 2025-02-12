import sys
import os
import numpy as np
import bayesvalidrox as bvr

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
    control_file="tel_ering_mu_restart.cas",
    model_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/",
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/synthetic-data-pool.csv",
    n_cpus=8,
    init_runs=30,
    calibration_parameters=["zone11", "zone12", "zone13", "zone14", "zone15"],
    calibration_quantities=["SCALAR VELOCITY","WATER DEPTH"],
    extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG"],
    dict_output_name="model-outputs-valid",
    parameter_sampling_method="sobol",
    friction_file="friction_ering_MU.tbl",
    tm_xd="1",
    results_filename_base="results2m3_mu",
    complete_bal_mode=False,
    only_bal_mode=False,
    check_inputs=False,
    delete_complex_outputs=True,
    multitask_selection="locations"
)

results_folder_path = full_complexity_model.asr_dir
restart_data_folder = full_complexity_model.restart_data_folder
plotter = BayesianPlotter(results_folder_path=results_folder_path)
obs = full_complexity_model.observations
err = full_complexity_model.measurement_errors
n_loc = full_complexity_model.nloc
n_quantities = full_complexity_model.num_calibration_quantities

# Import last trained surrogate model
sm = full_complexity_model.read_data(results_folder_path, f"surrogate-gpe/bal_dkl/gpr_gpy_TP30_bal_quantities_{full_complexity_model.calibration_quantities}_{full_complexity_model.calibration_parameters}_{full_complexity_model.multitask_selection}.pkl")

# Load validation sets from CSV
validation_sets = full_complexity_model.read_data(restart_data_folder, "collocation-points-validation.csv")

# Predict outputs using the surrogate model
sm_predictions = sm.predict_(input_sets=validation_sets, get_conf_int=True)

# Load complex model outputs from CSV
cm_outputs = full_complexity_model.output_processing(output_data_path=os.path.join(full_complexity_model.restart_data_folder,
                                                                                          f'collocation-points-validation.json'),
                                                            validation=full_complexity_model.validation,
                                                            filter_outputs=True,
                                                            run_range_filtering=(1, full_complexity_model.init_runs))

# Number of columns per quantity
num_simulations, num_columns = cm_outputs.shape
columns_per_quantity = num_columns // n_quantities

# Split outputs dynamically for each quantity
cm_outputs_split = {}
sm_outputs_split = {}
sm_upper_ci_split = {}
sm_lower_ci_split = {}
obs_split = {}
err_split = {}

for i in range(n_quantities):
    cm_outputs_split[f'cm_outputs_{i + 1}'] = cm_outputs[:, i::n_quantities]
    sm_outputs_split[f'sm_outputs_{i + 1}'] = sm_predictions["output"][:, i::n_quantities]
    sm_upper_ci_split[f'sm_upper_ci_{i + 1}'] = sm_predictions["upper_ci"][:, i::n_quantities]
    sm_lower_ci_split[f'sm_lower_ci_{i + 1}'] = sm_predictions["lower_ci"][:, i::n_quantities]
    obs_split[f'obs_{i + 1}'] = obs[:, i::n_quantities]
    err_split[f'err_{i + 1}'] = err[i::n_quantities]

# Plot results dynamically for each quantity
for i in range(n_quantities):
    cm_output = cm_outputs_split[f'cm_outputs_{i + 1}']
    sm_output = sm_outputs_split[f'sm_outputs_{i + 1}']
    sm_upper_ci = sm_upper_ci_split[f'sm_upper_ci_{i + 1}']
    sm_lower_ci = sm_lower_ci_split[f'sm_lower_ci_{i + 1}']
    obs_quantity = obs_split[f'obs_{i + 1}']
    err_quantity = err_split[f'err_{i + 1}']

    plotter.plot_validation_results(obs_quantity, sm_output, cm_output,gpe_lower_ci=sm_lower_ci ,gpe_upper_ci=sm_upper_ci,
                                    measurement_error=err_quantity, plot_ci=True,N=2)
    plotter.plot_validation_locations(sm_output, cm_output, sm_lower_ci, sm_upper_ci,selected_locations=[2,15,26,28])
    plotter.plot_model_outputs_vs_locations(
        observed_values=obs_quantity,
        surrogate_outputs=sm_output[-1, :].reshape(1, -1),
        complex_model_outputs=cm_output[-1, :].reshape(1, -1),
        gpe_lower_ci=sm_lower_ci[-1, :].reshape(1, -1),
        gpe_upper_ci=sm_upper_ci[-1, :].reshape(1, -1),
        measurement_error=err_quantity)

    # Plot Bayesian results
    # plotter.plot_bme_re(bayesian_dict=bayesian_data, num_bal_iterations=35, plot_type='both')
    # plotter.plot_posterior_updates(
    #     posterior_arrays=bayesian_data['posterior'],
    #     parameter_names=full_complexity_model.calibration_parameters,
    #     prior=bayesian_data['prior'],
    #     param_values=full_complexity_model.param_values,
    #     iterations_to_plot=[0, 35],
    #     bins=20,
    #     plot_prior=True,
    # )
    # print(sm_output1)
    # print(sm_output2)
    # print(complex_output)

    # collocation_points_validation = full_complexity_model.read_data(results_folder_path,"collocation-points-validation.csv")
    # validation_sets_metamodel_output = sm.predict_(input_sets=collocation_points_validation, get_conf_int=True)
    # validation_set_complexmodel_output = full_complexity_model.output_processing(os.path.join(full_complexity_model.asr_dir,
    #                                                                              f'{full_complexity_model.dict_output_name}.json'))
    # plotter.plot_correlation(sm_output1,complex_output1,["WATER DEPTH"],n_loc_=n_loc,fig_title="water depth")
    # plotter.plot_correlation(sm_output2,complex_output2,["SCALAR VELOCITY"],n_loc_=n_loc,fig_title="velocity")


