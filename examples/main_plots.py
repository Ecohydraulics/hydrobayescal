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
import sys

# Base directory
base_dir = Path(__file__).resolve().parent.parent

src_path = base_dir / 'src'
hydroBayesCal_path = src_path / 'hydroBayesCal'
sys.path.append(str(src_path))
sys.path.append(str(hydroBayesCal_path))

from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter

#Instance of Telemac Model for plotting results (calibration)
full_complexity_model = TelemacModel(
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements_VITESSE_filtered.csv",
    init_runs=30,
    calibration_parameters=["zone11", "zone9", "zone10", "zone1", "zone5", "zone7", "vg_zone7-par1",
                            "vg_zone7-par2", "vg_zone7-par3",
                            "ROUGHNESS COEFFICIENT OF BOUNDARIES", "VELOCITY DIFFUSIVITY"],
    param_values=[[0.01, 0.18], [0.002, 0.07], [0.01, 0.18], [0.002, 0.07], [0.15, 0.30], [0.02, 0.10], [0.7, 1.3],
                  [5, 6],
                  [0.3, 0.6], [0.013, 0.035], [0.000015, 0.0015]],
    calibration_quantities=["SCALAR VELOCITY"],
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
                        iterations_to_plot=[120])
# plotter.plot_model_comparisons(obs, sm_outputs[-1, :].reshape(1, -1), cm_outputs[-1, :].reshape(1, -1))
# plotter.plot_model_outputs_vs_locations(obs, sm_outputs[-1, :].reshape(1, -1), cm_outputs[-1, :].reshape(1, -1), sm_upper_ci[-1, :].reshape(1, -1), sm_lower_ci[-1, :].reshape(1, -1), err)
plotter.plot_bme_3d(param_sets=collocation_points,
                    param_ranges=full_complexity_model.param_values,
                    param_names=full_complexity_model.calibration_parameters,
                    bme_values=bayesian_data['BME'],
                    param_indices=(3, 6),
                    grid_size=400,
                    last_iterations=30,
                    plot_criteria="BME"
                    )
plotter.plot_bme_3d(param_sets=collocation_points,
                    param_ranges=full_complexity_model.param_values,
                    param_names=full_complexity_model.calibration_parameters,
                    bme_values=bayesian_data['RE'],
                    param_indices=(3, 6),
                    grid_size=400,
                    last_iterations=30,
                    plot_criteria="RE"
                    )
