from pathlib import Path
import sys

# Base directory
base_dir = Path(__file__).resolve().parent.parent

src_path = base_dir / 'src'
hydroBayesCal_path = src_path / 'hydroBayesCal'
sys.path.append(str(src_path))
sys.path.append(str(hydroBayesCal_path))

#from user_settings import user_inputs_tm
from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter

full_complexity_model = TelemacModel(
    res_dir="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/",
    calibration_pts_file_path="/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measurements_VITESSE_WDEPTH_filtered.csv",
    calibration_parameters=["zone11",
                            "zone9",
                            "zone8",
                            "zone10",
                            "zone1",
                            "zone4",
                            "zone5",
                            "zone7"],
    init_runs=20,
    param_values=[[0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.18],
                  [0.01, 0.03],
                  [0.15, 0.30],
                  [0.02, 0.10]],
    calibration_quantities=["SCALAR VELOCITY"],
    dict_output_name="model-outputs_scalar-velocity",
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
surrogate_outputs = full_complexity_model.read_data(results_folder_path,"surrogate_output_iter_70.pkl")


plotter.plot_bme_re(bayesian_dict=bayesian_data,
            num_bal_iterations=70,
                    plot_type='both')
plotter.plot_combined_bal(collocation_points = collocation_points,
                  n_init_tp = full_complexity_model.init_runs,
                  bayesian_dict = bayesian_data)
plotter.plot_posterior_updates(posterior_arrays = bayesian_data['posterior'],
                       parameter_names = full_complexity_model.calibration_parameters,
                       prior = bayesian_data['prior'],
                        iterations_to_plot=[70])
plotter.plot_bme_3d(param_sets=collocation_points,
                    param_ranges=full_complexity_model.param_values,
                    param_names=full_complexity_model.calibration_parameters,
                    bme_values=bayesian_data['BME'],
                    param_indices=(1, 4),
                    grid_size=200,
                    last_iterations=30
                    )
