from pathlib import Path
import sys

# Base directory
base_dir = Path(__file__).resolve().parent.parent

src_path = base_dir / 'src'
hydroBayesCal_path = src_path / 'hydroBayesCal'
sys.path.append(str(src_path))
sys.path.append(str(hydroBayesCal_path))

#from user_settings import user_inputs_tm
from hysim import HydroSimulations
from plots.plots import BayesianPlotter

results_folder_path = user_inputs_tm['results_folder_path']
auto_saved_results_folder = Path(results_folder_path) / 'auto-saved-results'

pickle_path_bayesian_dict = str(Path(auto_saved_results_folder ) / 'BAL_dictionary.pkl')
pickle_path_surrogate_outputs = str(Path(auto_saved_results_folder ) / 'surrogate_output_iter_5.pkl')
model_evaluations_path = str(Path(auto_saved_results_folder ) / 'model_results.npy')
collocation_points_path = str(Path(auto_saved_results_folder ) / 'colocation_points.npy')
#path_gpe_surrogate = str(Path(auto_saved_results_folder ) / 'surrogate_R_ering2m3_ndim_5_nout_34/bal_dkl/gpr_gpy_TP25_bal_quantities1.pickle')



full_complexity_model = HydroSimulations(user_inputs=user_inputs_tm)
plotter = BayesianPlotter(results_folder_path=results_folder_path)

bayesian_data = full_complexity_model.read_stored_data(pickle_path_bayesian_dict)
collocation_points = full_complexity_model.read_stored_data(collocation_points_path)
model_evaluations = full_complexity_model.read_stored_data(model_evaluations_path)
surrogate_outputs = full_complexity_model.read_stored_data(pickle_path_surrogate_outputs)
#sm = full_complexity_model.read_stored_data(path_gpe_surrogate)


num_calibration_quantities = 1

plotter.plot_bme_re(bayesian_dict=bayesian_data,
            num_bal_iterations=5,
                    plot_type='both')
plotter.plot_combined_bal(collocation_points = collocation_points,
                  n_init_tp = user_inputs_tm['init_runs'],
                  bayesian_dict = bayesian_data)
plotter.plot_posterior_updates(posterior_arrays = bayesian_data['posterior'],
                       parameter_names = user_inputs_tm['calibration_parameters'],
                       prior = bayesian_data['prior'],
                        iterations_to_plot=[5])
plotter.plot_bme_3d(param_values = collocation_points,
                            param_ranges=user_inputs_tm['param_values'],
                            param_names = user_inputs_tm['calibration_parameters'],
                            bme_values=bayesian_data['BME'],
                            param_indices=(1, 3),
                            grid_size=200,
                            last_iterations = 5
                             )
