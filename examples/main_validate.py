from pathlib import Path
import sys

# Base directory
base_dir = Path(__file__).resolve().parent.parent.parent
src_path = base_dir / 'src'
hydroBayesCal_path = src_path / 'hydroBayesCal'
sys.path.append(str(src_path))
sys.path.append(str(hydroBayesCal_path))

from user_settings import user_inputs_tm
from hydro_simulations import HydroSimulations
from plots.plots import BayesianPlotter

results_folder_path = user_inputs_tm['results_folder_path']
auto_saved_results_folder = Path(results_folder_path) / 'auto-saved-results'

pickle_path_bayesian_dict = str(Path(auto_saved_results_folder ) / 'BAL_dictionary.pkl')
pickle_path_surrogate_outputs = str(Path(auto_saved_results_folder ) / 'surrogate_output_iter_10.pkl')
model_evaluations_path = str(Path(auto_saved_results_folder ) / 'model_results.npy')
collocation_points_path = str(Path(auto_saved_results_folder ) / 'colocation_points.npy')
path_gpe_surrogate = str(Path(auto_saved_results_folder ) / 'surrogate_R_ering2m3_ndim_5_nout_34/bal_dkl/gpr_gpy_TP25_bal_quantities1.pickle')


user_inputs_tm['dict_output_name'] = 'MODEL_OUTPUTS_VALID'
user_inputs_tm['init_runs'] = 2
validation_samples = user_inputs_tm['init_runs']
full_complexity_model = HydroSimulations(user_inputs=user_inputs_tm)
plotter = BayesianPlotter(results_folder_path=results_folder_path)

bayesian_data = full_complexity_model.read_stored_data(pickle_path_bayesian_dict)
collocation_points = full_complexity_model.read_stored_data(collocation_points_path)
model_evaluations = full_complexity_model.read_stored_data(model_evaluations_path)
surrogate_outputs = full_complexity_model.read_stored_data(pickle_path_surrogate_outputs)
sm = full_complexity_model.read_stored_data(path_gpe_surrogate)
sm.Exp_Design.sampling_method = 'random'
sm.Exp_Design.generate_ED(n_samples=validation_samples,max_pce_deg=1)
validation_sets = sm.Exp_Design.X
validation_surrogate_output = sm.predict_(input_sets=validation_sets, get_conf_int=True)
full_complexity_model.run(collocation_points=validation_sets,
                          bal_iteration=0,
                          bal_new_set_parameters=None,
                          complete_bal_mode=False
                          )
validation_fullcomplex_outputs = full_complexity_model.model_evaluations

