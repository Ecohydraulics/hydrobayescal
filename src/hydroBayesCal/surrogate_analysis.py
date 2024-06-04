import pickle
from surrogate_modelling.bal_functions import BayesianInference, SequentialDesign
from surrogate_modelling.gpe_skl import *
from surrogate_modelling.gpe_gpytorch import *
from surrogate_modelling.inputs import Input
from surrogate_modelling.exp_design_ import ExpDesign
from user_settings import user_inputs

# Specify the path to your pickle file
pickle_file_path = '/home/IWS/hidalgo/Documents/hydrobayescal/examples/donau/auto-saved-results/surrogate_R_donau_ndim_4_nout_100/bal_dkl/gpr_gpy_TP25_bal.pickle'

# Open the pickle file in binary read mode
with open(pickle_file_path, 'rb') as file:
    # Load the contents of the pickle file
    data = pickle.load(file)

# Display the contents
print(data)
ndim=len(user_inputs['calib_parameter_list'])
Inputs = Input()
# # One "Marginal" for each parameter.
for i in range(ndim):
    Inputs.add_marginals()  # Create marginal for parameter "i"
    Inputs.Marginals[i].name = user_inputs['calib_parameter_list'][i]  # Parameter name
    Inputs.Marginals[i].dist_type = 'uniform'  # Parameter distribution (see exp_design.py --> build_dist()
    Inputs.Marginals[i].parameters = user_inputs['parameter_ranges_list'][i]  # Inputs needed for distribution

# # Experimental design: ....................................................................

exp_design = ExpDesign(input_object=Inputs,
                       exploit_method='bal',  # bal, space_filling, sobol
                       explore_method='random',  # method to sample from parameter set for active learning
                       training_step=1,  # No. of training points to sample in each iteration
                       sampling_method=user_inputs['parameter_sampling_method'],
                       # how to sample the initial training points
                       main_meta_model='gpr',  # main surrogate method: 'gpr' or 'apce'
                       n_initial_tp=user_inputs['init_runs'],  # Number of initial training points (min = n_trunc*2)
                       n_max_tp=user_inputs['n_max_tp'],  # max number of tp to use
                       training_method='sequential',  # normal (train only once) or sequential (Active Learning)
                       util_func='dkl',  # criteria for bal (dkl, bme, ie, dkl_bme) or SF (default: global_mc)
                       eval_step=user_inputs['eval_steps'],  # every how many iterations to evaluate the surrogate
                       secondary_meta_model=False  # only gpr is available
                       )

prior = exp_design.generate_samples(user_inputs['n_samples'])
prior_logpdf = np.log(exp_design.JDist.pdf(prior.T)).reshape(-1)
