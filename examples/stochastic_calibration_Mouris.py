# Stochastic calibration of model using surrogate assistance bayesian inversion.
# Surrogate model generated using gaussian processes

# Methodology and code logic by: Dr.-Ing. habil. Sergey Oladyshkin
# Transcription to python by: Eduardo Acuna and Farid Mohammadi

# Import libraries
import sys, os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from auxiliary_functions_BAL import *

# ---------------------------------------------------------------------------------------------------------------------
# USER INPUT  ---------------------------------------------------------------------------------------------------------

# Prior distribution of calibration parameters
N = 4  # number of calibration parameters (uncertainty parameters)
NS = 500000  # sample size I take from my prior distributions
input_distribution = np.zeros((NS, N))
input_distribution[:, 0] = np.random.uniform(0.01, 0.1, NS)
input_distribution[:, 1] = np.random.uniform(0.05, 0.4, NS)
input_distribution[:, 2] = np.random.uniform(200, 500, NS)
input_distribution[:, 3] = np.random.uniform(0.8, 1.7, NS)
np.savetxt("input_distribution.txt", input_distribution)
parameters_name = ["CLASSES CRITICAL SHEAR STRESS FOR MUD DEPOSITION",
                   "LAYERS CRITICAL EROSION SHEAR STRESS OF THE MUD",
                   "LAYERS MUD CONCENTRATION",
                   "CLASSES SETTLING VELOCITIES"]

# Observations (measured values that are going to be used for calibration)
temp = np.loadtxt(r"Y:\Abt1\Temp-Mitarbeiter\Mouris\Paper_Draft\Paper2\GPE_BAL_Telemac_V3b\55_it\main\calibration_points.txt")
n_points = temp.shape[0]
nodes = temp[:, 0].reshape(-1, 1)
observations = temp[:, 1].reshape(-1, 1)
observations_error = temp[:, 2]

# Calibration parameters
calibration_variable = "BOTTOM"

# Paths
path_results = r"Y:\Abt1\Temp-Mitarbeiter\Mouris\Paper_Draft\Paper2\GPE_BAL_Telemac_V3b\55_it\results"

# END OF USER INPUT  --------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Part 2. Read initial collocation points  ----------------------------------------------------------------------------
temp = np.loadtxt(os.path.abspath(os.path.expanduser(path_results))+"/parameter_file.txt", dtype=np.str, delimiter=';')
simulation_names = temp[:, 0]
collocation_points = temp[:, 1:].astype(np.float)
n_simulation = collocation_points.shape[0]

# Part 3. Read the previously computed simulations of the numerical model in the initial collocation points -----------
temp = np.loadtxt(os.path.abspath(os.path.expanduser(path_results)) + "/" + simulation_names[0] + "_" +
                  calibration_variable + ".txt")
model_results = np.zeros((collocation_points.shape[0], temp.shape[0]))
for i, name in enumerate(simulation_names):
    model_results[i, :] = np.loadtxt(os.path.abspath(os.path.expanduser(path_results))+"/" + name + "_" +
                                     calibration_variable + ".txt")[:, 1]

# Part 4. Computation of surrogate model prediction in MC points using gaussian processes --------------------------
surrogate_prediction = np.zeros((n_points, input_distribution.shape[0]))
surrogate_std = np.zeros((n_points, input_distribution.shape[0]))

for i, model in enumerate(model_results.T):
    kernel = RBF(length_scale=[0.05, 0.2, 150, 0.5], length_scale_bounds=[(0.001, 0.1), (0.001, 0.4), (5, 300), (0.02, 2)]) * np.var(model)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(collocation_points, model)
    surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(input_distribution, return_std=True)

loocv_error = np.loadtxt(r"Y:\Abt1\Temp-Mitarbeiter\Mouris\Paper_Draft\Paper2\GPE_BAL_Telemac_V3\main\loocv_error_variance.txt")[:, 1]
total_error = (observations_error ** 2 + loocv_error)*5
#total_error = (observations_error ** 2)

# Part 6. Compute likelihood-----------------------------------------------------
likelihood = compute_fast_likelihood(surrogate_prediction.T, observations.T, total_error).reshape(1, surrogate_prediction.T.shape[0])
print(np.max(likelihood))
print(np.count_nonzero(likelihood))
accepted = likelihood / np.amax(likelihood) >= np.random.rand(1, surrogate_prediction.shape[1])
print(np.sum(accepted))

best5lik = likelihood[0, np.argpartition(likelihood, -5, axis=1)[:, -5:][0,:]]
best5param = input_distribution[np.argpartition(likelihood, -5, axis=1)[:, -5:][0,:],:]
best5sim = surrogate_prediction[:, np.argpartition(likelihood, -5, axis=1)[:, -5:][0,:]]
np.savetxt("best5lik.txt", best5lik)
np.savetxt("best5param.txt", best5param)
np.savetxt("best5sim.txt", best5sim)
# Best parameter
posterior = input_distribution[accepted[0,:],:]
np.savetxt("posterior2.txt", posterior)
x=1

