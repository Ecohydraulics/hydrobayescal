import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from scipy.interpolate import griddata

# Assuming __file__ is defined, this may not work in some environments like Jupyter notebooks
base_dir = Path(__file__).resolve().parent.parent.parent.parent

examples_path = base_dir / 'examples'
auto_saved_results = examples_path / 'ering_1quantity' / 'auto-saved-results'
hydrobayesCal_path = base_dir / 'src' / 'hydroBayesCal'
sys.path.append(str(base_dir))
sys.path.append(str(examples_path))
sys.path.append(str(auto_saved_results))
sys.path.append(str(auto_saved_results))
sys.path.append(str(hydrobayesCal_path))

# Import necessary modules after appending paths
from user_settings_ering_1quantity import user_inputs_tm
from hydro_simulations import HydroSimulations
from surrogate_modelling.gpe_gpytorch import *

pickle_path_bayesian_dict = str(auto_saved_results / 'BAL_dictionary.pkl')
pickle_path_surrogate_outputs = str(auto_saved_results / 'surrogate_output_iter_85.pkl')
model_evaluations_path = str(auto_saved_results / 'model_results.npy')
collocation_points_path = str(auto_saved_results / 'colocation_points.npy')
num_calibration_quantities = 1

full_complexity_model = HydroSimulations(user_inputs=user_inputs_tm)
bayesian_data = full_complexity_model.read_stored_data(pickle_path_bayesian_dict)
surrogate_outputs = full_complexity_model.read_stored_data(pickle_path_surrogate_outputs)
model_evaluations = full_complexity_model.read_stored_data(model_evaluations_path)
collocation_points = full_complexity_model.read_stored_data(collocation_points_path)

num_samples = model_evaluations.shape[0]
num_collocation_points = collocation_points.shape[0]
train_ratio = 0.10

# Calculate the number of training and testing samples
num_train_samples = int(num_samples // num_calibration_quantities * train_ratio)
num_test_samples = num_samples // num_calibration_quantities - num_train_samples

# Split the data
first_quantity_data = model_evaluations[:num_samples // num_calibration_quantities]
second_quantity_data = model_evaluations[num_samples // num_calibration_quantities:]

first_quantity_train = first_quantity_data[:num_train_samples]
first_quantity_test = first_quantity_data[num_train_samples:]

second_quantity_train = second_quantity_data[:num_train_samples]
second_quantity_test = second_quantity_data[num_train_samples:]

collocation_points_model_train = collocation_points[:num_train_samples]
collocation_points_model_test = collocation_points[num_train_samples:]

# Stack vertically the 2 quantities for model train and model test
model_evaluations_train = np.vstack((first_quantity_train, second_quantity_train))
model_evaluations_test = np.vstack((first_quantity_test, second_quantity_test))

class Plotter:
    def __init__(self):
        pass

    def plot_predicted_vs_observed(self, observed, predicted, title):
        plt.figure(figsize=(10, 6))
        plt.scatter(observed, predicted, alpha=0.5)
        plt.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--', lw=2)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title(title)
        plt.show()

    def plot_residuals(self, observed, predicted, title):
        residuals = observed - predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.show()

    def plot_3d_surface(self, test_collocation_points, predicted_surrogate_outputs, param_indices=(0, 1), num_points=50):
        param1_range = np.linspace(np.min(test_collocation_points[:, param_indices[0]]), np.max(test_collocation_points[:, param_indices[0]]), num_points)
        param2_range = np.linspace(np.min(test_collocation_points[:, param_indices[1]]), np.max(test_collocation_points[:, param_indices[1]]), num_points)
        param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)

        param_grid_flattened = np.vstack([param1_grid.ravel(), param2_grid.ravel()]).T
        predicted_grid = griddata(test_collocation_points[:, param_indices], predicted_surrogate_outputs[:, 0].flatten(), param_grid_flattened, method='linear')
        predicted_grid = predicted_grid.reshape(param1_grid.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(param1_grid, param2_grid, predicted_grid, cmap='viridis', edgecolor='none')

        ax.set_xlabel(f'Parameter {param_indices[0] + 1}')
        ax.set_ylabel(f'Parameter {param_indices[1] + 1}')
        ax.set_zlabel('Model Prediction')
        ax.set_title('Surrogate Model Predictions')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()

plotter = Plotter()

if num_calibration_quantities == 1:
    ndim = len(user_inputs_tm['calibration_parameters'])
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ndim))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    likelihood.noise = 1e-5

    sm = GPyTraining(collocation_points=collocation_points, model_evaluations=model_evaluations, likelihood=likelihood, kernel=kernel, training_iter=100, optimizer="adam", verbose=False)
    sm.train_()
    predicted_outputs_test = sm.predict_(input_sets=collocation_points_model_test, get_conf_int=True)

    mse_1 = mean_squared_error(model_evaluations_test.flatten(), predicted_outputs_test['output'].flatten())
    r2_1 = r2_score(model_evaluations_test.flatten(), predicted_outputs_test['output'].flatten())
    print(f'MSE: {mse_1:.9f}')
    print(f'R²: {r2_1:.3f}')

    plotter.plot_predicted_vs_observed(model_evaluations_test.flatten(), predicted_outputs_test['output'].flatten(), 'Predicted vs Observed (First Quantity)')
    plotter.plot_residuals(model_evaluations_test.flatten(), predicted_outputs_test['output'].flatten(), 'Residuals Plot (Second Quantity)')
else:
    # combined_kernel = gpytorch.kernels.ScaleKernel(
    #    gpytorch.kernels.RBFKernel(lengthscale=1.0) * gpytorch.kernels.LinearKernel()
    # )
    ndim = len(user_inputs_tm['calibration_parameters'])
    combined_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ndim))

    kernel = (combined_kernel, combined_kernel)
    #kernel = gpytorch.kernels.RBFKernel() * gpytorch.kernels.LinearKernel()
    #kernel = (gpytorch.kernels.RBFKernel(), gpytorch.kernels.RBFKernel())
    multi_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    multi_sm = MultiGPyTraining(collocation_points_model_train, model_evaluations_train, kernel, training_iter=150, likelihood=multi_likelihood, optimizer="adam", lr=0.01, number_quantities=2)

    multi_sm.train()

    predicted_outputs_test = multi_sm.predict_(input_sets=collocation_points_model_test)
    mid_index_training_sets = model_evaluations_test.shape[0] // 2
    predicted_outputs_test = predicted_outputs_test['output']
    mid_index_locations = predicted_outputs_test.shape[1] // 2
    predicted_outputs_test1 = predicted_outputs_test[:, :mid_index_locations]
    predicted_outputs_test2 = predicted_outputs_test[:, mid_index_locations:]
    observed_output_test1 = model_evaluations_test[:mid_index_training_sets, :]
    observed_output_test2 = model_evaluations_test[mid_index_training_sets:, :]

    mse_1 = mean_squared_error(observed_output_test1.flatten(), predicted_outputs_test1.flatten())
    r2_1 = r2_score(observed_output_test1.flatten(), predicted_outputs_test1.flatten())
    mse_2 = mean_squared_error(observed_output_test2.flatten(), predicted_outputs_test2.flatten())
    r2_2 = r2_score(observed_output_test2.flatten(), predicted_outputs_test2.flatten())

    print(f'MSE: {mse_1:.9f}')
    print(f'R²: {r2_1:.3f}')
    print(f'MSE: {mse_2:.9f}')
    print(f'R²: {r2_2:.3f}')

    plotter.plot_predicted_vs_observed(observed_output_test1.flatten(), predicted_outputs_test1.flatten(), 'Predicted vs Observed (First Quantity)')
    plotter.plot_predicted_vs_observed(observed_output_test2.flatten(), predicted_outputs_test2.flatten(), 'Predicted vs Observed (Second Quantity)')
    plotter.plot_residuals(observed_output_test2.flatten(), predicted_outputs_test2.flatten(), 'Residuals Plot (Second Quantity)')
