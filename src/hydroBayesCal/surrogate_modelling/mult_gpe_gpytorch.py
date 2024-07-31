import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from scipy.interpolate import griddata


base_dir = Path(__file__).resolve().parent.parent.parent.parent

examples_path = base_dir / 'examples'
auto_saved_results = examples_path / 'ering_1quantity' / 'auto-saved-results'
hydrobayesCal_path = base_dir / 'src' / 'hydroBayesCal'
sys.path.append(str(base_dir))
sys.path.append(str(examples_path))
sys.path.append(str(auto_saved_results))
sys.path.append(str(auto_saved_results))
sys.path.append(str(hydrobayesCal_path))



from user_settings_ering_1quantity import user_inputs_tm
from hydro_simulations import HydroSimulations
from surrogate_modelling.gpe_gpytorch import *


pickle_path_bayesian_dict = str(auto_saved_results / 'BAL_dictionary.pkl')
pickle_path_surrogate_outputs = str(auto_saved_results / 'surrogate_output_iter_3.pkl')
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
train_ratio = 0.7

# Calculate the number of training and testing samples.
num_train_samples = int(num_samples // num_calibration_quantities * train_ratio)
num_test_samples = num_samples // num_calibration_quantities - num_train_samples

# Split into two halves
# (first half of the data corresponding to the first calibration quantity and the second half corresponding to the second calibration quantity).
first_quantity_data = model_evaluations[:num_samples // num_calibration_quantities]
second_quantity_data = model_evaluations[num_samples // num_calibration_quantities:]

# Split first model data
first_quantity_train = first_quantity_data[:num_train_samples]
first_quantity_test = first_quantity_data[num_train_samples:]

# Split second model data
second_quantity_train = second_quantity_data[:num_train_samples]
second_quantity_test = second_quantity_data[num_train_samples:]
# Split collocation points data

collocation_points_model_train = collocation_points[:num_train_samples]
collocation_points_model_test = collocation_points[num_train_samples:]

#Stack vertically the 2 quantities for model train and model test.
model_evaluations_train = np.vstack((first_quantity_train, second_quantity_train))
model_evaluations_test = np.vstack((first_quantity_test, second_quantity_test))


kernel = (RBFKernel(), RBFKernel())
multi_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
multi_sm = MultiGPyTraining(collocation_points_model_train,
                            model_evaluations_train,
                            kernel,
                            training_iter=100,
                            likelihood=multi_likelihood,
                            optimizer="adam", lr=0.1, number_quantities=2,
                            )

multi_sm.train()
predicted_outputs_test = multi_sm.predict_(input_sets=collocation_points_model_test)
mid_index_training_sets = model_evaluations_test.shape[0] // 2
predicted_outputs_test = predicted_outputs_test['output']
mid_index_locations = predicted_outputs_test.shape[1] // 2
predicted_outputs_test1 = predicted_outputs_test[:, :mid_index_locations]
predicted_outputs_test2 = predicted_outputs_test[:, mid_index_locations:]
observed_output_test1 = model_evaluations_test[:mid_index_training_sets, :]
observed_output_test2 = model_evaluations_test[mid_index_training_sets:, :]

# Calculate metrics first quantity
mse_1 = mean_squared_error(observed_output_test1.flatten(), predicted_outputs_test1.flatten())
r2_1 = r2_score(observed_output_test1.flatten(), predicted_outputs_test1.flatten())
# Calculate metrics second quantity
mse_2 = mean_squared_error(observed_output_test2.flatten(), predicted_outputs_test2.flatten())
r2_2 = r2_score(observed_output_test2.flatten(), predicted_outputs_test2.flatten())

print(f'MSE: {mse_1:.9f}')
print(f'R²: {r2_1:.3f}')

print(f'MSE: {mse_2:.9f}')
print(f'R²: {r2_2:.3f}')

def plot_predicted_vs_observed(observed, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(observed, predicted, alpha=0.5)
    plt.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()
def plot_residuals(observed, predicted, title):
    residuals = observed - predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()
def plot_3d_surface(test_collocation_points, predicted_surrogate_outputs, param_indices=(0, 1), num_points=50):
    """
    Plots a 3D surface of the surrogate model predictions based on two input parameters.

    Parameters:
    - test_collocation_points: The array of input parameters used for testing
    - predicted_surrogate_outputs: The predicted outputs from the surrogate model
    - param_indices: A tuple indicating which two parameters to use for the 3D plot (default: (0, 1))
    - num_points: The number of points to use for the grid in each dimension (default: 50)
    """
    # Extract the relevant parameters for the grid
    param1_range = np.linspace(np.min(test_collocation_points[:, param_indices[0]]), np.max(test_collocation_points[:, param_indices[0]]), num_points)
    param2_range = np.linspace(np.min(test_collocation_points[:, param_indices[1]]), np.max(test_collocation_points[:, param_indices[1]]), num_points)
    param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)

    # Interpolate predictions onto the grid
    param_grid_flattened = np.vstack([param1_grid.ravel(), param2_grid.ravel()]).T
    predicted_grid = griddata(test_collocation_points[:, param_indices], predicted_surrogate_outputs[:,0].flatten(), param_grid_flattened, method='linear')
    predicted_grid = predicted_grid.reshape(param1_grid.shape)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(param1_grid, param2_grid, predicted_grid, cmap='viridis', edgecolor='none')

    # Add labels and title
    ax.set_xlabel(f'Parameter {param_indices[0] + 1}')
    ax.set_ylabel(f'Parameter {param_indices[1] + 1}')
    ax.set_zlabel('Model Prediction')
    ax.set_title('Surrogate Model Predictions')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()

plot_predicted_vs_observed(observed_output_test1.flatten(), predicted_outputs_test1.flatten(), 'Predicted vs Observed (First Quantity)')
plot_predicted_vs_observed(observed_output_test2.flatten(), predicted_outputs_test2.flatten(), 'Predicted vs Observed (Second Quantity)')
plot_residuals(observed_output_test2.flatten(), predicted_outputs_test2.flatten(), 'Residuals Plot (Second Quantity)')
#plot_3d_surface(collocation_points_model_test, predicted_outputs_test1, param_indices=(0, 1), num_points=50)


#print(predicted_outputs_test)

# MultiGPyTraining class
# class MultiGPyTraining:
#     def __init__(self, collocation_points, model_evaluations, kernel, training_iter, likelihood,number_quantities,
#                  optimizer="adam", lr=0.1, n_restarts=1, parallelize=False,
#                  noise_constraint=GreaterThan(1e-6)):
#         # Basic attributes
#         self.training_points = collocation_points
#         self.model_evaluations = model_evaluations
#         self.number_quantities = number_quantities
#         self.n_obs = self.model_evaluations.shape[1]
#         self.n_params = collocation_points.shape[1]
#         self.gp_list = []
#
#         # Initialize likelihood and other hyperparameters
#         self.likelihood = likelihood
#         self.kernel = kernel
#         self.optimizer_ = optimizer
#         self.training_iter = training_iter
#         self.n_restarts = n_restarts
#         self.lr = lr
#         self.parallel = parallelize
#         self.noise_contraint = noise_constraint
#
#     def train(self):
#         X = torch.tensor(self.training_points, dtype=torch.float32)
#         Y = torch.tensor(self.model_evaluations, dtype=torch.float32)
#         rows_per_task = Y.shape[0] // self.number_quantities
#         for loc in range(Y.shape[1]):
#             Y_loc = torch.cat([Y[i * rows_per_task:(i + 1) * rows_per_task, loc].reshape(rows_per_task, 1)
#                                for i in range(self.number_quantities)], dim=1)
#
#             model = MultitaskGPModel(X, Y_loc, self.likelihood, self.kernel)
#
#             # Training mode
#             model.train()
#             self.likelihood.train()
#
#             # Set the optimizer
#             if self.optimizer_ == "adam":
#                 optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
#             else:
#                 raise ValueError(f"Optimizer '{self.optimizer_}' not supported.")
#
#             # Set the MLL objective
#             mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
#
#             # Training loop
#             for _ in range(self.training_iter):
#                 optimizer.zero_grad()
#                 output = model(X)
#                 loss = -mll(output, Y_loc)
#                 loss.backward()
#                 optimizer.step()
#
#             # Store trained model in the list
#             self.gp_list.append(model)
#
#     def predict_(self, input_sets):
#         input_sets = torch.tensor(input_sets, dtype=torch.float32)
#         means = []
#         stds = []
#
#         for model in self.gp_list:
#             model.eval()  # Set the model to evaluation mode
#             self.likelihood.eval()  # Set the likelihood to evaluation mode
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 predictions = model(input_sets)
#                 means.append(predictions.mean.detach().cpu().numpy())  # Convert to numpy
#                 stds.append(predictions.stddev.detach().cpu().numpy())  # Convert to numpy
#
#         # Concatenate means and stds along axis 0 (vertical stack)
#         means = np.vstack(means)
#         stds = np.vstack(stds)
#
#         surrogate_outputs = {'output': means, 'std': stds}
#         return surrogate_outputs
#
#
# class MultitaskGPModel(ExactGP):
#     def __init__(self, train_x, train_y, likelihood, kernel):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
#         self.covar_module = MultitaskKernel(
#             AdditiveKernel(
#                 ProductKernel(kernel[0], kernel[1]),  # Assuming kernel is a tuple of two components
#                 ScaleKernel(kernel[0])
#             ),
#             num_tasks=2, rank=1
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
#
#
# if __name__ == "__main__":
#
#     # Define the kernel and likelihood
#     kernel = (RBFKernel(), RBFKernel())
#     likelihood = MultitaskGaussianLikelihood(num_tasks=2)
#
#     # Initialize the MultiGPyTraining class
#     multi_gp = MultiGPyTraining(
#         collocation_points=collocation_points,
#         model_evaluations=model_evaluations,
#         kernel=kernel,
#         training_iter=50,
#         likelihood=likelihood,
#         lr=0.1,
#         noise_constraint=GreaterThan(1e-6),
#         number_quantities=2
#     )
#
#     # Train the model
#     multi_gp.train()
#
#     # Create test points
#     test_points = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
#
#     # Make predictions
#     predictions = multi_gp.predict_(test_points)
#
#     # Unstack the predictions into individual tasks
#     model_outputs = predictions['output']
#     model_stdv = predictions['std']
#     pred_f1 = predictions['output'][:, 0]  # First column
#     pred_f2 = predictions['output'][:, 1]  # Second column
#
#     # Plotting
#     plt.figure(figsize=(12, 6))
#
#     # Plot actual functions
#     plt.subplot(2, 1, 1)
#     plt.plot(test_points, f1(test_points), 'b-', label='Actual f1')
#     plt.plot(test_points, f2(test_points), 'g-', label='Actual f2')
#     plt.fill_between(test_points.flatten(), f1(test_points).flatten(), color='blue', alpha=0.1)
#     plt.fill_between(test_points.flatten(), f2(test_points).flatten(), color='green', alpha=0.1)
#     plt.title('Actual Functions')
#     plt.xlabel('Input')
#     plt.ylabel('Output')
#     plt.legend()
#
#     # Plot predicted mean and confidence intervals
#     plt.subplot(2, 1, 2)
#     plt.plot(test_points, pred_f1.flatten(), 'b-', label='Predicted f1')
#     plt.plot(test_points, pred_f2.flatten(), 'g-', label='Predicted f2')
#     plt.fill_between(test_points.flatten(),
#                      pred_f1.flatten() - 1.96 * model_stdv [:,0].flatten(),
#                      pred_f1.flatten() + 1.96 * model_stdv [:,0].flatten(),
#                      color='blue', alpha=0.1)
#     plt.fill_between(test_points.flatten(),
#                      pred_f2.flatten() - 1.96 * model_stdv [:,1].flatten(),
#                      pred_f2.flatten() + 1.96 * model_stdv [:,1].flatten(),
#                      color='green', alpha=0.1)
#     plt.title('Predicted Functions with Confidence Intervals')
#     plt.xlabel('Input')
#     plt.ylabel('Output')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()