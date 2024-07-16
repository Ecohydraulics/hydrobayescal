import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel, LinearKernel, ScaleKernel, ProductKernel, AdditiveKernel,MaternKernel,PeriodicKernel
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys

base_dir = Path(__file__).resolve().parent.parent.parent.parent
examples_path = base_dir / 'examples'
sys.path.append(str(examples_path))

from user_settings_ering_restart import user_inputs_tm
# Generate sample data
path_np_collocation_points = user_inputs_tm['results_folder_path'] + 'auto-saved-results/colocation_points.npy'
path_np_model_results = user_inputs_tm['results_folder_path'] + 'auto-saved-results/model_results.npy'

# Load the collocation points and model results if they exist
model_evaluations = np.load(path_np_model_results)
collocation_points = np.load(path_np_collocation_points)
number_tasks = 2 # number of calibration quantities
X = torch.tensor(collocation_points, dtype=torch.float32)
Y = torch.tensor(model_evaluations, dtype=torch.float32)

# Define the Multitask GP Model
# Define the Multitask GP Model
class MultitaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.covar_module = MultitaskKernel(
            AdditiveKernel(
                ProductKernel(MaternKernel(nu=2.5), PeriodicKernel()),
                ScaleKernel(MaternKernel(nu=2.5))
            ),
            num_tasks=number_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
# Initialize likelihood and list of models
likelihood = MultitaskGaussianLikelihood(num_tasks=number_tasks)
gp_models = []
rows_per_task = Y.shape[0] // number_tasks
# Train separate models for each location
for loc in range(Y.shape[1]):
    # Divide Y into two parts for the two output variables
    # Y1 = Y[:5, loc].reshape(5, 1)
    # Y2 = Y[5:10, loc].reshape(5, 1)
    # Y_loc = torch.cat((Y1, Y2), dim=1)  # Combine the two outputs
    Y_loc = torch.cat([Y[i*rows_per_task:(i+1)*rows_per_task, loc].reshape(rows_per_task, 1) for i in range(number_tasks)], dim=1)

    model = MultitaskGPModel(X, Y_loc, likelihood)

    # Training mode
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Set the MLL objective
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    num_iter = 200
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y_loc)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{num_iter} - Loss: {loss.item()}")

    # Store trained model in the list
    gp_models.append(model)

# Generate test points within specified parameter ranges
param_ranges = [
    [0.04, 0.09],
    [0.04, 0.09],
    [0.04, 0.09],
    [0.04, 0.09],
]

num_test_points = 100
num_dimensions = 4

X_test_raw = torch.rand(num_test_points, num_dimensions)
X_test = torch.empty(num_test_points, num_dimensions)
for i, (low, high) in enumerate(param_ranges):
    X_test[:, i] = X_test_raw[:, i] * (high - low) + low

# List to store means and standard deviations
means_Y1 = []
stds_Y1 = []
means_Y2 = []
stds_Y2 = []

# Make predictions and calculate means/stds for each location
for model in gp_models:
    model.eval()  # Set the model to evaluation mode
    likelihood.eval()  # Set the likelihood to evaluation mode
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(X_test)
        mean_Y1 = predictions.mean[:, 0].numpy()
        std_Y1 = predictions.stddev[:, 0].numpy()
        mean_Y2 = predictions.mean[:, 1].numpy()
        std_Y2 = predictions.stddev[:, 1].numpy()

        means_Y1.append(mean_Y1)
        stds_Y1.append(std_Y1)
        means_Y2.append(mean_Y2)
        stds_Y2.append(std_Y2)

# Convert lists to numpy arrays for easy manipulation
means_Y1 = np.array(means_Y1)
stds_Y1 = np.array(stds_Y1)
means_Y2 = np.array(means_Y2)
stds_Y2 = np.array(stds_Y2)
# Plotting the results for the first location (as an example)
plt.figure(figsize=(20, 16))

# Plotting for the first location
loc = 0

# Four subplots for four parameters
for i in range(4):
    plt.subplot(4, 2, 2*i+1)
    plt.title(f'Location {loc + 1} - Parameter {i + 1} - Output 1')
    plt.plot(X[:, i].numpy(), Y[:rows_per_task, loc].numpy(), 'k*', label='Observed Data')  # Plotting observed data for the first task
    plt.scatter(X_test[:, i].numpy(), means_Y1[loc], color='b', label='Prediction Y1')
    plt.fill_between(X_test[:, i].numpy(), means_Y1[loc] - stds_Y1[loc], means_Y1[loc] + stds_Y1[loc], color='b', alpha=0.2)
    plt.xlabel(f'X[:, {i}]')
    plt.ylabel(f'Y1')
    plt.legend()

    plt.subplot(4, 2, 2*i+2)
    plt.title(f'Location {loc + 1} - Parameter {i + 1} - Output 2')
    plt.plot(X[:, i].numpy(), Y[rows_per_task:2*rows_per_task, loc].numpy(), 'k*', label='Observed Data')  # Plotting observed data for the second task
    plt.scatter(X_test[:, i].numpy(), means_Y2[loc], color='g', label='Prediction Y2')
    plt.fill_between(X_test[:, i].numpy(), means_Y2[loc] - stds_Y2[loc], means_Y2[loc] + stds_Y2[loc], color='g', alpha=0.2)
    plt.xlabel(f'X[:, {i}]')
    plt.ylabel(f'Y2')
    plt.legend()

plt.tight_layout()
plt.show()