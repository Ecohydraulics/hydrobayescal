import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel, AdditiveKernel, ProductKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood


# Example functions
def f1(x):
    return np.sin(x)


def f2(x):
    return np.cos(x)


# MultiGPyTraining class
class MultiGPyTraining:
    def __init__(self, collocation_points, model_evaluations, kernel, training_iter, likelihood,number_quantities,
                 optimizer="adam", lr=0.1, n_restarts=1, parallelize=False,
                 noise_constraint=GreaterThan(1e-6)):
        # Basic attributes
        self.training_points = collocation_points
        self.model_evaluations = model_evaluations
        self.number_quantities = number_quantities
        self.n_obs = self.model_evaluations.shape[1]
        self.n_params = collocation_points.shape[1]
        self.gp_list = []

        # Initialize likelihood and other hyperparameters
        self.likelihood = likelihood
        self.kernel = kernel
        self.optimizer_ = optimizer
        self.training_iter = training_iter
        self.n_restarts = n_restarts
        self.lr = lr
        self.parallel = parallelize
        self.noise_contraint = noise_constraint

    def train(self):
        X = torch.tensor(self.training_points, dtype=torch.float32)
        Y = torch.tensor(self.model_evaluations, dtype=torch.float32)
        rows_per_task = Y.shape[0] // self.number_quantities
        for loc in range(Y.shape[1]):
            Y_loc = torch.cat([Y[i * rows_per_task:(i + 1) * rows_per_task, loc].reshape(rows_per_task, 1)
                               for i in range(self.number_quantities)], dim=1)

            model = MultitaskGPModel(X, Y_loc, self.likelihood, self.kernel)

            # Training mode
            model.train()
            self.likelihood.train()

            # Set the optimizer
            if self.optimizer_ == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            else:
                raise ValueError(f"Optimizer '{self.optimizer_}' not supported.")

            # Set the MLL objective
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

            # Training loop
            for _ in range(self.training_iter):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, Y_loc)
                loss.backward()
                optimizer.step()

            # Store trained model in the list
            self.gp_list.append(model)

    def predict_(self, input_sets):
        input_sets = torch.tensor(input_sets, dtype=torch.float32)
        means = []
        stds = []

        for model in self.gp_list:
            model.eval()  # Set the model to evaluation mode
            self.likelihood.eval()  # Set the likelihood to evaluation mode
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = model(input_sets)
                means.append(predictions.mean.detach().cpu().numpy())  # Convert to numpy
                stds.append(predictions.stddev.detach().cpu().numpy())  # Convert to numpy

        # Concatenate means and stds along axis 0 (vertical stack)
        means = np.vstack(means)
        stds = np.vstack(stds)

        surrogate_outputs = {'output': means, 'std': stds}
        return surrogate_outputs


class MultitaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.covar_module = MultitaskKernel(
            AdditiveKernel(
                ProductKernel(kernel[0], kernel[1]),  # Assuming kernel is a tuple of two components
                ScaleKernel(kernel[0])
            ),
            num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    # Generate collocation points
    collocation_points = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    number_quantities = 2
    # Evaluate the functions at the collocation points
    f1_values = f1(collocation_points)
    f2_values = f2(collocation_points)

    # Stack the evaluations vertically into a single column
    model_evaluations = np.vstack([f1_values, f2_values])

    # Define the kernel and likelihood
    kernel = (RBFKernel(), RBFKernel())
    likelihood = MultitaskGaussianLikelihood(num_tasks=2)

    # Initialize the MultiGPyTraining class
    multi_gp = MultiGPyTraining(
        collocation_points=collocation_points,
        model_evaluations=model_evaluations,
        kernel=kernel,
        training_iter=50,
        likelihood=likelihood,
        lr=0.1,
        noise_constraint=GreaterThan(1e-6),
        number_quantities=number_quantities
    )

    # Train the model
    multi_gp.train()

    # Create test points
    test_points = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

    # Make predictions
    predictions = multi_gp.predict_(test_points)

    # Unstack the predictions into individual tasks
    model_outputs = predictions['output']
    model_stdv = predictions['std']
    pred_f1 = predictions['output'][:, 0]  # First column
    pred_f2 = predictions['output'][:, 1]  # Second column

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot actual functions
    plt.subplot(2, 1, 1)
    plt.plot(test_points, f1(test_points), 'b-', label='Actual f1')
    plt.plot(test_points, f2(test_points), 'g-', label='Actual f2')
    plt.fill_between(test_points.flatten(), f1(test_points).flatten(), color='blue', alpha=0.1)
    plt.fill_between(test_points.flatten(), f2(test_points).flatten(), color='green', alpha=0.1)
    plt.title('Actual Functions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()

    # Plot predicted mean and confidence intervals
    plt.subplot(2, 1, 2)
    plt.plot(test_points, pred_f1.flatten(), 'b-', label='Predicted f1')
    plt.plot(test_points, pred_f2.flatten(), 'g-', label='Predicted f2')
    plt.fill_between(test_points.flatten(),
                     pred_f1.flatten() - 1.96 * model_stdv [:,0].flatten(),
                     pred_f1.flatten() + 1.96 * model_stdv [:,0].flatten(),
                     color='blue', alpha=0.1)
    plt.fill_between(test_points.flatten(),
                     pred_f2.flatten() - 1.96 * model_stdv [:,1].flatten(),
                     pred_f2.flatten() + 1.96 * model_stdv [:,1].flatten(),
                     color='green', alpha=0.1)
    plt.title('Predicted Functions with Confidence Intervals')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()

    plt.tight_layout()
    plt.show()