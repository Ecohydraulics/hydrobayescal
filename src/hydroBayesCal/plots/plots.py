import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,norm,linregress
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path

parameter_names = ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4']
# Generate a random prior following a uniform distribution
prior = np.random.uniform(low=0.0, high=1.0, size=(10000, len(parameter_names)))
posterior_vector = np.random.randn(10000, len(parameter_names))

num_updates = 3
posterior_vectors = np.random.randn(num_updates, 10000, len(parameter_names))

bayesian_dict = {
    'N_tp': {}, 'BME': {}, 'RE': {}, 'ELPD': {}, 'IE': {}, 'post_size': {}
}

for it in range(10):
    bayesian_dict['N_tp'][it] = np.random.randint(100, 200)
    bayesian_dict['BME'][it] = np.random.uniform(0, 1)
    bayesian_dict['RE'][it] = np.random.uniform(0, 1)
    bayesian_dict['ELPD'][it] = np.random.uniform(0, 1)
    bayesian_dict['IE'][it] = np.random.uniform(0, 1)
    bayesian_dict['post_size'][it] = np.random.randint(1000, 2000)
# Generate a posterior vector of size 10000


# Function to calculate likelihood for each parameter set

def plot_posterior(posterior_vector, parameter_name):
    colors = ['dimgray']
    bins = 30

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot posterior distribution
    counts, bins, patches = ax.hist(posterior_vector, bins=bins, density=True, alpha=0.5, color=colors[0], edgecolor='black')

    # Add mean line
    mean_posterior = np.mean(posterior_vector)
    ax.axvline(mean_posterior, color='blue', linestyle='dashed', linewidth=1, label='Mean')

    # Set labels and title
    ax.set_xlabel(parameter_name, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(direction='in', labelsize=12)

    # Ensure histogram starts from the vertical axis without a white space
    ax.set_xlim(left=bins[0])

    plt.tight_layout()
    plt.show()

#     # Adjust layout
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.4)  # Add more space between rows
#     plt.savefig('last_posterior.png')
#     plt.close()
    #plt.savefig('posterior_distributions.eps', format='eps')
    # Print the posterior probabilities
    # for i, prob in enumerate(posterior_vector):
    #     print(f"Posterior probability for parameter set {i + 1}: {prob:.4f}")
    #
    # # Identify the best parameter set
    # best_index = np.argmax(posterior_vector)
    # print(f"The best parameter set is set {best_index + 1} with posterior probability {posterior_vector[best_index]:.4f}")


def plot_posterior_updates(posterior_arrays, parameter_names, prior, iterations_to_plot=None,
                           save_folder='auto-saved-results'):
    """
    Plots the prior distributions and posterior updates for given parameters.

    Args:
        posterior_arrays: list of np.array
            List of 2D arrays with posterior samples for each update.
        parameter_names: list of str
            List of parameter names corresponding to the columns of the arrays.
        prior: np.array
            2D array with prior samples.
        iterations_to_plot: list of int or None
            List of iteration indices to plot. If None, only prior distributions are plotted.
        save_folder: str or Path
            Directory to save the plots.
    """
    # Convert save_folder to Path if it's a string
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)

    # Ensure save_folder exists
    save_folder.mkdir(parents=True, exist_ok=True)

    colors = ['darkgray', 'darkgray']
    parameter_num = len(parameter_names)

    # Calculate x_limits for each parameter from the prior data
    x_limits = np.zeros((parameter_num, 2))
    for i in range(parameter_num):
        x_limits[i] = (prior[:, i].min(), prior[:, i].max())

    # Calculate y_max for each parameter
    y_max_prior = np.zeros(parameter_num)
    y_max_posterior = np.zeros(parameter_num)
    for i in range(parameter_num):
        counts_prior, _ = np.histogram(prior[:, i], bins=40, density=True)
        y_max_prior[i] = max(counts_prior)
        for row in range(len(posterior_arrays)):
            if posterior_arrays[row] is not None:  # Check if the current array is not None
                counts_posterior, _ = np.histogram(posterior_arrays[row][:, i], bins=40, density=True)
                y_max_posterior[i] = max(y_max_posterior[i], max(counts_posterior))

    # Plot prior distributions
    for i in range(parameter_num):
        plt.figure(figsize=(4, 8))
        plt.hist(prior[:, i], bins=40, density=True, alpha=0.5, color=colors[0], label='Prior', edgecolor='black')
        plt.xlabel(parameter_names[i], fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(direction='in', labelsize=12)
        plt.ylim(0, y_max_prior[i])
        plt.xlim(x_limits[i])
        plt.tight_layout()
        plt.savefig(save_folder / f'prior_distribution_param_{i + 1}.png')
        plt.close()

    # Plot each selected update
    if iterations_to_plot is not None:
        for plot_index, iteration_idx in enumerate(iterations_to_plot):
            for col in range(parameter_num):
                plt.figure(figsize=(4, 8))
                posterior_vector = posterior_arrays[iteration_idx]
                plt.hist(posterior_vector[:, col], bins=40, density=True, alpha=0.5, color=colors[1], label='Posterior',
                         edgecolor='black')
                plt.xlabel(parameter_names[col], fontsize=14)
                plt.ylabel('Density', fontsize=14)
                plt.legend(fontsize=8)
                plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
                plt.tick_params(direction='in', labelsize=12)
                plt.ylim(0, y_max_posterior[col])
                plt.xlim(x_limits[col])
                plt.tight_layout()
                plt.savefig(save_folder / f'posterior_distribution_param_{col + 1}_iteration_{iteration_idx + 1}.png')
                plt.close()


def plot_bme_re(bayesian_dict, num_bal_iterations, save_folder='auto-saved-results'):
    """
    Plots BME and RE values over iterations.

    Args:
        bayesian_dict: dict
            Dictionary containing 'BME' and 'RE' values for each iteration.
        num_bal_iterations: int
            Number of iterations for which to plot data.
        save_folder: str or Path
            Directory to save the plot. If provided as a string, it will be converted to a Path object.
    """
    # Convert save_folder to Path if it's a string
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)

    # Extract BME and RE for plotting
    iterations = list(range(num_bal_iterations))
    bme_values = [bayesian_dict['BME'][it] for it in iterations]
    re_values = [bayesian_dict['RE'][it] for it in iterations]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot BME
    axes[0].plot(iterations, bme_values, marker='+', markersize=10, color='darkslategray', linestyle='-', linewidth=1.5,
                 label='BME')
    axes[0].set_title('Bayesian Model Evidence (BME)', fontsize=16)
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel('BME', fontsize=14)
    axes[0].grid(True, linestyle='--', linewidth=0.5)
    axes[0].legend()

    # Add a dashed tendency line for BME
    slope_bme, intercept_bme, _, _, _ = linregress(iterations, bme_values)
    trend_bme = [slope_bme * x + intercept_bme for x in iterations]
    axes[0].plot(iterations, trend_bme, color='darkslategray', linestyle='--', linewidth=1.5, label='Trend Line')
    axes[0].legend()

    # Set x-axis limits for BME plot
    axes[0].set_xlim(iterations[0], iterations[-1])
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(5))

    # Plot RE
    axes[1].plot(iterations, re_values, marker='x', markersize=10, color='dimgray', linestyle='-', linewidth=1.5,
                 label='RE')
    axes[1].set_title('Relative Entropy (RE)', fontsize=16)
    axes[1].set_xlabel('Iteration', fontsize=14)
    axes[1].set_ylabel('RE', fontsize=14)
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].legend()

    # Add a dashed tendency line for RE
    slope_re, intercept_re, _, _, _ = linregress(iterations, re_values)
    trend_re = [slope_re * x + intercept_re for x in iterations]
    axes[1].plot(iterations, trend_re, color='dimgray', linestyle='--', linewidth=1.5, label='Trend Line')
    axes[1].legend()

    # Set x-axis limits for RE plot
    axes[1].set_xlim(iterations[0], iterations[-1])
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(5))

    # Adjust layout
    plt.tight_layout()

    # Ensure the directory exists
    save_folder.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(save_folder / 'RE_BME_plots.png')
    plt.close()
def plot_combined_bal(collocation_points, n_init_tp, bayesian_dict, save_folder=None):
    """
    Plots the initial training points and points selected using different utility functions.
    Args:
        collocation_points: np.array[n_tp, n_param]
            Array with all collocation points, in order in which they were selected.
        n_init_tp: int
            Number of initial training points selected.
        bayesian_dict: dictionary
            With keys 'util_func', detailing which utility function was used in each iteration.
        save_folder: Path or None
            Directory where to save the plot. If None, the plot is not saved.
    """
    if collocation_points.shape[1] == 1:
        collocation_points = np.hstack((collocation_points, collocation_points))

    fig, ax = plt.subplots()

    # Initial TP:
    ax.scatter(collocation_points[0:n_init_tp, 0], collocation_points[0:n_init_tp, 1], label='Initial TP', c='black', s=100)
    selected_tp = collocation_points[n_init_tp:, :]

    # Get indexes for 'dkl'
    dkl_ind = np.where(bayesian_dict['util_func'] == 'dkl')
    ax.scatter(selected_tp[dkl_ind, 0], selected_tp[dkl_ind, 1], label='DKL', c='gold', s=200, alpha=0.5)

    # Get indexes for 'bme'
    bme_ind = np.where(bayesian_dict['util_func'] == 'bme')
    ax.scatter(selected_tp[bme_ind, 0], selected_tp[bme_ind, 1], label='BME', c='blue', s=200, alpha=0.5)

    # Get indexes for 'ie'
    ie_ind = np.where(bayesian_dict['util_func'] == 'ie')
    ax.scatter(selected_tp[ie_ind, 0], selected_tp[ie_ind, 1], label='IE', c='green', s=200, alpha=0.5)

    # Global MC
    mc_ind = np.where(bayesian_dict['util_func'] == 'global_mc')
    ax.scatter(selected_tp[mc_ind, 0], selected_tp[mc_ind, 1], label='MC', c='red', s=200, alpha=0.5)

    ax.set_xlabel('K_Zone 8')
    ax.set_ylabel('$K_Zone 9$')

    fig.legend(loc='lower center', ncol=5)
    plt.subplots_adjust(top=0.95, bottom=0.15, wspace=0.25, hspace=0.55)

    # Save the figure
    if save_folder:
        save_folder = Path(save_folder)  # Ensure save_folder is a Path object
        save_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_folder / 'collocation_points.png')  # Save with .png extension
    plt.close()

def simulate_bme(X, Y, iteration):
    # Simulated function for Bayesian Model Evidence (BME)
    Z = np.cos(np.sqrt(X ** 2 + Y ** 2)) * np.exp(-0.1 * (X ** 2 + Y ** 2)) + 0.5 * np.sin(X) + 0.5 * np.sin(Y) + 0.2 * np.random.randn(*X.shape) * (iteration + 1)
    return Z

# Example usage
#plot_bme_surface(num_iterations=6)


def plot_bme_surface_3d(num_iterations):
    # Generate sample data for illustration
    np.random.seed(0)
    x = np.linspace(0.1, 10, 20)  # Positive limits: 0.1 to 10
    y = np.linspace(0.1, 5, 20)   # Positive limits: 0.1 to 5
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(16, 8))

    # 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, aspect='equal')  # Set aspect to 'equal' for a square plot

    scatter_plotted = False  # Flag to ensure scatter plot legend is added only once

    # Lists to store coordinates of high and low BME regions
    high_bme_coords = []
    low_bme_coords = []

    # Iterate through the number of iterations
    for i in range(num_iterations):
        # Simulated Bayesian Model Evidence as a function of X and Y
        Z = simulate_bme(X, Y, iteration=i)

        # Apply Gaussian smoothing to Z
        Z_smooth = gaussian_filter(Z, sigma=1.0)

        # Find regions with higher BME
        threshold_high = np.percentile(Z_smooth, 95)  # Define threshold for high BME regions
        threshold_low = np.percentile(Z_smooth, 5)    # Define threshold for low BME regions
        high_bme_indices = np.argwhere(Z_smooth > threshold_high)
        low_bme_indices = np.argwhere(Z_smooth < threshold_low)

        # Plot 3D surface for each iteration
        surf = ax1.plot_surface(X, Y, Z_smooth, cmap='viridis', alpha=0.6)

        # Store coordinates of high and low BME regions for plain view plot
        high_bme_coords.extend(list(zip(X[high_bme_indices[:, 0], high_bme_indices[:, 1]],
                                        Y[high_bme_indices[:, 0], high_bme_indices[:, 1]])))
        low_bme_coords.extend(list(zip(X[low_bme_indices[:, 0], low_bme_indices[:, 1]],
                                       Y[low_bme_indices[:, 0], low_bme_indices[:, 1]])))

    ax1.set_title('Bayesian Model Evidence (BME)', fontsize=20)
    ax1.set_xlabel(r'$\omega_1$', fontsize=18)
    ax1.set_ylabel(r'$\omega_2$', fontsize=18)
    ax1.set_zlabel('BME', fontsize=18)
    ax1.set_zticklabels([])  # Remove Z-axis labels

    # Zoom in by adjusting the view limits
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.set_zlim(-2, 2)

    ax1.view_init(elev=30, azim=225)  # Adjust view angle

    # Add colorbar for the surface plot
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # Create a plain view grid
    grid_size = 100
    x_plain = np.linspace(0.1, 10, grid_size)
    y_plain = np.linspace(0.1, 5, grid_size)
    X_plain, Y_plain = np.meshgrid(x_plain, y_plain)

    # Compute BME values for the plain view grid
    Z_plain = simulate_bme(X_plain, Y_plain, iteration=0)  # Use iteration=0 or any preferred iteration for plotting

    # Plot the plain view surface
    ax2.contourf(X_plain, Y_plain, Z_plain, cmap='viridis', alpha=0.8)

    # Plot scatter points for high BME regions in red
    if high_bme_coords:  # Check if there are high BME points to plot
        ax2.scatter(*zip(*high_bme_coords), color='red', s=10, label='High BME Regions')

    # Plot scatter points for low BME regions in blue
    if low_bme_coords:  # Check if there are low BME points to plot
        ax2.scatter(*zip(*low_bme_coords), color='blue', s=10, label='Low BME Regions')

    ax2.set_title('Plain View of BME Regions', fontsize=20)
    ax2.set_xlabel(r'$\omega_1$', fontsize=18)
    ax2.set_ylabel(r'$\omega_2$', fontsize=18)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

# def simulate_bme(X, Y, iteration):
#     # Simulated function for Bayesian Model Evidence (BME)
#     Z = np.cos(np.sqrt(X ** 2 + Y ** 2)) * np.exp(-0.1 * (X ** 2 + Y ** 2)) + 0.5 * np.sin(X) + 0.5 * np.sin(Y) + 0.2 * np.random.randn(*X.shape) * (iteration + 1)
#     return Z

# Example usage
#plot_bme_surface_3d(num_iterations=6)

# def plot_posterior(posterior_vector, parameter_name):
#     colors = ['dimgray']
#     bins = 30
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     # Plot posterior distribution
#     ax.hist(posterior_vector, bins=bins, density=True, alpha=0.5, color=colors[0], edgecolor='black')
#
#     # Add mean line
#     mean_posterior = np.mean(posterior_vector)
#     ax.axvline(mean_posterior, color='blue', linestyle='dashed', linewidth=1, label='Mean')
#
#     # Set labels and title
#     ax.set_xlabel(parameter_name, fontsize=14)
#     ax.set_ylabel('Density', fontsize=14)
#     ax.legend(fontsize=12)
#     ax.grid(True)
#     ax.tick_params(direction='in', labelsize=12)
#
#     # Ensure histogram starts from the vertical axis
#     ax.set_xlim(left=posterior_vector.min(), right=posterior_vector.max())
#
#     plt.tight_layout()
#     plt.savefig('posterior_single_parameter.png')
#     plt.show()

# Example usage
# np.random.seed(0)
# posterior_example = np.random.uniform(0.04, 0.08, 900)
# plot_posterior(posterior_example, r'$\omega_1$')