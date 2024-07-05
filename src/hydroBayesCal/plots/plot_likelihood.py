import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,norm,linregress

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

def plot_posterior(posterior_vector, parameter_names, prior):
    colors = ['darkgray', 'dimgray']
    parameter_num = len(parameter_names)

    fig, axes = plt.subplots(2, parameter_num, figsize=(15, 10))

    # Font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 8
    bins = 40

    # Determine y limits
    y_max = 0

    # Calculate y_max by plotting without displaying
    if posterior_vector.ndim == 1:
        counts_posterior, _ = np.histogram(posterior_vector, bins=bins, density=True)
        y_max = np.max(counts_posterior)
    else:
        for i in range(parameter_num):
            counts_posterior, _ = np.histogram(posterior_vector[:, i], bins=bins, density=True)
            y_max = max(y_max, np.max(counts_posterior))

    # Plot prior distributions
    for i in range(parameter_num):
        ax = axes[0, i]
        ax.hist(prior[:, i], bins=bins, density=True, alpha=0.5, color=colors[0], label='Prior')
        mean_prior = np.mean(prior[:, i])
        ax.axvline(mean_prior, color='black', linestyle='dashed', linewidth=1, label='Mean')
        ax.set_title(f'Prior: {parameter_names[i]}', fontsize=title_fontsize)
        ax.set_xlabel(parameter_names[i], fontsize=label_fontsize)
        ax.set_ylabel('Density', fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True)
        ax.tick_params(direction='in', labelsize=tick_fontsize)

    # Plot posterior distributions
    for i in range(parameter_num):
        ax = axes[1, i]
        ax.hist(posterior_vector[:, i], bins=bins, density=True, alpha=0.5, color=colors[1], label='Posterior')
        mean_posterior = np.mean(posterior_vector[:, i])
        ax.axvline(mean_posterior, color='black', linestyle='dashed', linewidth=1, label='Mean')
        ax.set_title(f'Posterior: {parameter_names[i]}', fontsize=title_fontsize)
        ax.set_xlabel(parameter_names[i], fontsize=label_fontsize)
        ax.set_ylabel('Density', fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True)
        ax.tick_params(direction='in', labelsize=tick_fontsize)
        ax.set_ylim(0, 250)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Add more space between rows

    plt.show()

    #plt.savefig('posterior_distributions.eps', format='eps')
    # Print the posterior probabilities
    # for i, prob in enumerate(posterior_vector):
    #     print(f"Posterior probability for parameter set {i + 1}: {prob:.4f}")
    #
    # # Identify the best parameter set
    # best_index = np.argmax(posterior_vector)
    # print(f"The best parameter set is set {best_index + 1} with posterior probability {posterior_vector[best_index]:.4f}")



def plot_posterior_updates(posterior_arrays, parameter_names, prior):
    colors = ['darkgray', 'dimgray']  # Red for prior, green for posterior
    parameter_num = len(parameter_names)

    # Ensure posterior_arrays is a list of 2D arrays and combine them correctly
    posterior_vectors = [np.array(p) for p in posterior_arrays]
    num_updates = len(posterior_vectors)

    # Create the subplots with the correct number of rows and columns
    fig, axes = plt.subplots(num_updates, parameter_num, figsize=(15, (6 * num_updates) + 1))

    # Font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 8

    # Calculate fixed x_limits for each parameter from the prior data
    x_limits = np.zeros((parameter_num, 2))
    for i in range(parameter_num):
        x_limits[i] = (prior[:, i].min(), prior[:, i].max())

    # Calculate y_max for each parameter separately
    y_max = np.zeros(parameter_num)
    for i in range(parameter_num):
        for row in range(num_updates):
            counts_posterior, _ = np.histogram(posterior_vectors[row][:, i], bins=30, density=True)
            y_max[i] = max(y_max[i], max(counts_posterior))

    # Plot each update
    for row in range(num_updates):
        for col in range(parameter_num):
            ax = axes[row, col]
            posterior_vector = posterior_vectors[row]  # Get the posterior vector for this update
            counts_posterior, bins_posterior, _ = ax.hist(posterior_vector[:, col], bins=30, density=True, alpha=0.5, color=colors[1], label='Posterior')
            mean_posterior = np.mean(posterior_vector[:, col])
            ax.axvline(mean_posterior, color='black', linestyle='dashed', linewidth=1, label='Mean')

            # KDE line
            kde = gaussian_kde(posterior_vector[:, col])
            x = np.linspace(x_limits[col][0], x_limits[col][1], 1000)
            ax.plot(x, kde(x), color='blue', linestyle='-', linewidth=1.5, label='KDE')

            # Normal Fit
            mu, std = norm.fit(posterior_vector[:, col])
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, color='red', linestyle='--', linewidth=1.5, label='Normal Fit')

            ax.set_xlabel(parameter_names[col], fontsize=label_fontsize)
            ax.set_ylabel('Density', fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)
            ax.grid(True)
            ax.tick_params(direction='in', labelsize=tick_fontsize)
            ax.set_ylim(0, y_max[col])
            ax.set_xlim(x_limits[col])

        fig.suptitle(f'Posterior Evaluation: {row + 1}', fontsize=title_fontsize, y=1.02 - (row * 0.1))
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2)  # Add more space between rows and columns
    plt.savefig('combined_posterior_distributions.png')
    plt.close()


def plot_bme_re(bayesian_dict, num_bal_iterations):
    # Extract BME and RE for plotting
    iterations = list(range(num_bal_iterations))
    bme_values = [bayesian_dict['BME'][it] for it in iterations]
    re_values = [bayesian_dict['RE'][it] for it in iterations]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot BME
    axes[0].plot(iterations, bme_values, marker='+', markersize=10, color='darkslategray', linestyle='-', linewidth=1.5, label='BME')
    axes[0].set_title('Bayesian Model Evidence (BME)', fontsize=16)
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel('BME', fontsize=14)
    axes[0].grid(True, linestyle='--', linewidth=1)
    axes[0].legend()

    # Add a dashed tendency line for BME
    slope_bme, intercept_bme, _, _, _ = linregress(iterations, bme_values)
    trend_bme = [slope_bme * x + intercept_bme for x in iterations]
    axes[0].plot(iterations, trend_bme, color='darkslategray', linestyle='--', linewidth=1.5, label='Trend Line')
    axes[0].legend()

    # Set x-axis limits for BME plot to start from the very beginning
    axes[0].set_xlim(iterations[0], iterations[-1])

    # Set x-axis major locator to show ticks every 1 unit
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(1))

    # Plot RE
    axes[1].plot(iterations, re_values, marker='x', markersize=10, color='dimgray', linestyle='-', linewidth=1.5, label='RE')
    axes[1].set_title('Relative Entropy (RE)', fontsize=16)
    axes[1].set_xlabel('Iteration', fontsize=14)
    axes[1].set_ylabel('RE', fontsize=14)
    axes[1].grid(True, linestyle='--', linewidth=1)
    axes[1].legend()

    # Add a dashed tendency line for RE
    slope_re, intercept_re, _, _, _ = linregress(iterations, re_values)
    trend_re = [slope_re * x + intercept_re for x in iterations]
    axes[1].plot(iterations, trend_re, color='dimgray', linestyle='--', linewidth=1.5, label='Trend Line')
    axes[1].legend()

    # Set x-axis limits for RE plot to start from the very beginning
    axes[1].set_xlim(iterations[0], iterations[-1])

    # Set x-axis major locator to show ticks every 1 unit
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(1))

    # Adjust layout
    plt.tight_layout()
    plt.savefig('RE_BME_plots.png')
    plt.close()

def plot_combined_bal(collocation_points, n_init_tp, bayesian_dict, save_name=None):
    """
    Plots the initial training point and which points were selected using DKL and which ones using BME, when the
    combined utility function is chosen.
    Args:
        collocation_points: np.array[n_tp, n_param]
            Array with all collocation points, in order in which they were selected.
        n_init_tp: int
            Number of TP selected initially
        bayesian_dict: dictionary
            With keys 'util_function', which details which utility function was used in each iteration.
        save_name: Path file
            File name with which to save results. Default is None, so no file is saved.

    Returns:

    """

    if collocation_points.shape[1] == 1:
        collocation_points = np.hstack((collocation_points, collocation_points))

    fig, ax = plt.subplots()

    # initial TP:
    ax.scatter(collocation_points[0:n_init_tp, 0], collocation_points[0:n_init_tp, 1], label='InitialTP', c='black',
               s=100)
    selected_tp = collocation_points[n_init_tp:, :]

    # Get indexes for 'dkl'
    dkl_ind = np.where(bayesian_dict['util_func'] == 'dkl')
    ax.scatter(selected_tp[dkl_ind, 0], selected_tp[dkl_ind, 1], label='DKL', c='gold', s=200, alpha=0.5)

    # Get indexes for 'bme'
    bme_ind = np.where(bayesian_dict['util_func'] == 'bme')
    ax.scatter(selected_tp[bme_ind, 0], selected_tp[bme_ind, 1], label='BME', c='blue', s=200, alpha=0.5)

    # Get indexes for 'ie'
    ie_ind = np.where(bayesian_dict['util_func'] == 'ie')
    ax.scatter(selected_tp[ie_ind, 0], selected_tp[ie_ind, 1], label='BME', c='green', s=200, alpha=0.5)

    # Global MC
    ie_ind = np.where(bayesian_dict['util_func'] == 'global_mc')
    ax.scatter(selected_tp[ie_ind, 0], selected_tp[ie_ind, 1], label='MC', c='red', s=200, alpha=0.5)

    ax.set_xlabel('$\omega_1$')
    ax.set_ylabel('$\omega_2$')

    fig.legend(loc='lower center', ncol=5)
    plt.subplots_adjust(top=0.95, bottom=0.15, wspace=0.25, hspace=0.55)
    if save_name is not None:
        plt.savefig(save_name)
    #plt.show(block=False)

# plot_posterior(posterior_vector, parameter_names, prior)
# plot_posterior_updates(posterior_vectors, parameter_names, prior)
#plot_bme_re(bayesian_dict, 10)