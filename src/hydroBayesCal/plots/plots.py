"""
Code for plotting results in the context of Bayesian Calibration with GPE
"""

import numpy as np
import os
import pandas as pd
import seaborn as sns
import corner
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, linregress
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path
from matplotlib import gridspec


class BayesianPlotter:
    def __init__(
            self,
            results_folder_path='',
            plots_subfolder='plots',
            variable_name = ''
    ):
        """
        Constructor of BayesianPlotter class, which is used to create and save various plots related to Bayesian calibration.

        Parameters
        ----------
        results_folder_path : str
            Path to the folder where results (including plots) will be saved. Usually auto-saved-results.
        plots_subfolder : str, optional
            Name of the subfolder within the results folder where plots will be saved. Default folder name is 'plots'.

        Attributes
        ----------
        save_folder : pathlib.Path
            A Path object representing the directory where plots will be saved.
        """
        self.save_folder = Path(results_folder_path) / plots_subfolder / variable_name
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times'],
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'lines.linewidth': 1.5,
            'lines.markersize': 8,
            'axes.linewidth': 0.8
        })

    def _set_latex_format(self, ax):
        """
        Sets LaTeX formatting for the text in the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to set LaTeX formatting.
        """
        ax.set_xlabel(ax.get_xlabel(), fontsize=14, family='serif')
        ax.set_ylabel(ax.get_ylabel(), fontsize=14, family='serif')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['right'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)

    def plot_prior_posterior_kde(self, bayesian_data, parameter_names, iterations_to_plot):
        """
        Generates and saves prior and posterior distribution plots using KDEs and histograms.

        Parameters
        ----------
        bayesian_data : dict
            Dictionary containing 'prior' and 'posterior' data.
        """
        prior_forplot = bayesian_data['prior']
        posterior_forplot = bayesian_data['posterior'][iterations_to_plot]
        columns = parameter_names
        df_prior = pd.DataFrame(prior_forplot, columns=columns)
        df_post = pd.DataFrame(posterior_forplot, columns=columns)

        # Create a PairGrid for customized mapping
        g = sns.PairGrid(df_prior, diag_sharey=False, corner=True)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "grey_blue", ["lightgrey", "blue"]
        )
        # Map the lower triangle to filled contour KDE plots
        g.map_lower(sns.kdeplot, fill=True,
                    cmap=cmap
                    )

        # Optionally, map the diagonal to filled KDE plots as well
        g.map_diag(sns.kdeplot, fill=True, lw=2)

        # #% Define the output directory and file paths
        output_dir = self.save_folder  # change this to your desired path
        os.makedirs(output_dir, exist_ok=True)
        pdf_file = os.path.join(output_dir, 'pairplot_prior.pdf')
        png_file = os.path.join(output_dir, 'pairplot_prior.png')

        # # Save the figure as PDF and PNG
        g.fig.savefig(pdf_file, bbox_inches='tight', dpi=300)
        g.fig.savefig(png_file, bbox_inches='tight', dpi=300)

        # Show the plot
        plt.show()
        # %%%%%%%%%%%%%%%%%%%%%%
        # % posterior
        # columns = ["zone3", "zone4", "zone10", "zone12", "zone13", "zone14", "zone15", "zone16", "zone17"]

        df_post = pd.DataFrame(posterior_forplot, columns=columns)

        # Create a PairGrid for customized mapping
        g_post = sns.PairGrid(df_post, diag_sharey=False, corner=True)

        # Map the lower triangle to filled contour KDE plots
        g_post.map_lower(sns.kdeplot, fill=True,
                         cmap="inferno"
                         )

        # Optionally, map the diagonal to filled KDE plots as well
        g_post.map_diag(sns.kdeplot, fill=True, lw=2, color='red')

        # #% Define the output directory and file paths
        output_dir = self.save_folder  # change this to your desired path
        os.makedirs(output_dir, exist_ok=True)
        pdf_file = os.path.join(output_dir, 'pairplot_post.pdf')
        png_file = os.path.join(output_dir, 'pairplot_post.png')

        # Save the figure as PDF and PNG
        g_post.fig.savefig(pdf_file, bbox_inches='tight', dpi=300)
        g_post.fig.savefig(png_file, bbox_inches='tight', dpi=300)

        # Show the plot
        plt.show()

        # %%%%%%%%%%%%%%%%%%%%%%%%%
        # % plot prior and posterior together
        # Custom function for the lower triangle: overlay filled contour KDE plots
        def overlay_lower(x, y, **kwargs):
            ax = plt.gca()
            # Use the series name to get the corresponding column names
            col_x = x.name
            col_y = y.name
            # Remove potential conflicting keyword 'color'
            kwargs_lower = kwargs.copy()
            kwargs_lower.pop('color', None)

            # Plot prior joint KDE (filled)
            sns.kdeplot(x=x, y=y, fill=True, cmap="viridis", ax=ax, **kwargs_lower)
            # Overlay posterior joint KDE (filled) with transparency
            sns.kdeplot(x=df_post[col_x], y=df_post[col_y], fill=True, cmap="inferno", ax=ax,
                        # alpha=0.5
                        )

        # Custom function for the diagonal: overlay individual KDE plots
        def overlay_diag(x, **kwargs):
            ax = plt.gca()
            # Remove potential conflicting keyword 'color'
            kwargs_diag = kwargs.copy()
            kwargs_diag.pop('color', None)

            # Plot the prior KDE (green)
            sns.kdeplot(x=x, fill=True, lw=2, color='green', ax=ax, **kwargs_diag)
            # Overlay the posterior KDE (red with transparency)
            sns.kdeplot(x=df_post[x.name], fill=True, lw=2, color='red', ax=ax, alpha=0.5)

        # Create a PairGrid using df_prior as the layout basis
        g = sns.PairGrid(df_prior, diag_sharey=False, corner=True)

        # Map the custom functions to the lower triangle and diagonal
        g.map_lower(overlay_lower)
        g.map_diag(overlay_diag)

        # Create proxy legend handles
        prior_patch = mpatches.Patch(color='green', label='Prior')
        posterior_patch = mpatches.Patch(color='red', label='Posterior')

        # Add the legend to the figure; adjust bbox_to_anchor as needed
        g.fig.legend(handles=[prior_patch, posterior_patch], loc='upper right', bbox_to_anchor=(0.95, 0.95))
        plt.show()
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % plot the measured velocity, surrogate velocity and complex model velocty at each location.
        # Number of locations
        # n_locations = 36
        # nrows = 6
        # ncols = 6
        # obs_flat = obs.flatten()
        #
        # # Create a grid of subplots (6x6 for 36 locations)
        # fig, axs = plt.subplots(nrows, ncols, figsize=(25, 25), sharex=False, sharey=False)
        #
        # for i in range(n_locations):
        #     row = i // ncols
        #     col = i % ncols
        #     ax = axs[row, col]
        #
        #     # Extract simulation data for location i
        #     cm_vals = cm_outputs[:, i]  # Complex model values (11 values)
        #     sm_vals = sm_outputs[:, i]  # Surrogate model values (11 values)
        #
        #     # Plot KDE distributions using Seaborn.
        #     sns.kdeplot(x=cm_vals, ax=ax, fill=True, color='blue', alpha=0.5, label='CM')
        #     sns.kdeplot(x=sm_vals, ax=ax, fill=True, color='green', alpha=0.5, label='SM')
        #
        #     # Overlay the observation as a red marker and a dashed vertical line.
        #     obs_val = obs_flat[i]  # Get a scalar from the flattened observations
        #     ax.plot(obs_val, 0, 'ro', markersize=15)
        #     ax.axvline(obs_val, color='red', linestyle='--', alpha=0.7)
        #
        #     ax.set_title(f'Location {i + 1}', fontsize=20)
        #     # Remove the y-axis label for this subplot.
        #     ax.set_ylabel('')
        #     # ax.set_xlabel('Velocity')
        #     ax.tick_params(labelsize=20)
        #
        # # Create custom legend handles.
        # line_cm = Line2D([0], [0], color='blue', lw=2, label='Complex Model')
        # line_sm = Line2D([0], [0], color='green', lw=2, label='Surrogate Model')
        # # Set lw=1 (instead of 0) to avoid dash errors.
        # line_obs = Line2D([0], [0], marker='o', color='red', lw=1, markersize=5, linestyle='--', label='Observation')
        #
        # # Place a common legend for all subplots at the top center.
        # fig.legend(handles=[line_cm, line_sm, line_obs],
        #            loc='upper center', ncol=3, fontsize=20,
        #            # title_fontsize=10
        #            )
        #
        # # Add a single overall y-axis label for density.
        # fig.text(0.01, 0.5, 'Density', rotation='vertical', ha='center', va='center', fontsize=20
        #          # 'small'
        #          )
        # fig.text(0.5, 0.005, 'Velocity [m s$^{-1}$]', ha='center', va='center', fontsize=20)
        #
        # # Adjust layout to ensure nothing is clipped, leaving space for the legend.
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        #
        # # # Save the figure as PDF and PNG.
        # output_dir = '/home/ran-wei/Documents/coding2025/hydrodynamic_model_surrogate/hydrobayesian_2dhydrodynamic/exercises_rw/bal/results/telemac2d_test/auto-saved-results-1-quantities_WATER DEPTH/plots/distributions'
        # os.makedirs(output_dir, exist_ok=True)
        # pdf_file = os.path.join(output_dir, 'subplots_by_location2.pdf')
        # png_file = os.path.join(output_dir, 'subplots_by_location2.png')
        # fig.savefig(pdf_file, bbox_inches='tight', dpi=300)
        # fig.savefig(png_file, bbox_inches='tight', dpi=300)
        #
        # plt.show()

    def plot_posterior_updates(
            self,
            posterior_arrays,
            parameter_names,
            prior,
            param_values=None,
            iterations_to_plot=None,
            bins=40,
            density=True,
            plot_prior=False
    ):
        """
        Plots posterior distributions, highlighting the max density value for the considered iteration.

        Parameters
        ----------
        posterior_arrays: list of arrays
            List of 2D arrays with posterior samples for each update.
        parameter_names: list of str
            List of parameter names.
        prior: array
            2D array with prior samples.
        param_values: list of arrays, optional
            X-axis limits for each parameter.
        iterations_to_plot: list of int or None
            Iteration indices to plot.
        bins: int
            Number of bins for histograms.
        density: bool
            Normalize histograms to probability density.
        plot_prior: bool
            Whether to plot the prior distribution.

        Returns
        -------
        None
            Saves the plots.
        """
        save_folder = self.save_folder
        save_folder.mkdir(parents=True, exist_ok=True)

        parameter_num = len(parameter_names)

        # Define x-axis limits
        x_limits = np.zeros((parameter_num, 2))
        if param_values is None:
            for i in range(parameter_num):
                x_limits[i] = (prior[:, i].min(), prior[:, i].max())
        else:
            for i in range(parameter_num):
                x_limits[i] = param_values[i]

        # Calculate y-axis limits for the selected iterations
        y_min_posterior = np.zeros(parameter_num)
        y_max_posterior = np.zeros(parameter_num)

        if iterations_to_plot is not None:
            for i in range(parameter_num):
                y_min_posterior[i] = np.inf
                for iteration_idx in iterations_to_plot:
                    if posterior_arrays[iteration_idx] is not None:
                        counts_posterior, _ = np.histogram(posterior_arrays[iteration_idx][:, i], bins=bins,
                                                           density=density)
                        y_max_posterior[i] = max(y_max_posterior[i], max(counts_posterior)) * 1.15
                        y_min_posterior[i] = min(y_min_posterior[i], min(counts_posterior))
                y_min_posterior[i] = 0  # Ensure the minimum is set to 0

        # Plot posterior distributions
        if iterations_to_plot is not None:
            for plot_index, iteration_idx in enumerate(iterations_to_plot):
                for col in range(parameter_num):
                    fig, ax = plt.subplots(figsize=(6, 8))

                    # Get the posterior samples for the current iteration
                    posterior_vector = posterior_arrays[iteration_idx][:, col]

                    # Compute histogram density
                    counts_posterior, bin_edges = np.histogram(posterior_vector, bins=bins, density=density)

                    # Find the mode (max density value)
                    max_density_idx = np.argmax(counts_posterior)
                    mode_value = (bin_edges[max_density_idx] + bin_edges[max_density_idx + 1]) / 2

                    # Plot posterior histogram
                    ax.hist(posterior_vector, bins=bins, density=density, alpha=0.6, color='grey',
                            edgecolor='black', linewidth=0.8, label='Posterior')

                    # Optionally plot the prior
                    if plot_prior:
                        ax.hist(prior[:, col], bins=bins, density=density, alpha=0.2, color='#1E90FF',
                                edgecolor='black', linewidth=0.8, label='Prior')

                    # Set axis labels
                    ax.set_xlabel(f'{parameter_names[col]}')
                    ax.set_ylabel('Density')

                    # Apply LaTeX formatting
                    self._set_latex_format(ax)

                    # Set x and y limits
                    ax.set_xticks(np.round(np.linspace(x_limits[col][0], x_limits[col][1], 4), 3))
                    ax.set_ylim(y_min_posterior[col], y_max_posterior[col])
                    ax.set_xlim(x_limits[col])

                    # Add vertical lines for mode and mean
                    ax.axvline(mode_value, color='red', linestyle='--', linewidth=2,
                               label=f'Max Density: {mode_value:.3f}')  # Show exact mode value
                    mean_value = np.mean([x_limits[col][0], x_limits[col][1]])
                    ax.axvline(mean_value, color='blue', linestyle='--', linewidth=1.5, label='Mean Value')

                    # Add grid
                    ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='lightgrey')
                    ax.minorticks_on()
                    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='grey')

                    # Update legend
                    ax.legend(fontsize=12)

                    fig.tight_layout()

                    # Save the plot
                    fig.savefig(
                        save_folder / f'combined_distribution_{parameter_names[col].replace(" ", "_")}_iteration_{iteration_idx + 1}.png')
                    plt.close(fig)

    def plot_posterior_iteration(self, posterior_samples, parameter_names, param_values):
        """
        Generates a corner plot for the posterior distributions with custom axis limits.

        Parameters
        ----------
        posterior_samples : array
            2D array with posterior samples (N samples x D parameters).
        parameter_names : list
            Names of the parameters.
        param_values : list of lists
            Axis limits for each parameter in the form [[min1, max1], [min2, max2], ...]

        Returns
        -------
        None
            Saves the corner plot.
        """
        # Convert to DataFrame for easier handling
        df_posterior = pd.DataFrame(posterior_samples, columns=parameter_names)

        # Create a custom PairGrid with larger size
        g = sns.PairGrid(df_posterior, diag_sharey=False, height=4.5, aspect=1.5, corner=True)

        # Map scatter plot for posterior (small transparent dots)
        g.map_lower(plt.scatter, alpha=0.1, s=1, color='blue')

        # Add KDE contours for posterior
        g.map_lower(sns.kdeplot, levels=5, color='blue', alpha=0.8, fill=True)

        # Calculate the x-value where the highest density occurs for each histogram and store it for the legend
        max_density_values = {}

        # Plot histograms on diagonal for posterior (density values)
        g.map_diag(sns.histplot, bins=30, color='grey', alpha=0.6, stat='density', kde=True)

        # Set axis limits, modify grid lines, and add density labels
        for i in range(len(parameter_names)):  # Loop over rows
            for j in range(i + 1):  # Loop over columns (lower triangle + diagonal)
                ax = g.axes[i, j]
                if ax is None:
                    continue  # Skip empty plots due to corner=True

                # Get predefined limits
                x_min, x_max = param_values[j]  # X-axis follows column parameter
                ax.set_xlim(x_min, x_max)

                # Y-axis limits (set to 0 and 1 for density plots)
                if i == j:  # Diagonal plots
                    ax.set_ylabel("Density", fontsize=10)  # Add density label
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))  # Format density ticks

                    # Calculate histogram density values
                    counts, bin_edges = np.histogram(df_posterior[parameter_names[i]], bins=30, density=True)
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Get the center of the bins

                    # Find the x value where density is maximum (mode of the histogram)
                    max_density_index = np.argmax(counts)
                    max_density_x_value = bin_centers[max_density_index]

                    # Store this max density x-value for the legend
                    max_density_values[parameter_names[i]] = max_density_x_value

                # Set only min and max ticks (hidden but needed for grid alignment)
                mid_tick = (x_min + x_max) / 2  # Calculate the midpoint for the x-axis

                # Add min, max, and midpoint ticks
                ax.set_xticks([x_min, mid_tick, x_max])  # Set ticks at min, middle, and max
                # Add primary and secondary grid lines
                ax.grid(True, linestyle='--', alpha=1, linewidth=1.5, which='major')  # Main grid
                ax.grid(True, linestyle=':', alpha=1, linewidth=1, which='minor')  # Secondary grid
                ax.minorticks_on()  # Enable minor ticks (without labels)
                ax.axvline(x=x_min, linestyle='--', color='black', linewidth=1.5)  # Thicker vertical primary grid line
                ax.axvline(x=x_max, linestyle='--', color='black', linewidth=1.5)  # Thicker vertical primary grid line

                # Increase the width of the vertical secondary grid lines
                ax.axvline(x=x_min, linestyle=':', color='black', linewidth=1.5)  # Thinner vertical secondary grid line
                ax.axvline(x=x_max, linestyle=':', color='black', linewidth=1.5)  # Thinner vertical secondary grid line
                # Format the tick labels to three decimal places
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))

        # Set the first vertical label for clarity
        g.axes[0, 0].set_ylabel(parameter_names[0], fontsize=12)

        # Add the legend with the x-value of the maximum density for each parameter
        legend_labels = [f"{param}: {max_density:.3f}" for param, max_density in max_density_values.items()]
        g.fig.legend(legend_labels, loc='upper right', fontsize=50, title="Max Density X-value", title_fontsize=60,
                     frameon=True, fancybox=True, facecolor='white', edgecolor='black')

        # Improve layout
        plt.tight_layout()

        # Save figure
        save_path = self.save_folder / "plot_posterior.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    def plot_bme_re(
            self,
            bayesian_dict,
            num_bal_iterations,
            plot_type='both'
    ):
        """
        Plots BME and/or RE values over iterations.

        Parameters
        ----------
            bayesian_dict: dict
                Dictionary containing 'BME' and 'RE' values for each iteration.
            num_bal_iterations: int
                Number of iterations for which to plot data.
            plot_type: str
                Type of plot to generate, can be 'BME', 'RE', or 'both'.

        Returns
        -------
            None
                The function creates plots of BME or RE values over iterations and are saved
                as .png files in the /plots folder.
        """
        save_folder = self.save_folder

        # Ensure save_folder exists
        save_folder.mkdir(parents=True, exist_ok=True)

        # Extract BME and RE for plotting
        iterations = list(range(num_bal_iterations))
        bme_values = [bayesian_dict['BME'][it] for it in iterations]
        re_values = [bayesian_dict['RE'][it] for it in iterations]

        if plot_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Plot BME
            axes[0].plot(iterations, bme_values, marker='+', markersize=10, color='black', linestyle='-', linewidth=0.8,
                         label=r'BME')
            axes[0].set_title(r'Bayesian Model Evidence (BME)', fontsize=16, weight='normal')
            axes[0].set_xlabel(r'Iteration', fontsize=14)
            axes[0].set_ylabel(r'BME', fontsize=14)
            axes[0].grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            axes[0].legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for BME
            slope_bme, intercept_bme, _, _, _ = linregress(iterations, bme_values)
            trend_bme = [slope_bme * x + intercept_bme for x in iterations]
            axes[0].plot(iterations, trend_bme, color='darkslategray', linestyle='--', linewidth=0.8,
                         label=r'Trend Line')
            axes[0].legend(fontsize=12, loc='upper left')

            # Set x-axis limits and tick parameters for BME plot
            axes[0].set_xlim(iterations[0], iterations[-1])
            axes[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes[0].tick_params(axis='both', which='major', labelsize=12)
            axes[0].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            axes[0].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            axes[0].yaxis.get_offset_text().set_fontsize(12)

            # Reduce the thickness of the axes borders
            for spine in axes[0].spines.values():
                spine.set_linewidth(0.5)

            # Plot RE
            axes[1].plot(iterations, re_values, marker='x', markersize=10, color='black', linestyle='-', linewidth=0.8,
                         label=r'RE')
            axes[1].set_title(r'Relative Entropy (RE)', fontsize=16, weight='normal')
            axes[1].set_xlabel(r'Iteration', fontsize=14)
            axes[1].set_ylabel(r'RE', fontsize=14)
            axes[1].grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            axes[1].legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for RE
            slope_re, intercept_re, _, _, _ = linregress(iterations, re_values)
            trend_re = [slope_re * x + intercept_re for x in iterations]
            axes[1].plot(iterations, trend_re, color='dimgray', linestyle='--', linewidth=0.8, label=r'Trend Line')
            axes[1].legend(fontsize=12, loc='upper left')

            # Set x-axis limits and tick parameters for RE plot
            axes[1].set_xlim(iterations[0], iterations[-1])
            axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes[1].tick_params(axis='both', which='major', labelsize=12)
            axes[1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            axes[1].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            axes[1].yaxis.get_offset_text().set_fontsize(12)

            # Reduce the thickness of the axes borders
            for spine in axes[1].spines.values():
                spine.set_linewidth(0.5)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(save_folder / 'BME_RE_plots.png')

        elif plot_type == 'BME':
            plt.figure(figsize=(8, 6))

            # Plot BME
            plt.plot(iterations, bme_values, marker='+', markersize=10, color='black', linestyle='-', linewidth=0.8,
                     label=r'BME')
            plt.title(r'Bayesian Model Evidence (BME)', fontsize=16, weight='normal')
            plt.xlabel(r'Iteration', fontsize=14)
            plt.ylabel(r'BME', fontsize=14)
            plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            plt.legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for BME
            slope_bme, intercept_bme, _, _, _ = linregress(iterations, bme_values)
            trend_bme = [slope_bme * x + intercept_bme for x in iterations]
            plt.plot(iterations, trend_bme, color='darkslategray', linestyle='--', linewidth=0.8, label=r'Trend Line')
            plt.legend(fontsize=12, loc='upper left')

            # Set x-axis limits and tick parameters for BME plot
            plt.xlim(iterations[0], iterations[-1])
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            plt.gca().yaxis.get_offset_text().set_fontsize(12)

            # Reduce the thickness of the axes borders
            for spine in plt.gca().spines.values():
                spine.set_linewidth(0.5)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(save_folder / 'BME_plot.png')

        elif plot_type == 'RE':
            plt.figure(figsize=(8, 6))

            # Plot RE
            plt.plot(iterations, re_values, marker='x', markersize=10, color='black', linestyle='-', linewidth=0.8,
                     label=r'RE')
            plt.title(r'Relative Entropy (RE)', fontsize=16, weight='normal')
            plt.xlabel(r'Iteration', fontsize=14)
            plt.ylabel(r'RE', fontsize=14)
            plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            plt.legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for RE
            slope_re, intercept_re, _, _, _ = linregress(iterations, re_values)
            trend_re = [slope_re * x + intercept_re for x in iterations]
            plt.plot(iterations, trend_re, color='dimgray', linestyle='--', linewidth=0.8, label=r'Trend Line')
            plt.legend(fontsize=12, loc='upper left')

            # Set x-axis limits and tick parameters for RE plot
            plt.xlim(iterations[0], iterations[-1])
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            plt.gca().yaxis.get_offset_text().set_fontsize(12)

            # Reduce the thickness of the axes borders
            for spine in plt.gca().spines.values():
                spine.set_linewidth(0.5)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(save_folder / 'RE_plot.png')

        plt.close()

    def plot_combined_bal(
            self,
            collocation_points,
            n_init_tp,
            bayesian_dict
    ):
        """
        Plots the initial training points and points selected using different utility functions.

        Parameters
        ----------
            collocation_points: array [n_tp, n_param]
                Array with all collocation points, in order in which they were selected.
            n_init_tp: int
                Number of initial training points selected.
            bayesian_dict: dictionary
                With keys 'util_func', detailing which utility function was used in each iteration.
            save_folder: Path or None
                Directory where to save the plot. If None, the plot is not saved.

        Returns
        -------
            None
                The function creates a scattered plot of the collocation points differentiating them between initial collocation
                points and BAL-selected points, saved as .png files in the /plots folder.
        """
        save_folder = self.save_folder

        # Ensure save_folder exists
        save_folder.mkdir(parents=True, exist_ok=True)

        if collocation_points.shape[1] == 1:
            collocation_points = np.hstack((collocation_points, collocation_points))

        fig, ax = plt.subplots()

        # Plot each initial training point individually for better visibility
        for i in range(n_init_tp):
            ax.scatter(collocation_points[i, 0], collocation_points[i, 1], label='Initial TP' if i == 0 else "",
                       c='black', s=100, edgecolor='white', marker='o')

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

        # LaTeX formatting for labels and legend
        ax.set_xlabel(r'$K_{Zone \, 8}$', fontsize=14)
        ax.set_ylabel(r'$K_{Zone \, 9}$', fontsize=14)

        fig.legend(loc='lower center', ncol=5, fontsize=12)

        # Adjust layout to make space for the legend
        plt.subplots_adjust(top=0.95, bottom=0.15, wspace=0.25, hspace=0.55)

        # Save the figure
        if save_folder:
            plt.savefig(save_folder / 'collocation_points.png')  # Save with .png extension
        plt.close()

    def plot_bme_3d(
            self,
            collocation_points,
            param_ranges,
            param_names,
            bme_values,
            param_indices=(1, 4),
            extra_param_index=2,
            grid_size=100,
            iteration_range=(1, 20),  # Specify the range of iterations
            plot_criteria="metric"
    ):
        """
        Plots the BME scatter for the specified range of iterations, a 3d surface interpolated from the scatter BME values,
        and adds a 2D contour plot to show high BME regions for 2 selected parameters.

        Parameters
        ----------
            param_values: array
                2D array where each row corresponds to parameter values for each iteration.
            param_ranges: list of lists
                List of [min, max] values for each parameter.
            bme_values: list of float
                List of BME values, one for each iteration.
            param_indices: tuple of int
                Indices of the two parameters to plot.
            extra_param_index: int, optional
                Index of the extra parameter for the 3D scatter plot.
            grid_size: int
                Size of the grid for the surface and contour plots.
            iteration_range: tuple of int
                Range of iterations to consider for the plot, inclusive.
            plot_criteria: str
                The criteria being plotted (e.g., 'BME' or 'RE').

        Returns
        -------
            None
                The function creates BME plots and are saved as .png files in the /plots folder.
        """
        save_folder = self.save_folder
        if save_folder:
            save_folder = Path(save_folder)  # Ensure save_folder is a Path object
            save_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        # Validate iteration range
        start_iter, end_iter = iteration_range
        if start_iter < 0 or end_iter >= len(bme_values) or start_iter > end_iter:
            raise ValueError("Invalid iteration range specified")

        # Extract BME values and corresponding parameters for the specified iteration range
        bme_values = bme_values[start_iter:end_iter ]
        param_values = collocation_points[start_iter:end_iter , :]

        # Extract ranges for the selected parameters
        x_range = param_ranges[param_indices[0]]
        y_range = param_ranges[param_indices[1]]

        # Extract names for the selected parameters
        x_name = param_names[param_indices[0]]
        y_name = param_names[param_indices[1]]

        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)

        # Prepare data for interpolation
        points = param_values[:, param_indices]
        values = bme_values

        # Ensure points and values have the same length
        if len(points) != len(values):
            raise ValueError("Mismatch between number of points and BME values")

        # Interpolate BME values onto the grid
        Z = griddata(points, values, (X, Y), method='cubic')

        # Set Z-axis limits with margin based on BME values
        Z_min = min(values) * 0.98
        Z_max = max(values) * 1.05
        margin = (Z_max - Z_min)  # 10% margin
        Z = np.clip(Z, Z_min, Z_max)

        # Set universal font properties
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'font.weight': 'normal',
                             'axes.labelsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                             'axes.linewidth': 0.8})  # Reduced axes line width

        # Helper function to set grid style
        def set_grid_style(ax):
            ax.grid(True, linestyle='--', color='lightgrey', alpha=0.7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)  # Set axis border thickness

        # Helper function to adjust figure margins
        def adjust_margins(fig):
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)

        # Find the point with the highest BME value
        max_bme_index = np.argmax(values)
        max_bme_point = points[max_bme_index]
        max_bme_value = values[max_bme_index]

        # 3D Scatter Plot
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        scatter = ax1.scatter(points[:, 0], points[:, 1], values, c=values, cmap='plasma', edgecolor='none', alpha=0.7)
        ax1.set_title(f'{plot_criteria} Scatter Plot (Iterations {start_iter} to {end_iter})', fontsize=16,
                      weight='normal')
        ax1.set_xlabel(f'{x_name}', fontsize=12)
        ax1.set_ylabel(f'{y_name}', fontsize=12)
        ax1.set_zlabel(f'{plot_criteria}', fontsize=12, rotation=90)  # Make BME axis title vertical
        ax1.set_zlim(Z_min - margin, Z_max + margin)
        ax1.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar1 = fig1.colorbar(scatter, orientation='vertical')
        cbar1.set_label(f'{plot_criteria} Value', fontsize=12)
        cbar1.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Set grid style for 3D plot
        set_grid_style(ax1)

        adjust_margins(fig1)
        fig1.tight_layout()
        fig1.savefig(save_folder / f'{plot_criteria}_scatter.png')  # Save with .png extension

        # 2D Contour Plot
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        levels = np.linspace(Z_min, Z_max, 100)
        contour = ax2.contourf(X, Y, Z, cmap='viridis', levels=levels, alpha=0.8)  # Use 'plasma' for better visibility
        ax2.set_title(f'2D - {plot_criteria} Values (Iterations {start_iter} to {end_iter})', fontsize=16,
                      weight='normal')
        ax2.set_xlabel(f'{x_name}', fontsize=12)
        ax2.set_ylabel(f'{y_name}', fontsize=12)

        # Optional: Highlight high BME regions
        high_bme_indices = np.where(Z > np.percentile(values, 95))  # Example threshold for high BME
        ax2.scatter(X[high_bme_indices], Y[high_bme_indices], color='red', s=10, label=f'High {plot_criteria} Regions',
                    alpha=0.5)

        ax2.legend(fontsize=10)

        # Add a color bar for the contour plot
        cbar2 = fig2.colorbar(contour, orientation='vertical')
        cbar2.set_label(f'{plot_criteria} Value', fontsize=12)
        cbar2.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Set grid style for 2D plot
        set_grid_style(ax2)

        adjust_margins(fig2)
        fig2.tight_layout()
        fig2.savefig(save_folder / f'2D_{plot_criteria}_contour_values.png')  # Save with .png extension

        # Continue with other plots...

        # 3D Surface Plot
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        surf = ax3.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
        ax3.set_title(f'{plot_criteria} Surface Plot (Iterations {start_iter} to {end_iter})', fontsize=16,
                      weight='normal')
        ax3.set_xlabel(f'{x_name}', fontsize=12)
        ax3.set_ylabel(f'{y_name}', fontsize=12)
        ax3.set_zlabel(f'{plot_criteria}', fontsize=12, rotation=90)  # Make BME axis title vertical
        ax3.set_zlim(Z_min - margin, Z_max + margin)
        ax3.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar3 = fig3.colorbar(surf, orientation='vertical')
        cbar3.set_label(f'{plot_criteria}', fontsize=12)
        cbar3.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Set grid style for 3D plot
        set_grid_style(ax3)

        adjust_margins(fig3)
        fig3.tight_layout()
        fig3.savefig(save_folder / f'3D_{plot_criteria}_surface_plot.png')  #
        # Show the plot to the user
        #plt.show()

        if extra_param_index is not None:
            # Prepare data for interpolation with extra parameter
            x_extra_range = param_ranges[extra_param_index]
            x_extra = np.linspace(x_extra_range[0], x_extra_range[1], grid_size)
            X_extra, Y_extra = np.meshgrid(x_extra, y)

            points_extra = param_values[:, [extra_param_index, param_indices[1]]]
            Z_extra = griddata(points_extra, values, (X_extra, Y_extra), method='cubic')
            Z_extra = np.clip(Z_extra, Z_min - margin, Z_max + margin)

            # 3D Scatter Plot with extra parameter
            fig4 = plt.figure(figsize=(8, 6))
            ax4 = fig4.add_subplot(111, projection='3d')
            scatter4 = ax4.scatter(param_values[:, param_indices[0]], param_values[:, param_indices[1]],
                                   param_values[:, extra_param_index], c=values, cmap='viridis', edgecolor='none',
                                   alpha=0.7)  # Changed colormap to 'plasma' for better visibility
            ax4.set_title(f'3D - Scatter Plot', fontsize=16,
                          weight='normal')
            ax4.set_xlabel(f'{x_name}', fontsize=12)
            ax4.set_ylabel(f'{y_name}', fontsize=12)
            z_name = param_names[extra_param_index]
            ax4.set_zlabel(f'{z_name}', fontsize=12)
            ax4.view_init(elev=30, azim=225)  # Adjust view angle

            # Add a color bar
            cbar4 = fig4.colorbar(scatter4, orientation='vertical')
            cbar4.set_label(f'{plot_criteria} Value', fontsize=12)
            cbar4.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

            # Set grid style for 3D plot
            set_grid_style(ax4)

            adjust_margins(fig4)
            fig4.tight_layout()
            fig4.savefig(save_folder / '3-parameters scatter plot.png')  # Save with .png extension

    def plot_bme_comparison(
            self,
            param_sets,
            param_ranges,
            param_names,
            bme_values,
            param_indices=(0, 1),
            grid_size=100,
            total_iterations_range=(0, 100),  # Total range of iterations to consider
            iterations_per_subplot=10,  # Number of iterations per subplot
            plot_criteria="BME"
    ):
        """
        Creates comparison plots of 2D BME or RE values across specified iteration ranges
        in a single figure with subplots.

        Parameters
        ----------
            param_sets: array
                2D array where each row corresponds to parameter values for each iteration.
            param_ranges: list of lists
                List of [min, max] values for each parameter.
            param_names: list of str
                Names of the parameters.
            bme_values: list of float
                List of BME values, one for each iteration.
            param_indices: tuple of int
                Indices of the two parameters to plot.
            grid_size: int
                Size of the grid for the surface and contour plots.
            total_iterations_range: tuple of int
                Total range of iterations to consider (start, end).
            iterations_per_subplot: int
                Number of iterations to display in each subplot.
            plot_criteria: str
                The criteria being plotted (e.g., 'BME' or 'RE').

        Returns
        -------
            None
                The function creates a comparison plot and saves it as a .png file in the /plots folder.
        """
        save_folder = self.save_folder
        if save_folder:
            save_folder = Path(save_folder)  # Ensure save_folder is a Path object
            save_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        start_iter, end_iter = total_iterations_range
        if start_iter < 0 or end_iter >= len(bme_values) or start_iter > end_iter:
            raise ValueError(f"Invalid total iteration range specified: {total_iterations_range}")

        # Calculate the iteration ranges for subplots
        iteration_ranges = [(i, min(i + iterations_per_subplot , end_iter)) for i in
                            range(start_iter, end_iter , iterations_per_subplot)]

        num_ranges = len(iteration_ranges)
        ncols = min(num_ranges, 4)  # Maximum 4 subplots per row
        nrows = (num_ranges + 3) // 4  # Calculate number of rows needed
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 6), sharey=True)

        # Flatten the axes array for easy iteration
        axes = axes.flatten() if num_ranges > 1 else [axes]

        # Set universal font properties
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.weight': 'normal',
                             'axes.labelsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                             'axes.linewidth': 0.8})  # Reduced axes line width

        # Helper function to set grid style
        def set_grid_style(ax):
            ax.grid(True, linestyle='--', color='lightgrey', alpha=0.7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)  # Set axis border thickness

        for i, iteration_range in enumerate(iteration_ranges):
            start, end = iteration_range

            # Extract BME values and corresponding parameters for the specified iteration range
            bme_range_values = bme_values[start:end ]
            param_range_values = param_sets[start:end, :]

            # Extract ranges for the selected parameters
            x_range = param_ranges[param_indices[0]]
            y_range = param_ranges[param_indices[1]]

            # Extract names for the selected parameters
            x_name = param_names[param_indices[0]]
            y_name = param_names[param_indices[1]]

            x = np.linspace(x_range[0], x_range[1], grid_size)
            y = np.linspace(y_range[0], y_range[1], grid_size)
            X, Y = np.meshgrid(x, y)

            # Prepare data for interpolation
            points = param_range_values[:, param_indices]
            values = bme_range_values

            # Interpolate BME values onto the grid
            Z = griddata(points, values, (X, Y), method='cubic')

            # Set Z-axis limits with margin based on BME values
            Z_min = min(values) * 0.98
            Z_max = max(values) * 1.05
            margin = (Z_max - Z_min)  # 10% margin
            Z = np.clip(Z, Z_min, Z_max)

            # Plot in the current subplot
            ax = axes[i]
            levels = np.linspace(Z_min, Z_max, 100)
            contour = ax.contourf(X, Y, Z, cmap='plasma', levels=levels,
                                  alpha=0.8)  # Use 'plasma' for better visibility
            ax.set_title(f'{plot_criteria} Values (Iterations {start} to {end})', fontsize=14)
            ax.set_xlabel(f'{x_name}', fontsize=12)
            ax.set_ylabel(f'{y_name}', fontsize=12)

            # Optional: Highlight high BME regions
            high_bme_indices = np.where(Z > np.percentile(values, 95))  # Example threshold for high BME
            ax.scatter(X[high_bme_indices], Y[high_bme_indices], color='red', s=10,
                       label=f'High {plot_criteria} Regions',
                       alpha=0.5)

            ax.legend(fontsize=10)
            self._set_latex_format(ax)  # Use the LaTeX formatting function

            # Add color bar for the current subplot
            cbar = fig.colorbar(contour, ax=ax, orientation='vertical')
            cbar.set_label(f'{plot_criteria} Value', fontsize=12)
            cbar.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Hide unused axes
        for j in range(num_ranges, len(axes)):
            axes[j].axis('off')

        # Adjust layout and save figure
        fig.tight_layout()
        fig.savefig(save_folder / f'{plot_criteria}_comparison.png')  # Save with .png extension
        plt.show()
    def plot_bme_surface_3d(
            self,
            collocation_points,
            param_ranges,
            bme_values,
            param_indices=(0, 1),
            grid_size=100,
            last_iterations=25,
    ):
        """
        Plots the BME surface for the last specified iterations and adds a 2D contour plot to show high BME regions.
        TODO: complete docstrings
        Args:
            collocation_points: np.array
                2D array where each row corresponds to parameter values for each iteration.
            param_ranges: list of lists
                List of [min, max] values for each parameter.
            bme_values: list of float
                List of BME values, one for each iteration.
            param_indices: tuple of int
                Indices of the two parameters to plot.
            grid_size: int
                Size of the grid for the surface and contour plots.
            last_iterations: int
                TODO
        """
        num_iterations = len(bme_values) - 1  # -1 because bme_values has iterations + 1 values
        if num_iterations < last_iterations:
            raise ValueError("Number of iterations is less than the last iterations specified")

        # Extract the last iterations + 1 BME values and corresponding parameters
        bme_values = bme_values[-(last_iterations + 1):]
        param_values = collocation_points[-(last_iterations + 1):, :]

        # Extract ranges for the selected parameters
        x_range = param_ranges[param_indices[0]]
        y_range = param_ranges[param_indices[1]]

        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)

        # Prepare data for interpolation
        points = param_values[:, param_indices]
        values = bme_values

        # Ensure points and values have the same length
        if len(points) != len(values):
            raise ValueError("Mismatch between number of points and BME values")

        # Interpolate BME values onto the grid
        Z = griddata(points, values, (X, Y), method='cubic')

        # Set Z-axis limits based on the min and max of BME values
        Z_min = min(values)
        Z_max = max(values)
        Z = np.clip(Z, Z_min, Z_max)

        # Plot the surface and contour
        fig = plt.figure(figsize=(16, 8))

        # 3D Plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
        ax1.set_title('BME Surface Plot (Last Iterations)', fontsize=20)
        ax1.set_xlabel(r'$\omega_{}$'.format(param_indices[0] + 1), fontsize=18)
        ax1.set_ylabel(r'$\omega_{}$'.format(param_indices[1] + 1), fontsize=18)
        ax1.set_zlabel('BME', fontsize=18)
        ax1.set_zlim(Z_min, Z_max)
        ax1.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        cbar.set_label('BME Value', fontsize=14)

        # 2D Contour Plot
        ax2 = fig.add_subplot(122, aspect='equal')
        contour = ax2.contourf(X, Y, Z, cmap='viridis', levels=np.linspace(Z_min, Z_max, 100), alpha=0.8)
        ax2.set_title('Contour Plot of BME Values', fontsize=20)
        ax2.set_xlabel(r'$\omega_{}$'.format(param_indices[0] + 1), fontsize=18)
        ax2.set_ylabel(r'$\omega_{}$'.format(param_indices[1] + 1), fontsize=18)

        # Optional: Plot high BME regions as scatter points
        high_bme_indices = np.where(Z > np.percentile(values, 95))  # Example threshold for high BME
        ax2.scatter(X[high_bme_indices], Y[high_bme_indices], color='red', s=10, label='High BME Regions')

        # Add a color bar for the contour plot
        cbar2 = fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
        cbar2.set_label('BME Value', fontsize=14)

        ax2.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_model_comparisons(self,observed_values, surrogate_outputs, complex_model_outputs):
        """
        Plots comparisons between model outputs and observed values in separate figures,
        and includes statistical measurements in the plot legends.

        Parameters
        ----------
        observed_values : numpy.ndarray
            1D array of observed values.
        model1_outputs : numpy.ndarray
            1D array of outputs from the first model.
        complex_model_outputs : numpy.ndarray
            1D array of outputs from the second model.

        Returns
        -------
        None
        """
        if not (len(observed_values) == len(surrogate_outputs) == len(complex_model_outputs)):
            raise ValueError("All input arrays must have the same length.")

        # Calculate statistical measurements
        mse_model1 = mean_squared_error(observed_values, surrogate_outputs)
        mse_model2 = mean_squared_error(observed_values, complex_model_outputs)
        r2_model1 = r2_score(observed_values.flatten(), surrogate_outputs.flatten())
        r2_model2 = r2_score(observed_values.flatten(), complex_model_outputs.flatten())
        corr_model1_model2 = np.corrcoef(surrogate_outputs, complex_model_outputs)[0, 1]

        # Find min and max values for the reference line
        min_value = min(np.min(observed_values), np.min(surrogate_outputs), np.min(complex_model_outputs))
        max_value = max(np.max(observed_values), np.max(surrogate_outputs), np.max(complex_model_outputs))

        # Figure 1: Observed vs Model 1 Outputs
        plt.figure(figsize=(8, 6))
        plt.scatter(observed_values, surrogate_outputs, color='blue', label='Metamodel Outputs')
        plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Observed vs Observed')
        plt.xlabel('Observed Values')
        plt.ylabel('Metamodel')
        plt.title('Metamodel vs Observed Values')
        plt.legend(title=f'MSE: {mse_model1:.4f}\nR²: {r2_model1:.4f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Figure 2: Observed vs Model 2 Outputs
        plt.figure(figsize=(8, 6))
        plt.scatter(observed_values, complex_model_outputs, color='green', label='Complex model Outputs')
        plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Observed vs Observed')
        plt.xlabel('Observed Values')
        plt.ylabel('Complex model Outputs')
        plt.title('Complex model vs Observed Values')
        plt.legend(title=f'MSE: {mse_model2:.4f}\nR²: {r2_model2:.4f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Figure 3: Model 1 vs Model 2 Outputs
        plt.figure(figsize=(8, 6))
        plt.scatter(surrogate_outputs, complex_model_outputs, color='purple', label='Model Outputs')
        plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Metamodel = Complex model')
        plt.xlabel('Metamodel Outputs')
        plt.ylabel('Complex model Outputs')
        plt.title('Metamodel vs Complex model Outputs')
        plt.legend(title=f'Correlation: {corr_model1_model2:.4f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_model_outputs_vs_locations(self, observed_values, surrogate_outputs,quantity_name, complex_model_outputs,
                                     selected_locations, gpe_lower_ci=None, gpe_upper_ci=None,
                                     measurement_error=None, plot_ci=True, plot_error=False):
        """
        Plots the outputs (velocities) of two models along a "Talweg" axis,
        preserving the exact order of selected locations.

        Parameters
        ----------
        observed_values : numpy.ndarray
            2D array (or 1D) of observed values.
        surrogate_outputs : numpy.ndarray
            2D array (or 1D) of outputs from the surrogate model.
        complex_model_outputs : numpy.ndarray
            2D array (or 1D) of outputs from the complex model.
        selected_locations : list
            List of 1-based locations to be plotted (order is preserved).
        gpe_lower_ci : numpy.ndarray, optional
            2D array (or 1D) of lower confidence intervals from GPE analysis.
        gpe_upper_ci : numpy.ndarray, optional
            2D array (or 1D) of upper confidence intervals from GPE analysis.
        measurement_error : numpy.ndarray, optional
            2D array (or 1D) of measurement errors (standard deviations) for each observed value.
        plot_ci : bool, optional
            Whether to plot the confidence interval from GPE analysis as a shaded area. Default is True.
        plot_error : bool, optional
            Whether to plot measurement error bars. Default is False.

        Returns
        -------
        None
        """

        # Flatten arrays to ensure they are 1D
        observed_values = observed_values.flatten()
        surrogate_outputs = surrogate_outputs.flatten()
        complex_model_outputs = complex_model_outputs.flatten()

        if gpe_lower_ci is not None and gpe_upper_ci is not None:
            gpe_lower_ci = gpe_lower_ci.flatten()
            gpe_upper_ci = gpe_upper_ci.flatten()
        else:
            plot_ci = False

        if measurement_error is not None:
            measurement_error = measurement_error.flatten()

        # Convert 1-based locations to 0-based indices
        selected_indices = np.array([loc - 1 for loc in selected_locations])

        # Validate indices
        max_index = len(observed_values) - 1
        if np.any((selected_indices < 0) | (selected_indices > max_index)):
            raise ValueError(f"Some selected locations are out of range. Valid range: 1 to {len(observed_values)}")

        # Extract values in the exact order given by the user
        observed_selected = observed_values[selected_indices]
        surrogate_selected = surrogate_outputs[selected_indices]
        complex_selected = complex_model_outputs[selected_indices]

        if plot_ci:
            gpe_lower_selected = gpe_lower_ci[selected_indices]
            gpe_upper_selected = gpe_upper_ci[selected_indices]

        if measurement_error is not None and plot_error:
            error_selected = 1 * measurement_error[selected_indices]
        else:
            error_selected = None

        # Compute errors
        surrogate_rmse = mean_squared_error(observed_selected, surrogate_selected, squared=False)
        complex_rmse = mean_squared_error(observed_selected, complex_selected, squared=False)
        surrogate_rmse_all = mean_squared_error(observed_values, surrogate_outputs, squared=False)
        complex_rmse_all = mean_squared_error(observed_values, complex_model_outputs, squared=False)
        surrogate_r2_all = r2_score(observed_values, surrogate_outputs)
        complex_r2_all = r2_score(observed_values, complex_model_outputs)
        surrogate_r2 = r2_score(observed_selected, surrogate_selected)
        complex_r2 = r2_score(observed_selected, complex_selected)

        print(f"Surrogate Model RMSE (selected points) {quantity_name}: {surrogate_rmse:.4f}, R²: {surrogate_r2:.4f}")
        print(f"Complex Model RMSE (selected points) {quantity_name}: {complex_rmse:.4f}, R²: {complex_r2:.4f}")
        print(f"Surrogate Model RMSE (all points) {quantity_name}: {surrogate_rmse_all:.4f}, R²: {surrogate_r2_all:.4f}")
        print(f"Complex Model RMSE (all points) {quantity_name}: {complex_rmse_all:.4f}, R²: {complex_r2_all:.4f}")
        # Define "Talweg" axis (1, 2, 3, ... in the order of input)
        talweg_positions = np.arange(1, len(selected_locations) + 1)

        # Plot
        plt.figure(figsize=(10, 6))

        # Plot confidence interval as shaded region
        if plot_ci:
            plt.fill_between(talweg_positions, gpe_lower_selected, gpe_upper_selected,
                             color='gray', alpha=0.3, label='GPE Confidence Interval')

        # Plot observed data with error bars
        if plot_error and error_selected is not None:
            plt.errorbar(talweg_positions, observed_selected, yerr=error_selected, fmt='o',
                         color='black', label='Observed Data', capsize=4, zorder=3)

        # Plot observed values as a line in exact input order
        plt.plot(talweg_positions, observed_selected, '-o', color='black',
                 label='Observed Values', markersize=6, zorder=4)

        # Plot surrogate model outputs as a line in exact input order
        plt.plot(talweg_positions, surrogate_selected, '-o', color='blue',
                 label=f'Surrogate Model (RMSE: {surrogate_rmse:.4f}, R²: {surrogate_r2:.4f})',
                 markersize=6, zorder=3)

        # Plot complex model outputs as a line in exact input order
        plt.plot(talweg_positions, complex_selected, '-o', color='green',
                 label=f'Complex Model (RMSE: {complex_rmse:.4f}, R²: {complex_r2:.4f})',
                 markersize=6, zorder=3)

        # Labels, title, and legend
        plt.xlabel('Talweg')
        plt.ylabel('Values')
        plt.title('Model Outputs vs Observed Data (Talweg)')
        plt.xticks(talweg_positions,
                   labels=[str(loc) for loc in selected_locations])  # Keep original locations as labels
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, linestyle='--', color='gray', alpha=0.7)
        plt.tight_layout()
        plt.show()
    def plot_correlation(self,
            sm_out,
            valid_eval,
            output_names,
            label_list=None,  # Make label_list optional
            n_loc_=1,
            fig_title=''
    ):
        """Function plots the scatter plots for the outputs, comparing the validation output (x-axis) and the
        surrogate outputs (y-axis).

        Args:
            sm_out (np.array): Surrogate outputs, of size [mc_size, n_obs].
            valid_eval (np.array): Array [mc_size, n_obs], with the validation output.
            output_names (list): Names of the different output types.
            label_list (list, optional): Contains the R2 information to add to each subplot label.
            n_loc_ (int, optional): Number of locations where each output name is read. Defaults to 1.
            fig_title (str, optional): Title of the plot. Defaults to ''.
        """
        colormap = plt.cm.tab20
        color_indices = np.linspace(0, 1, n_loc_)
        colors_obs = [colormap(color_index) for color_index in color_indices]

        # Create subplots
        fig, axs = plt.subplots(1, len(output_names), figsize=(10, 5))

        # Ensure axs is always iterable, even if there is only one subplot
        if len(output_names) == 1:
            axs = [axs]

        c = 0
        for o, ot in enumerate(output_names):
            for i in range(n_loc_):
                axs[o].scatter(valid_eval[:, i + c], sm_out[:, i + c], color=colors_obs[i], label=f'{i + 1}')

            # Set plot limits and add the identity line
            mn = np.min(np.hstack((valid_eval[:, c:n_loc_ + c], sm_out[:, c:n_loc_ + c])))
            mx = np.max(np.hstack((valid_eval[:, c:n_loc_ + c], sm_out[:, c:n_loc_ + c])))
            axs[o].plot([mn, mx], [mn, mx], color='black', linestyle='--')

            # Set titles and labels
            title = f'{ot}'
            if label_list is not None:
                title += f' - R2: {label_list[o]}'
            axs[o].set_title(title, loc='left')
            axs[o].set_xlabel('Full complexity model outputs')

            if o == 0:
                axs[o].set_ylabel('Simulator outputs')

            c += n_loc_

        # Set the overall title and legend
        fig.suptitle(fig_title)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc="center right", ncol=1)
        plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.2, hspace=0.5)
        plt.show()

    def plot_validation_results(self,obs, surrogate_outputs, complex_model_outputs, gpe_lower_ci=None, gpe_upper_ci=None,
                                measurement_error=None, plot_ci=True, N=5):
        """
        Plots N randomly selected realizations of surrogate and complex model outputs versus observed values.
        Each realization is plotted in a separate subplot with confidence intervals.
        """
        obs = obs.flatten()
        if surrogate_outputs.ndim == 1:
            surrogate_outputs = surrogate_outputs.reshape(1, -1)
        if complex_model_outputs.ndim == 1:
            complex_model_outputs = complex_model_outputs.reshape(1, -1)

        num_realizations_surrogate, num_points = surrogate_outputs.shape
        num_realizations_complex, _ = complex_model_outputs.shape

        if measurement_error is not None:
            measurement_error = measurement_error.flatten()
        obs_error = 2 * measurement_error if measurement_error is not None else None

        # Randomly select N realizations to plot
        N = min(N, num_realizations_surrogate, num_realizations_complex)
        selected_indices = random.sample(range(num_realizations_surrogate), N)

        fig, axes = plt.subplots(N, 1, figsize=(12, 3 * N), sharex=True)
        if N == 1:
            axes = [axes]  # Ensure axes is iterable

        for idx, realization in enumerate(selected_indices):
            ax = axes[idx]
            locations = np.arange(1, num_points + 1)

            # Extract CI for the specific realization
            gpe_lower = gpe_lower_ci[realization, :] if gpe_lower_ci is not None else None
            gpe_upper = gpe_upper_ci[realization, :] if gpe_upper_ci is not None else None

            # Compute MSE and R² for the current realization
            surrogate_mse = mean_squared_error(obs, surrogate_outputs[realization, :])
            complex_mse = mean_squared_error(obs, complex_model_outputs[realization, :])
            surrogate_r2 = r2_score(obs, surrogate_outputs[realization, :])
            complex_r2 = r2_score(obs, complex_model_outputs[realization, :])

            # Plot observed data with measurement error bars
            ax.errorbar(locations, obs, yerr=obs_error, fmt='o', color='black', label='Observed Data', capsize=4,
                        zorder=3)

            # Plot confidence interval for the selected realization
            if plot_ci and gpe_lower is not None and gpe_upper is not None:
                ax.fill_between(locations, gpe_lower, gpe_upper, color='gray', alpha=0.3,
                                label='GPE Confidence Interval')

            # Plot surrogate and complex model outputs
            ax.plot(locations, surrogate_outputs[realization, :], color='blue', alpha=0.8, linewidth=1.5,
                    label='Surrogate Model')
            ax.plot(locations, complex_model_outputs[realization, :], color='green', alpha=0.8, linewidth=1.5,
                    label='Complex Model')

            # Labels and legend
            ax.set_ylabel('Values')
            ax.set_title(
                f'Realization {realization + 1}: MSE(SM)={surrogate_mse:.4f}, R²(SM)={surrogate_r2:.4f}, MSE(CM)={complex_mse:.4f}, R²(CM)={complex_r2:.4f}')
            ax.legend()
            ax.grid(True, linestyle='--', color='gray', alpha=0.7)

        axes[-1].set_xlabel('Location')
        plt.tight_layout()
        plt.show()

    def plot_validation_locations(self, surrogate_outputs, complex_model_outputs, gpe_lower_ci, gpe_upper_ci,
                                  selected_locations, plot_residuals=False):
        """
        Plots surrogate model confidence intervals and complex model realizations for multiple locations.
        Adds surrogate model realization values as points along with hatch bars for the confidence intervals.
        Computes and displays metrics between the complex model and surrogate model, including residuals in a separate plot.

        Parameters:
        -----------
        surrogate_outputs : numpy.ndarray
            2D array of surrogate model outputs (rows = realizations, columns = locations).
        complex_model_outputs : numpy.ndarray
            2D array of complex model outputs (rows = realizations, columns = locations).
        gpe_lower_ci : numpy.ndarray
            2D array of lower confidence bounds from the surrogate model.
        gpe_upper_ci : numpy.ndarray
            2D array of upper confidence bounds from the surrogate model.
        selected_locations : list of int
            List of location indices to visualize.
        plot_residuals : bool
            If True, plots residuals; if False, plots comparison (CM vs SM).

        Returns:
        --------
        None
        """

        def compute_metrics(cm_values, sm_values):
            """Compute various metrics between complex model and surrogate model."""
            mse = mean_squared_error(cm_values, sm_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(cm_values, sm_values)
            correlation = np.corrcoef(cm_values, sm_values)[0, 1]
            return mse, rmse, mae, correlation

        num_locations = len(selected_locations)
        fig, axes = plt.subplots(1, num_locations, figsize=(5 * num_locations, 5), sharey=False)
        if num_locations == 1:
            axes = [axes]

        # Create a separate figure for residuals (if requested)
        if plot_residuals:
            fig_residuals, axes_residuals = plt.subplots(1, num_locations, figsize=(5 * num_locations, 5), sharey=False)
            if num_locations == 1:
                axes_residuals = [axes_residuals]
        else:
            axes_residuals = [None] * num_locations


        for ax, selected_location, ax_residual in zip(axes, selected_locations, axes_residuals):
            # Extract values at the selected location
            sm_lower_ci = gpe_lower_ci[:, selected_location]  # CI lower bound per realization
            sm_upper_ci = gpe_upper_ci[:, selected_location]  # CI upper bound per realization
            sm_values = surrogate_outputs[:, selected_location]  # SM realizations
            cm_values = complex_model_outputs[:, selected_location]  # CM realizations

            # Compute residuals: the difference between complex model and surrogate model outputs
            residuals = cm_values - sm_values

            if plot_residuals:
                # Plot residuals in a separate plot
                ax_residual.scatter(range(len(residuals)), residuals, color='red', alpha=0.6, label='Residuals')
                ax_residual.set_title(f"Residuals - Location {selected_location}")
                ax_residual.set_xlabel("Realization Index")
                ax_residual.set_ylabel("Residual Value")
                ax_residual.grid(True, linestyle="--", alpha=0.6)
            else:
                # Plot individual realizations of the complex model
                ax.scatter(range(len(cm_values)), cm_values, color='green', alpha=0.6, label='CM Realizations')

                # Plot hatch bars for the surrogate model confidence intervals
                for i in range(len(sm_lower_ci)):
                    ax.bar(i, sm_upper_ci[i] - sm_lower_ci[i], bottom=sm_lower_ci[i], width=0.8, color='gray',
                           alpha=0.5, hatch='/')

                    # Plot the surrogate model realization values as points
                    ax.scatter(i, sm_values[i], color='blue', marker='x')

                # Compute metrics between complex model and surrogate model
                mse, rmse, mae, correlation = compute_metrics(cm_values, sm_values)

                # Formatting for comparison plot
                ax.set_title(f"Location {selected_location}")
                ax.set_xlabel("Realization Index")
                ax.set_ylabel("Output Value")
                ax.grid(True, linestyle="--", alpha=0.6)

                # Prepare metrics for the legend and text box
                metrics_text = (f"MSE: {mse:.2e}\n"
                                f"RMSE: {rmse:.2e}\n"
                                f"MAE: {mae:.2e}\n"
                                f"Correlation: {correlation:.2f}")

                # Plot legend for comparison plot
                ax.legend(loc='upper right', fontsize=4, bbox_to_anchor=(0.5, 0.5))

                # Display the metrics in the top-right corner as a text box
                plt.figtext(0.95, 0.95, metrics_text, horizontalalignment='right', verticalalignment='top', fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

        # Adjust layout for both figures
        plt.tight_layout()
        plt.show()

        if plot_residuals:
            # Show residuals plots
            fig_residuals.tight_layout()
            plt.show()

    def plot_realizations(self,surrogate_outputs, complex_model_outputs, gpe_lower_ci, gpe_upper_ci):
        """
        Plots selected realizations comparing the complex model and surrogate model outputs.
        Each realization is displayed in a separate subplot with confidence intervals.

        Parameters:
        -----------
        surrogate_outputs : numpy.ndarray
            2D array of surrogate model outputs (rows = realizations, columns = locations).
        complex_model_outputs : numpy.ndarray
            2D array of complex model outputs (rows = realizations, columns = locations).
        gpe_lower_ci : numpy.ndarray
            2D array of lower confidence bounds from the surrogate model.
        gpe_upper_ci : numpy.ndarray
            2D array of upper confidence bounds from the surrogate model.

        Returns:
        --------
        None
        """
        num_realizations = surrogate_outputs.shape[0]

        # Ask user which realizations to plot
        selected_realizations = input(
            f"Enter the realizations to plot (0 to {num_realizations - 1}, comma-separated): ")
        selected_realizations = list(map(int, selected_realizations.split(',')))

        num_plots = len(selected_realizations)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        for ax, realization in zip(axes, selected_realizations):
            cm_values = complex_model_outputs[realization, :]
            sm_values = surrogate_outputs[realization, :]
            lower_ci = gpe_lower_ci[realization, :]
            upper_ci = gpe_upper_ci[realization, :]
            locations = np.arange(len(cm_values))

            # Compute residuals
            residuals = cm_values - sm_values
            mse = mean_squared_error(cm_values, sm_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(cm_values, sm_values)
            correlation = np.corrcoef(cm_values, sm_values)[0, 1]

            # Plot data
            ax.plot(locations, cm_values, 'g-o', label='Complex Model')
            ax.plot(locations, sm_values, 'b-x', label='Surrogate Model')
            ax.fill_between(locations, lower_ci, upper_ci, color='gray', alpha=0.3, hatch='//',
                            label='Confidence Interval')

            # Labels and title
            ax.set_title(f'Realization {realization}')
            ax.set_ylabel('Output Value')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Metrics
            metrics_text = (f"MSE: {mse:.2e}\nRMSE: {rmse:.2e}\nMAE: {mae:.2e}\nCorrelation: {correlation:.2f}")
            ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

            ax.legend()

        axes[-1].set_xlabel('Locations')
        plt.tight_layout()
        plt.show()

# def plot_bme_concentration_last_iterations(param_values, param_ranges, bme_values, param_indices=(0, 1), grid_size=100,
#                                            last_iterations=10, interval=1):
#     """
#     Plots the BME concentration for the last specified iterations and adds 2D contour plots to show the evolution.
#
#     Args:
#         param_values: np.array
#             2D array where each row corresponds to parameter values for each iteration.
#         param_ranges: list of lists
#             List of [min, max] values for each parameter.
#         bme_values: list of float
#             List of BME values, one for each iteration.
#         param_indices: tuple of int
#             Indices of the two parameters to plot.
#         grid_size: int
#             Size of the grid for the contour plots.
#         last_iterations: int
#             Number of last iterations to consider for the plot.
#         interval: int
#             Interval at which to plot the evolution of BME values.
#     """
#     num_iterations = len(bme_values) - 1  # -1 because bme_values has iterations + 1 values
#
#     if num_iterations < last_iterations:
#         raise ValueError("Number of iterations is less than the last iterations specified")
#
#     # Extract the last iterations + 1 BME values and corresponding parameters
#     bme_values = bme_values[-(last_iterations + 1):]
#     param_values = param_values[-(last_iterations + 1):, :]
#
#     # Extract ranges for the selected parameters
#     x_range = param_ranges[param_indices[0]]
#     y_range = param_ranges[param_indices[1]]
#
#     x = np.linspace(x_range[0], x_range[1], grid_size)
#     y = np.linspace(y_range[0], y_range[1], grid_size)
#     X, Y = np.meshgrid(x, y)
#
#     # Calculate number of subplots needed
#     num_plots = last_iterations // interval
#     if last_iterations % interval != 0:
#         num_plots += 1
#
#     # Create subplots for the evolution of BME values
#     fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, num_plots * 4))
#     if num_plots == 1:
#         axes = [axes]  # Ensure axes is iterable
#
#     for i in range(0, last_iterations, interval):
#         ax = axes[i // interval]
#
#         # Prepare data for interpolation
#         current_points = param_values[:i + 2, param_indices]
#         current_values = bme_values[:i + 2]
#
#         # Check if there are enough points for interpolation
#         if len(current_points) >= 4:
#             try:
#                 # Interpolate BME values onto the grid
#                 Z = griddata(current_points, current_values, (X, Y), method='cubic')
#             except Exception as e:
#                 # If cubic interpolation fails, use linear interpolation
#                 Z = griddata(current_points, current_values, (X, Y), method='linear')
#                 print(f"Warning: {e}. Using linear interpolation instead.")
#
#             # Set Z-axis limits based on the min and max of BME values
#             Z_min = min(current_values)
#             Z_max = max(current_values)
#             Z = np.clip(Z, Z_min, Z_max)
#
#             # 2D Contour Plot
#             contour = ax.contourf(X, Y, Z, cmap='viridis', levels=np.linspace(Z_min, Z_max, 100), alpha=0.8)
#             ax.set_title(f'BME Values (Iteration {num_iterations - last_iterations + i + 1})', fontsize=12)
#             ax.set_xlabel(r'$\omega_{}$'.format(param_indices[0] + 1), fontsize=10)
#             ax.set_ylabel(r'$\omega_{}$'.format(param_indices[1] + 1), fontsize=10)
#             ax.set_aspect('equal', 'box')
#
#             # Optional: Plot high BME regions as scatter points
#             high_bme_indices = np.where(Z > np.percentile(current_values, 95))  # Example threshold for high BME
#             ax.scatter(X[high_bme_indices], Y[high_bme_indices], color='red', s=10, label='High BME Regions')
#
#             # Add a color bar for the contour plot
#             cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)
#             cbar.set_label('BME Value', fontsize=10)
#             ax.legend(fontsize=8)
#         else:
#             ax.set_title(f'Not enough points (Iteration {num_iterations - last_iterations + i + 1})', fontsize=12)
#             ax.text(0.5, 0.5, 'Insufficient points for interpolation', ha='center', va='center', fontsize=12)
#
#     plt.tight_layout()
#     plt.show()

# def plot_bme_surface_3d(num_iterations, param_ranges, bme_values, param_indices=(0, 1), grid_size=100):
#     """
#     Plots the BME surface for the last 5 selected iterations and adds a 2D contour plot to show high BME regions.
#
#     Args:
#         num_iterations: int
#             Total number of iterations (length of bme_values).
#         param_ranges: list of lists
#             List of [min, max] values for each parameter.
#         bme_values: list of float
#             List of BME values, one for each iteration.
#         param_indices: tuple of int
#             Indices of the two parameters to plot.
#         grid_size: int
#             Size of the grid for the surface and contour plots.
#     """
#     # Extract the last 5 BME values and corresponding parameters
#     last_iterations = 10
#     if num_iterations < last_iterations:
#         raise ValueError("Number of iterations is less than the last iterations specified")
#
#     bme_values = bme_values[-last_iterations:]
#
#     # Extract ranges for the selected parameters
#     x_range = param_ranges[param_indices[0]]
#     y_range = param_ranges[param_indices[1]]
#
#     x = np.linspace(x_range[0], x_range[1], grid_size)
#     y = np.linspace(y_range[0], y_range[1], grid_size)
#     X, Y = np.meshgrid(x, y)
#
#     # Prepare data for interpolation
#     points = []
#     values = []
#
#     for i in range(last_iterations):
#         # Generate or retrieve parameter values for this iteration (use your actual parameter values)
#         param_x = np.random.uniform(x_range[0], x_range[1])
#         param_y = np.random.uniform(y_range[0], y_range[1])
#
#         points.append((param_x, param_y))
#         values.append(bme_values[i])
#
#     points = np.array(points)
#     values = np.array(values)
#
#     # Ensure points and values have the same length
#     if len(points) != len(values):
#         raise ValueError("Mismatch between number of points and BME values")
#
#     # Interpolate BME values onto the grid
#     Z = griddata(points, values, (X, Y), method='cubic')
#
#     # Set Z-axis limits based on the min and max of BME values
#     Z_min = min(values)
#     Z_max = max(values)
#     Z = np.clip(Z, Z_min, Z_max)
#
#     # Plot the surface and contour
#     fig = plt.figure(figsize=(16, 8))
#
#     # 3D Plot
#     ax1 = fig.add_subplot(121, projection='3d')
#     surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
#     ax1.set_title('BME Surface Plot (Last 5 Iterations)', fontsize=20)
#     ax1.set_xlabel(r'$\omega_{}$'.format(param_indices[0] + 1), fontsize=18)
#     ax1.set_ylabel(r'$\omega_{}$'.format(param_indices[1] + 1), fontsize=18)
#     ax1.set_zlabel('BME', fontsize=18)
#     ax1.set_zlim(Z_min, Z_max)
#     ax1.view_init(elev=30, azim=225)  # Adjust view angle
#
#     # Add a color bar
#     cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
#     cbar.set_label('BME Value', fontsize=14)
#
#     # 2D Contour Plot
#     ax2 = fig.add_subplot(122, aspect='equal')
#     contour = ax2.contourf(X, Y, Z, cmap='viridis', levels=np.linspace(Z_min, Z_max, 100), alpha=0.8)
#     ax2.set_title('Contour Plot of BME Values', fontsize=20)
#     ax2.set_xlabel(r'$\omega_{}$'.format(param_indices[0] + 1), fontsize=18)
#     ax2.set_ylabel(r'$\omega_{}$'.format(param_indices[1] + 1), fontsize=18)
#
#     # Optional: Plot high BME regions as scatter points
#     high_bme_indices = np.where(Z > np.percentile(values, 95))  # Example threshold for high BME
#     ax2.scatter(X[high_bme_indices], Y[high_bme_indices], color='red', s=10, label='High BME Regions')
#
#     # Add a color bar for the contour plot
#     cbar2 = fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
#     cbar2.set_label('BME Value', fontsize=14)
#
#     ax2.legend(fontsize=12)
#
#     plt.tight_layout()
#     plt.show()
#
