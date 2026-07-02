"""
Prior and posterior distribution plots for Bayesian calibration with GPE.
"""

import math
import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


class PosteriorPlots:
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

    def plot_posterior_updates(
            self,
            posterior_arrays,
            parameter_names,
            prior,
            param_values=None,
            iterations_to_plot=None,
            bins=40,
            density=True,
            plot_prior=False,
            parameter_units=None,
            parameter_indices=None,
            best_estimate_value="posterior_MAP"  # "posterior_MAP" or "posterior_mean"
    ):

        save_folder = self.save_folder

        # Select indices of parameters to plot
        if parameter_indices is None:
            selected_indices = list(range(len(parameter_names)))
        else:
            selected_indices = parameter_indices

        parameter_num = len(selected_indices)

        if parameter_units is None:
            parameter_units = [''] * len(parameter_names)

        # Determine x-axis limits
        x_limits = np.zeros((parameter_num, 2))
        if param_values is None:
            for idx, param_idx in enumerate(selected_indices):
                x_limits[idx] = (prior[:, param_idx].min(), prior[:, param_idx].max())
        else:
            for idx, param_idx in enumerate(selected_indices):
                x_limits[idx] = param_values[param_idx]

        # Loop over iterations
        for plot_index, iteration_idx in enumerate(iterations_to_plot):

            num_rows = 3
            num_cols = math.ceil(parameter_num / num_rows)

            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(6.5 * num_cols, 5 * num_rows)
            )

            axes = axes.flatten()

            for col, param_idx in enumerate(selected_indices):

                ax = axes[col]
                posterior_vector = posterior_arrays[iteration_idx][:, param_idx]

                # --------------------------------------------------
                # Histogram values for normalization
                # --------------------------------------------------
                hist_values, _ = np.histogram(
                    posterior_vector,
                    bins=bins,
                    density=density
                )

                max_density = max(hist_values)

                if plot_prior:
                    prior_vector = prior[:, param_idx]
                    prior_hist_values, _ = np.histogram(
                        prior_vector,
                        bins=bins,
                        density=density
                    )
                    max_density = max(max_density, max(prior_hist_values))

                # --------------------------------------------------
                # Posterior histogram
                # --------------------------------------------------
                hist_values, bins_edges, patches = ax.hist(
                    posterior_vector,
                    bins=bins,
                    density=density,
                    alpha=0.75,
                    color='0.35',
                    edgecolor='black',
                    linewidth=0.8
                )

                if max_density > 0:
                    for patch in patches:
                        patch.set_height(patch.get_height() / max_density)

                # --------------------------------------------------
                # MAP directly from posterior vector
                # --------------------------------------------------
                mean_value = np.mean(posterior_vector)

                hist_counts, hist_bin_edges = np.histogram(
                    posterior_vector,
                    bins=bins,
                    density=False
                )

                map_bin_index = np.argmax(hist_counts)

                # Samples inside the most populated bin
                bin_left = hist_bin_edges[map_bin_index]
                bin_right = hist_bin_edges[map_bin_index + 1]

                samples_in_map_bin = posterior_vector[
                    (posterior_vector >= bin_left) &
                    (posterior_vector <= bin_right)
                    ]

                # Empirical MAP value from posterior samples
                # Using the mean of samples inside the most populated bin
                map_value = np.mean(samples_in_map_bin)

                if best_estimate_value == "posterior_mean":
                    value = mean_value
                elif best_estimate_value == "posterior_MAP":
                    value = map_value
                else:
                    raise ValueError(
                        "best_estimate_value must be either 'posterior_MAP' or 'posterior_mean'"
                    )

                ax.axvline(
                    value,
                    color='red',
                    linestyle='--',
                    linewidth=2
                )

                ax.text(
                    value,
                    1.05,
                    f'{value:.3f}',
                    color='black',
                    fontsize=28,
                    rotation=90,
                    verticalalignment='bottom',
                    horizontalalignment='right'
                )

                # --------------------------------------------------
                # Prior histogram optional
                # --------------------------------------------------
                if plot_prior:
                    prior_hist_values, prior_bins_edges, prior_patches = ax.hist(
                        prior_vector,
                        bins=bins,
                        density=density,
                        alpha=0.35,
                        color='0.75',
                        edgecolor='0.6',
                        linewidth=0.8
                    )

                    if max_density > 0:
                        for patch in prior_patches:
                            patch.set_height(patch.get_height() / max_density)

                # --------------------------------------------------
                # Axis labels
                # --------------------------------------------------
                unit = f' [{parameter_units[param_idx]}]' if parameter_units[param_idx] else ''
                ax.set_xlabel(f'{parameter_names[param_idx]}{unit}', fontsize=40)
                ax.set_ylabel('Posterior\n density [-]', fontsize=40)

                ax.tick_params(axis='both', which='major', labelsize=35)

                ax.set_xticks(
                    np.round(
                        np.linspace(
                            x_limits[col][0],
                            x_limits[col][1],
                            4
                        ),
                        3
                    )
                )

                ax.set_xlim(x_limits[col])
                ax.set_ylim(0, 1.2)

                ax.grid(
                    True,
                    which='both',
                    linestyle='--',
                    linewidth=0.7,
                    color='lightgrey'
                )

                ax.minorticks_on()

                ax.grid(
                    True,
                    which='minor',
                    linestyle=':',
                    linewidth=0.5,
                    color='grey'
                )

            # Remove unused axes
            for j in range(parameter_num, len(axes)):
                fig.delaxes(axes[j])

            # --------------------------------------------------
            # Global horizontal legend
            # --------------------------------------------------
            estimate_label = (
                'Posterior MAP'
                if best_estimate_value == "posterior_MAP"
                else 'Posterior mean'
            )

            legend_elements = [
                mpatches.Patch(
                    facecolor='0.75',
                    edgecolor='0.6',
                    alpha=0.35,
                    label='Prior'
                ),
                mpatches.Patch(
                    facecolor='0.35',
                    edgecolor='black',
                    alpha=0.75,
                    label='Posterior'
                ),
                Line2D(
                    [0],
                    [0],
                    color='red',
                    lw=2,
                    linestyle='--',
                    label=estimate_label
                )
            ]

            fig.legend(
                handles=legend_elements,
                loc='upper center',
                ncol=3,
                fontsize=28,
                frameon=False,
                bbox_to_anchor=(0.5, 1.02)
            )

            fig.tight_layout(rect=[0, 0.01, 1, 0.95])

            fig.savefig(
                save_folder / f'posterior_distributions_iteration_{iteration_idx + 1}.svg',
                format='svg',
                bbox_inches='tight',
                transparent=True,
            )

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

