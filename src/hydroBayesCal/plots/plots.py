"""
Code for plotting results in the context of Bayesian Calibration with GPE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, linregress
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path


class BayesianPlotter:
    def __init__(
            self,
            results_folder_path='',
            plots_subfolder='plots'
    ):
        """
        Constructor of BayesianPlotter class, which is used to create and save various plots related to Bayesian calibration.

        Parameters
        ----------
        results_folder_path : str
            Path to the folder where results (including plots) will be saved. Usually auto-saved-results
        plots_subfolder : str, optional
            Name of the subfolder within the results folder where plots will be saved. Default folder name is 'plots'.

        Attributes
        ----------
        save_folder : pathlib.Path
            A Path object representing the directory where plots will be saved.
        """
        self.save_folder = Path(results_folder_path) / plots_subfolder

    def plot_posterior(
            self,
            posterior_vector,
            parameter_name
    ):
        """
        Plots the posterior distribution of a Bayesian parameter.

        Parameters
        ----------
        posterior_vector : array
            A vector containing samples from the posterior distribution of the selected parameter.
        parameter_name : str
            The name of the parameter whose posterior distribution is being plotted.
            This will be used as the label for the x-axis.

        Returns
        -------
        None
            The function creates a plot of the posterior distribution and is saved
            as a .png file in the /plots folder.
        """
        save_folder = self.save_folder

        colors = ['dimgray']
        bins = 30

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot posterior distribution
        counts, bins, patches = ax.hist(posterior_vector, bins=bins, density=True, alpha=0.5, color=colors[0],
                                        edgecolor='black')

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
        plt.savefig(save_folder / f'posterior_{parameter_name}.png')
        plt.close()

    def plot_posterior_updates(
            self,
            posterior_arrays,
            parameter_names,
            prior,
            iterations_to_plot=None,
            bins=30,
            density=True
    ):
        """
        Plots the prior distributions and posterior updates for given parameters.

        Parameters
        ----------
            posterior_arrays: list of arrays
                List of 2D arrays with posterior samples for each update.
            parameter_names: list of str
                List of parameter names corresponding to the columns of the arrays.
            prior: array
                2D array with prior samples.
            iterations_to_plot: list of int or None
                List of iteration indices to plot. If None, only prior distributions are plotted.
            bins: int
                Number of bins to use for the histograms. Default 30.
            density: bool
                Whether to normalize the histograms to form a probability density.
         Returns
        -------
            None
                The function creates plots of the prior and posterior distribution functions  and are saved
                as  .png files in the /plots folder.
        """
        save_folder = self.save_folder

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
            counts_prior, _ = np.histogram(prior[:, i], bins=bins, density=density)
            y_max_prior[i] = max(counts_prior)
            for row in range(len(posterior_arrays)):
                if posterior_arrays[row] is not None:  # Check if the current array is not None
                    counts_posterior, _ = np.histogram(posterior_arrays[row][:, i], bins=bins, density=density)
                    y_max_posterior[i] = max(y_max_posterior[i], max(counts_posterior))

        # Helper function to set grid and border style
        def set_plot_style():
            plt.grid(True, linestyle='--', color='lightgrey', alpha=0.7)
            plt.tick_params(axis='both', which='both', direction='in', labelsize=12)
            plt.gca().spines['top'].set_linewidth(0.8)
            plt.gca().spines['right'].set_linewidth(0.8)
            plt.gca().spines['bottom'].set_linewidth(0.8)
            plt.gca().spines['left'].set_linewidth(0.8)

        # Plot prior distributions
        for i in range(parameter_num):
            plt.figure(figsize=(4, 8))
            plt.hist(prior[:, i], bins=bins, density=density, alpha=0.5, color=colors[0], label='Prior',
                     edgecolor='black', linewidth=0.8)
            mean_prior = np.mean(prior[:, i])
            plt.axvline(mean_prior, color='blue', linestyle='dashed', linewidth=1, label='Mean')
            plt.xlabel(parameter_names[i], fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(fontsize=8)
            set_plot_style()
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
                    plt.hist(posterior_vector[:, col], bins=bins, density=density, alpha=0.5, color=colors[1],
                             label='Posterior',
                             edgecolor='black', linewidth=0.8)
                    mean_posterior = np.mean(posterior_vector[:, col])
                    plt.axvline(mean_posterior, color='blue', linestyle='dashed', linewidth=1, label='Mean')
                    plt.xlabel(parameter_names[col], fontsize=12)
                    plt.ylabel('Density', fontsize=12)
                    plt.legend(fontsize=8)
                    set_plot_style()
                    plt.ylim(0, y_max_posterior[col])
                    plt.xlim(x_limits[col])
                    plt.tight_layout()
                    plt.savefig(
                        save_folder / f'posterior_distribution_param_{col + 1}_iteration_{iteration_idx + 1}.png')
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
                The function creates plots of BME or RE values over iterations  and are saved
                as  .png files in the /plots folder.
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
                         label='BME')
            axes[0].set_title('Bayesian Model Evidence (BME)', fontsize=16, weight='normal')
            axes[0].set_xlabel('Iteration', fontsize=14)
            axes[0].set_ylabel('BME', fontsize=14)
            axes[0].grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            axes[0].legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for BME
            slope_bme, intercept_bme, _, _, _ = linregress(iterations, bme_values)
            trend_bme = [slope_bme * x + intercept_bme for x in iterations]
            axes[0].plot(iterations, trend_bme, color='darkslategray', linestyle='--', linewidth=0.8,
                         label='Trend Line')
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
                         label='RE')
            axes[1].set_title('Relative Entropy (RE)', fontsize=16, weight='normal')
            axes[1].set_xlabel('Iteration', fontsize=14)
            axes[1].set_ylabel('RE', fontsize=14)
            axes[1].grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            axes[1].legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for RE
            slope_re, intercept_re, _, _, _ = linregress(iterations, re_values)
            trend_re = [slope_re * x + intercept_re for x in iterations]
            axes[1].plot(iterations, trend_re, color='dimgray', linestyle='--', linewidth=0.8, label='Trend Line')
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
                     label='BME')
            plt.title('Bayesian Model Evidence (BME)', fontsize=16, weight='normal')
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('BME', fontsize=14)
            plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            plt.legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for BME
            slope_bme, intercept_bme, _, _, _ = linregress(iterations, bme_values)
            trend_bme = [slope_bme * x + intercept_bme for x in iterations]
            plt.plot(iterations, trend_bme, color='darkslategray', linestyle='--', linewidth=0.8, label='Trend Line')
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
                     label='RE')
            plt.title('Relative Entropy (RE)', fontsize=16, weight='normal')
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('RE', fontsize=14)
            plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
            plt.legend(fontsize=12, loc='upper left')

            # Add a dashed tendency line for RE
            slope_re, intercept_re, _, _, _ = linregress(iterations, re_values)
            trend_re = [slope_re * x + intercept_re for x in iterations]
            plt.plot(iterations, trend_re, color='dimgray', linestyle='--', linewidth=0.8, label='Trend Line')
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
                The function creates scattered plot of the collocation points differentiating them between initial collocation
                points and BAL-selected are saved as  .png files in the /plots folder.
        """
        save_folder = self.save_folder

        # Ensure save_folder exists
        save_folder.mkdir(parents=True, exist_ok=True)

        if collocation_points.shape[1] == 1:
            collocation_points = np.hstack((collocation_points, collocation_points))

        fig, ax = plt.subplots()

        # Initial TP:
        ax.scatter(collocation_points[0:n_init_tp, 0], collocation_points[0:n_init_tp, 1], label='Initial TP',
                   c='black', s=100)
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

    def plot_bme_3d(
            self,
            param_sets,
            param_ranges,
            param_names,
            bme_values,
            param_indices=(0, 1),
            extra_param_index=4,
            grid_size=100,
            last_iterations=25
    ):
        """
        Plots the BME scatter for the last specified iterations, a 3d surface interpolated from the scatter BME values
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
            last_iterations: int
                Number of last iterations to consider for the plot.

        Returns
        -------
            None
                The function creates BME plots and are saved as  .png files in the /plots folder.
        """
        save_folder = self.save_folder
        if save_folder:
            save_folder = Path(save_folder)  # Ensure save_folder is a Path object
            save_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        num_iterations = len(bme_values) - 1  # -1 because bme_values has iterations + 1 values
        if num_iterations < last_iterations:
            raise ValueError("Number of iterations is less than the last iterations specified")

        # Extract the last iterations + 1 BME values and corresponding parameters
        bme_values = bme_values[-(last_iterations + 1):]
        param_values = param_sets[-(last_iterations + 1):, :]

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

        # Set Z-axis limits based on the min and max of BME values
        Z_min = min(values)
        Z_max = max(values)
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

        # 3D Scatter Plot
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        scatter = ax1.scatter(points[:, 0], points[:, 1], values, c=values, cmap='plasma', edgecolor='none', alpha=0.7)
        ax1.set_title(f'BME Scatter Plot ({last_iterations} Last Iterations)', fontsize=16, weight='normal')
        ax1.set_xlabel(f'{x_name}', fontsize=12)
        ax1.set_ylabel(f'{y_name}', fontsize=12)
        ax1.set_zlabel('BME', fontsize=12, rotation=90)  # Make BME axis title vertical
        ax1.set_zlim(Z_min, Z_max)
        ax1.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar1 = fig1.colorbar(scatter, orientation='vertical')
        cbar1.set_label('BME Value', fontsize=12)
        cbar1.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Set grid style for 3D plot
        set_grid_style(ax1)

        adjust_margins(fig1)
        fig1.tight_layout()
        fig1.savefig(save_folder / 'BME_scatter.png')  # Save with .png extension

        # 2D Contour Plot
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        contour = ax2.contourf(X, Y, Z, cmap='viridis', levels=np.linspace(Z_min, Z_max, 100), alpha=0.8)
        ax2.set_title(f'2D - BME Values ({last_iterations} Last Iterations)', fontsize=16, weight='normal')
        ax2.set_xlabel(f'{x_name}', fontsize=12)
        ax2.set_ylabel(f'{y_name}', fontsize=12)

        # Optional: Plot high BME regions as scatter points
        # high_bme_indices = np.where(Z > np.percentile(values, 95))  # Example threshold for high BME
        # ax2.scatter(X[high_bme_indices], Y[high_bme_indices], color='red', s=10, label='High BME Regions')

        ax2.legend(fontsize=10)

        # Add a color bar for the contour plot
        cbar2 = fig2.colorbar(contour, orientation='vertical')
        cbar2.set_label('BME Value', fontsize=12)
        cbar2.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Set grid style for 2D plot
        set_grid_style(ax2)

        adjust_margins(fig2)
        fig2.tight_layout()
        fig2.savefig(save_folder / '2D_BME_contour_values.png')  # Save with .png extension

        # 3D Surface Plot
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        surf = ax3.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
        ax3.set_title(f'BME Surface Plot ({last_iterations} Last Iterations)', fontsize=16, weight='normal')
        ax3.set_xlabel(f'{x_name}', fontsize=12)
        ax3.set_ylabel(f'{y_name}', fontsize=12)
        ax3.set_zlabel('BME', fontsize=12, rotation=90)  # Make BME axis title vertical
        ax3.set_zlim(Z_min, Z_max)
        ax3.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar3 = fig3.colorbar(surf, orientation='vertical')
        cbar3.set_label('BME', fontsize=12)
        cbar3.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

        # Set grid style for 3D plot
        set_grid_style(ax3)

        adjust_margins(fig3)
        fig3.tight_layout()
        fig3.savefig(save_folder / '3D_BME_surface_plot.png')  # Save with .png extension

        if extra_param_index is not None:
            # Prepare data for interpolation with extra parameter
            x_extra_range = param_ranges[extra_param_index]
            x_extra = np.linspace(x_extra_range[0], x_extra_range[1], grid_size)
            X_extra, Y_extra = np.meshgrid(x_extra, y)

            points_extra = param_values[:, [extra_param_index, param_indices[1]]]
            Z_extra = griddata(points_extra, values, (X_extra, Y_extra), method='cubic')
            Z_extra = np.clip(Z_extra, Z_min, Z_max)

            # 3D Scatter Plot with extra parameter
            fig4 = plt.figure(figsize=(8, 6))
            ax4 = fig4.add_subplot(111, projection='3d')
            scatter4 = ax4.scatter(param_values[:, param_indices[0]], param_values[:, param_indices[1]],
                                   param_values[:, extra_param_index], c=values, cmap='inferno', edgecolor='none',
                                   alpha=0.7)  # Changed colormap to 'plasma' for better visibility
            ax4.set_title(f'3D - Scatter (3 parameters) ({last_iterations} Last Iterations)', fontsize=16,
                          weight='normal')
            ax4.set_xlabel(f'{x_name}', fontsize=12)
            ax4.set_ylabel(f'{y_name}', fontsize=12)
            z_name = param_names[extra_param_index]
            ax4.set_zlabel(f'{z_name}', fontsize=12)
            ax4.view_init(elev=30, azim=225)  # Adjust view angle

            # Add a color bar
            cbar4 = fig4.colorbar(scatter4, orientation='vertical')
            cbar4.set_label('BME Value', fontsize=12)
            cbar4.ax.tick_params(labelsize=12)  # Set font size for color bar ticks

            # Set grid style for 3D plot
            set_grid_style(ax4)

            adjust_margins(fig4)
            fig4.tight_layout()
            fig4.savefig(save_folder / '3d_scatter_plot_with_extra_param.png')  # Save with .png extension

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
        plt.scatter(observed_values, surrogate_outputs, color='blue', label='Model 1 Outputs')
        plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Observed vs Observed')
        plt.xlabel('Observed Values')
        plt.ylabel('Model 1 Outputs')
        plt.title('Model 1 vs Observed Values')
        plt.legend(title=f'MSE: {mse_model1:.4f}\nR²: {r2_model1:.4f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Figure 2: Observed vs Model 2 Outputs
        plt.figure(figsize=(8, 6))
        plt.scatter(observed_values, complex_model_outputs, color='green', label='Model 2 Outputs')
        plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Observed vs Observed')
        plt.xlabel('Observed Values')
        plt.ylabel('Model 2 Outputs')
        plt.title('Model 2 vs Observed Values')
        plt.legend(title=f'MSE: {mse_model2:.4f}\nR²: {r2_model2:.4f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Figure 3: Model 1 vs Model 2 Outputs
        plt.figure(figsize=(8, 6))
        plt.scatter(surrogate_outputs, complex_model_outputs, color='purple', label='Model Outputs')
        plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Model 1 = Model 2')
        plt.xlabel('Model 1 Outputs')
        plt.ylabel('Model 2 Outputs')
        plt.title('Model 1 vs Model 2 Outputs')
        plt.legend(title=f'Correlation: {corr_model1_model2:.4f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_model_outputs_vs_locations(self, observed_values, surrogate_outputs, complex_model_outputs, gpe_lower_ci,
                                        gpe_upper_ci, measurement_error):
        """
        Plots the outputs (velocities) of two models versus locations in a single figure,
        including observed data, a confidence interval from GPE analysis, and measurement error.

        Parameters
        ----------
        self : object
            The instance of the class, if using class-based method.
        observed_values : numpy.ndarray
            1D array of observed values.
        surrogate_outputs : numpy.ndarray
            1D array of outputs from the surrogate model.
        complex_model_outputs : numpy.ndarray
            1D array of outputs from the complex model.
        gpe_lower_ci : numpy.ndarray
            1D array of lower confidence intervals from GPE analysis.
        gpe_upper_ci : numpy.ndarray
            1D array of upper confidence intervals from GPE analysis.
        measurement_error : numpy.ndarray
            1D array of measurement errors (standard deviations) for each observed value.

        Returns
        -------
        None
        """
        # Ensure all inputs are 1D arrays
        observed_values = observed_values.flatten()
        surrogate_outputs = surrogate_outputs.flatten()
        complex_model_outputs = complex_model_outputs.flatten()
        gpe_lower_ci = gpe_lower_ci.flatten()
        gpe_upper_ci = gpe_upper_ci.flatten()
        measurement_error = measurement_error.flatten()

        if not (len(observed_values) == len(surrogate_outputs) == len(complex_model_outputs) == len(
                gpe_lower_ci) == len(gpe_upper_ci) == len(measurement_error)):
            raise ValueError("All input arrays must have the same length.")

        locations = np.arange(1, len(observed_values) + 1)

        # Calculate the measurement confidence interval (2 standard deviations at each location)
        obs_lower_bound = observed_values - 2 * measurement_error
        obs_upper_bound = observed_values + 2 * measurement_error

        # Plot
        plt.figure(figsize=(12, 8))

        # Plot observed data with measurement error confidence interval
        plt.plot(locations, observed_values, marker='o', color='black', label='Observed Data')
        plt.fill_between(locations, obs_lower_bound, obs_upper_bound, color='red', alpha=0.2,
                         label='Measurement Error (±2 SD)')

        # Plot confidence interval from GPE analysis
        plt.fill_between(locations, gpe_lower_ci, gpe_upper_ci, color='gray', alpha=0.3, hatch='/',
                         label='GPE Confidence Interval')

        # Plot model outputs
        plt.plot(locations, surrogate_outputs, marker='o', color='blue', linestyle='--',
                 label='Surrogate Model Outputs')
        plt.plot(locations, complex_model_outputs, marker='s', color='green', linestyle='--',
                 label='Complex Model Outputs')

        # Add labels, title, and legend with smaller font size
        plt.xlabel('Location')
        plt.ylabel('Values')
        plt.title('Model Outputs vs Locations with Observed Data, Measurement Error, and Confidence Intervals')
        plt.legend(fontsize='small', loc='upper left')  # Set legend font size to small
        plt.grid(True)
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
