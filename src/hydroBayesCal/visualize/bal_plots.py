"""
Bayesian active learning (BAL) diagnostics: BME and RE evolution, BME
surfaces, and collocation-point plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import linregress

from hydroBayesCal.visualize.axis_utils import adjust_margins, set_grid_style


class BALPlots:
    def _plot_series_with_trend(self, ax, iterations, values, ylabel, marker,
                                trend_color, trend_linewidth):
        """Plot one BME/RE series with its linear trend and shared axis style."""
        ax.plot(iterations, values, marker=marker, color='black', linestyle='-')
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)

        # Fit on the finite points only. A single inf or nan in the series makes
        # linregress return nan for every coefficient, which silently removes the
        # trend line from the figure without any indication that it failed.
        x_values = np.asarray(iterations, dtype=float)
        y_values = np.asarray(values, dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if finite.sum() >= 2:
            slope, intercept, _, _, _ = linregress(x_values[finite], y_values[finite])
            trend = slope * x_values + intercept
            ax.plot(iterations, trend, color=trend_color, linestyle='--',
                    linewidth=trend_linewidth)

        ax.set_xlim(iterations[0], iterations[-1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_offset_text().set_fontsize(20)
        self._set_latex_format(ax)

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
            The function creates plots of BME or RE values over iterations and saves them
            as .png files in the /plots folder.
        """
        save_folder = self.save_folder
        save_folder.mkdir(parents=True, exist_ok=True)

        iterations = list(range(num_bal_iterations))
        # Prefer log_BME where the result file has it. The linear BME underflows to
        # 0.0 and overflows to inf at realistic problem sizes, and a series spanning
        # many orders of magnitude is unreadable on a linear axis. Result files
        # written before log_BME existed fall back to BME and render as before.
        if bayesian_dict.get('log_BME') is not None:
            bme_values = [bayesian_dict['log_BME'][it] for it in iterations]
            bme_label = r'$\log$ BME'
        else:
            bme_values = [bayesian_dict['BME'][it] for it in iterations]
            bme_label = r'BME'
        re_values = [bayesian_dict['RE'][it] for it in iterations]

        if plot_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            self._plot_series_with_trend(axes[0], iterations, bme_values, bme_label,
                                         marker='.', trend_color='darkslategray', trend_linewidth=0.8)
            self._plot_series_with_trend(axes[1], iterations, re_values, r'RE',
                                         marker='.', trend_color='dimgray', trend_linewidth=0.5)

            plt.tight_layout()
            plt.savefig(save_folder / 'BME_RE_plots.svg', dpi=300)
            plt.close()

        elif plot_type == 'BME':
            fig, ax = plt.subplots(figsize=(8, 6))

            self._plot_series_with_trend(ax, iterations, bme_values, bme_label,
                                         marker='+', trend_color='darkslategray', trend_linewidth=0.5)

            plt.tight_layout()
            plt.savefig(save_folder / 'BME_plot.svg', dpi=300)
            plt.close()

        elif plot_type == 'RE':
            fig, ax = plt.subplots(figsize=(8, 6))

            self._plot_series_with_trend(ax, iterations, re_values, r'RE',
                                         marker='x', trend_color='dimgray', trend_linewidth=0.5)

            plt.tight_layout()
            plt.savefig(save_folder / 'RE_plot.svg', dpi=300)
            plt.close()

    def plot_combined_bal_3d(
            self,
            collocation_points,
            n_init_tp,
            bayesian_dict,
            param_indices=(0, 6, 10)
    ):
        """
        Plots the initial training points and points selected using different utility functions in 3D.

        Parameters
        ----------
            collocation_points: array [n_tp, n_param]
                Array with all collocation points, in order in which they were selected.
            n_init_tp: int
                Number of initial training points selected.
            bayesian_dict: dictionary
                With keys 'util_func', detailing which utility function was used in each iteration.
            param_indices: tuple of ints
                Three column indices of collocation_points to plot in 3D.

        Returns
        -------
            None
                The function creates a 3D scatter plot of the collocation points differentiating them between initial collocation
                points and BAL-selected points, saved as .png files in the /plots folder.
        """
        save_folder = self.save_folder
        save_folder.mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot initial training points
        for i in range(n_init_tp):
            ax.scatter(
                collocation_points[i, param_indices[0]],
                collocation_points[i, param_indices[1]],
                collocation_points[i, param_indices[2]],
                label='Initial TP' if i == 0 else "",
                c='black', s=100, edgecolor='white', marker='o'
            )

        selected_tp = collocation_points[n_init_tp:, :]

        # Plot points by utility function
        util_funcs = {
            'dkl': 'gold',
            'bme': 'blue',
            'ie': 'green',
            'global_mc': 'red'
        }

        for uf, color in util_funcs.items():
            ind = np.where(bayesian_dict['util_func'] == uf)
            ax.scatter(
                selected_tp[ind, param_indices[0]],
                selected_tp[ind, param_indices[1]],
                selected_tp[ind, param_indices[2]],
                label=uf.upper(),
                c=color, s=200, alpha=0.5
            )

        # Labels
        ax.set_xlabel(f'Param {param_indices[0]}', fontsize=12)
        ax.set_ylabel(f'Param {param_indices[1]}', fontsize=12)
        ax.set_zlabel(f'Param {param_indices[2]}', fontsize=12)

        # Legend
        ax.legend(loc='lower center', ncol=4, fontsize=10)

        # Save figure
        if save_folder:
            plt.savefig(save_folder / 'collocation_points_3d.png')
        plt.show()
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
        plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif', 'font.weight': 'normal',
                             'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
                             'axes.linewidth': 0.8})  # Reduced axes line width

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
        ax1.set_xlabel(f'{x_name}', fontsize=18)
        ax1.set_ylabel(f'{y_name}', fontsize=18)
        ax1.set_zlabel(f'{plot_criteria}', fontsize=18, rotation=90)  # Make BME axis title vertical
        ax1.set_zlim(Z_min - margin, Z_max + margin)
        ax1.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar1 = fig1.colorbar(scatter, orientation='vertical')
        cbar1.set_label(f'{plot_criteria} Value', fontsize=12)
        cbar1.ax.tick_params(labelsize=18)  # Set font size for color bar ticks

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
        ax3.set_title(f'{plot_criteria} Surface Plot (Iterations {start_iter} to {end_iter})', fontsize=18,
                      weight='normal')
        ax3.set_xlabel(f'{x_name}', fontsize=18)
        ax3.set_ylabel(f'{y_name}', fontsize=18)
        ax3.set_zlabel(f'{plot_criteria}', fontsize=18, rotation=90)  # Make BME axis title vertical
        ax3.set_zlim(Z_min - margin, Z_max + margin)
        ax3.view_init(elev=30, azim=225)  # Adjust view angle

        # Add a color bar
        cbar3 = fig3.colorbar(surf, orientation='vertical')
        cbar3.set_label(f'{plot_criteria}', fontsize=18)
        cbar3.ax.tick_params(labelsize=18)  # Set font size for color bar ticks

        # Set grid style for 3D plot
        set_grid_style(ax3)

        adjust_margins(fig3)
        fig3.tight_layout()
        fig3.savefig(save_folder / f'3D_{plot_criteria}_surface_plot.png')  #
        # Show the plot to the user
        plt.show()

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

