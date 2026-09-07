"""
Bayesian active learning (BAL) diagnostics: BME and RE evolution, BME
surfaces, and collocation-point plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
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
        plot_type='both',
        window=10,
        stabilization_iteration=None
    ):
        """
        Plot BME and/or RE over Bayesian active-learning iterations.

        Parameters
        ----------
        bayesian_dict : dict
            Dictionary containing 'BME'/'log_BME' and 'RE' values.
        num_bal_iterations : int
            Number of BAL iterations to plot.
        plot_type : str
            'BME', 'RE', or 'both'.
        window : int
            Window size for the moving mean and rolling standard deviation.
        stabilization_iteration : int or None
            Iteration from which the stabilization phase is highlighted.
            Set to None to omit the stabilization indication.
        """

        save_folder = self.save_folder
        save_folder.mkdir(parents=True, exist_ok=True)

        iterations = np.arange(num_bal_iterations)

        # ---------------------------------------------------------
        # BME
        # ---------------------------------------------------------
        if bayesian_dict.get('log_BME') is not None:

            bme_values = np.asarray([
                bayesian_dict['log_BME'][it]
                for it in iterations
            ])

            bme_label = r'$\log(\mathrm{BME})$'

        else:

            bme_values = np.asarray([
                bayesian_dict['BME'][it]
                for it in iterations
            ])

            bme_label = r'$\mathrm{BME}$'

        # ---------------------------------------------------------
        # RE
        # ---------------------------------------------------------
        re_values = np.asarray([
            bayesian_dict['RE'][it]
            for it in iterations
        ])

        # ---------------------------------------------------------
        # Internal plotting function
        # ---------------------------------------------------------
        def plot_metric(
                ax,
                iterations,
                values,
                ylabel
        ):

            values = np.asarray(values)

            # -----------------------------------------------------
            # Moving statistics
            # -----------------------------------------------------
            series = pd.Series(values)

            moving_mean = (
                series
                .rolling(
                    window=window,
                    min_periods=1
                )
                .mean()
                .to_numpy()
            )

            moving_std = (
                series
                .rolling(
                    window=window,
                    min_periods=2
                )
                .std()
                .to_numpy()
            )

            # -----------------------------------------------------
            # Raw BAL iteration values
            # -----------------------------------------------------
            ax.plot(
                iterations,
                values,
                'o-',
                markersize=3,
                linewidth=0.7,
                alpha=0.45,
                label='BAL iteration'
            )

            # -----------------------------------------------------
            # Moving mean
            # -----------------------------------------------------
            ax.plot(
                iterations,
                moving_mean,
                linewidth=2.0,
                label=f'{window}-iteration moving mean'
            )

            # -----------------------------------------------------
            # Rolling standard deviation
            # -----------------------------------------------------
            ax.fill_between(
                iterations,
                moving_mean - moving_std,
                moving_mean + moving_std,
                alpha=0.12,
                label='Rolling SD'
            )

            # -----------------------------------------------------
            # Stabilization phase
            # -----------------------------------------------------
            if stabilization_iteration is not None:

                ax.axvline(
                    stabilization_iteration,
                    linestyle='--',
                    linewidth=0.9
                )

                ymin, ymax = ax.get_ylim()

                ax.text(
                    stabilization_iteration + 1,
                    ymax - 0.08 * (ymax - ymin),
                    'Stabilization',
                    fontsize=9,
                    va='top'
                )

            # -----------------------------------------------------
            # Axis formatting
            # -----------------------------------------------------
            ax.set_xlabel('BAL iteration')
            ax.set_ylabel(ylabel)

            # Show every 10 iterations
            ax.set_xticks(
                np.arange(
                    0,
                    len(iterations),
                    10
                )
            )

            ax.grid(
                True,
                linestyle=':',
                linewidth=0.5,
                alpha=0.5
            )

            # Small legend
            ax.legend(
                loc='lower right',
                fontsize=8,
                frameon=False
            )

        # =========================================================
        # BOTH
        # =========================================================
        if plot_type == 'both':

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(11, 4),
                sharex=True
            )

            # BME
            plot_metric(
                axes[0],
                iterations,
                bme_values,
                bme_label
            )

            # RE
            plot_metric(
                axes[1],
                iterations,
                re_values,
                r'$\mathrm{RE}$'
            )

            # -----------------------------------------------------
            # Panel labels
            # -----------------------------------------------------
            axes[0].text(
                0.02,
                0.97,
                '(a)',
                transform=axes[0].transAxes,
                va='top',
                fontsize=11,
                fontweight='bold'
            )

            axes[1].text(
                0.02,
                0.97,
                '(b)',
                transform=axes[1].transAxes,
                va='top',
                fontsize=11,
                fontweight='bold'
            )

            fig.tight_layout()

            fig.savefig(
                save_folder / 'BME_RE_plots.svg',
                dpi=300,
                bbox_inches='tight'
            )

            plt.close(fig)

        # =========================================================
        # BME ONLY
        # =========================================================
        elif plot_type == 'BME':

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            plot_metric(
                ax,
                iterations,
                bme_values,
                bme_label
            )

            fig.tight_layout()

            fig.savefig(
                save_folder / 'BME_plot.svg',
                dpi=300,
                bbox_inches='tight'
            )

            plt.close(fig)

        # =========================================================
        # RE ONLY
        # =========================================================
        elif plot_type == 'RE':

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            plot_metric(
                ax,
                iterations,
                re_values,
                r'$\mathrm{RE}$'
            )

            fig.tight_layout()

            fig.savefig(
                save_folder / 'RE_plot.svg',
                dpi=300,
                bbox_inches='tight'
            )

            plt.close(fig)

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
    def _locate_collocation_points_file(folder, variable_name):
        """
        Build the expected collocation-points file path from the same
        variable_name convention used elsewhere in the class
        (collocation-points-<variable_name>.csv), falling back to a glob
        match if the exact name isn't found (e.g. slightly different casing
        or quantity ordering).
        """
        expected_path = os.path.join(folder, f"collocation-points-{variable_name}.csv")
        if os.path.isfile(expected_path):
            return expected_path
    
        candidates = sorted(glob.glob(os.path.join(folder, "collocation-points-*.csv")))
        if len(candidates) == 1:
            print(
                f"Note: '{os.path.basename(expected_path)}' not found; "
                f"using '{os.path.basename(candidates[0])}' instead."
            )
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Expected '{expected_path}' not found, and multiple "
                f"collocation-points-*.csv candidates exist in {folder}: "
                f"{[os.path.basename(c) for c in candidates]}. "
                f"Pass collocation_points_path explicitly to disambiguate."
            )
        raise FileNotFoundError(
            f"No collocation-points csv found in {folder} "
            f"(expected '{os.path.basename(expected_path)}')."
        )
    
    
    def plot_collocation_points(
        self,
        bayesian_dict,
        collocation_points=None,
        collocation_points_folder=None,
        collocation_points_path=None,
        variable_name=None,
        parameter_names=None,
        parameter_indices=None,
        parameter_units=None,
        param_values=None,
        num_bal_iterations=None,
        color_by="util_func",
        annotate_iterations=False,
        density_background=False,
        density_points="bal",
        density_cmap="viridis",
        density_alpha=0.55,
        density_grid_size=150,
        save_folder=None,
        plot_name="BAL_collocation_points_distribution",
        show=True,
    ):
        """
        Plot initial + BAL-selected collocation points in 2D or 3D, colored by
        the acquisition/utility function that selected each BAL point.
    
        Parameters
        ----------
        bayesian_dict : dict
            The BAL results dictionary (same one passed to plot_bme_re etc.).
            Must contain "N_tp" and "util_func"; "calibration_parameters" is
            used to name columns when collocation_points has no header.
        collocation_points : pandas.DataFrame or array-like, optional
            The collocation-point coordinates already in memory
            (n_points x n_params, initial design first then BAL-added points,
            same row order as they were evaluated). If given, this is used
            directly and no file is read at all - takes priority over
            collocation_points_path / collocation_points_folder. If it's a
            plain array (not a DataFrame), columns are named from
            bayesian_dict["calibration_parameters"].
        collocation_points_folder : str, optional
            Folder containing "collocation-points-<variable_name>.csv"
            (the same folder BAL_dictionary.pkl was read from). Only used
            if collocation_points is not given. Required (together with
            variable_name) unless collocation_points_path is given directly.
        collocation_points_path : str, optional
            Exact path to the collocation-points csv, bypassing the
            folder + variable_name lookup. Only used if collocation_points
            is not given.
        variable_name : str, optional
            The quantities_str used to name the file, e.g.
            "_".join(full_complexity_model.calibration_quantities). Only
            used if collocation_points and collocation_points_path are both
            omitted.
        parameter_names : list of str, optional
            Names of the 2 or 3 parameters to plot. Takes priority over
            parameter_indices. Must match column names in collocation_points
            (or bayesian_dict["calibration_parameters"] if unnamed).
        parameter_indices : list of int, optional
            Column indices of the 2 or 3 parameters to plot, used if
            parameter_names is not given.
        parameter_units : dict, optional
            Maps parameter name -> unit string, used for axis labels.
        param_values : list of [min, max], optional
            Parameter bounds (same order as bayesian_dict["calibration_parameters"])
            used to set axis limits.
        num_bal_iterations : int, optional
            Number of BAL iterations to show, counting from the first one.
            Defaults to all available (len(bayesian_dict["util_func"])).
        color_by : {"util_func", "iteration"}
            "util_func": color BAL points by which acquisition function chose
            them (legend groups by function name).
            "iteration": color BAL points by a continuous colormap over
            iteration order (shows the BAL sequence/progression instead).
        annotate_iterations : bool
            If True, label each BAL point with its iteration number.
        density_background : bool
            2D plots only. If True, draws a smooth Gaussian-KDE density
            heatmap behind the scatter points, showing where collocation
            points are concentrated. Ignored (with a warning) for 3D plots.
        density_points : {"bal", "initial", "all"}
            Which points the density estimate is computed from: just the
            BAL-selected points (default, highlights where BAL concentrated
            its search), just the initial design, or all points combined.
        density_cmap : str
            Colormap for the density heatmap.
        density_alpha : float
            Opacity of the density heatmap (0-1).
        density_grid_size : int
            Resolution (per axis) of the density evaluation grid.
        save_folder : str, optional
            If given, saves the figure to this folder as "<plot_name>.png".
        plot_name : str
            Base file name used when saving.
        show : bool
            Whether to call plt.show() at the end.
    
        Returns
        -------
        fig, ax : the matplotlib Figure and Axes objects.
        """
        # ------------------------------------------------------------------
        # 1) Get the collocation-point coordinates: prefer in-memory data,
        #    fall back to reading a csv from disk.
        # ------------------------------------------------------------------
        save_folder = self.save_folder
        save_folder.mkdir(parents=True, exist_ok=True)
        calibration_parameters = list(bayesian_dict.get("calibration_parameters", []))
    
        if collocation_points is not None:
            if isinstance(collocation_points, pd.DataFrame):
                points_df = collocation_points.copy()
            else:
                points_arr = np.asarray(collocation_points)
                columns = calibration_parameters if calibration_parameters else [
                    f"param_{i}" for i in range(points_arr.shape[1])
                ]
                points_df = pd.DataFrame(points_arr, columns=columns)
        else:
            if collocation_points_path is None:
                folder = collocation_points_folder or getattr(self, "results_folder_path", None)
                if folder is None:
                    raise ValueError(
                        "Provide collocation_points (array/DataFrame), "
                        "collocation_points_path, or collocation_points_folder "
                        "so the data can be found."
                    )
                var_name = variable_name or getattr(self, "variable_name", None)
                if var_name is None:
                    raise ValueError(
                        "Provide variable_name (e.g. \"_\".join(calibration_quantities)) "
                        "or collocation_points_path directly - could not find "
                        "self.variable_name on the plotter instance."
                    )
                collocation_points_path = _locate_collocation_points_file(folder, var_name)
    
            points_df = pd.read_csv(collocation_points_path)
    
        if calibration_parameters and list(points_df.columns) != calibration_parameters:
            # Columns present but possibly reordered/relabeled - only rename
            # positionally if counts match, otherwise trust the existing header.
            if len(points_df.columns) == len(calibration_parameters):
                points_df.columns = calibration_parameters
    
        # ------------------------------------------------------------------
        # 2) Work out which 2 or 3 parameters to plot.
        #
        #    calibration_parameters (from bayesian_dict) holds the raw column
        #    names used in the collocation-points data (e.g. "zone2"...).
        #    parameter_names, when given as a FULL list matching
        #    calibration_parameters in length, holds the pretty/LaTeX display
        #    labels in the same order - selected via parameter_indices, same
        #    convention as plot_posterior_updates. If parameter_names is
        #    instead given as a short list (len 2 or 3), it's treated as the
        #    direct column selection (previous behaviour).
        # ------------------------------------------------------------------
        cols_full = calibration_parameters if calibration_parameters else list(points_df.columns)
    
        if parameter_indices is not None:
            sel_names = [cols_full[i] for i in parameter_indices]
            if parameter_names is not None and len(parameter_names) == len(cols_full):
                sel_labels = [parameter_names[i] for i in parameter_indices]
            else:
                sel_labels = sel_names
        elif parameter_names is not None and len(parameter_names) in (2, 3):
            sel_names = list(parameter_names)
            sel_labels = sel_names
        elif parameter_names is not None:
            raise ValueError(
                f"parameter_names has {len(parameter_names)} entries. Either pass "
                f"exactly 2 or 3 names directly, or pass the full list together "
                f"with parameter_indices=[i, j, (k)] to select a subset."
            )
        else:
            sel_names = list(points_df.columns)[: min(3, points_df.shape[1])]
            sel_labels = sel_names
    
        if len(sel_names) not in (2, 3):
            raise ValueError(
                f"Select 2 or 3 parameters to plot, got {len(sel_names)}: {sel_names}"
            )
        missing = [n for n in sel_names if n not in points_df.columns]
        if missing:
            raise ValueError(
                f"Parameter(s) {missing} not found in collocation_points columns "
                f"{list(points_df.columns)}"
            )
    
        n_dims = len(sel_names)
    
        # ------------------------------------------------------------------
        # 3) Split into initial design vs. BAL-added points
        # ------------------------------------------------------------------
        n_tp = np.asarray(bayesian_dict["N_tp"])
        util_func = np.asarray(bayesian_dict["util_func"], dtype=object)
    
        num_initial = int(n_tp[0])
        total_bal_available = len(util_func)
    
        if num_bal_iterations is None:
            num_bal_iterations = total_bal_available
        num_bal_iterations = int(min(num_bal_iterations, total_bal_available))
    
        n_points_needed = num_initial + num_bal_iterations
        if n_points_needed > len(points_df):
            raise ValueError(
                f"Requested {num_initial} initial + {num_bal_iterations} BAL points "
                f"({n_points_needed} total) but collocation_points only has "
                f"{len(points_df)} rows."
            )
    
        initial_pts = points_df.iloc[:num_initial][sel_names].to_numpy()
        bal_pts = points_df.iloc[num_initial:n_points_needed][sel_names].to_numpy()
        bal_labels = util_func[:num_bal_iterations]
    
        # ------------------------------------------------------------------
        # 4) Plot
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(9, 7.5))
        ax = fig.add_subplot(111, projection="3d") if n_dims == 3 else fig.add_subplot(111)
    
        # Optional density heatmap background (2D only), drawn first so
        # scatter points sit on top of it.
        if density_background:
            if n_dims == 3:
                print(
                    "Note: density_background is only supported for 2D plots "
                    "(matplotlib 3D axes can't render a KDE heatmap); skipping it."
                )
            else:
                from scipy.stats import gaussian_kde
    
                if density_points == "initial":
                    density_src = initial_pts
                elif density_points == "all":
                    density_src = np.vstack([initial_pts, bal_pts]) if len(bal_pts) else initial_pts
                else:  # "bal" (default)
                    density_src = bal_pts if len(bal_pts) else initial_pts
    
                if len(density_src) >= 3:
                    x_min, x_max = points_df[sel_names[0]].min(), points_df[sel_names[0]].max()
                    y_min, y_max = points_df[sel_names[1]].min(), points_df[sel_names[1]].max()
                    # pad slightly so points near the edge aren't clipped
                    x_pad = 0.05 * (x_max - x_min or 1.0)
                    y_pad = 0.05 * (y_max - y_min or 1.0)
                    xx, yy = np.mgrid[
                        x_min - x_pad: x_max + x_pad: complex(density_grid_size),
                        y_min - y_pad: y_max + y_pad: complex(density_grid_size),
                    ]
                    try:
                        kde = gaussian_kde(density_src.T)
                        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                        mesh = ax.pcolormesh(
                            xx, yy, zz, cmap=density_cmap, alpha=density_alpha,
                            shading="gouraud", zorder=1,
                        )
                        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.85)
                        cbar.set_label(f"Point density ({density_points})")
                    except np.linalg.LinAlgError:
                        print(
                            "Note: density estimate failed (points may be "
                            "degenerate/collinear); skipping heatmap."
                        )
                else:
                    print(
                        f"Note: need at least 3 points to estimate density "
                        f"(got {len(density_src)} for density_points="
                        f"'{density_points}'); skipping heatmap."
                    )
    
        def _scatter(ax, pts, **kwargs):
            if n_dims == 3:
                return ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
            return ax.scatter(pts[:, 0], pts[:, 1], **kwargs)
    
        # initial points
        _scatter(
            ax, initial_pts,
            color="lightgray", edgecolor="k", marker="o", s=55, alpha=0.8,
            label=f"Initial points (n={num_initial})",
            zorder=2,
        )
    
        # BAL points
        if len(bal_pts):
            if color_by == "iteration":
                order = np.arange(1, len(bal_pts) + 1)
                sc = _scatter(
                    ax, bal_pts,
                    c=order, cmap="viridis", marker="^", s=65,
                    edgecolor="k", zorder=3,
                )
                cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.75)
                cbar.set_label("BAL iteration")
            else:  # color_by == "util_func"
                unique_funcs = list(dict.fromkeys(bal_labels))  # preserve order, unique
                cmap = plt.get_cmap("tab10" if len(unique_funcs) <= 10 else "tab20")
                for i, func_name in enumerate(unique_funcs):
                    mask = bal_labels == func_name
                    _scatter(
                        ax, bal_pts[mask],
                        color=cmap(i), edgecolor="k", marker="^", s=65, alpha=0.9,
                        label=f"BAL - {func_name} (n={mask.sum()})",
                        zorder=3,
                    )
    
            if annotate_iterations:
                for i, pt in enumerate(bal_pts, start=1):
                    if n_dims == 3:
                        ax.text(pt[0], pt[1], pt[2], str(i), fontsize=7, zorder=4)
                    else:
                        ax.text(pt[0], pt[1], str(i), fontsize=7, zorder=4)
    
        # ------------------------------------------------------------------
        # 5) Labels, limits, title
        # ------------------------------------------------------------------
        def _label(name, label):
            if parameter_units and label in parameter_units:
                return f"{label} [{parameter_units[label]}]"
            if parameter_units and name in parameter_units:
                return f"{label} [{parameter_units[name]}]"
            return label
    
        ax.set_xlabel(_label(sel_names[0], sel_labels[0]))
        ax.set_ylabel(_label(sel_names[1], sel_labels[1]))
        if n_dims == 3:
            ax.set_zlabel(_label(sel_names[2], sel_labels[2]))
    
        if param_values is not None and calibration_parameters:
            for axis_setter, name in zip(
                [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if n_dims == 3 else []),
                sel_names,
            ):
                if name in calibration_parameters:
                    idx = calibration_parameters.index(name)
                    axis_setter(param_values[idx][0], param_values[idx][1])
    
        ax.set_title(
            f"Collocation points: {num_initial} initial + {num_bal_iterations} BAL "
            f"({n_dims}D view)"
        )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
    
        # ------------------------------------------------------------------
        # 6) Save / show
        # ------------------------------------------------------------------
        if save_folder:
            fig.savefig(save_folder /f"{plot_name}.pdf", dpi=300)
            fig.savefig(save_folder / f"{plot_name}.png", dpi=300)
        plt.close(fig)
    
        return fig, ax

