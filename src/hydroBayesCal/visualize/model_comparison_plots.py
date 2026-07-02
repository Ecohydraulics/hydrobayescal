"""
Comparisons of surrogate model, complex (full-complexity) model, and observed
values: scatter plots, location series, and realization plots.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelComparisonPlots:
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

        # Compute MSE and RMSE between Surrogate and Complex Model
        mse_sm_vs_cm = mean_squared_error(complex_selected, surrogate_selected)
        rmse_sm_vs_cm = np.sqrt(mse_sm_vs_cm)

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
                 label=(f'Surrogate Model\n'
                        f'RMSE (obs): {surrogate_rmse:.4f}, R²: {surrogate_r2:.4f}\n'
                        f'vs Complex: MSE: {mse_sm_vs_cm:.2e}, RMSE: {rmse_sm_vs_cm:.4f}'),
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

    def plot_validation_results(self, obs, surrogate_outputs, complex_model_outputs,
                                gpe_lower_ci=None, gpe_upper_ci=None,
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
            surrogate_vs_complex_mse = mean_squared_error(complex_model_outputs[realization, :],
                                                          surrogate_outputs[realization, :])

            # Plot observed data
            ax.errorbar(locations, obs, yerr=obs_error, fmt='o', color='black', capsize=4,
                        zorder=3)

            # Plot confidence interval
            if plot_ci and gpe_lower is not None and gpe_upper is not None:
                ax.fill_between(locations, gpe_lower, gpe_upper, color='gray', alpha=0.3)

            # Plot model outputs
            surrogate_line, = ax.plot(locations, surrogate_outputs[realization, :], color='blue', alpha=0.8,
                                      linewidth=1.5)
            complex_line, = ax.plot(locations, complex_model_outputs[realization, :], color='green', alpha=0.8,
                                    linewidth=1.5)

            # Title
            ax.set_title(f'Realization {realization + 1}')
            ax.set_ylabel('Values')
            ax.grid(True, linestyle='--', color='gray', alpha=0.7)

            # Only add legend once (on first subplot)
            if idx == 0:
                lines = [
                    surrogate_line,
                    complex_line,
                    ax.errorbar([], [], yerr=[], fmt='o', color='black')[0],  # dummy observed
                    plt.Line2D([0], [0], color='gray', lw=6, alpha=0.3)  # dummy CI
                ]
                labels = [
                    f'Surrogate Model\nMSE={surrogate_mse:.4f}, R²={surrogate_r2:.4f}',
                    f'Complex Model\nMSE={complex_mse:.4f}, R²={complex_r2:.4f}',
                    'Observed Data',
                    'GPE Confidence Interval\nMSE(SM vs CM)=' + f'{surrogate_vs_complex_mse:.4f}'
                ]
                ax.legend(lines, labels, loc='upper right', fontsize='small', frameon=True)

        axes[-1].set_xlabel('Location')
        plt.tight_layout()
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

