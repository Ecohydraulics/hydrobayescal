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
import pickle
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr
from scipy.stats import gaussian_kde



from hydroBayesCal.surrogate.posterior_analysis import marginal_optima, track_iteration


def _latex_safe(text):
    """Escape characters that break the LaTeX text renderer enabled in PlotterBase."""
    return str(text).replace('\\', ' ').replace('_', r'\_').replace('%', r'\%')


class PosteriorPlots:
    @staticmethod
    def _iteration_series(bayesian_dict, parameter_names, param_values=None):
        """Per-iteration marginal optima, either as stored or rebuilt from the posteriors.

        The diagnostics are written into ``bayesian_dict`` by the BAL drivers. Result
        files produced before those keys existed carry only ``posterior``, which is
        enough to reconstruct the whole series after the fact, so archived
        calibrations can be plotted without being re-run.

        Returns
        -------
        dict
            ``n_tp``, ``peak`` ``[n_iter, ndim]``, ``hdi`` ``[n_iter, ndim, 2]``,
            ``variance_reduction`` ``[n_iter, ndim]``, ``joint`` ``[n_iter, ndim]``,
            ``n_modes`` ``[n_iter]``, ``density_percentile`` ``[n_iter]``,
            ``max_abs_correlation`` ``[n_iter]`` and ``iterations``.
        """
        posteriors = bayesian_dict.get('posterior', [])
        ndim = len(parameter_names)
        stored = bayesian_dict.get('marginal_optima')
        rebuild = stored is None or all(entry is None for entry in stored)

        iterations, peaks, hdis, reductions, percentiles, correlations = [], [], [], [], [], []
        joints, mode_counts = [], []
        for it, posterior in enumerate(posteriors):
            if posterior is None or np.asarray(posterior).size == 0:
                continue
            if rebuild:
                summary = track_iteration(posterior, prior=bayesian_dict.get('prior'),
                                          parameter_names=parameter_names,
                                          prior_bounds=param_values)
                peak = summary['peak']
                hdi = summary['hdi']
                reduction = summary['variance_reduction']
                gap = summary['gap']
                joint = summary['joint']
                n_modes = summary['n_modes']
            else:
                peak = stored[it]
                if peak is None:
                    continue
                hdi = bayesian_dict['marginal_hdi'][it]
                reduction = bayesian_dict['variance_reduction'][it]
                gap = bayesian_dict['marginal_joint_gap'][it] or {}
                # Additive keys: a result file written before the joint optimum was
                # tracked has neither, and its panels simply carry no joint trace.
                stored_joint = (bayesian_dict.get('joint_optimum') or [None] * len(posteriors))[it]
                joint = (np.full(ndim, np.nan) if stored_joint is None
                         else np.asarray(stored_joint, dtype=float))
                n_modes = (bayesian_dict.get('posterior_modes')
                           or [np.nan] * len(posteriors))[it]
            iterations.append(it)
            peaks.append(np.asarray(peak, dtype=float))
            hdis.append(np.asarray(hdi, dtype=float).reshape(ndim, 2))
            reductions.append(np.asarray(reduction, dtype=float))
            percentiles.append(float(gap.get('density_percentile', np.nan)))
            correlations.append(float(gap.get('max_abs_correlation', np.nan)))
            joints.append(np.asarray(joint, dtype=float).ravel())
            mode_counts.append(float(n_modes) if n_modes is not None else np.nan)

        if not iterations:
            raise ValueError("No iteration with accepted posterior samples to plot.")

        n_tp = bayesian_dict.get('N_tp')
        x_values = (np.asarray(n_tp, dtype=float)[iterations] if n_tp is not None
                    else np.asarray(iterations, dtype=float))

        return {
            'iterations': np.asarray(iterations),
            'n_tp': x_values,
            'peak': np.asarray(peaks),
            'hdi': np.asarray(hdis),
            'variance_reduction': np.asarray(reductions),
            'joint': np.asarray(joints),
            'n_modes': np.asarray(mode_counts, dtype=float),
            'density_percentile': np.asarray(percentiles),
            'max_abs_correlation': np.asarray(correlations),
        }

    def plot_parameter_optimum_convergence(
            self,
            bayesian_dict,
            parameter_names,
            param_values=None,
            parameter_units=None,
            num_rows=3,
            show_hdi=True,
            plot_variance_reduction=True,
            file_stem='parameter_optimum_convergence',
    ):
        """Trace each calibration parameter's own optimum over the BAL iterations.

        One panel per parameter showing the peak of that parameter's posterior
        marginal against the number of training points, with its credible interval
        and the calibration range. A trace that is still drifting means the
        calibration has not converged for that parameter; a trace that sits on a
        prior bound means the parameter is pinned and the range or the parameter
        choice needs revisiting.

        Each panel also carries the *joint* posterior maximum, which is the calibration
        result proper. Where the two traces converge onto each other, reading the
        posterior per parameter and reading it jointly give the same answer; where they
        stay apart, the parameters are coupled and only the joint trace is a parameter
        set that can be run.

        The companion figure shows the posterior-to-prior variance reduction, i.e.
        how much the measurements actually constrain each parameter.
        """
        series = self._iteration_series(bayesian_dict, parameter_names, param_values)
        ndim = len(parameter_names)
        if parameter_units is None:
            parameter_units = [''] * ndim

        num_cols = math.ceil(ndim / num_rows)
        fig, axes = plt.subplots(num_rows, num_cols,
                                 figsize=(6.5 * num_cols, 4.5 * num_rows),
                                 squeeze=False)
        axes = axes.flatten()

        for i, name in enumerate(parameter_names):
            ax = axes[i]
            ax.plot(series['n_tp'], series['peak'][:, i], color='tab:blue',
                    marker='o', markersize=4, linewidth=2, label='marginal optimum')
            if series['joint'].size and np.any(np.isfinite(series['joint'][:, i])):
                ax.plot(series['n_tp'], series['joint'][:, i], color='tab:green',
                        marker='s', markersize=4, linewidth=2, linestyle='--',
                        label='joint optimum')
            if show_hdi:
                ax.fill_between(series['n_tp'], series['hdi'][:, i, 0],
                                series['hdi'][:, i, 1], color='tab:blue', alpha=0.18,
                                label='credible interval')
            if param_values is not None:
                low, high = param_values[i]
                ax.axhline(low, color='grey', linestyle=':', linewidth=1.5)
                ax.axhline(high, color='grey', linestyle=':', linewidth=1.5,
                           label='calibration range')
                margin = 0.05 * (high - low)
                pinned = ((series['peak'][:, i] - low < margin)
                          | (high - series['peak'][:, i] < margin))
                if np.any(pinned):
                    ax.plot(series['n_tp'][pinned], series['peak'][pinned, i],
                            linestyle='none', marker='x', markersize=11,
                            color='tab:red', label='pinned at bound')

            unit = f" [{_latex_safe(parameter_units[i])}]" if parameter_units[i] else ''
            ax.set_title(_latex_safe(name))
            ax.set_xlabel('number of training points')
            ax.set_ylabel(f"optimum{unit}")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc='best', fontsize='small')

        for ax in axes[ndim:]:
            ax.set_visible(False)

        plt.tight_layout()
        for extension in ('pdf', 'png'):
            fig.savefig(os.path.join(self.save_folder, f'{file_stem}.{extension}'),
                        bbox_inches='tight', dpi=300)
        plt.close(fig)

        if not plot_variance_reduction:
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, name in enumerate(parameter_names):
            ax.plot(series['n_tp'], 100.0 * series['variance_reduction'][:, i],
                    marker='o', markersize=4, linewidth=2, label=_latex_safe(name))
        ax.axhline(10.0, color='tab:red', linestyle='--', linewidth=1.5,
                   label='non-identifiable below')
        ax.set_xlabel('number of training points')
        ax.set_ylabel('posterior variance reduction (0-100)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize='small', ncol=2)
        plt.tight_layout()
        for extension in ('pdf', 'png'):
            fig.savefig(os.path.join(self.save_folder,
                                     f'{file_stem}_variance_reduction.{extension}'),
                        bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_marginal_vs_joint(
            self,
            bayesian_dict,
            parameter_names,
            param_values=None,
            alarm_percentile=10.0,
            file_stem='marginal_vs_joint',
    ):
        """Is the combination of the per-parameter optima a plausible parameter set?

        Each parameter has its own posterior marginal and therefore its own optimum,
        but stacking those optima into one vector assumes the parameters are
        independent. This plot tracks where that assembled vector sits in the joint
        posterior density, as a percentile of the accepted samples, together with the
        strongest parameter correlation driving the discrepancy.

        A trace that stays near the top means the per-parameter optima do form a
        valid calibrated parameter set. A trace that stays low is a quantitative
        equifinality warning: the parameters trade off against each other and only a
        jointly selected parameter vector is defensible.
        """
        series = self._iteration_series(bayesian_dict, parameter_names, param_values)

        fig, ax_density = plt.subplots(figsize=(12, 7))
        ax_density.plot(series['n_tp'], series['density_percentile'], color='tab:blue',
                        marker='o', markersize=5, linewidth=2)
        ax_density.axhspan(0, alarm_percentile, color='tab:red', alpha=0.12)
        ax_density.axhline(alarm_percentile, color='tab:red', linestyle='--',
                           linewidth=1.5)
        ax_density.set_xlabel('number of training points')
        # Kept short: at the package's default label size a full sentence does not
        # fit along the axis and is clipped. The legend carries the meaning.
        ax_density.set_ylabel('density percentile (0-100)')
        ax_density.set_ylim(0, 100)
        ax_density.grid(alpha=0.3)

        ax_corr = ax_density.twinx()
        ax_corr.plot(series['n_tp'], series['max_abs_correlation'], color='tab:orange',
                     marker='s', markersize=4, linewidth=1.5, linestyle='--')
        ax_corr.set_ylabel(r'largest $|r|$')
        ax_corr.set_ylim(0, 1)

        ax_density.legend(handles=[
            Line2D([], [], color='tab:blue', marker='o',
                   label='density percentile of the marginal-peak vector'),
            Line2D([], [], color='tab:orange', marker='s', linestyle='--',
                   label=r'largest $|r|$ between parameters'),
            mpatches.Patch(color='tab:red', alpha=0.12,
                           label='marginal peaks not jointly plausible'),
        ], loc='best', fontsize='small')

        plt.tight_layout()
        for extension in ('pdf', 'png'):
            fig.savefig(os.path.join(self.save_folder, f'{file_stem}.{extension}'),
                        bbox_inches='tight', dpi=300)
        plt.close(fig)

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
            best_estimate_value="posterior_marginal_peak",
            post_loglikelihood_arrays=None
    ):
        """
        Plot marginal posterior distributions for selected Bayesian iterations.

        All posterior subplots use the same histogram bin width and the same
        histogram bin edges. The common histogram range is determined from the
        minimum and maximum parameter limits across all selected parameters.

        Parameters
        ----------
        posterior_arrays : list of array-like
            Posterior parameter samples for each Bayesian iteration.

            Each valid entry must have shape:

                (number_of_posterior_samples, number_of_parameters)

        parameter_names : list of str
            Names of all calibration parameters.

        prior : array-like
            Original prior parameter samples with shape:

                (number_of_prior_samples, number_of_parameters)

        param_values : array-like, optional
            Parameter bounds with shape:

                (number_of_parameters, 2)

        iterations_to_plot : iterable of int, optional
            Bayesian iterations to plot.

        bins : int, default=40
            Number of histogram bins across the complete global parameter range.
            Because the same bin edges are used for every subplot, all histograms
            have exactly the same numerical bin width.

        density : bool, default=True
            Whether the histogram represents probability density.

        plot_prior : bool, default=False
            Whether to plot the prior histograms.

        parameter_units : list of str, optional
            Units corresponding to each calibration parameter.

        parameter_indices : list of int, optional
            Indices of parameters to plot.

        best_estimate_value : str, default="posterior_marginal_peak"
            Estimate represented by the vertical red line.

            Available options:

            - "posterior_mean"
            - "posterior_marginal_peak"
            - "joint_posterior_MAP"

            Backward-compatible aliases:

            - "posterior_peak" -> "posterior_marginal_peak"
            - "posterior_MAP" -> "joint_posterior_MAP"

        post_loglikelihood_arrays : list of array-like or array-like, optional
            Post-rejection log-likelihood values.

            For an iteration-indexed object:

                post_loglikelihood_arrays[i][j]

            must correspond exactly to:

                posterior_arrays[i][j, :]

            A direct one-dimensional log-likelihood vector can also be passed
            when only one iteration is plotted.

        Returns
        -------
        dict
            Estimated parameter values for every plotted iteration.
        """

        save_folder = self.save_folder

        # ------------------------------------------------------------------
        # Backward-compatible estimator names
        # ------------------------------------------------------------------
        estimate_aliases = {
            "posterior_peak": "posterior_marginal_peak",
            "posterior_MAP": "joint_posterior_MAP"
        }

        best_estimate_value = estimate_aliases.get(
            best_estimate_value,
            best_estimate_value
        )

        valid_estimate_options = {
            "posterior_mean",
            "posterior_marginal_peak",
            "joint_posterior_MAP"
        }

        if best_estimate_value not in valid_estimate_options:
            raise ValueError(
                "best_estimate_value must be one of: "
                "'posterior_mean', "
                "'posterior_marginal_peak', or "
                "'joint_posterior_MAP'."
            )

        # ------------------------------------------------------------------
        # Determine iterations to plot
        # ------------------------------------------------------------------
        if iterations_to_plot is None:

            iterations_to_plot = [
                iteration_idx
                for iteration_idx, posterior in enumerate(
                    posterior_arrays
                )
                if (
                        posterior is not None
                        and np.asarray(posterior).size > 0
                )
            ]

        elif np.isscalar(iterations_to_plot):

            iterations_to_plot = [
                int(iterations_to_plot)
            ]

        else:

            iterations_to_plot = list(
                iterations_to_plot
            )

        if len(iterations_to_plot) == 0:
            raise ValueError(
                "No posterior iterations were selected for plotting."
            )

        # ------------------------------------------------------------------
        # Select parameters
        # ------------------------------------------------------------------
        if parameter_indices is None:

            selected_indices = list(
                range(len(parameter_names))
            )

        else:

            selected_indices = list(
                parameter_indices
            )

        parameter_num = len(selected_indices)

        if parameter_num == 0:
            raise ValueError(
                "No parameter indices were selected."
            )

        for param_idx in selected_indices:

            if (
                    param_idx < 0
                    or param_idx >= len(parameter_names)
            ):
                raise IndexError(
                    f"Parameter index {param_idx} is outside the "
                    f"valid range 0 to "
                    f"{len(parameter_names) - 1}."
                )

        if parameter_units is None:
            parameter_units = [
                                  ''
                              ] * len(parameter_names)

        if len(parameter_units) != len(parameter_names):
            raise ValueError(
                "parameter_units and parameter_names must have "
                "the same length."
            )

        # ------------------------------------------------------------------
        # Validate prior
        # ------------------------------------------------------------------
        prior = np.asarray(
            prior,
            dtype=float
        )

        if prior.ndim != 2:
            raise ValueError(
                "prior must be a two-dimensional array with shape "
                "(number_of_samples, number_of_parameters)."
            )

        if prior.shape[1] != len(parameter_names):
            raise ValueError(
                "The number of prior columns does not match the "
                "number of parameter names. "
                f"Prior columns: {prior.shape[1]}; "
                f"parameter names: {len(parameter_names)}."
            )

        # ------------------------------------------------------------------
        # Determine x-axis limits
        # ------------------------------------------------------------------
        x_limits = np.zeros(
            (parameter_num, 2),
            dtype=float
        )

        if param_values is None:

            for col, param_idx in enumerate(
                    selected_indices
            ):
                x_limits[col] = (
                    np.nanmin(prior[:, param_idx]),
                    np.nanmax(prior[:, param_idx])
                )

        else:

            param_values = np.asarray(
                param_values,
                dtype=float
            )

            if (
                    param_values.ndim != 2
                    or param_values.shape[0] != len(parameter_names)
                    or param_values.shape[1] != 2
            ):
                raise ValueError(
                    "param_values must have shape "
                    "(number_of_parameters, 2)."
                )

            for col, param_idx in enumerate(
                    selected_indices
            ):
                x_limits[col] = param_values[
                    param_idx
                ]

        # ------------------------------------------------------------------
        # Define COMMON histogram bins for all posterior subplots
        # ------------------------------------------------------------------
        # The histogram domain is the complete range covered by all selected
        # calibration parameters. Therefore every subplot uses exactly the
        # same numerical bin width and exactly the same bin-edge positions.
        global_bin_min = float(
            np.min(
                x_limits[:, 0]
            )
        )

        global_bin_max = float(
            np.max(
                x_limits[:, 1]
            )
        )

        if (
                not np.isfinite(global_bin_min)
                or not np.isfinite(global_bin_max)
        ):
            raise ValueError(
                "The global histogram limits must be finite."
            )

        if global_bin_max <= global_bin_min:
            raise ValueError(
                "Cannot construct histogram bins because the global "
                "parameter range is zero or negative."
            )

        if (
                not isinstance(
                    bins,
                    (int, np.integer)
                )
                or bins <= 0
        ):
            raise ValueError(
                "bins must be a positive integer."
            )

        common_bin_edges = np.linspace(
            global_bin_min,
            global_bin_max,
            bins + 1
        )

        common_bin_width = float(
            common_bin_edges[1]
            - common_bin_edges[0]
        )

        print(
            "Common histogram configuration:"
        )

        print(
            f"  global minimum: "
            f"{global_bin_min}"
        )

        print(
            f"  global maximum: "
            f"{global_bin_max}"
        )

        print(
            f"  number of bins: "
            f"{bins}"
        )

        print(
            f"  common bin width: "
            f"{common_bin_width}"
        )

        # Results returned by the function
        estimate_results = {}

        # ------------------------------------------------------------------
        # Loop over Bayesian iterations
        # ------------------------------------------------------------------
        for iteration_idx in iterations_to_plot:

            if (
                    iteration_idx < 0
                    or iteration_idx >= len(posterior_arrays)
            ):
                raise IndexError(
                    f"Posterior iteration {iteration_idx} is outside "
                    f"the valid range 0 to "
                    f"{len(posterior_arrays) - 1}."
                )

            if posterior_arrays[iteration_idx] is None:
                raise ValueError(
                    f"Posterior iteration {iteration_idx} is None."
                )

            posterior_matrix = np.asarray(
                posterior_arrays[iteration_idx],
                dtype=float
            )

            if posterior_matrix.ndim != 2:
                raise ValueError(
                    f"posterior_arrays[{iteration_idx}] must be a "
                    "two-dimensional array with shape "
                    "(number_of_samples, number_of_parameters)."
                )

            if posterior_matrix.shape[0] == 0:
                raise ValueError(
                    f"Posterior iteration {iteration_idx} contains "
                    "no samples."
                )

            if posterior_matrix.shape[1] != len(parameter_names):
                raise ValueError(
                    f"The posterior at iteration {iteration_idx} has "
                    f"{posterior_matrix.shape[1]} parameter columns, "
                    f"but {len(parameter_names)} parameter names were "
                    "provided."
                )

            print(
                f"Plotting posterior iteration: "
                f"{iteration_idx}"
            )

            # --------------------------------------------------------------
            # Determine the joint posterior MAP
            # --------------------------------------------------------------
            joint_map_vector = None
            joint_map_sample_index = None
            maximum_loglikelihood = None

            if best_estimate_value == "joint_posterior_MAP":

                if post_loglikelihood_arrays is None:
                    raise ValueError(
                        "post_loglikelihood_arrays must be provided "
                        "when best_estimate_value="
                        "'joint_posterior_MAP'."
                    )

                direct_loglikelihood_vector = None

                # ----------------------------------------------------------
                # Case 1:
                # A direct one-dimensional log-likelihood vector was passed
                # ----------------------------------------------------------
                try:

                    converted_scores = np.asarray(
                        post_loglikelihood_arrays,
                        dtype=float
                    )

                    if (
                            converted_scores.ndim == 1
                            and converted_scores.size
                            == posterior_matrix.shape[0]
                    ):
                        direct_loglikelihood_vector = (
                            converted_scores.reshape(-1)
                        )

                except (TypeError, ValueError):

                    # Expected when the complete object is a list containing
                    # arrays of different lengths.
                    direct_loglikelihood_vector = None

                if direct_loglikelihood_vector is not None:

                    if len(iterations_to_plot) != 1:
                        raise ValueError(
                            "A single log-likelihood vector was passed "
                            "while multiple iterations were requested. "
                            "Pass the complete iteration-indexed "
                            "post_loglikelihood_arrays object."
                        )

                    log_likelihood_vector = (
                        direct_loglikelihood_vector
                    )

                # ----------------------------------------------------------
                # Case 2:
                # Complete iteration-indexed object was passed
                # ----------------------------------------------------------
                else:

                    try:
                        iteration_loglikelihood = (
                            post_loglikelihood_arrays[
                                iteration_idx
                            ]
                        )

                    except (IndexError, TypeError) as error:
                        raise ValueError(
                            "Could not extract post-rejection "
                            "log-likelihood values for iteration "
                            f"{iteration_idx}."
                        ) from error

                    if iteration_loglikelihood is None:
                        raise ValueError(
                            "No post-rejection log-likelihood values "
                            f"were stored for iteration "
                            f"{iteration_idx}."
                        )

                    log_likelihood_vector = np.asarray(
                        iteration_loglikelihood,
                        dtype=float
                    ).reshape(-1)

                # ----------------------------------------------------------
                # Confirm row alignment
                # ----------------------------------------------------------
                if (
                        log_likelihood_vector.size
                        != posterior_matrix.shape[0]
                ):
                    raise ValueError(
                        "The posterior samples and post-rejection "
                        "log-likelihood values are not aligned for "
                        f"iteration {iteration_idx}. "
                        f"Posterior samples: "
                        f"{posterior_matrix.shape[0]}; "
                        f"log-likelihood values: "
                        f"{log_likelihood_vector.size}. "
                        "Each log-likelihood value must correspond to "
                        "the posterior sample at the same row index."
                    )

                valid_scores = np.isfinite(
                    log_likelihood_vector
                )

                if not np.any(valid_scores):
                    raise ValueError(
                        "All post-rejection log-likelihood values are "
                        f"non-finite for iteration {iteration_idx}."
                    )

                valid_sample_indices = np.flatnonzero(
                    valid_scores
                )

                joint_map_sample_index = int(
                    valid_sample_indices[
                        np.argmax(
                            log_likelihood_vector[
                                valid_scores
                            ]
                        )
                    ]
                )

                maximum_loglikelihood = float(
                    log_likelihood_vector[
                        joint_map_sample_index
                    ]
                )

                # All parameter components are taken from the same
                # posterior sample.
                joint_map_vector = posterior_matrix[
                                   joint_map_sample_index,
                                   :
                                   ].copy()

                print(
                    f"Joint posterior MAP for iteration "
                    f"{iteration_idx}:"
                )

                print(
                    f"  posterior sample index: "
                    f"{joint_map_sample_index}"
                )

                print(
                    f"  maximum log-likelihood: "
                    f"{maximum_loglikelihood}"
                )

                print(
                    f"  parameter combination: "
                    f"{joint_map_vector}"
                )

            # --------------------------------------------------------------
            # Create figure
            # --------------------------------------------------------------
            num_rows = 3

            num_cols = math.ceil(
                parameter_num / num_rows
            )

            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(
                    6.5 * num_cols,
                    5 * num_rows
                )
            )

            axes = np.asarray(
                axes
            ).reshape(-1)

            iteration_parameter_estimates = {}

            # --------------------------------------------------------------
            # Loop over selected parameters
            # --------------------------------------------------------------
            for col, param_idx in enumerate(
                    selected_indices
            ):

                ax = axes[col]

                posterior_vector = posterior_matrix[
                                   :,
                                   param_idx
                                   ]

                finite_posterior_values = posterior_vector[
                    np.isfinite(
                        posterior_vector
                    )
                ]

                if finite_posterior_values.size == 0:
                    raise ValueError(
                        f"Parameter "
                        f"'{parameter_names[param_idx]}' contains no "
                        f"finite posterior samples at iteration "
                        f"{iteration_idx}."
                    )

                # ----------------------------------------------------------
                # Histogram values used for common normalization
                # ----------------------------------------------------------
                posterior_hist_values, _ = np.histogram(
                    finite_posterior_values,
                    bins=common_bin_edges,
                    density=density
                )

                max_density = (
                    float(
                        np.max(
                            posterior_hist_values
                        )
                    )
                    if posterior_hist_values.size > 0
                    else 0.0
                )

                finite_prior_values = None

                if plot_prior:

                    prior_vector = prior[
                                   :,
                                   param_idx
                                   ]

                    finite_prior_values = prior_vector[
                        np.isfinite(
                            prior_vector
                        )
                    ]

                    prior_hist_values, _ = np.histogram(
                        finite_prior_values,
                        bins=common_bin_edges,
                        density=density
                    )

                    if prior_hist_values.size > 0:
                        max_density = max(
                            max_density,
                            float(
                                np.max(
                                    prior_hist_values
                                )
                            )
                        )

                # ----------------------------------------------------------
                # Plot posterior histogram
                # ----------------------------------------------------------
                (
                    posterior_hist_values,
                    posterior_bin_edges,
                    posterior_patches
                ) = ax.hist(
                    finite_posterior_values,
                    bins=common_bin_edges,
                    density=density,
                    alpha=0.75,
                    color='0.35',
                    edgecolor='black',
                    linewidth=0.8
                )

                if max_density > 0:
                    for patch in posterior_patches:
                        patch.set_height(
                            patch.get_height()
                            / max_density
                        )

                # ----------------------------------------------------------
                # Calculate selected best estimate
                # ----------------------------------------------------------
                if best_estimate_value == "posterior_mean":

                    value = float(
                        np.mean(
                            finite_posterior_values
                        )
                    )

                elif (
                        best_estimate_value
                        == "posterior_marginal_peak"
                ):

                    # ------------------------------------------------------
                    # Marginal posterior peak
                    # ------------------------------------------------------
                    # The peak is calculated from exactly the same common
                    # histogram bins displayed in the plot.
                    peak_bin_index = int(
                        np.argmax(
                            posterior_hist_values
                        )
                    )

                    peak_bin_left = posterior_bin_edges[
                        peak_bin_index
                    ]

                    peak_bin_right = posterior_bin_edges[
                        peak_bin_index + 1
                        ]

                    # NumPy histograms include the right edge only for the
                    # final histogram bin.
                    if (
                            peak_bin_index
                            == len(posterior_hist_values) - 1
                    ):

                        samples_in_peak_bin = (
                            finite_posterior_values[
                                (
                                        finite_posterior_values
                                        >= peak_bin_left
                                )
                                & (
                                        finite_posterior_values
                                        <= peak_bin_right
                                )
                                ]
                        )

                    else:

                        samples_in_peak_bin = (
                            finite_posterior_values[
                                (
                                        finite_posterior_values
                                        >= peak_bin_left
                                )
                                & (
                                        finite_posterior_values
                                        < peak_bin_right
                                )
                                ]
                        )

                    # Use the mean of the actual posterior samples located
                    # inside the densest marginal posterior bin.
                    if samples_in_peak_bin.size > 0:

                        value = float(
                            np.mean(
                                samples_in_peak_bin
                            )
                        )

                    else:

                        # Defensive fallback: use the center of the densest
                        # histogram bin.
                        value = float(
                            0.5 * (
                                    peak_bin_left
                                    + peak_bin_right
                            )
                        )

                elif (
                        best_estimate_value
                        == "joint_posterior_MAP"
                ):

                    value = float(
                        joint_map_vector[
                            param_idx
                        ]
                    )

                else:

                    raise RuntimeError(
                        "Unsupported best-estimate option."
                    )

                iteration_parameter_estimates[
                    parameter_names[param_idx]
                ] = value

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

                # ----------------------------------------------------------
                # Optional prior histogram
                # ----------------------------------------------------------
                if plot_prior:

                    (
                        prior_hist_values,
                        prior_bin_edges,
                        prior_patches
                    ) = ax.hist(
                        finite_prior_values,
                        bins=common_bin_edges,
                        density=density,
                        alpha=0.35,
                        color='0.75',
                        edgecolor='0.6',
                        linewidth=0.8
                    )

                    if max_density > 0:
                        for patch in prior_patches:
                            patch.set_height(
                                patch.get_height()
                                / max_density
                            )

                # ----------------------------------------------------------
                # Axis labels and formatting
                # ----------------------------------------------------------
                unit = (
                    f' [{parameter_units[param_idx]}]'
                    if parameter_units[param_idx]
                    else ''
                )

                ax.set_xlabel(
                    f'{parameter_names[param_idx]}{unit}',
                    fontsize=40
                )

                ax.set_ylabel(
                    'Posterior\n density [-]',
                    fontsize=40
                )

                ax.tick_params(
                    axis='both',
                    which='major',
                    labelsize=35
                )

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

                ax.set_xlim(
                    x_limits[col]
                )

                ax.set_ylim(
                    0,
                    1.2
                )

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

            # --------------------------------------------------------------
            # Remove unused axes
            # --------------------------------------------------------------
            for axis_idx in range(
                    parameter_num,
                    len(axes)
            ):
                fig.delaxes(
                    axes[axis_idx]
                )

            # --------------------------------------------------------------
            # Global legend
            # --------------------------------------------------------------
            estimate_labels = {
                "posterior_mean":
                    "Marginal posterior mean",

                "posterior_marginal_peak":
                    "Marginal posterior peak",

                "joint_posterior_MAP":
                    "Joint posterior MAP"
            }

            legend_elements = []

            if plot_prior:
                legend_elements.append(
                    mpatches.Patch(
                        facecolor='0.75',
                        edgecolor='0.6',
                        alpha=0.35,
                        label='Prior'
                    )
                )

            legend_elements.extend([
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
                    linewidth=2,
                    linestyle='--',
                    label=estimate_labels[
                        best_estimate_value
                    ]
                )
            ])

            fig.legend(
                handles=legend_elements,
                loc='upper center',
                ncol=len(legend_elements),
                fontsize=28,
                frameon=False,
                bbox_to_anchor=(0.5, 1.02)
            )

            fig.tight_layout(
                rect=[0, 0.01, 1, 0.95]
            )

            fig.savefig(
                save_folder
                / (
                    f'posterior_distributions_iteration_'
                    f'{iteration_idx + 1}.svg'
                ),
                format='svg',
                bbox_inches='tight',
                transparent=True
            )

            plt.close(fig)

            # --------------------------------------------------------------
            # Store returned results
            # --------------------------------------------------------------
            estimate_results[iteration_idx] = {
                "estimate_type":
                    best_estimate_value,

                "parameters":
                    iteration_parameter_estimates
            }

            if best_estimate_value == "joint_posterior_MAP":
                estimate_results[iteration_idx].update({
                    "sample_index":
                        joint_map_sample_index,

                    "log_likelihood":
                        maximum_loglikelihood,

                    "parameter_vector":
                        joint_map_vector.copy()
                })

        return estimate_results


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

    def plot_posterior_model_comparison(
            self,
            bal_dictionary_1,
            bal_dictionary_2,
            model_names=("MO-GPE", "SO-GPE"),
            parameter_names=None,
            parameter_units=None,
            param_values=None,
            parameter_indices=None,
            iteration_1=None,
            iteration_2=None,
            plot_prior=False,
            n_bins=12,
            ncols=2,
            figsize_per_panel=(5.5, 4.0),
            colors=("black", "blue"),
            prior_color="0.85",
            line_width=2.0,
            prior_alpha=0.40,
            normalize_density=True,
            show_title=False,
            filename=None
    ):
        """
        Compare marginal posterior distributions from two BAL_dictionary objects
        using overlaid step histograms.

        All distributions for a given parameter use exactly the same histogram
        bin edges.

        If ``normalize_density=True``:

        1. Each posterior probability density is calculated independently using
           ``np.histogram(..., density=True)``.

        2. Each posterior is then divided by its OWN maximum density:

               max(Model 1 posterior) = 1
               max(Model 2 posterior) = 1

        3. The prior is NOT independently peak-normalized. Instead, the prior
           density is divided by the largest raw posterior peak:

               prior_scaled =
                   prior_density /
                   max(raw_peak_model_1, raw_peak_model_2)

           Therefore, a diffuse prior remains below 1 when the posterior
           distributions are more concentrated than the prior.

        This provides a visualization in which both posterior shapes can be
        compared independently while retaining a visual indication of
        prior-to-posterior concentration.

        Parameters
        ----------
        bal_dictionary_1, bal_dictionary_2 : dict or str or pathlib.Path
            Loaded BAL_dictionary objects or file paths to BAL_dictionary
            pickle files.

        model_names : tuple/list of str, default=("MO-GPE", "SO-GPE")
            Labels used in the legend.

        parameter_names : list of str, optional
            Names of calibration parameters. If None, they are obtained from
            bal_dictionary_1['calibration_parameters'].

        parameter_units : list of str, optional
            Units corresponding to calibration parameters.

        param_values : array-like, optional
            Parameter bounds with shape (n_parameters, 2). If None, they are
            obtained from bal_dictionary_1['param_values'], if available.

        parameter_indices : list of int, optional
            Parameters to plot. If None, all parameters are plotted.

        iteration_1, iteration_2 : int, optional
            Posterior iteration to use for each calibration. If None, the last
            non-empty posterior iteration is used.

        plot_prior : bool, default=False
            If True, show the prior as a light-gray filled histogram.

        n_bins : int, default=12
            Number of histogram bins.

        ncols : int, default=2
            Number of subplot columns.

        figsize_per_panel : tuple of float, default=(5.5, 4.0)
            Width and height allocated to each subplot.

        colors : tuple of str, default=("black", "blue")
            Colors for Model 1 and Model 2.

        prior_color : str, default="0.85"
            Gray level used for prior distribution.

        line_width : float, default=2.0
            Width of posterior histogram lines.

        prior_alpha : float, default=0.40
            Transparency of the prior.

        normalize_density : bool, default=True
            If True, independently peak-normalize each posterior while scaling
            the prior relative to the largest raw posterior peak.

            If False, plot the actual probability-density estimates returned
            by ``np.histogram(..., density=True)``.

        show_title : bool, default=False
            Whether to add an overall figure title.

        filename : str, optional
            Output file name. If None, an automatic SVG filename is generated.

        Returns
        -------
        dict
            Metadata describing the generated figure.
        """

        save_folder = self.save_folder

        # ================================================================
        # Helper functions
        # ================================================================

        def _load_bal_dictionary(obj):

            if isinstance(obj, (str, os.PathLike)):
                with open(obj, 'rb') as f:
                    return pickle.load(f)

            if isinstance(obj, dict):
                return obj

            raise TypeError(
                "Each BAL dictionary input must be either a dict "
                "or a path to a pickle file."
            )

        def _last_valid_iteration(posterior_list):

            valid_indices = [
                idx
                for idx, arr in enumerate(posterior_list)
                if arr is not None and np.asarray(arr).size > 0
            ]

            if len(valid_indices) == 0:
                raise ValueError(
                    "No non-empty posterior iterations were found."
                )

            return valid_indices[-1]

        def _format_parameter_label(param_idx):

            name = str(
                parameter_names[param_idx]
            )

            if parameter_units[param_idx]:
                return (
                    f"{name} "
                    f"[{parameter_units[param_idx]}]"
                )

            return name

        # ================================================================
        # Load BAL dictionaries
        # ================================================================

        bal1 = _load_bal_dictionary(
            bal_dictionary_1
        )

        bal2 = _load_bal_dictionary(
            bal_dictionary_2
        )

        if 'posterior' not in bal1:
            raise KeyError(
                "The first BAL dictionary does not contain 'posterior'."
            )

        if 'posterior' not in bal2:
            raise KeyError(
                "The second BAL dictionary does not contain 'posterior'."
            )

        if plot_prior and 'prior' not in bal1:
            raise KeyError(
                "plot_prior=True, but the first BAL dictionary "
                "does not contain 'prior'."
            )

        # ================================================================
        # Parameter information
        # ================================================================

        if parameter_names is None:

            if 'calibration_parameters' not in bal1:
                raise KeyError(
                    "parameter_names was not provided and "
                    "'calibration_parameters' was not found "
                    "in the first BAL dictionary."
                )

            parameter_names = list(
                bal1['calibration_parameters']
            )

        n_parameters = len(
            parameter_names
        )

        if parameter_units is None:

            parameter_units = (
                    [''] * n_parameters
            )

        elif len(parameter_units) != n_parameters:

            raise ValueError(
                "parameter_units and parameter_names "
                "must have the same length."
            )

        # ================================================================
        # Parameter selection
        # ================================================================

        if parameter_indices is None:

            selected_indices = list(
                range(n_parameters)
            )

        else:

            selected_indices = list(
                parameter_indices
            )

        if len(selected_indices) == 0:
            raise ValueError(
                "No parameter indices were selected."
            )

        for idx in selected_indices:

            if idx < 0 or idx >= n_parameters:
                raise IndexError(
                    f"Parameter index {idx} is outside "
                    f"the valid range "
                    f"0 to {n_parameters - 1}."
                )

        # ================================================================
        # Posterior iterations
        # ================================================================

        posterior_list_1 = bal1[
            'posterior'
        ]

        posterior_list_2 = bal2[
            'posterior'
        ]

        if iteration_1 is None:
            iteration_1 = (
                _last_valid_iteration(
                    posterior_list_1
                )
            )

        if iteration_2 is None:
            iteration_2 = (
                _last_valid_iteration(
                    posterior_list_2
                )
            )

        if (
                iteration_1 < 0
                or iteration_1 >= len(
            posterior_list_1
        )
        ):
            raise IndexError(
                f"iteration_1={iteration_1} is outside "
                f"the valid range."
            )

        if (
                iteration_2 < 0
                or iteration_2 >= len(
            posterior_list_2
        )
        ):
            raise IndexError(
                f"iteration_2={iteration_2} is outside "
                f"the valid range."
            )

        # ================================================================
        # Extract posterior arrays
        # ================================================================

        posterior_1 = np.asarray(
            posterior_list_1[
                iteration_1
            ],
            dtype=float
        )

        posterior_2 = np.asarray(
            posterior_list_2[
                iteration_2
            ],
            dtype=float
        )

        if (
                posterior_1.ndim != 2
                or posterior_1.shape[1] != n_parameters
        ):
            raise ValueError(
                f"Posterior matrix for {model_names[0]} "
                f"must have shape "
                f"(n_samples, {n_parameters})."
            )

        if (
                posterior_2.ndim != 2
                or posterior_2.shape[1] != n_parameters
        ):
            raise ValueError(
                f"Posterior matrix for {model_names[1]} "
                f"must have shape "
                f"(n_samples, {n_parameters})."
            )

        # ================================================================
        # Prior
        # ================================================================

        prior = None

        if plot_prior:

            prior = np.asarray(
                bal1['prior'],
                dtype=float
            )

            if (
                    prior.ndim != 2
                    or prior.shape[1] != n_parameters
            ):
                raise ValueError(
                    "Prior must have shape "
                    f"(n_samples, {n_parameters})."
                )

        # ================================================================
        # Parameter bounds
        # ================================================================

        if param_values is None:

            if 'param_values' in bal1:

                param_values = np.asarray(
                    bal1['param_values'],
                    dtype=float
                )

            else:

                param_values = None

        else:

            param_values = np.asarray(
                param_values,
                dtype=float
            )

        if param_values is not None:

            if (
                    param_values.ndim != 2
                    or param_values.shape
                    != (n_parameters, 2)
            ):
                raise ValueError(
                    "param_values must have shape "
                    "(number_of_parameters, 2)."
                )

        # ================================================================
        # Figure layout
        # ================================================================

        n_panels = len(
            selected_indices
        )

        nrows = math.ceil(
            n_panels / ncols
        )

        fig_width = (
                figsize_per_panel[0]
                * ncols
        )

        fig_height = (
                figsize_per_panel[1]
                * nrows
        )

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                fig_width,
                fig_height
            )
        )

        axes = (
            np.atleast_1d(
                axes
            ).reshape(-1)
        )

        # Panel labels:
        # a), b), c), ...
        panel_letters = [
            f"{chr(97 + i)})"
            for i in range(
                n_panels
            )
        ]

        # ================================================================
        # Plot distributions
        # ================================================================

        for local_idx, param_idx in enumerate(
                selected_indices
        ):

            ax = axes[
                local_idx
            ]

            # ------------------------------------------------------------
            # Extract Model 1 posterior samples
            # ------------------------------------------------------------

            values_1 = posterior_1[
                       :, param_idx
                       ]

            values_1 = values_1[
                np.isfinite(
                    values_1
                )
            ]

            # ------------------------------------------------------------
            # Extract Model 2 posterior samples
            # ------------------------------------------------------------

            values_2 = posterior_2[
                       :, param_idx
                       ]

            values_2 = values_2[
                np.isfinite(
                    values_2
                )
            ]

            if values_1.size < 2:
                raise ValueError(
                    f"{model_names[0]} posterior "
                    f"for parameter "
                    f"'{parameter_names[param_idx]}' "
                    f"contains fewer than two finite samples."
                )

            if values_2.size < 2:
                raise ValueError(
                    f"{model_names[1]} posterior "
                    f"for parameter "
                    f"'{parameter_names[param_idx]}' "
                    f"contains fewer than two finite samples."
                )

            # ------------------------------------------------------------
            # Extract prior samples
            # ------------------------------------------------------------

            values_prior = None

            if plot_prior:

                values_prior = prior[
                               :, param_idx
                               ]

                values_prior = values_prior[
                    np.isfinite(
                        values_prior
                    )
                ]

                if values_prior.size < 2:
                    raise ValueError(
                        f"Prior for parameter "
                        f"'{parameter_names[param_idx]}' "
                        f"contains fewer than two finite samples."
                    )

            # ------------------------------------------------------------
            # Determine x-axis limits
            # ------------------------------------------------------------

            if param_values is not None:

                x_min, x_max = (
                    param_values[
                        param_idx
                    ]
                )

            else:

                combined_values = np.concatenate(
                    [
                        values_1,
                        values_2
                    ]
                )

                if plot_prior:
                    combined_values = np.concatenate(
                        [
                            combined_values,
                            values_prior
                        ]
                    )

                x_min = np.min(
                    combined_values
                )

                x_max = np.max(
                    combined_values
                )

            if (
                    not np.isfinite(x_min)
                    or not np.isfinite(x_max)
                    or x_max <= x_min
            ):
                raise ValueError(
                    f"Invalid x-axis bounds "
                    f"for parameter "
                    f"'{parameter_names[param_idx]}': "
                    f"[{x_min}, {x_max}]."
                )

            # ------------------------------------------------------------
            # Common histogram bins
            #
            # IMPORTANT:
            # Prior and both posterior distributions use exactly
            # the same bin boundaries.
            # ------------------------------------------------------------

            bin_edges = np.linspace(
                x_min,
                x_max,
                n_bins + 1
            )

            # ============================================================
            # TRUE PROBABILITY DENSITIES
            # ============================================================

            # ------------------------------------------------------------
            # Model 1 posterior
            #
            # density=True uses the number of samples in THIS
            # posterior independently.
            # ------------------------------------------------------------

            hist_1, edges = np.histogram(
                values_1,
                bins=bin_edges,
                density=True
            )

            # ------------------------------------------------------------
            # Model 2 posterior
            #
            # Again, this is normalized independently according
            # to the number of retained Model 2 samples.
            # ------------------------------------------------------------

            hist_2, _ = np.histogram(
                values_2,
                bins=bin_edges,
                density=True
            )

            # ------------------------------------------------------------
            # Prior
            # ------------------------------------------------------------

            hist_prior = None

            if plot_prior:
                hist_prior, _ = np.histogram(
                    values_prior,
                    bins=bin_edges,
                    density=True
                )

            # ============================================================
            # STORE RAW DENSITY PEAKS
            #
            # IMPORTANT:
            # These must be calculated BEFORE any plotting normalization.
            # ============================================================

            max_1_raw = np.max(
                hist_1
            )

            max_2_raw = np.max(
                hist_2
            )

            max_prior_raw = None

            if plot_prior:
                max_prior_raw = np.max(
                    hist_prior
                )

            # ============================================================
            # NORMALIZATION FOR VISUALIZATION
            # ============================================================
            #
            # Posterior 1:
            #     divided by its OWN maximum
            #     -> peak = 1
            #
            # Posterior 2:
            #     divided by its OWN maximum
            #     -> peak = 1
            #
            # Prior:
            #     NOT divided by its own maximum
            #
            #     Instead:
            #
            #     prior /
            #     max(raw posterior 1 peak,
            #         raw posterior 2 peak)
            #
            # Consequently, a broader / less concentrated prior
            # remains below the posterior peaks.
            # ============================================================

            posterior_reference_peak = None

            if normalize_density:

                # --------------------------------------------------------
                # Model 1 posterior
                # --------------------------------------------------------

                if max_1_raw > 0:
                    hist_1 = (
                            hist_1
                            / max_1_raw
                    )

                # --------------------------------------------------------
                # Model 2 posterior
                # --------------------------------------------------------

                if max_2_raw > 0:
                    hist_2 = (
                            hist_2
                            / max_2_raw
                    )

                # --------------------------------------------------------
                # Prior
                # --------------------------------------------------------

                if plot_prior:

                    posterior_reference_peak = max(
                        max_1_raw,
                        max_2_raw
                    )

                    if posterior_reference_peak > 0:
                        hist_prior = (
                                hist_prior
                                / posterior_reference_peak
                        )

            # ============================================================
            # Diagnostic output
            # ============================================================

            if normalize_density:

                if plot_prior:

                    print(
                        f"{parameter_names[param_idx]} | "
                        f"N prior={values_prior.size}, "
                        f"N {model_names[0]}={values_1.size}, "
                        f"N {model_names[1]}={values_2.size}"
                    )

                    print(
                        f"  Raw density peaks: "
                        f"Prior={max_prior_raw:.4f}, "
                        f"{model_names[0]}={max_1_raw:.4f}, "
                        f"{model_names[1]}={max_2_raw:.4f}"
                    )

                    print(
                        f"  Plotted peaks: "
                        f"Prior={np.max(hist_prior):.4f}, "
                        f"{model_names[0]}={np.max(hist_1):.4f}, "
                        f"{model_names[1]}={np.max(hist_2):.4f}"
                    )

                else:

                    print(
                        f"{parameter_names[param_idx]} | "
                        f"N {model_names[0]}={values_1.size}, "
                        f"N {model_names[1]}={values_2.size}"
                    )

                    print(
                        f"  Plotted peaks: "
                        f"{model_names[0]}={np.max(hist_1):.4f}, "
                        f"{model_names[1]}={np.max(hist_2):.4f}"
                    )

            # ============================================================
            # Plot prior
            # ============================================================

            if plot_prior:
                ax.stairs(
                    hist_prior,
                    edges,
                    fill=True,
                    color=prior_color,
                    alpha=prior_alpha,
                    linewidth=0.8,
                    label='Prior',
                    zorder=1
                )

            # ============================================================
            # Plot Model 1 posterior
            # ============================================================

            ax.stairs(
                hist_1,
                edges,
                linewidth=line_width,
                color=colors[0],
                linestyle='-',
                label=str(
                    model_names[0]
                ),
                zorder=3
            )

            # ============================================================
            # Plot Model 2 posterior
            # ============================================================

            ax.stairs(
                hist_2,
                edges,
                linewidth=line_width,
                color=colors[1],
                linestyle='--',
                label=str(
                    model_names[1]
                ),
                zorder=4
            )

            # ============================================================
            # Axes
            # ============================================================

            ax.set_xlim(
                x_min,
                x_max
            )

            if normalize_density:

                # --------------------------------------------------------
                # Usually the two posterior curves peak at exactly 1.
                #
                # If the prior is unexpectedly more concentrated than
                # both posteriors, it can exceed 1. In that case we do
                # not clip it; the y-axis is expanded accordingly.
                # --------------------------------------------------------

                y_max = 1.0

                if plot_prior:
                    prior_plot_max = np.max(
                        hist_prior
                    )

                    y_max = max(
                        y_max,
                        prior_plot_max
                    )

                ax.set_ylim(
                    0.0,
                    y_max * 1.05
                )

                ax.set_ylabel(
                    'Relative density [-]',
                    fontsize=16
                )

            else:

                ax.set_ylabel(
                    'Probability density',
                    fontsize=16
                )

            ax.set_xlabel(
                _format_parameter_label(
                    param_idx
                ),
                fontsize=18
            )

            ax.tick_params(
                axis='both',
                which='major',
                labelsize=14
            )

            # ============================================================
            # Grid
            # ============================================================

            ax.grid(
                True,
                which='major',
                linestyle='--',
                linewidth=0.6,
                color='lightgrey'
            )

            ax.minorticks_on()

            ax.grid(
                True,
                which='minor',
                linestyle=':',
                linewidth=0.4,
                color='0.85'
            )

            # ============================================================
            # Panel labels
            # ============================================================

            ax.text(
                0.02,
                1.03,
                panel_letters[
                    local_idx
                ],
                transform=ax.transAxes,
                fontsize=20,
                fontstyle='italic',
                ha='left',
                va='bottom'
            )

        # ================================================================
        # Handle unused subplot(s)
        # ================================================================

        remaining_axes = (
                len(axes)
                - n_panels
        )

        # ------------------------------------------------------------
        # Exactly one empty subplot:
        # use it for the legend
        # ------------------------------------------------------------

        if remaining_axes == 1:

            legend_ax = axes[-1]

            legend_ax.axis(
                'off'
            )

            handles, labels = (
                axes[0]
                .get_legend_handles_labels()
            )

            # Remove duplicate labels
            unique = {}

            for handle, label in zip(
                    handles,
                    labels
            ):

                if label not in unique:
                    unique[label] = (
                        handle
                    )

            legend_ax.legend(
                unique.values(),
                unique.keys(),
                loc='center',
                fontsize=16,
                frameon=False
            )

        # ------------------------------------------------------------
        # More than one empty subplot
        # ------------------------------------------------------------

        elif remaining_axes > 1:

            for extra_idx in range(
                    n_panels,
                    len(axes)
            ):
                fig.delaxes(
                    axes[
                        extra_idx
                    ]
                )

            handles, labels = (
                axes[0]
                .get_legend_handles_labels()
            )

            unique = {}

            for handle, label in zip(
                    handles,
                    labels
            ):

                if label not in unique:
                    unique[label] = (
                        handle
                    )

            fig.legend(
                unique.values(),
                unique.keys(),
                loc='lower center',
                bbox_to_anchor=(
                    0.5,
                    0.01
                ),
                ncol=len(
                    unique
                ),
                fontsize=14,
                frameon=False
            )

        # ================================================================
        # Overall title
        # ================================================================

        if show_title:

            fig.suptitle(
                f"Comparison of posterior distributions: "
                f"{model_names[0]} vs "
                f"{model_names[1]}",
                fontsize=18,
                y=0.995
            )

            fig.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.97
                ]
            )

        else:

            fig.tight_layout()

        # ================================================================
        # Output filename
        # ================================================================

        if filename is None:
            safe_model_1 = str(
                model_names[0]
            ).replace(
                ' ',
                '_'
            )

            safe_model_2 = str(
                model_names[1]
            ).replace(
                ' ',
                '_'
            )

            filename = (
                f"posterior_histogram_comparison_"
                f"{safe_model_1}_vs_"
                f"{safe_model_2}_"
                f"iter_{iteration_1 + 1}_vs_"
                f"{iteration_2 + 1}.svg"
            )

        output_path = (
                save_folder
                / filename
        )

        # ================================================================
        # Save
        # ================================================================

        fig.savefig(
            output_path,
            format='svg',
            bbox_inches='tight',
            transparent=True
        )

        plt.close(
            fig
        )

        # ================================================================
        # Return metadata
        # ================================================================

        return {
            'file_path':
                output_path,

            'model_names':
                tuple(
                    model_names
                ),

            'iteration_1':
                iteration_1,

            'iteration_2':
                iteration_2,

            'selected_parameter_indices':
                selected_indices,

            'selected_parameter_names':
                [
                    parameter_names[i]
                    for i in selected_indices
                ],

            'n_bins':
                n_bins,

            'plot_prior':
                plot_prior,

            'normalize_density':
                normalize_density,

            'normalization':
                (
                    'posterior_independent_peak_'
                    'prior_relative_to_largest_posterior_peak'
                    if normalize_density
                    else 'probability_density'
                )
        }