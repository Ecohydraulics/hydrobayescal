#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import os
from itertools import combinations, cycle
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
# Load the mplstyle
plt.style.use(os.path.join(os.path.split(__file__)[0],
                           '../', 'bayesvalidrox.mplstyle'))


class PostProcessing:
    """
    This class provides many helper functions to post-process the trained
    meta-model.

    Attributes
    ----------
    MetaModel : obj
        MetaModel object to do postprocessing on.
    name : str
        Type of the anaylsis. The default is `'calib'`. If a validation is
        expected to be performed change this to `'valid'`.
    """

    def __init__(self, engine, name='calib'):
        self.engine = engine
        self.MetaModel = engine.MetaModel
        self.ExpDesign = engine.ExpDesign
        self.ModelObj = engine.Model
        self.name = name

    # -------------------------------------------------------------------------
    def plot_moments(self, xlabel='Time [s]', plot_type=None):
        """
        Plots the moments in a pdf format in the directory
        `Outputs_PostProcessing`.

        Parameters
        ----------
        xlabel : str, optional
            String to be displayed as x-label. The default is `'Time [s]'`.
        plot_type : str, optional
            Options: bar or line. The default is `None`.

        Returns
        -------
        pce_means: dict
            Mean of the model outputs.
        pce_means: dict
            Standard deviation of the model outputs.

        """

        bar_plot = True if plot_type == 'bar' else False
        meta_model_type = self.MetaModel.meta_model_type
        Model = self.ModelObj

        # Read Monte-Carlo reference
        self.mc_reference = Model.read_observation('mc_ref')

        # Set the x values
        x_values_orig = self.engine.ExpDesign.x_values

        # Compute the moments with the PCEModel object
        self.pce_means, self.pce_stds = self.compute_pce_moments()

        # Get the variables
        out_names = Model.Output.names

        # Open a pdf for the plots
        newpath = (f'Outputs_PostProcessing_{self.name}/')
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # Plot the best fit line, set the linewidth (lw), color and
        # transparency (alpha) of the line
        for key in out_names:
            fig, ax = plt.subplots(nrows=1, ncols=2)

            # Extract mean and std
            mean_data = self.pce_means[key]
            std_data = self.pce_stds[key]

            # Extract a list of x values
            if type(x_values_orig) is dict:
                x = x_values_orig[key]
            else:
                x = x_values_orig

            # Plot: bar plot or line plot
            if bar_plot:
                ax[0].bar(list(map(str, x)), mean_data, color='b',
                          width=0.25)
                ax[1].bar(list(map(str, x)), std_data, color='b',
                          width=0.25)
                ax[0].legend(labels=[meta_model_type])
                ax[1].legend(labels=[meta_model_type])
            else:
                ax[0].plot(x, mean_data, lw=3, color='k', marker='x',
                           label=meta_model_type)
                ax[1].plot(x, std_data, lw=3, color='k', marker='x',
                           label=meta_model_type)

            if self.mc_reference is not None:
                if bar_plot:
                    ax[0].bar(list(map(str, x)), self.mc_reference['mean'],
                              color='r', width=0.25)
                    ax[1].bar(list(map(str, x)), self.mc_reference['std'],
                              color='r', width=0.25)
                    ax[0].legend(labels=[meta_model_type])
                    ax[1].legend(labels=[meta_model_type])
                else:
                    ax[0].plot(x, self.mc_reference['mean'], lw=3, marker='x',
                               color='r', label='Ref.')
                    ax[1].plot(x, self.mc_reference['std'], lw=3, marker='x',
                               color='r', label='Ref.')

            # Label the axes and provide a title
            ax[0].set_xlabel(xlabel)
            ax[1].set_xlabel(xlabel)
            ax[0].set_ylabel(key)
            ax[1].set_ylabel(key)

            # Provide a title
            ax[0].set_title('Mean of ' + key)
            ax[1].set_title('Std of ' + key)

            if not bar_plot:
                ax[0].legend(loc='best')
                ax[1].legend(loc='best')

            plt.tight_layout()

            # save the current figure
            fig.savefig(
                f'./{newpath}Mean_Std_PCE_{key}.pdf',
                bbox_inches='tight'
                )

        return self.pce_means, self.pce_stds

    # -------------------------------------------------------------------------
    def valid_metamodel(self, n_samples=1, samples=None, model_out_dict=None,
                        x_axis='Time [s]'):
        """
        Evaluates and plots the meta model and the PCEModel outputs for the
        given number of samples or the given samples.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to be evaluated. The default is 1.
        samples : array of shape (n_samples, n_params), optional
            Samples to be evaluated. The default is None.
        model_out_dict: dict
            The model runs using the samples provided.
        x_axis : str, optional
            Label of x axis. The default is `'Time [s]'`.

        Returns
        -------
        None.

        """
        MetaModel = self.MetaModel
        Model = self.ModelObj

        if samples is None:
            self.n_samples = n_samples
            samples = self._get_sample()
        else:
            self.n_samples = samples.shape[0]

        # Extract x_values
        x_values = self.engine.ExpDesign.x_values

        if model_out_dict is not None:
            self.model_out_dict = model_out_dict
        else:
            self.model_out_dict = self._eval_model(samples, key_str='valid')
        self.pce_out_mean, self.pce_out_std = MetaModel.eval_metamodel(samples)

        try:
            key = Model.Output.names[1]
        except IndexError:
            key = Model.Output.names[0]

        n_obs = self.model_out_dict[key].shape[1]

        if n_obs == 1:
            self._plot_validation()
        else:
            self._plot_validation_multi(x_values=x_values, x_axis=x_axis)

    # -------------------------------------------------------------------------
    def check_accuracy(self, n_samples=None, samples=None, outputs=None):
        """
        Checks accuracy of the metamodel by computing the root mean square
        error and validation error for all outputs.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples. The default is None.
        samples : array of shape (n_samples, n_params), optional
            Parameter sets to be checked. The default is None.
        outputs : dict, optional
            Output dictionary with model outputs for all given output types in
            `Model.Output.names`. The default is None.

        Raises
        ------
        Exception
            When neither n_samples nor samples are provided.

        Returns
        -------
        rmse: dict
            Root mean squared error for each output.
        valid_error : dict
            Validation error for each output.

        """
        MetaModel = self.MetaModel
        Model = self.ModelObj

        # Set the number of samples
        if n_samples:
            self.n_samples = n_samples
        elif samples is not None:
            self.n_samples = samples.shape[0]
        else:
            raise Exception("Please provide either samples or pass the number"
                            " of samples!")

        # Generate random samples if necessary
        Samples = self._get_sample() if samples is None else samples

        # Run the original model with the generated samples
        if outputs is None:
            outputs = self._eval_model(Samples, key_str='validSet')

        # Run the PCE model with the generated samples
        pce_outputs, _ = MetaModel.eval_metamodel(samples=Samples)

        self.rmse = {}
        self.valid_error = {}
        # Loop over the keys and compute RMSE error.
        for key in Model.Output.names:
            # Root mena square
            self.rmse[key] = mean_squared_error(outputs[key], pce_outputs[key],
                                                squared=False,
                                                multioutput='raw_values')
            # Validation error
            self.valid_error[key] = (self.rmse[key]**2) / \
                np.var(outputs[key], ddof=1, axis=0)

            # Print a report table
            print("\n>>>>> Errors of {} <<<<<".format(key))
            print("\nIndex  |  RMSE   |  Validation Error")
            print('-'*35)
            print('\n'.join(f'{i+1}  |  {k:.3e}  |  {j:.3e}' for i, (k, j)
                            in enumerate(zip(self.rmse[key],
                                             self.valid_error[key]))))
        # Save error dicts in PCEModel object
        self.MetaModel.rmse = self.rmse
        self.MetaModel.valid_error = self.valid_error

        return

    # -------------------------------------------------------------------------
    def plot_seq_design_diagnostics(self, ref_BME_KLD=None):
        """
        Plots the Bayesian Model Evidence (BME) and Kullback-Leibler divergence
        (KLD) for the sequential design.

        Parameters
        ----------
        ref_BME_KLD : array, optional
            Reference BME and KLD . The default is `None`.

        Returns
        -------
        None.

        """
        engine = self.engine
        PCEModel = self.MetaModel
        n_init_samples = engine.ExpDesign.n_init_samples
        n_total_samples = engine.ExpDesign.X.shape[0]

        newpath = f'Outputs_PostProcessing_{self.name}/seq_design_diagnostics/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        plotList = ['Modified LOO error', 'Validation error', 'KLD', 'BME',
                    'RMSEMean', 'RMSEStd', 'Hellinger distance']
        seqList = [engine.SeqModifiedLOO, engine.seqValidError,
                   engine.SeqKLD, engine.SeqBME, engine.seqRMSEMean,
                   engine.seqRMSEStd, engine.SeqDistHellinger]

        markers = ('x', 'o', 'd', '*', '+')
        colors = ('k', 'darkgreen', 'b', 'navy', 'darkred')

        # Plot the evolution of the diagnostic criteria of the
        # Sequential Experimental Design.
        for plotidx, plot in enumerate(plotList):
            fig, ax = plt.subplots()
            seq_dict = seqList[plotidx]
            name_util = list(seq_dict.keys())

            if len(name_util) == 0:
                continue

            # Box plot when Replications have been detected.
            if any(int(name.split("rep_", 1)[1]) > 1 for name in name_util):
                # Extract the values from dict
                sorted_seq_opt = {}
                # Number of replications
                n_reps = engine.ExpDesign.n_replication

                # Get the list of utility function names
                # Handle if only one UtilityFunction is provided
                if not isinstance(engine.ExpDesign.util_func, list):
                    util_funcs = [engine.ExpDesign.util_func]
                else:
                    util_funcs = engine.ExpDesign.util_func

                for util in util_funcs:
                    sortedSeq = {}
                    # min number of runs available from reps
                    n_runs = min([seq_dict[f'{util}_rep_{i+1}'].shape[0]
                                 for i in range(n_reps)])

                    for runIdx in range(n_runs):
                        values = []
                        for key in seq_dict.keys():
                            if util in key:
                                values.append(seq_dict[key][runIdx].mean())
                        sortedSeq['SeqItr_'+str(runIdx)] = np.array(values)
                    sorted_seq_opt[util] = sortedSeq

                # BoxPlot
                def draw_plot(data, labels, edge_color, fill_color, idx):
                    pos = labels - (idx-1)
                    bp = plt.boxplot(data, positions=pos, labels=labels,
                                     patch_artist=True, sym='', widths=0.75)
                    elements = ['boxes', 'whiskers', 'fliers', 'means',
                                'medians', 'caps']
                    for element in elements:
                        plt.setp(bp[element], color=edge_color[idx])

                    for patch in bp['boxes']:
                        patch.set(facecolor=fill_color[idx])

                if engine.ExpDesign.n_new_samples != 1:
                    step1 = engine.ExpDesign.n_new_samples
                    step2 = 1
                else:
                    step1 = 5
                    step2 = 5
                edge_color = ['red', 'blue', 'green']
                fill_color = ['tan', 'cyan', 'lightgreen']
                plot_label = plot
                # Plot for different Utility Functions
                for idx, util in enumerate(util_funcs):
                    all_errors = np.empty((n_reps, 0))

                    for key in list(sorted_seq_opt[util].keys()):
                        errors = sorted_seq_opt.get(util, {}).get(key)[:, None]
                        all_errors = np.hstack((all_errors, errors))

                    # Special cases for BME and KLD
                    if plot == 'KLD' or plot == 'BME':
                        # BME convergence if refBME is provided
                        if ref_BME_KLD is not None:
                            if plot == 'BME':
                                refValue = ref_BME_KLD[0]
                                plot_label = r'BME/BME$^{Ref.}$'
                            if plot == 'KLD':
                                refValue = ref_BME_KLD[1]
                                plot_label = '$D_{KL}[p(\\theta|y_*),p(\\theta)]'\
                                    ' / D_{KL}^{Ref.}[p(\\theta|y_*), '\
                                    'p(\\theta)]$'

                            # Difference between BME/KLD and the ref. values
                            all_errors = np.divide(all_errors,
                                                   np.full((all_errors.shape),
                                                           refValue))

                            # Plot baseline for zero, i.e. no difference
                            plt.axhline(y=1.0, xmin=0, xmax=1, c='green',
                                        ls='--', lw=2)

                    # Plot each UtilFuncs
                    labels = np.arange(n_init_samples, n_total_samples+1, step1)
                    draw_plot(all_errors[:, ::step2], labels, edge_color,
                              fill_color, idx)

                plt.xticks(labels, labels)
                # Set the major and minor locators
                ax.xaxis.set_major_locator(ticker.AutoLocator())
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.xaxis.grid(True, which='major', linestyle='-')
                ax.xaxis.grid(True, which='minor', linestyle='--')

                # Legend
                legend_elements = []
                for idx, util in enumerate(util_funcs):
                    legend_elements.append(Patch(facecolor=fill_color[idx],
                                                 edgecolor=edge_color[idx],
                                                 label=util))
                plt.legend(handles=legend_elements[::-1], loc='best')

                if plot != 'BME' and plot != 'KLD':
                    plt.yscale('log')
                plt.autoscale(True)
                plt.xlabel('\\# of training samples')
                plt.ylabel(plot_label)
                plt.title(plot)

                # save the current figure
                plot_name = plot.replace(' ', '_')
                fig.savefig(
                    f'./{newpath}/seq_{plot_name}.pdf',
                    bbox_inches='tight'
                    )
                # Destroy the current plot
                plt.clf()
                # Save arrays into files
                f = open(f'./{newpath}/seq_{plot_name}.txt', 'w')
                f.write(str(sorted_seq_opt))
                f.close()
            else:
                for idx, name in enumerate(name_util):
                    seq_values = seq_dict[name]
                    if engine.ExpDesign.n_new_samples != 1:
                        step = engine.ExpDesign.n_new_samples
                    else:
                        step = 1
                    x_idx = np.arange(n_init_samples, n_total_samples+1, step)
                    if n_total_samples not in x_idx:
                        x_idx = np.hstack((x_idx, n_total_samples))

                    if plot == 'KLD' or plot == 'BME':
                        # BME convergence if refBME is provided
                        if ref_BME_KLD is not None:
                            if plot == 'BME':
                                refValue = ref_BME_KLD[0]
                                plot_label = r'BME/BME$^{Ref.}$'
                            if plot == 'KLD':
                                refValue = ref_BME_KLD[1]
                                plot_label = '$D_{KL}[p(\\theta|y_*),p(\\theta)]'\
                                    ' / D_{KL}^{Ref.}[p(\\theta|y_*), '\
                                    'p(\\theta)]$'

                            # Difference between BME/KLD and the ref. values
                            values = np.divide(seq_values,
                                               np.full((seq_values.shape),
                                                       refValue))

                            # Plot baseline for zero, i.e. no difference
                            plt.axhline(y=1.0, xmin=0, xmax=1, c='green',
                                        ls='--', lw=2)

                            # Set the limits
                            plt.ylim([1e-1, 1e1])

                            # Create the plots
                            plt.semilogy(x_idx, values, marker=markers[idx],
                                         color=colors[idx], ls='--', lw=2,
                                         label=name.split("_rep", 1)[0])
                        else:
                            plot_label = plot

                            # Create the plots
                            plt.plot(x_idx, seq_values, marker=markers[idx],
                                     color=colors[idx], ls='--', lw=2,
                                     label=name.split("_rep", 1)[0])

                    else:
                        plot_label = plot
                        seq_values = np.nan_to_num(seq_values)

                        # Plot the error evolution for each output
                        plt.semilogy(x_idx, seq_values.mean(axis=1),
                                     marker=markers[idx], ls='--', lw=2,
                                     color=colors[idx],
                                     label=name.split("_rep", 1)[0])

                # Set the major and minor locators
                ax.xaxis.set_major_locator(ticker.AutoLocator())
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.xaxis.grid(True, which='major', linestyle='-')
                ax.xaxis.grid(True, which='minor', linestyle='--')

                ax.tick_params(axis='both', which='major', direction='in',
                               width=3, length=10)
                ax.tick_params(axis='both', which='minor', direction='in',
                               width=2, length=8)
                plt.xlabel('Number of runs')
                plt.ylabel(plot_label)
                plt.title(plot)
                plt.legend(frameon=True)

                # save the current figure
                plot_name = plot.replace(' ', '_')
                fig.savefig(
                    f'./{newpath}/seq_{plot_name}.pdf',
                    bbox_inches='tight'
                    )
                # Destroy the current plot
                plt.clf()

                # ---------------- Saving arrays into files ---------------
                np.save(f'./{newpath}/seq_{plot_name}.npy', seq_values)

        return

    # -------------------------------------------------------------------------
    def sobol_indices(self, xlabel='Time [s]', plot_type=None):
        """
        Provides Sobol indices as a sensitivity measure to infer the importance
        of the input parameters. See Eq. 27 in [1] for more details. For the
        case with Principal component analysis refer to [2].

        [1] Global sensitivity analysis: A flexible and efficient framework
        with an example from stochastic hydrogeology S. Oladyshkin, F.P.
        de Barros, W. Nowak  https://doi.org/10.1016/j.advwatres.2011.11.001

        [2] Nagel, J.B., Rieckermann, J. and Sudret, B., 2020. Principal
        component analysis and sparse polynomial chaos expansions for global
        sensitivity analysis and model calibration: Application to urban
        drainage simulation. Reliability Engineering & System Safety, 195,
        p.106737.

        Parameters
        ----------
        xlabel : str, optional
            Label of the x-axis. The default is `'Time [s]'`.
        plot_type : str, optional
            Plot type. The default is `None`. This corresponds to line plot.
            Bar chart can be selected by `bar`.

        Returns
        -------
        sobol_cell: dict
            Sobol indices.
        total_sobol: dict
            Total Sobol indices.

        """
        # Extract the necessary variables
        PCEModel = self.MetaModel
        basis_dict = PCEModel.basis_dict
        coeffs_dict = PCEModel.coeffs_dict
        n_params = PCEModel.n_params
        max_order = np.max(PCEModel.pce_deg)
        sobol_cell_b = {}
        total_sobol_b = {}
        cov_Z_p_q = np.zeros((n_params))

        for b_i in range(PCEModel.n_bootstrap_itrs):

            sobol_cell_, total_sobol_ = {}, {}

            for output in self.ModelObj.Output.names:

                n_meas_points = len(coeffs_dict[f'b_{b_i+1}'][output])

                # Initialize the (cell) array containing the (total) Sobol indices.
                sobol_array = dict.fromkeys(range(1, max_order+1), [])
                sobol_cell_array = dict.fromkeys(range(1, max_order+1), [])

                for i_order in range(1, max_order+1):
                    n_comb = math.comb(n_params, i_order)

                    sobol_cell_array[i_order] = np.zeros((n_comb, n_meas_points))

                total_sobol_array = np.zeros((n_params, n_meas_points))

                # Initialize the cell to store the names of the variables
                TotalVariance = np.zeros((n_meas_points))
                # Loop over all measurement points and calculate sobol indices
                for pIdx in range(n_meas_points):

                    # Extract the basis indices (alpha) and coefficients
                    Basis = basis_dict[f'b_{b_i+1}'][output][f'y_{pIdx+1}']

                    try:
                        clf_poly = PCEModel.clf_poly[f'b_{b_i+1}'][output][f'y_{pIdx+1}']
                        PCECoeffs = clf_poly.coef_
                    except:
                        PCECoeffs = coeffs_dict[f'b_{b_i+1}'][output][f'y_{pIdx+1}']

                    # Compute total variance
                    TotalVariance[pIdx] = np.sum(np.square(PCECoeffs[1:]))

                    nzidx = np.where(PCECoeffs != 0)[0]
                    # Set all the Sobol indices equal to zero in the presence of a
                    # null output.
                    if len(nzidx) == 0:
                        # This is buggy.
                        for i_order in range(1, max_order+1):
                            sobol_cell_array[i_order][:, pIdx] = 0

                    # Otherwise compute them by summing well-chosen coefficients
                    else:
                        nz_basis = Basis[nzidx]
                        for i_order in range(1, max_order+1):
                            idx = np.where(np.sum(nz_basis > 0, axis=1) == i_order)
                            subbasis = nz_basis[idx]
                            Z = np.array(list(combinations(range(n_params), i_order)))

                            for q in range(Z.shape[0]):
                                Zq = Z[q]
                                subsubbasis = subbasis[:, Zq]
                                subidx = np.prod(subsubbasis, axis=1) > 0
                                sum_ind = nzidx[idx[0][subidx]]
                                if TotalVariance[pIdx] == 0.0:
                                    sobol_cell_array[i_order][q, pIdx] = 0.0
                                else:
                                    sobol = np.sum(np.square(PCECoeffs[sum_ind]))
                                    sobol /= TotalVariance[pIdx]
                                    sobol_cell_array[i_order][q, pIdx] = sobol

                        # Compute the TOTAL Sobol indices.
                        for ParIdx in range(n_params):
                            idx = nz_basis[:, ParIdx] > 0
                            sum_ind = nzidx[idx]

                            if TotalVariance[pIdx] == 0.0:
                                total_sobol_array[ParIdx, pIdx] = 0.0
                            else:
                                sobol = np.sum(np.square(PCECoeffs[sum_ind]))
                                sobol /= TotalVariance[pIdx]
                                total_sobol_array[ParIdx, pIdx] = sobol

                    # ----- if PCA selected: Compute covariance -----
                    if PCEModel.dim_red_method.lower() == 'pca':
                        # Extract the basis indices (alpha) and coefficients for
                        # next component
                        if pIdx < n_meas_points-1:
                            nextBasis = basis_dict[f'b_{b_i+1}'][output][f'y_{pIdx+2}']
                            if PCEModel.bootstrap_method != 'fast' or b_i == 0:
                                clf_poly = PCEModel.clf_poly[f'b_{b_i+1}'][output][f'y_{pIdx+2}']
                                nextPCECoeffs = clf_poly.coef_
                            else:
                                nextPCECoeffs = coeffs_dict[f'b_{b_i+1}'][output][f'y_{pIdx+2}']

                            # Choose the common non-zero basis
                            mask = (Basis[:, None] == nextBasis).all(-1).any(-1)
                            n_mask = (nextBasis[:, None] == Basis).all(-1).any(-1)

                            # Compute the covariance in Eq 17.
                            for ParIdx in range(n_params):
                                idx = (mask) & (Basis[:, ParIdx] > 0)
                                n_idx = (n_mask) & (nextBasis[:, ParIdx] > 0)
                                try:
                                    cov_Z_p_q[ParIdx] += np.sum(np.dot(
                                        PCECoeffs[idx], nextPCECoeffs[n_idx])
                                        )
                                except:
                                    pass

                # Compute the sobol indices according to Ref. 2
                if PCEModel.dim_red_method.lower() == 'pca':
                    n_c_points = self.engine.ExpDesign.Y[output].shape[1]
                    PCA = PCEModel.pca[f'b_{b_i+1}'][output]
                    compPCA = PCA.components_
                    nComp = compPCA.shape[0]
                    var_Z_p = PCA.explained_variance_

                    # Extract the sobol index of the components
                    for i_order in range(1, max_order+1):
                        n_comb = math.comb(n_params, i_order)
                        sobol_array[i_order] = np.zeros((n_comb, n_c_points))
                        Z = np.array(list(combinations(range(n_params), i_order)))

                        # Loop over parameters
                        for q in range(Z.shape[0]):
                            S_Z_i = sobol_cell_array[i_order][q]

                            for tIdx in range(n_c_points):
                                var_Y_t = np.var(
                                    self.engine.ExpDesign.Y[output][:, tIdx])
                                if var_Y_t == 0.0:
                                    term1, term2 = 0.0, 0.0
                                else:
                                    # Eq. 17
                                    term1 = 0.0
                                    for i in range(nComp):
                                        a = S_Z_i[i] * var_Z_p[i]
                                        a *= compPCA[i, tIdx]**2
                                        term1 += a

                                    # TODO: Term 2
                                    # term2 = 0.0
                                    # for i in range(nComp-1):
                                    #     term2 += cov_Z_p_q[q] * compPCA[i, tIdx]
                                    #     term2 *= compPCA[i+1, tIdx]
                                    # term2 *= 2

                                sobol_array[i_order][q, tIdx] = term1 #+ term2

                                # Devide over total output variance Eq. 18
                                sobol_array[i_order][q, tIdx] /= var_Y_t

                    # Compute the TOTAL Sobol indices.
                    total_sobol = np.zeros((n_params, n_c_points))
                    for ParIdx in range(n_params):
                        S_Z_i = total_sobol_array[ParIdx]

                        for tIdx in range(n_c_points):
                            var_Y_t = np.var(self.engine.ExpDesign.Y[output][:, tIdx])
                            if var_Y_t == 0.0:
                                term1, term2 = 0.0, 0.0
                            else:
                                term1 = 0
                                for i in range(nComp):
                                    term1 += S_Z_i[i] * var_Z_p[i] * \
                                        (compPCA[i, tIdx]**2)

                                # Term 2
                                term2 = 0
                                for i in range(nComp-1):
                                    term2 += cov_Z_p_q[ParIdx] * compPCA[i, tIdx] \
                                        * compPCA[i+1, tIdx]
                                term2 *= 2

                            total_sobol[ParIdx, tIdx] = term1 #+ term2

                            # Devide over total output variance Eq. 18
                            total_sobol[ParIdx, tIdx] /= var_Y_t

                    sobol_cell_[output] = sobol_array
                    total_sobol_[output] = total_sobol
                else:
                    sobol_cell_[output] = sobol_cell_array
                    total_sobol_[output] = total_sobol_array

            # Save for each bootsrtap iteration
            sobol_cell_b[b_i] = sobol_cell_
            total_sobol_b[b_i] = total_sobol_

        # Average total sobol indices
        total_sobol_all = {}
        for i in sorted(total_sobol_b):
            for k, v in total_sobol_b[i].items():
                if k not in total_sobol_all:
                    total_sobol_all[k] = [None] * len(total_sobol_b)
                total_sobol_all[k][i] = v

        self.total_sobol = {}
        for output in self.ModelObj.Output.names:
            self.total_sobol[output] = np.mean(total_sobol_all[output], axis=0)

        # ---------------- Plot -----------------------
        par_names = self.engine.ExpDesign.par_names
        x_values_orig = self.engine.ExpDesign.x_values

        newpath = (f'Outputs_PostProcessing_{self.name}/')
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        fig = plt.figure()

        for outIdx, output in enumerate(self.ModelObj.Output.names):

            # Extract total Sobol indices
            total_sobol = self.total_sobol[output]

            # Compute quantiles
            q_5 = np.quantile(total_sobol_all[output], q=0.05, axis=0)
            q_97_5 = np.quantile(total_sobol_all[output], q=0.975, axis=0)

            # Extract a list of x values
            if type(x_values_orig) is dict:
                x = x_values_orig[output]
            else:
                x = x_values_orig

            if plot_type == 'bar':
                ax = fig.add_axes([0, 0, 1, 1])
                dict1 = {xlabel: x}
                dict2 = {param: sobolIndices for param, sobolIndices
                         in zip(par_names, total_sobol)}

                df = pd.DataFrame({**dict1, **dict2})
                df.plot(x=xlabel, y=par_names, kind="bar", ax=ax, rot=0,
                        colormap='Dark2', yerr=q_97_5-q_5)
                ax.set_ylabel('Total Sobol indices, $S^T$')

            else:
                for i, sobolIndices in enumerate(total_sobol):
                    plt.plot(x, sobolIndices, label=par_names[i],
                             marker='x', lw=2.5)
                    plt.fill_between(x, q_5[i], q_97_5[i], alpha=0.15)

                plt.ylabel('Total Sobol indices, $S^T$')
                plt.xlabel(xlabel)

            plt.title(f'Sensitivity analysis of {output}')
            if plot_type != 'bar':
                plt.legend(loc='best', frameon=True)

            # Save indices
            np.savetxt(f'./{newpath}totalsobol_' +
                       output.replace('/', '_') + '.csv',
                       total_sobol.T, delimiter=',',
                       header=','.join(par_names), comments='')

            # save the current figure
            fig.savefig(
                f'./{newpath}Sobol_indices_{output}.pdf',
                bbox_inches='tight'
                )

            # Destroy the current plot
            plt.clf()

        return self.total_sobol

    # -------------------------------------------------------------------------
    def check_reg_quality(self, n_samples=1000, samples=None):
        """
        Checks the quality of the metamodel for single output models based on:
        https://towardsdatascience.com/how-do-you-check-the-quality-of-your-regression-model-in-python-fa61759ff685


        Parameters
        ----------
        n_samples : int, optional
            Number of parameter sets to use for the check. The default is 1000.
        samples : array of shape (n_samples, n_params), optional
            Parameter sets to use for the check. The default is None.

        Returns
        -------
        None.

        """
        MetaModel = self.MetaModel

        if samples is None:
            self.n_samples = n_samples
            samples = self._get_sample()
        else:
            self.n_samples = samples.shape[0]

        # Evaluate the original and the surrogate model
        y_val = self._eval_model(samples, key_str='valid')
        y_pce_val, _ = MetaModel.eval_metamodel(samples=samples)

        # Open a pdf for the plots
        newpath = f'Outputs_PostProcessing_{self.name}/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # Fit the data(train the model)
        for key in y_pce_val.keys():

            y_pce_val_ = y_pce_val[key]
            y_val_ = y_val[key]
            residuals = y_val_ - y_pce_val_

            # ------ Residuals vs. predicting variables ------
            # Check the assumptions of linearity and independence
            fig1 = plt.figure()
            for i, par in enumerate(self.engine.ExpDesign.par_names):
                plt.title(f"{key}: Residuals vs. {par}")
                plt.scatter(
                    x=samples[:, i], y=residuals, color='blue', edgecolor='k')
                plt.grid(True)
                xmin, xmax = min(samples[:, i]), max(samples[:, i])
                plt.hlines(y=0, xmin=xmin*0.9, xmax=xmax*1.1, color='red',
                           lw=3, linestyle='--')
                plt.xlabel(par)
                plt.ylabel('Residuals')
                plt.show()

                # save the current figure
                fig1.savefig(f'./{newpath}/Residuals_vs_Par_{i+1}.pdf',
                             bbox_inches='tight')
                # Destroy the current plot
                plt.clf()

            # ------ Fitted vs. residuals ------
            # Check the assumptions of linearity and independence
            fig2 = plt.figure()
            plt.title(f"{key}: Residuals vs. fitted values")
            plt.scatter(x=y_pce_val_, y=residuals, color='blue', edgecolor='k')
            plt.grid(True)
            xmin, xmax = min(y_val_), max(y_val_)
            plt.hlines(y=0, xmin=xmin*0.9, xmax=xmax*1.1, color='red', lw=3,
                       linestyle='--')
            plt.xlabel(key)
            plt.ylabel('Residuals')
            plt.show()

            # save the current figure
            fig2.savefig(f'./{newpath}/Fitted_vs_Residuals.pdf',
                         bbox_inches='tight')
            # Destroy the current plot
            plt.clf()

            # ------ Histogram of normalized residuals ------
            fig3 = plt.figure()
            resid_pearson = residuals / (max(residuals)-min(residuals))
            plt.hist(resid_pearson, bins=20, edgecolor='k')
            plt.ylabel('Count')
            plt.xlabel('Normalized residuals')
            plt.title(f"{key}: Histogram of normalized residuals")

            # Normality (Shapiro-Wilk) test of the residuals
            ax = plt.gca()
            _, p = stats.shapiro(residuals)
            if p < 0.01:
                annText = "The residuals seem to come from a Gaussian Process."
            else:
                annText = "The normality assumption may not hold."
            at = AnchoredText(annText, prop=dict(size=30), frameon=True,
                              loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

            plt.show()

            # save the current figure
            fig3.savefig(f'./{newpath}/Hist_NormResiduals.pdf',
                         bbox_inches='tight')
            # Destroy the current plot
            plt.clf()

            # ------ Q-Q plot of the normalized residuals ------
            plt.figure()
            stats.probplot(residuals[:, 0], plot=plt)
            plt.xticks()
            plt.yticks()
            plt.xlabel("Theoretical quantiles")
            plt.ylabel("Sample quantiles")
            plt.title(f"{key}: Q-Q plot of normalized residuals")
            plt.grid(True)
            plt.show()

            # save the current figure
            plt.savefig(f'./{newpath}/QQPlot_NormResiduals.pdf',
                        bbox_inches='tight')
            # Destroy the current plot
            plt.clf()

    # -------------------------------------------------------------------------
    def eval_pce_model_3d(self):

        self.n_samples = 1000

        PCEModel = self.MetaModel
        Model = self.ModelObj
        n_samples = self.n_samples

        # Create 3D-Grid
        # TODO: Make it general
        x = np.linspace(-5, 10, n_samples)
        y = np.linspace(0, 15, n_samples)

        X, Y = np.meshgrid(x, y)
        PCE_Z = np.zeros((self.n_samples, self.n_samples))
        Model_Z = np.zeros((self.n_samples, self.n_samples))

        for idxMesh in range(self.n_samples):
            sample_mesh = np.vstack((X[:, idxMesh], Y[:, idxMesh])).T

            univ_p_val = PCEModel.univ_basis_vals(sample_mesh)

            for Outkey, ValuesDict in PCEModel.coeffs_dict.items():

                pce_out_mean = np.zeros((len(sample_mesh), len(ValuesDict)))
                pce_out_std = np.zeros((len(sample_mesh), len(ValuesDict)))
                model_outs = np.zeros((len(sample_mesh), len(ValuesDict)))

                for Inkey, InIdxValues in ValuesDict.items():
                    idx = int(Inkey.split('_')[1]) - 1
                    basis_deg_ind = PCEModel.basis_dict[Outkey][Inkey]
                    clf_poly = PCEModel.clf_poly[Outkey][Inkey]

                    PSI_Val = PCEModel.create_psi(basis_deg_ind, univ_p_val)

                    # Perdiction with error bar
                    y_mean, y_std = clf_poly.predict(PSI_Val, return_std=True)

                    pce_out_mean[:, idx] = y_mean
                    pce_out_std[:, idx] = y_std

                    # Model evaluation
                    model_out_dict, _ = Model.run_model_parallel(sample_mesh,
                                                                 key_str='Valid3D')
                    model_outs[:, idx] = model_out_dict[Outkey].T

                PCE_Z[:, idxMesh] = y_mean
                Model_Z[:, idxMesh] = model_outs[:, 0]

        # ---------------- 3D plot for PCEModel -----------------------
        fig_PCE = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, PCE_Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('PCEModel')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1,x_2)$')

        plt.grid()
        plt.show()

        #  Saving the figure
        newpath = f'Outputs_PostProcessing_{self.name}/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # save the figure to file
        fig_PCE.savefig(f'./{newpath}/3DPlot_PCEModel.pdf',
                        bbox_inches='tight')
        plt.close(fig_PCE)

        # ---------------- 3D plot for Model -----------------------
        fig_Model = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, PCE_Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('Model')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1,x_2)$')

        plt.grid()
        plt.show()

        # Save the figure
        fig_Model.savefig(f'./{newpath}/3DPlot_Model.pdf',
                          bbox_inches='tight')
        plt.close(fig_Model)

        return

    # -------------------------------------------------------------------------
    def compute_pce_moments(self):
        """
        Computes the first two moments using the PCE-based meta-model.

        Returns
        -------
        pce_means: dict
            The first moments (mean) of outpust.
        pce_means: dict
            The first moments (mean) of outpust.

        """

        MetaModel = self.MetaModel
        outputs = self.ModelObj.Output.names
        pce_means_b = {}
        pce_stds_b = {}

        # Loop over bootstrap iterations
        for b_i in range(MetaModel.n_bootstrap_itrs):
            # Loop over the metamodels
            coeffs_dicts = MetaModel.coeffs_dict[f'b_{b_i+1}'].items()
            means = {}
            stds = {}
            for output, coef_dict in coeffs_dicts:

                pce_mean = np.zeros((len(coef_dict)))
                pce_var = np.zeros((len(coef_dict)))

                for index, values in coef_dict.items():
                    idx = int(index.split('_')[1]) - 1
                    coeffs = MetaModel.coeffs_dict[f'b_{b_i+1}'][output][index]

                    # Mean = c_0
                    if coeffs[0] != 0:
                        pce_mean[idx] = coeffs[0]
                    else:
                        clf_poly = MetaModel.clf_poly[f'b_{b_i+1}'][output]
                        pce_mean[idx] = clf_poly[index].intercept_
                    # Var = sum(coeffs[1:]**2)
                    pce_var[idx] = np.sum(np.square(coeffs[1:]))

                # Save predictions for each output
                if MetaModel.dim_red_method.lower() == 'pca':
                    PCA = MetaModel.pca[f'b_{b_i+1}'][output]
                    means[output] = PCA.inverse_transform(pce_mean)
                    stds[output] = np.sqrt(np.dot(pce_var,
                                                  PCA.components_**2))
                else:
                    means[output] = pce_mean
                    stds[output] = np.sqrt(pce_var)

            # Save predictions for each bootstrap iteration
            pce_means_b[b_i] = means
            pce_stds_b[b_i] = stds

        # Change the order of nesting
        mean_all = {}
        for i in sorted(pce_means_b):
            for k, v in pce_means_b[i].items():
                if k not in mean_all:
                    mean_all[k] = [None] * len(pce_means_b)
                mean_all[k][i] = v
        std_all = {}
        for i in sorted(pce_stds_b):
            for k, v in pce_stds_b[i].items():
                if k not in std_all:
                    std_all[k] = [None] * len(pce_stds_b)
                std_all[k][i] = v

        # Back transformation if PCA is selected.
        pce_means, pce_stds = {}, {}
        for output in outputs:
            pce_means[output] = np.mean(mean_all[output], axis=0)
            pce_stds[output] = np.mean(std_all[output], axis=0)

            # Print a report table
            print("\n>>>>> Moments of {} <<<<<".format(output))
            print("\nIndex  |  Mean   |  Std. deviation")
            print('-'*35)
            print('\n'.join(f'{i+1}  |  {k:.3e}  |  {j:.3e}' for i, (k, j)
                            in enumerate(zip(pce_means[output],
                                             pce_stds[output]))))
        print('-'*40)

        return pce_means, pce_stds

    # -------------------------------------------------------------------------
    def _get_sample(self, n_samples=None):
        """
        Generates random samples taken from the input parameter space.

        Returns
        -------
        samples : array of shape (n_samples, n_params)
            Generated samples.

        """
        if n_samples is None:
            n_samples = self.n_samples
        self.samples = self.ExpDesign.generate_samples(
            n_samples,
            sampling_method='random')
        return self.samples

    # -------------------------------------------------------------------------
    def _eval_model(self, samples=None, key_str='Valid'):
        """
        Evaluates Forward Model for the given number of self.samples or given
        samples.

        Parameters
        ----------
        samples : array of shape (n_samples, n_params), optional
            Samples to evaluate the model at. The default is None.
        key_str : str, optional
            Key string pass to the model. The default is 'Valid'.

        Returns
        -------
        model_outs : dict
            Dictionary of results.

        """
        Model = self.ModelObj

        if samples is None:
            samples = self._get_sample()
            self.samples = samples
        else:
            self.n_samples = len(samples)

        model_outs, _ = Model.run_model_parallel(samples, key_str=key_str)

        return model_outs

    # -------------------------------------------------------------------------
    def _plot_validation(self):
        """
        Plots outputs for visual comparison of metamodel outputs with that of
        the (full) original model.

        Returns
        -------
        None.

        """
        PCEModel = self.MetaModel

        # get the samples
        x_val = self.samples
        y_pce_val = self.pce_out_mean
        y_val = self.model_out_dict

        # Open a pdf for the plots
        newpath = f'Outputs_PostProcessing_{self.name}/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        fig = plt.figure()
        # Fit the data(train the model)
        for key in y_pce_val.keys():

            y_pce_val_ = y_pce_val[key]
            y_val_ = y_val[key]

            regression_model = LinearRegression()
            regression_model.fit(y_pce_val_, y_val_)

            # Predict
            x_new = np.linspace(np.min(y_pce_val_), np.max(y_val_), 100)
            y_predicted = regression_model.predict(x_new[:, np.newaxis])

            plt.scatter(y_pce_val_, y_val_, color='gold', linewidth=2)
            plt.plot(x_new, y_predicted, color='k')

            # Calculate the adjusted R_squared and RMSE
            # the total number of explanatory variables in the model
            # (not including the constant term)
            length_list = []
            for key, value in PCEModel.coeffs_dict['b_1'][key].items():
                length_list.append(len(value))
            n_predictors = min(length_list)
            n_samples = x_val.shape[0]

            R2 = r2_score(y_pce_val_, y_val_)
            AdjR2 = 1 - (1 - R2) * (n_samples - 1) / \
                (n_samples - n_predictors - 1)
            rmse = mean_squared_error(y_pce_val_, y_val_, squared=False)

            plt.annotate(f'RMSE = {rmse:.3f}\n Adjusted $R^2$ = {AdjR2:.3f}',
                         xy=(0.05, 0.85), xycoords='axes fraction')

            plt.ylabel("Original Model")
            plt.xlabel("PCE Model")
            plt.grid()
            plt.show()

            # save the current figure
            plot_name = key.replace(' ', '_')
            fig.savefig(f'./{newpath}/Model_vs_PCEModel_{plot_name}.pdf',
                        bbox_inches='tight')

            # Destroy the current plot
            plt.clf()

    # -------------------------------------------------------------------------
    def _plot_validation_multi(self, x_values=[], x_axis="x [m]"):
        """
        Plots outputs for visual comparison of metamodel outputs with that of
        the (full) multioutput original model

        Parameters
        ----------
        x_values : list or array, optional
            List of x values. The default is [].
        x_axis : str, optional
            Label of the x axis. The default is "x [m]".

        Returns
        -------
        None.

        """
        Model = self.ModelObj

        newpath = f'Outputs_PostProcessing_{self.name}/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # List of markers and colors
        color = cycle((['b', 'g', 'r', 'y', 'k']))
        marker = cycle(('x', 'd', '+', 'o', '*'))

        fig = plt.figure()
        # Plot the model vs PCE model
        for keyIdx, key in enumerate(Model.Output.names):

            y_pce_val = self.pce_out_mean[key]
            y_pce_val_std = self.pce_out_std[key]
            y_val = self.model_out_dict[key]
            try:
                x = self.model_out_dict['x_values'][key]
            except (TypeError, IndexError):
                x = x_values

            for idx in range(y_val.shape[0]):
                Color = next(color)
                Marker = next(marker)

                plt.plot(x, y_val[idx], color=Color, marker=Marker,
                         label='$Y_{%s}^M$'%(idx+1))

                plt.plot(x, y_pce_val[idx], color=Color, marker=Marker,
                         linestyle='--',
                         label='$Y_{%s}^{PCE}$'%(idx+1))
                plt.fill_between(x, y_pce_val[idx]-1.96*y_pce_val_std[idx],
                                 y_pce_val[idx]+1.96*y_pce_val_std[idx],
                                 color=Color, alpha=0.15)

            # Calculate the RMSE
            rmse = mean_squared_error(y_pce_val, y_val, squared=False)
            R2 = r2_score(y_pce_val[idx].reshape(-1, 1),
                          y_val[idx].reshape(-1, 1))

            plt.annotate(f'RMSE = {rmse:.3f}\n $R^2$ = {R2:.3f}',
                         xy=(0.85, 0.1), xycoords='axes fraction')

            plt.ylabel(key)
            plt.xlabel(x_axis)
            plt.legend(loc='best')
            plt.grid()

            # save the current figure
            plot_name = key.replace(' ', '_')
            fig.savefig(f'./{newpath}/Model_vs_PCEModel_{plot_name}.pdf',
                        bbox_inches='tight')

            # Destroy the current plot
            plt.clf()

        # Zip the subdirectories
        Model.zip_subdirs(f'{Model.name}valid', f'{Model.name}valid_')
