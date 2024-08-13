#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from scipy import stats
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pylab as plt
from .bayes_inference import BayesInference

# Load the mplstyle
plt.style.use(os.path.join(os.path.split(__file__)[0],
                           '../', 'bayesvalidrox.mplstyle'))


class BayesModelComparison:
    """
    A class to perform Bayesian Analysis.


    Attributes
    ----------
    justifiability : bool, optional
        Whether to perform the justifiability analysis. The default is
        `True`.
    perturbed_data : array of shape (n_bootstrap_itrs, n_obs), optional
        User defined perturbed data. The default is `None`.
    n_bootstarp : int
        Number of bootstrap iteration. The default is `1000`.
    data_noise_level : float
        A noise level to perturb the data set. The default is `0.01`.
    just_n_meas : int
        Number of measurements considered for visualization of the
        justifiability results.

    """

    def __init__(self, justifiability=True, perturbed_data=None,
                 n_bootstarp=1000, data_noise_level=0.01, just_n_meas=2):

        self.justifiability = justifiability
        self.perturbed_data = perturbed_data
        self.n_bootstarp = n_bootstarp
        self.data_noise_level = data_noise_level
        self.just_n_meas = just_n_meas

    # --------------------------------------------------------------------------
    def create_model_comparison(self, model_dict, opts_dict):
        """
        Starts the two-stage model comparison.
        Stage I: Compare models using Bayes factors.
        Stage II: Compare models via justifiability analysis.

        Parameters
        ----------
        model_dict : dict
            A dictionary including the metamodels.
        opts_dict : dict
            A dictionary given the `BayesInference` options.

            Example:

                >>> opts_bootstrap = {
                    "bootstrap": True,
                    "n_samples": 10000,
                    "Discrepancy": DiscrepancyOpts,
                    "emulator": True,
                    "plot_post_pred": True
                    }

        Returns
        -------
        output : dict
            A dictionary containing the objects and the model weights for the
            comparison using Bayes factors and justifiability analysis.

        """

        # Bayes factor
        bayes_dict_bf, model_weights_dict_bf = self.compare_models(
            model_dict, opts_dict
            )

        output = {
            'Bayes objects BF': bayes_dict_bf,
            'Model weights BF': model_weights_dict_bf
            }

        # Justifiability analysis
        if self.justifiability:
            bayes_dict_ja, model_weights_dict_ja = self.compare_models(
                model_dict, opts_dict, justifiability=True
                )

            output['Bayes objects JA'] = bayes_dict_ja
            output['Model weights JA'] = model_weights_dict_ja

        return output

    # --------------------------------------------------------------------------
    def compare_models(self, model_dict, opts_dict, justifiability=False):
        """
        Passes the options to instantiates the BayesInference class for each
        model and passes the options from `opts_dict`. Then, it starts the
        computations.
        It also creates a folder and saves the diagrams, e.g., Bayes factor
        plot, confusion matrix, etc.

        Parameters
        ----------
        model_dict : dict
            A dictionary including the metamodels.
        opts_dict : dict
            A dictionary given the `BayesInference` options.
        justifiability : bool, optional
            Whether to perform the justifiability analysis. The default is
            `False`.

        Returns
        -------
        bayes_dict : dict
            A dictionary with `BayesInference` objects.
        model_weights_dict : dict
            A dictionary containing the model weights.

        """

        if not isinstance(model_dict, dict):
            raise Exception("To run model comparsion, you need to pass a "
                            "dictionary of models.")

        # Extract model names
        self.model_names = [*model_dict]

        # Compute total number of the measurement points
        Engine = list(model_dict.items())[0][1]
        Engine.Model.read_observation()
        self.n_meas = Engine.Model.n_obs

        # ----- Generate data -----
        # Find n_bootstrap
        if self.perturbed_data is None:
            n_bootstarp = self.n_bootstarp
        else:
            n_bootstarp = self.perturbed_data.shape[0]

        # Create dataset
        justData = self.generate_dataset(
            model_dict, justifiability, n_bootstarp=n_bootstarp)

        # Run create Interface for each model
        bayes_dict = {}
        for model in model_dict.keys():
            print("-"*20)
            print("Bayesian inference of {}.\n".format(model))

            BayesOpts = BayesInference(model_dict[model])

            # Set BayesInference options
            for key, value in opts_dict.items():
                if key in BayesOpts.__dict__.keys():
                    if key == "Discrepancy" and isinstance(value, dict):
                        setattr(BayesOpts, key, value[model])
                    else:
                        setattr(BayesOpts, key, value)

            # Pass justifiability data as perturbed data
            BayesOpts.perturbed_data = justData
            BayesOpts.just_analysis = justifiability

            bayes_dict[model] = BayesOpts.create_inference()
            print("-"*20)

        # Compute model weights
        BME_Dict = dict()
        for modelName, bayesObj in bayes_dict.items():
            BME_Dict[modelName] = np.exp(bayesObj.log_BME, dtype=np.longdouble)#float128)

        # BME correction in BayesInference class
        model_weights = self.cal_model_weight(
            BME_Dict, justifiability, n_bootstarp=n_bootstarp)

        # Plot model weights
        if justifiability:
            model_names = self.model_names
            model_names.insert(0, 'Observation')

            # Split the model weights and save in a dict
            list_ModelWeights = np.split(
                model_weights, model_weights.shape[1]/self.n_meas, axis=1)
            model_weights_dict = {key: weights for key, weights in
                                  zip(model_names, list_ModelWeights)}

            #self.plot_just_analysis(model_weights_dict)
        else:
            # Create box plot for model weights
            self.plot_model_weights(model_weights, 'model_weights')

            # Create kde plot for bayes factors
            self.plot_bayes_factor(BME_Dict, 'kde_plot')

            # Store model weights in a dict
            model_weights_dict = {key: weights for key, weights in
                                  zip(self.model_names, model_weights)}

        return bayes_dict, model_weights_dict

    # -------------------------------------------------------------------------
    def generate_dataset(self, model_dict, justifiability=False,
                         n_bootstarp=1):
        """
        Generates the perturbed data set for the Bayes factor calculations and
        the data set for the justifiability analysis.

        Parameters
        ----------
        model_dict : dict
            A dictionary including the metamodels.
        bool, optional
            Whether to perform the justifiability analysis. The default is
            `False`.
        n_bootstarp : int, optional
            Number of bootstrap iterations. The default is `1`.

        Returns
        -------
        all_just_data: array
            Created data set.

        """
        # Compute some variables
        all_just_data = []
        Engine = list(model_dict.items())[0][1]
        out_names = Engine.Model.Output.names

        # Perturb observations for Bayes Factor
        if self.perturbed_data is None:
            self.perturbed_data = self.__perturb_data(
                    Engine.Model.observations, out_names, n_bootstarp,
                    noise_level=self.data_noise_level)

        # Only for Bayes Factor
        if not justifiability:
            return self.perturbed_data

        # Evaluate metamodel
        runs = {}
        for key, metaModel in model_dict.items():
            y_hat, _ = metaModel.eval_metamodel(nsamples=n_bootstarp)
            runs[key] = y_hat

        # Generate data
        for i in range(n_bootstarp):
            y_data = self.perturbed_data[i].reshape(1, -1)
            justData = np.tril(np.repeat(y_data, y_data.shape[1], axis=0))
            # Use surrogate runs for data-generating process
            for key, metaModel in model_dict.items():
                model_data = np.array(
                    [runs[key][out][i] for out in out_names]).reshape(y_data.shape)
                justData = np.vstack((
                    justData,
                    np.tril(np.repeat(model_data, model_data.shape[1], axis=0))
                    ))
            # Save in a list
            all_just_data.append(justData)

        # Squeeze the array
        all_just_data = np.array(all_just_data).transpose(1, 0, 2).reshape(
            -1, np.array(all_just_data).shape[2]
            )

        return all_just_data

    # -------------------------------------------------------------------------
    def __perturb_data(self, data, output_names, n_bootstrap, noise_level):
        """
        Returns an array with n_bootstrap_itrs rowsof perturbed data.
        The first row includes the original observation data.
        If `self.bayes_loocv` is True, a 2d-array will be returned with
        repeated rows and zero diagonal entries.

        Parameters
        ----------
        data : pandas DataFrame
            Observation data.
        output_names : list
            List of the output names.

        Returns
        -------
        final_data : array
            Perturbed data set.

        """
        obs_data = data[output_names].values
        n_measurement, n_outs = obs_data.shape
        n_tot_measurement = obs_data[~np.isnan(obs_data)].shape[0]
        final_data = np.zeros(
            (n_bootstrap, n_tot_measurement)
            )
        final_data[0] = obs_data.T[~np.isnan(obs_data.T)]
        for itrIdx in range(1, n_bootstrap):
            data = np.zeros((n_measurement, n_outs))
            for idx in range(len(output_names)):
                std = np.nanstd(obs_data[:, idx])
                if std == 0:
                    std = 0.001
                noise = std * noise_level
                data[:, idx] = np.add(
                    obs_data[:, idx],
                    np.random.normal(0, 1, obs_data.shape[0]) * noise,
                )

            final_data[itrIdx] = data.T[~np.isnan(data.T)]

        return final_data

    # -------------------------------------------------------------------------
    def cal_model_weight(self, BME_Dict, justifiability=False, n_bootstarp=1):
        """
        Normalize the BME (Asumption: Model Prior weights are equal for models)

        Parameters
        ----------
        BME_Dict : dict
            A dictionary containing the BME values.

        Returns
        -------
        model_weights : array
            Model weights.

        """
        # Stack the BME values for all models
        all_BME = np.vstack(list(BME_Dict.values()))

        if justifiability:
            # Compute expected log_BME for justifiabiliy analysis
            all_BME = all_BME.reshape(
                all_BME.shape[0], -1, n_bootstarp).mean(axis=2)

        # Model weights
        model_weights = np.divide(all_BME, np.nansum(all_BME, axis=0))

        return model_weights

    # -------------------------------------------------------------------------
    def plot_just_analysis(self, model_weights_dict):
        """
        Visualizes the confusion matrix and the model wights for the
        justifiability analysis.

        Parameters
        ----------
        model_weights_dict : dict
            Model weights.

        Returns
        -------
        None.

        """

        directory = 'Outputs_Comparison/'
        os.makedirs(directory, exist_ok=True)
        Color = [*mcolors.TABLEAU_COLORS]
        names = [*model_weights_dict]

        model_names = [model.replace('_', '$-$') for model in self.model_names]
        for name in names:
            fig, ax = plt.subplots()
            for i, model in enumerate(model_names[1:]):
                plt.plot(list(range(1, self.n_meas+1)),
                         model_weights_dict[name][i],
                         color=Color[i], marker='o',
                         ms=10, linewidth=2, label=model
                         )

            plt.title(f"Data generated by: {name.replace('_', '$-$')}")
            plt.ylabel("Weights")
            plt.xlabel("No. of measurement points")
            ax.set_xticks(list(range(1, self.n_meas+1)))
            plt.legend(loc="best")
            fig.savefig(
                f'{directory}modelWeights_{name}.svg', bbox_inches='tight'
                )
            plt.close()

        # Confusion matrix for some measurement points
        epsilon = 1 if self.just_n_meas != 1 else 0
        for index in range(0, self.n_meas+epsilon, self.just_n_meas):
            weights = np.array(
                [model_weights_dict[key][:, index] for key in model_weights_dict]
                )
            g = sns.heatmap(
                weights.T, annot=True, cmap='Blues', xticklabels=model_names,
                yticklabels=model_names[1:], annot_kws={"size": 24}
                )

            # x axis on top
            g.xaxis.tick_top()
            g.xaxis.set_label_position('top')
            g.set_xlabel(r"\textbf{Data generated by:}", labelpad=15)
            g.set_ylabel(r"\textbf{Model weight for:}", labelpad=15)
            g.figure.savefig(
                f"{directory}confusionMatrix_ND_{index+1}.pdf",
                bbox_inches='tight'
                )
            plt.close()

    # -------------------------------------------------------------------------
    def plot_model_weights(self, model_weights, plot_name):
        """
        Visualizes the model weights resulting from BMS via the observation
        data.

        Parameters
        ----------
        model_weights : array
            Model weights.
        plot_name : str
            Plot name.

        Returns
        -------
        None.

        """
        font_size = 40
        # mkdir for plots
        directory = 'Outputs_Comparison/'
        os.makedirs(directory, exist_ok=True)

        # Create figure
        fig, ax = plt.subplots()

        # Filter data using np.isnan
        mask = ~np.isnan(model_weights.T)
        filtered_data = [d[m] for d, m in zip(model_weights, mask.T)]

        # Create the boxplot
        bp = ax.boxplot(filtered_data, patch_artist=True, showfliers=False)

        # change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set(color='#7570b3', linewidth=4)
            # change fill color
            box.set(facecolor='#1b9e77')

        # change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)

        # change the style of fliers and their fill
        # for flier in bp['fliers']:
        #     flier.set(marker='o', color='#e7298a', alpha=0.75)

        # Custom x-axis labels
        model_names = [model.replace('_', '$-$') for model in self.model_names]
        ax.set_xticklabels(model_names)

        ax.set_ylabel('Weight', fontsize=font_size)

        # Title
        plt.title('Posterior Model Weights')

        # Set y lim
        ax.set_ylim((-0.05, 1.05))

        # Set size of the ticks
        for t in ax.get_xticklabels():
            t.set_fontsize(font_size)
        for t in ax.get_yticklabels():
            t.set_fontsize(font_size)

        # Save the figure
        fig.savefig(
            f'./{directory}{plot_name}.pdf', bbox_inches='tight'
            )

        plt.close()

    # -------------------------------------------------------------------------
    def plot_bayes_factor(self, BME_Dict, plot_name=''):
        """
        Plots the Bayes factor distibutions in a :math:`N_m \\times N_m`
        matrix, where :math:`N_m` is the number of the models.

        Parameters
        ----------
        BME_Dict : dict
            A dictionary containing the BME values of the models.
        plot_name : str, optional
            Plot name. The default is ''.

        Returns
        -------
        None.

        """

        font_size = 40

        # mkdir for plots
        directory = 'Outputs_Comparison/'
        os.makedirs(directory, exist_ok=True)

        Colors = ["blue", "green", "gray", "brown"]

        model_names = list(BME_Dict.keys())
        nModels = len(model_names)

        # Plots
        fig, axes = plt.subplots(
            nrows=nModels, ncols=nModels, sharex=True, sharey=True
            )

        for i, key_i in enumerate(model_names):

            for j, key_j in enumerate(model_names):
                ax = axes[i, j]
                # Set size of the ticks
                for t in ax.get_xticklabels():
                    t.set_fontsize(font_size)
                for t in ax.get_yticklabels():
                    t.set_fontsize(font_size)

                if j != i:

                    # Null hypothesis: key_j is the better model
                    BayesFactor = np.log10(
                        np.divide(BME_Dict[key_i], BME_Dict[key_j])
                        )

                    # sns.kdeplot(BayesFactor, ax=ax, color=Colors[i], shade=True)
                    # sns.histplot(BayesFactor, ax=ax, stat="probability",
                    #              kde=True, element='step',
                    #              color=Colors[j])

                    # taken from seaborn's source code (utils.py and
                    # distributions.py)
                    def seaborn_kde_support(data, bw, gridsize, cut, clip):
                        if clip is None:
                            clip = (-np.inf, np.inf)
                        support_min = max(data.min() - bw * cut, clip[0])
                        support_max = min(data.max() + bw * cut, clip[1])
                        return np.linspace(support_min, support_max, gridsize)

                    kde_estim = stats.gaussian_kde(
                        BayesFactor, bw_method='scott'
                        )

                    # manual linearization of data
                    # linearized = np.linspace(
                    #     quotient.min(), quotient.max(), num=500)

                    # or better: mimic seaborn's internal stuff
                    bw = kde_estim.scotts_factor() * np.std(BayesFactor)
                    linearized = seaborn_kde_support(
                        BayesFactor, bw, 100, 3, None)

                    # computes values of the estimated function on the
                    # estimated linearized inputs
                    Z = kde_estim.evaluate(linearized)

                    # https://stackoverflow.com/questions/29661574/normalize-
                    # numpy-array-columns-in-python
                    def normalize(x):
                        return (x - x.min(0)) / x.ptp(0)

                    # normalize so it is between 0;1
                    Z2 = normalize(Z)
                    ax.plot(linearized, Z2, "-", color=Colors[i], linewidth=4)
                    ax.fill_between(
                        linearized, 0, Z2, color=Colors[i], alpha=0.25
                        )

                    # Draw BF significant levels according to Jeffreys 1961
                    # Strong evidence for both models
                    ax.axvline(
                        x=np.log10(3), ymin=0, linewidth=4, color='dimgrey'
                        )
                    # Strong evidence for one model
                    ax.axvline(
                        x=np.log10(10), ymin=0, linewidth=4, color='orange'
                        )
                    # Decisive evidence for one model
                    ax.axvline(
                        x=np.log10(100), ymin=0, linewidth=4, color='r'
                        )

                    # legend
                    BF_label = key_i.replace('_', '$-$') + \
                        '/' + key_j.replace('_', '$-$')
                    legend_elements = [
                        patches.Patch(facecolor=Colors[i], edgecolor=Colors[i],
                                      label=f'BF({BF_label})')
                        ]
                    ax.legend(
                        loc='upper left', handles=legend_elements,
                        fontsize=font_size-(nModels+1)*5
                        )

                elif j == i:
                    # build a rectangle in axes coords
                    left, width = 0, 1
                    bottom, height = 0, 1

                    # axes coordinates are 0,0 is bottom left and 1,1 is upper
                    # right
                    p = patches.Rectangle(
                        (left, bottom), width, height, color='white',
                        fill=True, transform=ax.transAxes, clip_on=False
                        )
                    ax.grid(False)
                    ax.add_patch(p)
                    # ax.text(0.5*(left+right), 0.5*(bottom+top), key_i,
                    fsize = font_size+20 if nModels < 4 else font_size
                    ax.text(0.5, 0.5, key_i.replace('_', '$-$'),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=fsize, color=Colors[i],
                            transform=ax.transAxes)

        # Defining custom 'ylim' values.
        custom_ylim = (0, 1.05)

        # Setting the values for all axes.
        plt.setp(axes, ylim=custom_ylim)

        # set labels
        for i in range(nModels):
            axes[-1, i].set_xlabel('log$_{10}$(BF)', fontsize=font_size)
            axes[i, 0].set_ylabel('Probability', fontsize=font_size)

        # Adjust subplots
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

        plt.savefig(
            f'./{directory}Bayes_Factor{plot_name}.pdf', bbox_inches='tight'
            )

        plt.close()
