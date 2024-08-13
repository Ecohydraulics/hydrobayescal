#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import copy
import pandas as pd
from tqdm import tqdm
from scipy import stats
import scipy.linalg as spla
import joblib
import seaborn as sns
import corner
import h5py
import multiprocessing
import gc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as plt

from .mcmc import MCMC

# Load the mplstyle
plt.style.use(os.path.join(os.path.split(__file__)[0],
                           '../', 'bayesvalidrox.mplstyle'))


class BayesInference:
    """
    A class to perform Bayesian Analysis.


    Attributes
    ----------
    MetaModel : obj
        Meta model object.
    discrepancy : obj
        The discrepancy object for the sigma2s, i.e. the diagonal entries
        of the variance matrix for a multivariate normal likelihood.
    name : str, optional
        The type of analysis, either calibration (`Calib`) or validation
        (`Valid`). The default is `'Calib'`.
    emulator : bool, optional
        Analysis with emulator (MetaModel). The default is `True`.
    bootstrap : bool, optional
        Bootstrap the analysis. The default is `False`.
    req_outputs : list, optional
        The list of requested output to be used for the analysis.
        The default is `None`. If None, all the defined outputs for the model
        object is used.
    selected_indices : dict, optional
        A dictionary with the selected indices of each model output. The
        default is `None`. If `None`, all measurement points are used in the
        analysis.
    samples : array of shape (n_samples, n_params), optional
        The samples to be used in the analysis. The default is `None`. If
        None the samples are drawn from the probablistic input parameter
        object of the MetaModel object.
    n_samples : int, optional
        Number of samples to be used in the analysis. The default is `500000`.
        If samples is not `None`, this argument will be assigned based on the
        number of samples given.
    measured_data : dict, optional
        A dictionary containing the observation data. The default is `None`.
        if `None`, the observation defined in the Model object of the
        MetaModel is used.
    inference_method : str, optional
        A method for approximating the posterior distribution in the Bayesian
        inference step. The default is `'rejection'`, which stands for
        rejection sampling. A Markov Chain Monte Carlo sampler can be simply
        selected by passing `'MCMC'`.
    mcmc_params : dict, optional
        A dictionary with args required for the Bayesian inference with
        `MCMC`. The default is `None`.

        Pass the mcmc_params like the following:

            >>> mcmc_params:{
                'init_samples': None,  # initial samples
                'n_walkers': 100,  # number of walkers (chain)
                'n_steps': 100000,  # number of maximum steps
                'n_burn': 200,  # number of burn-in steps
                'moves': None,  # Moves for the emcee sampler
                'multiprocessing': False,  # multiprocessing
                'verbose': False # verbosity
                }
        The items shown above are the default values. If any parmeter is
        not defined, the default value will be assigned to it.
    bayes_loocv : bool, optional
        Bayesian Leave-one-out Cross Validation. The default is `False`. If
        `True`, the LOOCV procedure is used to estimate the bayesian Model
        Evidence (BME).
    n_bootstrap_itrs : int, optional
        Number of bootstrap iteration. The default is `1`. If bayes_loocv is
        `True`, this is qualt to the total length of the observation data
        set.
    perturbed_data : array of shape (n_bootstrap_itrs, n_obs), optional
        User defined perturbed data. The default is `[]`.
    bootstrap_noise : float, optional
        A noise level to perturb the data set. The default is `0.05`.
    just_analysis : bool, optional
        Justifiability analysis. The default is False.
    valid_metrics : list, optional
        List of the validation metrics. The following metrics are supported:

        1. log_BME : logarithm of the Bayesian model evidence
        2. KLD : Kullback-Leibler Divergence
        3. inf_entropy: Information entropy
        The default is `['log_BME']`.
    plot_post_pred : bool, optional
        Plot posterior predictive plots. The default is `True`.
    plot_map_pred : bool, optional
        Plot the model outputs vs the metamodel predictions for the maximum
        a posteriori (defined as `max_a_posteriori`) parameter set. The
        default is `False`.
    max_a_posteriori : str, optional
        Maximum a posteriori. `'mean'` and `'mode'` are available. The default
        is `'mean'`.
    corner_title_fmt : str, optional
        Title format for the posterior distribution plot with python
        package `corner`. The default is `'.2e'`.

    """

    def __init__(self, engine, MetaModel = None, discrepancy=None, emulator=True,
                 name='Calib', bootstrap=False, req_outputs=None,
                 selected_indices=None, samples=None, n_samples=100000,
                 measured_data=None, inference_method='rejection',
                 mcmc_params=None, bayes_loocv=False, n_bootstrap_itrs=1,
                 perturbed_data=[], bootstrap_noise=0.05, just_analysis=False,
                 valid_metrics=['BME'], plot_post_pred=True,
                 plot_map_pred=False, max_a_posteriori='mean',
                 corner_title_fmt='.2e'):

        self.engine = engine
        self.MetaModel = engine.MetaModel
        self.Discrepancy = discrepancy
        self.emulator = emulator
        self.name = name
        self.bootstrap = bootstrap
        self.req_outputs = req_outputs
        self.selected_indices = selected_indices
        self.samples = samples
        self.n_samples = n_samples
        self.measured_data = measured_data
        self.inference_method = inference_method
        self.mcmc_params = mcmc_params
        self.perturbed_data = perturbed_data
        self.bayes_loocv = bayes_loocv
        self.n_bootstrap_itrs = n_bootstrap_itrs
        self.bootstrap_noise = bootstrap_noise
        self.just_analysis = just_analysis
        self.valid_metrics = valid_metrics
        self.plot_post_pred = plot_post_pred
        self.plot_map_pred = plot_map_pred
        self.max_a_posteriori = max_a_posteriori
        self.corner_title_fmt = corner_title_fmt

    # -------------------------------------------------------------------------
    def create_inference(self):
        """
        Starts the inference.

        Returns
        -------
        BayesInference : obj
            The Bayes inference object.

        """

        # Set some variables
        MetaModel = self.MetaModel
        Model = self.engine.Model
        n_params = MetaModel.n_params
        output_names = Model.Output.names
        par_names = self.engine.ExpDesign.par_names

        # If the prior is set by the user, take it.
        if self.samples is None:
            self.samples = self.engine.ExpDesign.generate_samples(
                self.n_samples, 'random')
        else:
            try:
                samples = self.samples.values
            except AttributeError:
                samples = self.samples

            # Take care of an additional Sigma2s
            self.samples = samples[:, :n_params]

            # Update number of samples
            self.n_samples = self.samples.shape[0]

        # ---------- Preparation of observation data ----------
        # Read observation data and perturb it if requested.
        if self.measured_data is None:
            self.measured_data = Model.read_observation(case=self.name)
        # Convert measured_data to a data frame
        if not isinstance(self.measured_data, pd.DataFrame):
            self.measured_data = pd.DataFrame(self.measured_data)

        # Extract the total number of measurement points
        if self.name.lower() == 'calib':
            self.n_tot_measurement = Model.n_obs
        else:
            self.n_tot_measurement = Model.n_obs_valid

        # Find measurement error (if not given) for post predictive plot
        if not hasattr(self, 'measurement_error'):
            if isinstance(self.Discrepancy, dict):
                Disc = self.Discrepancy['known']
            else:
                Disc = self.Discrepancy
            if isinstance(Disc.parameters, dict):
                self.measurement_error = {k: np.sqrt(Disc.parameters[k]) for k
                                          in Disc.parameters.keys()}
            else:
                try:
                    self.measurement_error = np.sqrt(Disc.parameters)
                except TypeError:
                    pass

        # ---------- Preparation of variance for covariance matrix ----------
        # Independent and identically distributed
        total_sigma2 = dict()
        opt_sigma_flag = isinstance(self.Discrepancy, dict)
        opt_sigma = None
        for key_idx, key in enumerate(output_names):

            # Find opt_sigma
            if opt_sigma_flag and opt_sigma is None:
                # Option A: known error with unknown bias term
                opt_sigma = 'A'
                known_discrepancy = self.Discrepancy['known']
                self.Discrepancy = self.Discrepancy['infer']
                sigma2 = np.array(known_discrepancy.parameters[key])

            elif opt_sigma == 'A' or self.Discrepancy.parameters is not None:
                # Option B: The sigma2 is known (no bias term)
                if opt_sigma == 'A':
                    sigma2 = np.array(known_discrepancy.parameters[key])
                else:
                    opt_sigma = 'B'
                    sigma2 = np.array(self.Discrepancy.parameters[key])

            elif not isinstance(self.Discrepancy.InputDisc, str):
                # Option C: The sigma2 is unknown (bias term including error)
                opt_sigma = 'C'
                self.Discrepancy.opt_sigma = opt_sigma
                n_measurement = self.measured_data[key].values.shape
                sigma2 = np.zeros((n_measurement[0]))

            total_sigma2[key] = sigma2

            self.Discrepancy.opt_sigma = opt_sigma
            self.Discrepancy.total_sigma2 = total_sigma2

        # If inferred sigma2s obtained from e.g. calibration are given
        try:
            self.sigma2s = self.Discrepancy.get_sample(self.n_samples)
        except:
            pass

        # ---------------- Bootstrap & TOM --------------------
        if self.bootstrap or self.bayes_loocv or self.just_analysis:
            if len(self.perturbed_data) == 0:
                # zero mean noise Adding some noise to the observation function
                self.perturbed_data = self._perturb_data(
                    self.measured_data, output_names
                    )
            else:
                self.n_bootstrap_itrs = len(self.perturbed_data)

            # -------- Model Discrepancy -----------
            if hasattr(self, 'error_model') and self.error_model \
               and self.name.lower() != 'calib':
                # Select posterior mean as MAP
                MAP_theta = self.samples.mean(axis=0).reshape((1, n_params))
                # MAP_theta = stats.mode(self.samples,axis=0)[0]

                # Evaluate the (meta-)model at the MAP
                y_MAP, y_std_MAP = MetaModel.eval_metamodel(samples=MAP_theta)

                # Train a GPR meta-model using MAP
                self.error_MetaModel = MetaModel.create_model_error(
                    self.bias_inputs, y_MAP, Name=self.name
                    )

            # -----------------------------------------------------
            # ----- Loop over the perturbed observation data ------
            # -----------------------------------------------------
            # Initilize arrays
            logLikelihoods = np.zeros((self.n_samples, self.n_bootstrap_itrs),
                                      dtype=np.float16)
            BME_Corr = np.zeros((self.n_bootstrap_itrs))
            log_BME = np.zeros((self.n_bootstrap_itrs))
            KLD = np.zeros((self.n_bootstrap_itrs))
            inf_entropy = np.zeros((self.n_bootstrap_itrs))

            # Compute the prior predtions
            # Evaluate the MetaModel
            if self.emulator:
                y_hat, y_std = MetaModel.eval_metamodel(samples=self.samples)
                self.__mean_pce_prior_pred = y_hat
                self._std_pce_prior_pred = y_std

                # Correct the predictions with Model discrepancy
                if hasattr(self, 'error_model') and self.error_model:
                    y_hat_corr, y_std = self.error_MetaModel.eval_model_error(
                        self.bias_inputs, self.__mean_pce_prior_pred
                        )
                    self.__mean_pce_prior_pred = y_hat_corr
                    self._std_pce_prior_pred = y_std

                # Surrogate model's error using RMSE of test data
                if hasattr(MetaModel, 'rmse'):
                    surrError = MetaModel.rmse
                else:
                    surrError = None

            else:
                # Evaluate the original model
                self.__model_prior_pred = self._eval_model(
                    samples=self.samples, key='PriorPred'
                    )
                surrError = None

            # Start the likelihood-BME computations for the perturbed data
            for itr_idx, data in tqdm(
                    enumerate(self.perturbed_data),
                    total=self.n_bootstrap_itrs,
                    desc="Bootstrapping the BME calculations", ascii=True
                    ):

                # ---------------- Likelihood calculation ----------------
                if self.emulator:
                    model_evals = self.__mean_pce_prior_pred
                else:
                    model_evals = self.__model_prior_pred

                # Leave one out
                if self.bayes_loocv or self.just_analysis:
                    self.selected_indices = np.nonzero(data)[0]

                # Prepare data dataframe
                nobs = list(self.measured_data.count().values[1:])
                numbers = list(np.cumsum(nobs))
                indices = list(zip([0] + numbers, numbers))
                data_dict = {
                    output_names[i]: data[j:k] for i, (j, k) in
                    enumerate(indices)
                    }
                #print(output_names)
                #print(indices)
                #print(numbers)
                #print(nobs)
                #print(self.measured_data)
                #for i, (j, k) in enumerate(indices):
                #    print(i,j,k)
                #print(data)
                #print(data_dict)
                #stop

                # Unknown sigma2
                if opt_sigma == 'C' or hasattr(self, 'sigma2s'):
                    logLikelihoods[:, itr_idx] = self.normpdf(
                        model_evals, data_dict, total_sigma2,
                        sigma2=self.sigma2s, std=surrError
                        )
                else:
                    # known sigma2
                    logLikelihoods[:, itr_idx] = self.normpdf(
                        model_evals, data_dict, total_sigma2,
                        std=surrError
                        )

                # ---------------- BME Calculations ----------------
                # BME (log)
                log_BME[itr_idx] = np.log(
                    np.nanmean(np.exp(logLikelihoods[:, itr_idx],
                                      dtype=np.longdouble))#float128))
                    )

                # BME correction when using Emulator
                if self.emulator:
                    BME_Corr[itr_idx] = self.__corr_factor_BME(
                        data_dict, total_sigma2, log_BME[itr_idx]
                        )

                # Rejection Step
                if 'kld' in list(map(str.lower, self.valid_metrics)) and\
                   'inf_entropy' in list(map(str.lower, self.valid_metrics)):
                    # Random numbers between 0 and 1
                    unif = np.random.rand(1, self.n_samples)[0]

                    # Reject the poorly performed prior
                    Likelihoods = np.exp(logLikelihoods[:, itr_idx],
                                         dtype=np.float64)
                    accepted = (Likelihoods/np.max(Likelihoods)) >= unif
                    posterior = self.samples[accepted]

                    # Posterior-based expectation of likelihoods
                    postExpLikelihoods = np.mean(
                        logLikelihoods[:, itr_idx][accepted]
                        )

                    # Calculate Kullback-Leibler Divergence
                    KLD[itr_idx] = postExpLikelihoods - log_BME[itr_idx]

                # Posterior-based expectation of prior densities
                if 'inf_entropy' in list(map(str.lower, self.valid_metrics)):
                    n_thread = int(0.875 * multiprocessing.cpu_count())
                    with multiprocessing.Pool(n_thread) as p:
                        postExpPrior = np.mean(np.concatenate(
                            p.map(
                                self.engine.ExpDesign.JDist.pdf,
                                np.array_split(posterior.T, n_thread, axis=1))
                            )
                            )
                    # Information Entropy based on Entropy paper Eq. 38
                    inf_entropy[itr_idx] = log_BME[itr_idx] - postExpPrior - \
                        postExpLikelihoods

                # Clear memory
                gc.collect(generation=2)

            # ---------- Store metrics for perturbed data set ----------------
            # Likelihoods (Size: n_samples, n_bootstrap_itr)
            self.log_likes = logLikelihoods

            # BME (log), KLD, infEntropy (Size: 1,n_bootstrap_itr)
            self.log_BME = log_BME

            # BMECorrFactor (log) (Size: 1,n_bootstrap_itr)
            if self.emulator:
                self.log_BME_corr_factor = BME_Corr

            if 'kld' in list(map(str.lower, self.valid_metrics)):
                self.KLD = KLD
            if 'inf_entropy' in list(map(str.lower, self.valid_metrics)):
                self.inf_entropy = inf_entropy

            # BME = BME + BMECorrFactor
            if self.emulator:
                self.log_BME += self.log_BME_corr_factor

        # ---------------- Parameter Bayesian inference ----------------
        if self.inference_method.lower() == 'mcmc':
            # Instantiate the MCMC object
            MCMC_Obj = MCMC(self)
            self.posterior_df = MCMC_Obj.run_sampler(
                self.measured_data, total_sigma2
                )

        elif self.name.lower() == 'valid':
            # Convert to a dataframe if samples are provided after calibration.
            self.posterior_df = pd.DataFrame(self.samples, columns=par_names)

        else:
            # Rejection sampling
            self.posterior_df = self._rejection_sampling()

        # Provide posterior's summary
        print('\n')
        print('-'*15 + 'Posterior summary' + '-'*15)
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        print(self.posterior_df.describe())
        print('-'*50)

        # -------- Model Discrepancy -----------
        if hasattr(self, 'error_model') and self.error_model \
           and self.name.lower() == 'calib':
            if self.inference_method.lower() == 'mcmc':
                self.error_MetaModel = MCMC_Obj.error_MetaModel
            else:
                # Select posterior mean as MAP
                if opt_sigma == "B":
                    posterior_df = self.posterior_df.values
                else:
                    posterior_df = self.posterior_df.values[:, :-Model.n_outputs]

                # Select posterior mean as Maximum a posteriori
                map_theta = posterior_df.mean(axis=0).reshape((1, n_params))
                # map_theta = stats.mode(Posterior_df,axis=0)[0]

                # Evaluate the (meta-)model at the MAP
                y_MAP, y_std_MAP = MetaModel.eval_metamodel(samples=map_theta)

                # Train a GPR meta-model using MAP
                self.error_MetaModel = MetaModel.create_model_error(
                    self.bias_inputs, y_MAP, Name=self.name
                    )

        # -------- Posterior perdictive -----------
        self._posterior_predictive()

        # -----------------------------------------------------
        # ------------------ Visualization --------------------
        # -----------------------------------------------------
        # Create Output directory, if it doesn't exist already.
        out_dir = f'Outputs_Bayes_{Model.name}_{self.name}'
        os.makedirs(out_dir, exist_ok=True)

        # -------- Posteior parameters --------
        if opt_sigma != "B":
            par_names.extend(
                [self.Discrepancy.InputDisc.Marginals[i].name for i
                 in range(len(self.Discrepancy.InputDisc.Marginals))]
                )
        # Pot with corner
        figPosterior = corner.corner(self.posterior_df.to_numpy(),
                                     labels=par_names,
                                     quantiles=[0.15, 0.5, 0.85],
                                     show_titles=True,
                                     title_fmt=self.corner_title_fmt,
                                     labelpad=0.2,
                                     use_math_text=True,
                                     title_kwargs={"fontsize": 28},
                                     plot_datapoints=False,
                                     plot_density=False,
                                     fill_contours=True,
                                     smooth=0.5,
                                     smooth1d=0.5)

        # Loop over axes and set x limits
        if opt_sigma == "B":
            axes = np.array(figPosterior.axes).reshape(
                (len(par_names), len(par_names))
                )
            for yi in range(len(par_names)):
                ax = axes[yi, yi]
                ax.set_xlim(self.engine.ExpDesign.bound_tuples[yi])
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.set_xlim(self.engine.ExpDesign.bound_tuples[xi])
        plt.close()

        # Turn off gridlines
        for ax in figPosterior.axes:
            ax.grid(False)

        if self.emulator:
            plotname = f'/Posterior_Dist_{Model.name}_emulator'
        else:
            plotname = f'/Posterior_Dist_{Model.name}'

        figPosterior.set_size_inches((24, 16))
        figPosterior.savefig(f'./{out_dir}{plotname}.pdf',
                             bbox_inches='tight')

        # -------- Plot MAP --------
        if self.plot_map_pred:
            self._plot_max_a_posteriori()

        # -------- Plot log_BME dist --------
        if self.bootstrap:

            # Computing the TOM performance
            self.log_BME_tom = stats.chi2.rvs(
                self.n_tot_measurement, size=self.log_BME.shape[0]
                )

            fig, ax = plt.subplots()
            sns.kdeplot(self.log_BME_tom, ax=ax, color="green", shade=True)
            sns.kdeplot(
                self.log_BME, ax=ax, color="blue", shade=True,
                label='Model BME')

            ax.set_xlabel('log$_{10}$(BME)')
            ax.set_ylabel('Probability density')

            legend_elements = [
                Patch(facecolor='green', edgecolor='green', label='TOM BME'),
                Patch(facecolor='blue', edgecolor='blue', label='Model BME')
                ]
            ax.legend(handles=legend_elements)

            if self.emulator:
                plotname = f'/BME_hist_{Model.name}_emulator'
            else:
                plotname = f'/BME_hist_{Model.name}'

            plt.savefig(f'./{out_dir}{plotname}.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

        # -------- Posteior perdictives --------
        if self.plot_post_pred:
            # Plot the posterior predictive
            self._plot_post_predictive()

        return self

    # -------------------------------------------------------------------------
    def _perturb_data(self, data, output_names):
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
        noise_level = self.bootstrap_noise
        obs_data = data[output_names].values
        n_measurement, n_outs = obs_data.shape
        self.n_tot_measurement = obs_data[~np.isnan(obs_data)].shape[0]
        # Number of bootstrap iterations
        if self.bayes_loocv:
            self.n_bootstrap_itrs = self.n_tot_measurement

        # Pass loocv dataset
        if self.bayes_loocv:
            obs = obs_data.T[~np.isnan(obs_data.T)]
            final_data = np.repeat(np.atleast_2d(obs), self.n_bootstrap_itrs,
                                   axis=0)
            np.fill_diagonal(final_data, 0)
            return final_data

        else:
            final_data = np.zeros(
                (self.n_bootstrap_itrs, self.n_tot_measurement)
                )
            final_data[0] = obs_data.T[~np.isnan(obs_data.T)]
            for itrIdx in range(1, self.n_bootstrap_itrs):
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
    def _logpdf(self, x, mean, cov):
        """
        computes the likelihood based on a multivariate normal distribution.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        mean : array_like
            Observation data.
        cov : 2d array
            Covariance matrix of the distribution.

        Returns
        -------
        log_lik : float
            Log likelihood.

        """
        n = len(mean)
        L = spla.cholesky(cov, lower=True)
        beta = np.sum(np.log(np.diag(L)))
        dev = x - mean
        alpha = dev.dot(spla.cho_solve((L, True), dev))
        log_lik = -0.5 * alpha - beta - n / 2. * np.log(2 * np.pi)
        return log_lik

    # -------------------------------------------------------------------------
    def _eval_model(self, samples=None, key='MAP'):
        """
        Evaluates Forward Model.

        Parameters
        ----------
        samples : array of shape (n_samples, n_params), optional
            Parameter sets. The default is None.
        key : str, optional
            Key string to be passed to the run_model_parallel method.
            The default is 'MAP'.

        Returns
        -------
        model_outputs : dict
            Model outputs.

        """
        MetaModel = self.MetaModel
        Model = self.engine.Model

        if samples is None:
            self.samples = self.engine.ExpDesign.generate_samples(
                self.n_samples, 'random')
        else:
            self.samples = samples
            self.n_samples = len(samples)

        model_outputs, _ = Model.run_model_parallel(
            self.samples, key_str=key+self.name)

        # Clean up
        # Zip the subdirectories
        try:
            dir_name = f'{Model.name}MAP{self.name}'
            key = dir_name + '_'
            Model.zip_subdirs(dir_name, key)
        except:
            pass

        return model_outputs

    # -------------------------------------------------------------------------
    def _kernel_rbf(self, X, hyperparameters):
        """
        Isotropic squared exponential kernel.

        Higher l values lead to smoother functions and therefore to coarser
        approximations of the training data. Lower l values make functions
        more wiggly with wide uncertainty regions between training data points.

        sigma_f controls the marginal variance of b(x)

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)

        hyperparameters : Dict
            Lambda characteristic length
            sigma_f controls the marginal variance of b(x)
            sigma_0 unresolvable error nugget term, interpreted as random
                    error that cannot be attributed to measurement error.
        Returns
        -------
        var_cov_matrix : ndarray of shape (n_samples_X,n_samples_X)
            Kernel k(X, X).

        """
        from sklearn.gaussian_process.kernels import RBF
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minmax = min_max_scaler.fit_transform(X)

        nparams = len(hyperparameters)
        # characteristic length (0,1]
        Lambda = hyperparameters[0]
        # sigma_f controls the marginal variance of b(x)
        sigma2_f = hyperparameters[1]

        # cov_matrix = sigma2_f*rbf_kernel(X_minmax, gamma = 1/Lambda**2)

        rbf = RBF(length_scale=Lambda)
        cov_matrix = sigma2_f * rbf(X_minmax)
        if nparams > 2:
            # (unresolvable error) nugget term that is interpreted as random
            # error that cannot be attributed to measurement error.
            sigma2_0 = hyperparameters[2:]
            for i, j in np.ndindex(cov_matrix.shape):
                cov_matrix[i, j] += np.sum(sigma2_0) if i == j else 0

        return cov_matrix

    # -------------------------------------------------------------------------
    def normpdf(self, outputs, obs_data, total_sigma2s, sigma2=None, std=None):
        """
        Calculates the likelihood of simulation outputs compared with
        observation data.

        Parameters
        ----------
        outputs : dict
            A dictionary containing the simulation outputs as array of shape
            (n_samples, n_measurement) for each model output.
        obs_data : dict
            A dictionary/dataframe containing the observation data.
        total_sigma2s : dict
            A dictionary with known values of the covariance diagonal entries,
            a.k.a sigma^2.
        sigma2 : array, optional
            An array of the sigma^2 samples, when the covariance diagonal
            entries are unknown and are being jointly inferred. The default is
            None.
        std : dict, optional
            A dictionary containing the root mean squared error as array of
            shape (n_samples, n_measurement) for each model output. The default
            is None.

        Returns
        -------
        logLik : array of shape (n_samples)
            Likelihoods.

        """
        Model = self.engine.Model
        logLik = 0.0

        # Extract the requested model outputs for likelihood calulation
        if self.req_outputs is None:
            req_outputs = Model.Output.names
        else:
            req_outputs = list(self.req_outputs)

        # Loop over the outputs
        for idx, out in enumerate(req_outputs):

            # (Meta)Model Output
            nsamples, nout = outputs[out].shape

            # Prepare data and remove NaN
            try:
                data = obs_data[out].values[~np.isnan(obs_data[out])]
            except AttributeError:
                data = obs_data[out][~np.isnan(obs_data[out])]

            # Prepare sigma2s
            non_nan_indices = ~np.isnan(total_sigma2s[out])
            tot_sigma2s = total_sigma2s[out][non_nan_indices][:nout]

            # Add the std of the PCE is chosen as emulator.
            if self.emulator:
                if std is not None:
                    tot_sigma2s += std[out]**2

            # Covariance Matrix
            covMatrix = np.diag(tot_sigma2s)

            # Select the data points to compare
            try:
                indices = self.selected_indices[out]
            except:
                indices = list(range(nout))
            covMatrix = np.diag(covMatrix[indices, indices])

            # If sigma2 is not given, use given total_sigma2s
            if sigma2 is None:
                logLik += stats.multivariate_normal.logpdf(
                    outputs[out][:, indices], data[indices], covMatrix)
                continue

            # Loop over each run/sample and calculate logLikelihood
            logliks = np.zeros(nsamples)
            for s_idx in range(nsamples):

                # Simulation run
                tot_outputs = outputs[out]

                # Covariance Matrix
                covMatrix = np.diag(tot_sigma2s)

                if sigma2 is not None:
                    # Check the type error term
                    if hasattr(self, 'bias_inputs') and \
                       not hasattr(self, 'error_model'):
                        # Infer a Bias model usig Gaussian Process Regression
                        bias_inputs = np.hstack(
                            (self.bias_inputs[out],
                             tot_outputs[s_idx].reshape(-1, 1)))

                        params = sigma2[s_idx, idx*3:(idx+1)*3]
                        covMatrix = self._kernel_rbf(bias_inputs, params)
                    else:
                        # Infer equal sigma2s
                        try:
                            sigma_2 = sigma2[s_idx, idx]
                        except TypeError:
                            sigma_2 = 0.0

                        covMatrix += sigma_2 * np.eye(nout)
                        # covMatrix = np.diag(sigma2 * total_sigma2s)

                # Select the data points to compare
                try:
                    indices = self.selected_indices[out]
                except:
                    indices = list(range(nout))
                covMatrix = np.diag(covMatrix[indices, indices])

                # Compute loglikelihood
                logliks[s_idx] = self._logpdf(
                    tot_outputs[s_idx, indices], data[indices], covMatrix
                    )

            logLik += logliks
        return logLik

    # -------------------------------------------------------------------------
    def _corr_factor_BME_old(self, Data, total_sigma2s, posterior):
        """
        Calculates the correction factor for BMEs.
        """
        MetaModel = self.MetaModel
        OrigModelOutput = self.engine.ExpDesign.Y
        Model = self.engine.Model

        # Posterior with guassian-likelihood
        postDist = stats.gaussian_kde(posterior.T)

        # Remove NaN
        Data = Data[~np.isnan(Data)]
        total_sigma2s = total_sigma2s[~np.isnan(total_sigma2s)]

        # Covariance Matrix
        covMatrix = np.diag(total_sigma2s[:self.n_tot_measurement])

        # Extract the requested model outputs for likelihood calulation
        if self.req_outputs is None:
            OutputType = Model.Output.names
        else:
            OutputType = list(self.req_outputs)

        # SampleSize = OrigModelOutput[OutputType[0]].shape[0]


        # Flatten the OutputType for OrigModel
        TotalOutputs = np.concatenate([OrigModelOutput[x] for x in OutputType], 1)

        NrofBayesSamples = self.n_samples
        # Evaluate MetaModel on the experimental design
        Samples = self.engine.ExpDesign.X
        OutputRS, stdOutputRS = MetaModel.eval_metamodel(samples=Samples)

        # Reset the NrofSamples to NrofBayesSamples
        self.n_samples = NrofBayesSamples

        # Flatten the OutputType for MetaModel
        TotalPCEOutputs = np.concatenate([OutputRS[x] for x in OutputRS], 1)
        TotalPCEstdOutputRS= np.concatenate([stdOutputRS[x] for x in stdOutputRS], 1)

        logweight = 0
        for i, sample in enumerate(Samples):
            # Compute likelilhood output vs RS
            covMatrix = np.diag(TotalPCEstdOutputRS[i]**2)
            logLik = self._logpdf(TotalOutputs[i], TotalPCEOutputs[i], covMatrix)
            # Compute posterior likelihood of the collocation points
            logpostLik = np.log(postDist.pdf(sample[:, None]))[0]
            if logpostLik != -np.inf:
                logweight += logLik + logpostLik
        return logweight

    # -------------------------------------------------------------------------
    def __corr_factor_BME(self, obs_data, total_sigma2s, logBME):
        """
        Calculates the correction factor for BMEs.
        """
        MetaModel = self.MetaModel
        samples = self.engine.ExpDesign.X
        model_outputs = self.engine.ExpDesign.Y
        Model = self.engine.Model
        n_samples = samples.shape[0]

        # Extract the requested model outputs for likelihood calulation
        output_names = Model.Output.names

        # Evaluate MetaModel on the experimental design and ValidSet
        OutputRS, stdOutputRS = MetaModel.eval_metamodel(samples=samples)

        logLik_data = np.zeros((n_samples))
        logLik_model = np.zeros((n_samples))
        # Loop over the outputs
        for idx, out in enumerate(output_names):

            # (Meta)Model Output
            nsamples, nout = model_outputs[out].shape

            # Prepare data and remove NaN
            try:
                data = obs_data[out].values[~np.isnan(obs_data[out])]
            except AttributeError:
                data = obs_data[out][~np.isnan(obs_data[out])]

            # Prepare sigma2s
            non_nan_indices = ~np.isnan(total_sigma2s[out])
            tot_sigma2s = total_sigma2s[out][non_nan_indices][:nout]

            # Covariance Matrix
            covMatrix_data = np.diag(tot_sigma2s)

            for i, sample in enumerate(samples):

                # Simulation run
                y_m = model_outputs[out][i]

                # Surrogate prediction
                y_m_hat = OutputRS[out][i]

                # CovMatrix with the surrogate error
                covMatrix = np.eye(len(y_m)) * 1/(2*np.pi)

                # Select the data points to compare
                try:
                    indices = self.selected_indices[out]
                except:
                    indices = list(range(nout))
                covMatrix = np.diag(covMatrix[indices, indices])
                covMatrix_data = np.diag(covMatrix_data[indices, indices])

                # Compute likelilhood output vs data
                logLik_data[i] += self._logpdf(
                    y_m_hat[indices], data[indices],
                    covMatrix_data
                    )

                # Compute likelilhood output vs surrogate
                logLik_model[i] += self._logpdf(
                    y_m_hat[indices], y_m[indices],
                    covMatrix
                    )

        # Weight
        logLik_data -= logBME
        weights = np.mean(np.exp(logLik_model+logLik_data))

        return np.log(weights)

    # -------------------------------------------------------------------------
    def _rejection_sampling(self):
        """
        Performs rejection sampling to update the prior distribution on the
        input parameters.

        Returns
        -------
        posterior : pandas.dataframe
            Posterior samples of the input parameters.

        """

        MetaModel = self.MetaModel
        try:
            sigma2_prior = self.Discrepancy.sigma2_prior
        except:
            sigma2_prior = None

        # Check if the discrepancy is defined as a distribution:
        samples = self.samples

        if sigma2_prior is not None:
            samples = np.hstack((samples, sigma2_prior))

        # Take the first column of Likelihoods (Observation data without noise)
        if self.just_analysis or self.bayes_loocv:
            index = self.n_tot_measurement-1
            likelihoods = np.exp(self.log_likes[:, index], dtype=np.longdouble)#np.float128)
        else:
            likelihoods = np.exp(self.log_likes[:, 0], dtype=np.longdouble)#np.float128)

        n_samples = len(likelihoods)
        norm_ikelihoods = likelihoods / np.max(likelihoods)

        # Normalize based on min if all Likelihoods are zero
        if all(likelihoods == 0.0):
            likelihoods = self.log_likes[:, 0]
            norm_ikelihoods = likelihoods / np.min(likelihoods)

        # Random numbers between 0 and 1
        unif = np.random.rand(1, n_samples)[0]

        # Reject the poorly performed prior
        accepted_samples = samples[norm_ikelihoods >= unif]

        # Output the Posterior
        par_names = self.engine.ExpDesign.par_names
        if sigma2_prior is not None:
            for name in self.Discrepancy.name:
                par_names.append(name)

        return pd.DataFrame(accepted_samples, columns=sigma2_prior)

    # -------------------------------------------------------------------------
    def _posterior_predictive(self):
        """
        Stores the prior- and posterior predictive samples, i.e. model
        evaluations using the samples, into hdf5 files.

        priorPredictive.hdf5 : Prior predictive samples.
        postPredictive_wo_noise.hdf5 : Posterior predictive samples without
        the additive noise.
        postPredictive.hdf5 : Posterior predictive samples with the additive
        noise.

        Returns
        -------
        None.

        """

        MetaModel = self.MetaModel
        Model = self.engine.Model

        # Make a directory to save the prior/posterior predictive
        out_dir = f'Outputs_Bayes_{Model.name}_{self.name}'
        os.makedirs(out_dir, exist_ok=True)

        # Read observation data and perturb it if requested
        if self.measured_data is None:
            self.measured_data = Model.read_observation(case=self.name)

        if not isinstance(self.measured_data, pd.DataFrame):
            self.measured_data = pd.DataFrame(self.measured_data)

        # X_values
        x_values = self.engine.ExpDesign.x_values

        try:
            sigma2_prior = self.Discrepancy.sigma2_prior
        except:
            sigma2_prior = None

        # Extract posterior samples
        posterior_df = self.posterior_df

        # Take care of the sigma2
        if sigma2_prior is not None:
            try:
                sigma2s = posterior_df[self.Discrepancy.name].values
                posterior_df = posterior_df.drop(
                    labels=self.Discrepancy.name, axis=1
                    )
            except:
                sigma2s = self.sigma2s

        # Posterior predictive
        if self.emulator:
            if self.inference_method == 'rejection':
                prior_pred = self.__mean_pce_prior_pred
            if self.name.lower() != 'calib':
                post_pred = self.__mean_pce_prior_pred
                post_pred_std = self._std_pce_prior_pred
            else:
                post_pred, post_pred_std = MetaModel.eval_metamodel(
                    samples=posterior_df.values
                    )

        else:
            if self.inference_method == 'rejection':
                prior_pred = self.__model_prior_pred
            if self.name.lower() != 'calib':
                post_pred = self.__mean_pce_prior_pred,
                post_pred_std = self._std_pce_prior_pred
            else:
                post_pred = self._eval_model(
                    samples=posterior_df.values, key='PostPred'
                    )
        # Correct the predictions with Model discrepancy
        if hasattr(self, 'error_model') and self.error_model:
            y_hat, y_std = self.error_MetaModel.eval_model_error(
                self.bias_inputs, post_pred
                )
            post_pred, post_pred_std = y_hat, y_std

        # Add discrepancy from likelihood Sample to the current posterior runs
        total_sigma2 = self.Discrepancy.total_sigma2
        post_pred_withnoise = copy.deepcopy(post_pred)
        for varIdx, var in enumerate(Model.Output.names):
            for i in range(len(post_pred[var])):
                pred = post_pred[var][i]

                # Known sigma2s
                clean_sigma2 = total_sigma2[var][~np.isnan(total_sigma2[var])]
                tot_sigma2 = clean_sigma2[:len(pred)]
                cov = np.diag(tot_sigma2)

                # Check the type error term
                if sigma2_prior is not None:
                    # Inferred sigma2s
                    if hasattr(self, 'bias_inputs') and \
                       not hasattr(self, 'error_model'):
                        # TODO: Infer a Bias model usig GPR
                        bias_inputs = np.hstack((
                            self.bias_inputs[var], pred.reshape(-1, 1)))
                        params = sigma2s[i, varIdx*3:(varIdx+1)*3]
                        cov = self._kernel_rbf(bias_inputs, params)
                    else:
                        # Infer equal sigma2s
                        try:
                            sigma2 = sigma2s[i, varIdx]
                        except TypeError:
                            sigma2 = 0.0

                        # Convert biasSigma2s to a covMatrix
                        cov += sigma2 * np.eye(len(pred))

                if self.emulator:
                    if hasattr(MetaModel, 'rmse') and \
                       MetaModel.rmse is not None:
                        stdPCE = MetaModel.rmse[var]
                    else:
                        stdPCE = post_pred_std[var][i]
                    # Expected value of variance (Assump: i.i.d stds)
                    cov += np.diag(stdPCE**2)

                # Sample a multivariate normal distribution with mean of
                # prediction and variance of cov
                post_pred_withnoise[var][i] = np.random.multivariate_normal(
                    pred, cov, 1
                    )

        # ----- Prior Predictive -----
        if self.inference_method.lower() == 'rejection':
            # Create hdf5 metadata
            hdf5file = f'{out_dir}/priorPredictive.hdf5'
            hdf5_exist = os.path.exists(hdf5file)
            if hdf5_exist:
                os.remove(hdf5file)
            file = h5py.File(hdf5file, 'a')

            # Store x_values
            if type(x_values) is dict:
                grp_x_values = file.create_group("x_values/")
                for varIdx, var in enumerate(Model.Output.names):
                    grp_x_values.create_dataset(var, data=x_values[var])
            else:
                file.create_dataset("x_values", data=x_values)

            # Store posterior predictive
            grpY = file.create_group("EDY/")
            for varIdx, var in enumerate(Model.Output.names):
                grpY.create_dataset(var, data=prior_pred[var])

        # ----- Posterior Predictive only model evaluations -----
        # Create hdf5 metadata
        hdf5file = out_dir+'/postPredictive_wo_noise.hdf5'
        hdf5_exist = os.path.exists(hdf5file)
        if hdf5_exist:
            os.remove(hdf5file)
        file = h5py.File(hdf5file, 'a')

        # Store x_values
        if type(x_values) is dict:
            grp_x_values = file.create_group("x_values/")
            for varIdx, var in enumerate(Model.Output.names):
                grp_x_values.create_dataset(var, data=x_values[var])
        else:
            file.create_dataset("x_values", data=x_values)

        # Store posterior predictive
        grpY = file.create_group("EDY/")
        for varIdx, var in enumerate(Model.Output.names):
            grpY.create_dataset(var, data=post_pred[var])

        # ----- Posterior Predictive with noise -----
        # Create hdf5 metadata
        hdf5file = out_dir+'/postPredictive.hdf5'
        hdf5_exist = os.path.exists(hdf5file)
        if hdf5_exist:
            os.remove(hdf5file)
        file = h5py.File(hdf5file, 'a')

        # Store x_values
        if type(x_values) is dict:
            grp_x_values = file.create_group("x_values/")
            for varIdx, var in enumerate(Model.Output.names):
                grp_x_values.create_dataset(var, data=x_values[var])
        else:
            file.create_dataset("x_values", data=x_values)

        # Store posterior predictive
        grpY = file.create_group("EDY/")
        for varIdx, var in enumerate(Model.Output.names):
            grpY.create_dataset(var, data=post_pred_withnoise[var])

        return

    # -------------------------------------------------------------------------
    def _plot_max_a_posteriori(self):
        """
        Plots the response of the model output against that of the metamodel at
        the maximum a posteriori sample (mean or mode of posterior.)

        Returns
        -------
        None.

        """

        MetaModel = self.MetaModel
        Model = self.engine.Model
        out_dir = f'Outputs_Bayes_{Model.name}_{self.name}'
        opt_sigma = self.Discrepancy.opt_sigma

        # -------- Find MAP and run MetaModel and origModel --------
        # Compute the MAP
        if self.max_a_posteriori.lower() == 'mean':
            if opt_sigma == "B":
                Posterior_df = self.posterior_df.values
            else:
                Posterior_df = self.posterior_df.values[:, :-Model.n_outputs]
            map_theta = Posterior_df.mean(axis=0).reshape(
                (1, MetaModel.n_params))
        else:
            map_theta = stats.mode(Posterior_df.values, axis=0)[0]
        # Prin report
        print("\nPoint estimator:\n", map_theta[0])

        # Run the models for MAP
        # MetaModel
        map_metamodel_mean, map_metamodel_std = MetaModel.eval_metamodel(
            samples=map_theta)
        self.map_metamodel_mean = map_metamodel_mean
        self.map_metamodel_std = map_metamodel_std

        # origModel
        map_orig_model = self._eval_model(samples=map_theta)
        self.map_orig_model = map_orig_model

        # Extract slicing index
        x_values = map_orig_model['x_values']

        # List of markers and colors
        Color = ['k', 'b', 'g', 'r']
        Marker = 'x'

        # Create a PdfPages object
        pdf = PdfPages(f'./{out_dir}MAP_PCE_vs_Model_{self.name}.pdf')
        fig = plt.figure()
        for i, key in enumerate(Model.Output.names):

            y_val = map_orig_model[key]
            y_pce_val = map_metamodel_mean[key]
            y_pce_val_std = map_metamodel_std[key]

            plt.plot(x_values, y_val, color=Color[i], marker=Marker,
                     lw=2.0, label='$Y_{MAP}^{M}$')

            plt.plot(
                x_values, y_pce_val[i], color=Color[i], lw=2.0,
                marker=Marker, linestyle='--', label='$Y_{MAP}^{PCE}$'
                )
            # plot the confidence interval
            plt.fill_between(
                x_values, y_pce_val[i] - 1.96*y_pce_val_std[i],
                y_pce_val[i] + 1.96*y_pce_val_std[i],
                color=Color[i], alpha=0.15
                )

            # Calculate the adjusted R_squared and RMSE
            R2 = r2_score(y_pce_val.reshape(-1, 1), y_val.reshape(-1, 1))
            rmse = np.sqrt(mean_squared_error(y_pce_val, y_val))

            plt.ylabel(key)
            plt.xlabel("Time [s]")
            plt.title(f'Model vs MetaModel {key}')

            ax = fig.axes[0]
            leg = ax.legend(loc='best', frameon=True)
            fig.canvas.draw()
            p = leg.get_window_extent().inverse_transformed(ax.transAxes)
            ax.text(
                p.p0[1]-0.05, p.p1[1]-0.25,
                f'RMSE = {rmse:.3f}\n$R^2$ = {R2:.3f}',
                transform=ax.transAxes, color='black',
                bbox=dict(facecolor='none', edgecolor='black',
                          boxstyle='round,pad=1'))

            plt.show()

            # save the current figure
            pdf.savefig(fig, bbox_inches='tight')

            # Destroy the current plot
            plt.clf()

        pdf.close()

    # -------------------------------------------------------------------------
    def _plot_post_predictive(self):
        """
        Plots the posterior predictives against the observation data.

        Returns
        -------
        None.

        """

        Model = self.engine.Model
        out_dir = f'Outputs_Bayes_{Model.name}_{self.name}'
        # Plot the posterior predictive
        for out_idx, out_name in enumerate(Model.Output.names):
            fig, ax = plt.subplots()
            with sns.axes_style("ticks"):
                x_key = list(self.measured_data)[0]

                # --- Read prior and posterior predictive ---
                if self.inference_method == 'rejection' and \
                   self.name.lower() != 'valid':
                    #  --- Prior ---
                    # Load posterior predictive
                    f = h5py.File(
                        f'{out_dir}/priorPredictive.hdf5', 'r+')

                    try:
                        x_coords = np.array(f[f"x_values/{out_name}"])
                    except:
                        x_coords = np.array(f["x_values"])

                    X_values = np.repeat(x_coords, 10000)

                    prior_pred_df = {}
                    prior_pred_df[x_key] = X_values
                    prior_pred_df[out_name] = np.array(
                        f[f"EDY/{out_name}"])[:10000].flatten('F')
                    prior_pred_df = pd.DataFrame(prior_pred_df)

                    tags_post = ['prior'] * len(prior_pred_df)
                    prior_pred_df.insert(
                        len(prior_pred_df.columns), "Tags", tags_post,
                        True)
                    f.close()

                    # --- Posterior ---
                    f = h5py.File(f"{out_dir}/postPredictive.hdf5", 'r+')

                    X_values = np.repeat(
                        x_coords, np.array(f[f"EDY/{out_name}"]).shape[0])

                    post_pred_df = {}
                    post_pred_df[x_key] = X_values
                    post_pred_df[out_name] = np.array(
                        f[f"EDY/{out_name}"]).flatten('F')

                    post_pred_df = pd.DataFrame(post_pred_df)

                    tags_post = ['posterior'] * len(post_pred_df)
                    post_pred_df.insert(
                        len(post_pred_df.columns), "Tags", tags_post, True)
                    f.close()
                    # Concatenate two dataframes based on x_values
                    frames = [prior_pred_df, post_pred_df]
                    all_pred_df = pd.concat(frames)

                    # --- Plot posterior predictive ---
                    sns.violinplot(
                        x_key, y=out_name, data=all_pred_df, hue="Tags",
                        legend=False, ax=ax, split=True, inner=None,
                        color=".8")

                    # --- Plot Data ---
                    # Find the x,y coordinates for each point
                    x_coords = np.arange(x_coords.shape[0])
                    first_header = list(self.measured_data)[0]
                    obs_data = self.measured_data.round({first_header: 6})
                    sns.pointplot(
                        x=first_header, y=out_name, color='g', markers='x',
                        linestyles='', capsize=16, data=obs_data, ax=ax)

                    ax.errorbar(
                        x_coords, obs_data[out_name].values,
                        yerr=1.96*self.measurement_error[out_name],
                        ecolor='g', fmt=' ', zorder=-1)

                    # Add labels to the legend
                    handles, labels = ax.get_legend_handles_labels()
                    labels.append('Data')

                    data_marker = mlines.Line2D(
                        [], [], color='lime', marker='+', linestyle='None',
                        markersize=10)
                    handles.append(data_marker)

                    # Add legend
                    ax.legend(handles=handles, labels=labels, loc='best',
                              fontsize='large', frameon=True)

                else:
                    # Load posterior predictive
                    f = h5py.File(f"{out_dir}/postPredictive.hdf5", 'r+')

                    try:
                        x_coords = np.array(f[f"x_values/{out_name}"])
                    except:
                        x_coords = np.array(f["x_values"])

                    mu = np.mean(np.array(f[f"EDY/{out_name}"]), axis=0)
                    std = np.std(np.array(f[f"EDY/{out_name}"]), axis=0)

                    # --- Plot posterior predictive ---
                    plt.plot(
                        x_coords, mu, marker='o', color='b',
                        label='Mean Post. Predictive')
                    plt.fill_between(
                        x_coords, mu-1.96*std, mu+1.96*std, color='b',
                        alpha=0.15)

                    # --- Plot Data ---
                    ax.plot(
                        x_coords, self.measured_data[out_name].values,
                        'ko', label='data', markeredgecolor='w')

                    # --- Plot ExpDesign ---
                    orig_ED_Y = self.engine.ExpDesign.Y[out_name]
                    for output in orig_ED_Y:
                        plt.plot(
                            x_coords, output, color='grey', alpha=0.15
                            )

                    # Add labels for axes
                    plt.xlabel('Time [s]')
                    plt.ylabel(out_name)

                    # Add labels to the legend
                    handles, labels = ax.get_legend_handles_labels()

                    patch = Patch(color='b', alpha=0.15)
                    handles.insert(1, patch)
                    labels.insert(1, '95 $\\%$ CI')

                    # Add legend
                    ax.legend(handles=handles, labels=labels, loc='best',
                              frameon=True)

                # Save figure in pdf format
                if self.emulator:
                    plotname = f'/Post_Prior_Perd_{Model.name}_emulator'
                else:
                    plotname = f'/Post_Prior_Perd_{Model.name}'

                fig.savefig(f'./{out_dir}{plotname}_{out_name}.pdf',
                            bbox_inches='tight')
