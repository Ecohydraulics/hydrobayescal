#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing
import scipy.stats as st
from scipy.linalg import cholesky as chol
import warnings
import shutil
os.environ["OMP_NUM_THREADS"] = "1"


class MCMC:
    """
    A class for bayesian inference via a Markov-Chain Monte-Carlo (MCMC)
    Sampler to approximate the posterior distribution of the Bayes theorem:
    $$p(\\theta|\\mathcal{y}) = \\frac{p(\\mathcal{y}|\\theta) p(\\theta)}
                                         {p(\\mathcal{y})}.$$

    This class make inference with emcee package [1] using an Affine Invariant
    Ensemble sampler (AIES) [2].

    [1] Foreman-Mackey, D., Hogg, D.W., Lang, D. and Goodman, J., 2013.emcee:
        the MCMC hammer. Publications of the Astronomical Society of the
        Pacific, 125(925), p.306. https://emcee.readthedocs.io/en/stable/

    [2] Goodman, J. and Weare, J., 2010. Ensemble samplers with affine
        invariance. Communications in applied mathematics and computational
        science, 5(1), pp.65-80.


    Attributes
    ----------
    BayesOpts : obj
        Bayes object.
    """

    def __init__(self, BayesOpts):

        self.BayesOpts = BayesOpts

    def run_sampler(self, observation, total_sigma2):

        BayesObj = self.BayesOpts
        MetaModel = BayesObj.engine.MetaModel
        Model = BayesObj.engine.Model
        Discrepancy = self.BayesOpts.Discrepancy
        n_cpus = Model.n_cpus
        priorDist = BayesObj.engine.ExpDesign.JDist
        ndim = MetaModel.n_params
        self.counter = 0
        output_dir = f'Outputs_Bayes_{Model.name}_{self.BayesOpts.name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.observation = observation
        self.total_sigma2 = total_sigma2

        # Unpack mcmc parameters given to BayesObj.mcmc_params
        self.initsamples = None
        self.nwalkers = 100
        self.nburn = 200
        self.nsteps = 100000
        self.moves = None
        self.mp = False
        self.verbose = False

        # Extract initial samples
        if 'init_samples' in BayesObj.mcmc_params:
            self.initsamples = BayesObj.mcmc_params['init_samples']
            if isinstance(self.initsamples, pd.DataFrame):
                self.initsamples = self.initsamples.values

        # Extract number of steps per walker
        if 'n_steps' in BayesObj.mcmc_params:
            self.nsteps = int(BayesObj.mcmc_params['n_steps'])
        # Extract number of walkers (chains)
        if 'n_walkers' in BayesObj.mcmc_params:
            self.nwalkers = int(BayesObj.mcmc_params['n_walkers'])
        # Extract moves
        if 'moves' in BayesObj.mcmc_params:
            self.moves = BayesObj.mcmc_params['moves']
        # Extract multiprocessing
        if 'multiprocessing' in BayesObj.mcmc_params:
            self.mp = BayesObj.mcmc_params['multiprocessing']
        # Extract verbose
        if 'verbose' in BayesObj.mcmc_params:
            self.verbose = BayesObj.mcmc_params['verbose']

        # Set initial samples
        np.random.seed(0)
        if self.initsamples is None:
            try:
                initsamples = priorDist.sample(self.nwalkers).T
            except:
                # when aPCE selected - gaussian kernel distribution
                inputSamples = MetaModel.ExpDesign.raw_data.T
                random_indices = np.random.choice(
                    len(inputSamples), size=self.nwalkers, replace=False
                    )
                initsamples = inputSamples[random_indices]

        else:
            if self.initsamples.ndim == 1:
                # When MAL is given.
                theta = self.initsamples
                initsamples = [theta + 1e-1*np.multiply(
                    np.random.randn(ndim), theta) for i in
                               range(self.nwalkers)]
            else:
                # Pick samples based on a uniform dist between min and max of
                # each dim
                initsamples = np.zeros((self.nwalkers, ndim))
                bound_tuples = []
                for idx_dim in range(ndim):
                    lower = np.min(self.initsamples[:, idx_dim])
                    upper = np.max(self.initsamples[:, idx_dim])
                    bound_tuples.append((lower, upper))
                    dist = st.uniform(loc=lower, scale=upper-lower)
                    initsamples[:, idx_dim] = dist.rvs(size=self.nwalkers)

                # Update lower and upper
                MetaModel.ExpDesign.bound_tuples = bound_tuples

        # Check if sigma^2 needs to be inferred
        if Discrepancy.opt_sigma != 'B':
            sigma2_samples = Discrepancy.get_sample(self.nwalkers)

            # Update initsamples
            initsamples = np.hstack((initsamples, sigma2_samples))

            # Update ndim
            ndim = initsamples.shape[1]

            # Discrepancy bound
            disc_bound_tuple = Discrepancy.ExpDesign.bound_tuples

            # Update bound_tuples
            BayesObj.engine.ExpDesign.bound_tuples += disc_bound_tuple

        print("\n>>>> Bayesian inference with MCMC for "
              f"{self.BayesOpts.name} started. <<<<<<")

        # Set up the backend
        filename = f"{output_dir}/emcee_sampler.h5"
        backend = emcee.backends.HDFBackend(filename)
        # Clear the backend in case the file already exists
        backend.reset(self.nwalkers, ndim)

        # Define emcee sampler
        # Here we'll set up the computation. emcee combines multiple "walkers",
        # each of which is its own MCMC chain. The number of trace results will
        # be nwalkers * nsteps.
        if self.mp:
            # Run in parallel
            if n_cpus is None:
                n_cpus = multiprocessing.cpu_count()

            with multiprocessing.Pool(n_cpus) as pool:
                sampler = emcee.EnsembleSampler(
                    self.nwalkers, ndim, self.log_posterior, moves=self.moves,
                    pool=pool, backend=backend
                    )

                # Check if a burn-in phase is needed!
                if self.initsamples is None:
                    # Burn-in
                    print("\n Burn-in period is starting:")
                    pos = sampler.run_mcmc(
                        initsamples, self.nburn, progress=True
                        )

                    # Reset sampler
                    sampler.reset()
                    pos = pos.coords
                else:
                    pos = initsamples

                # Production run
                print("\n Production run is starting:")
                pos, prob, state = sampler.run_mcmc(
                    pos, self.nsteps, progress=True
                    )

        else:
            # Run in series and monitor the convergence
            sampler = emcee.EnsembleSampler(
                self.nwalkers, ndim, self.log_posterior, moves=self.moves,
                backend=backend, vectorize=True
                )

            # Check if a burn-in phase is needed!
            if self.initsamples is None:
                # Burn-in
                print("\n Burn-in period is starting:")
                pos = sampler.run_mcmc(
                    initsamples, self.nburn, progress=True
                    )

                # Reset sampler
                sampler.reset()
                pos = pos.coords
            else:
                pos = initsamples

            # Production run
            print("\n Production run is starting:")

            # Track how the average autocorrelation time estimate changes
            autocorrIdx = 0
            autocorr = np.empty(self.nsteps)
            tauold = np.inf
            autocorreverynsteps = 50

            # sample step by step using the generator sampler.sample
            for sample in sampler.sample(pos,
                                         iterations=self.nsteps,
                                         tune=True,
                                         progress=True):

                # only check convergence every autocorreverynsteps steps
                if sampler.iteration % autocorreverynsteps:
                    continue

                # Train model discrepancy/error
                if hasattr(BayesObj, 'errorModel') and BayesObj.errorModel \
                   and not sampler.iteration % 3 * autocorreverynsteps:
                    try:
                        self.error_MetaModel = self.train_error_model(sampler)
                    except:
                        pass

                # Print the current mean acceptance fraction
                if self.verbose:
                    print("\nStep: {}".format(sampler.iteration))
                    acc_fr = np.mean(sampler.acceptance_fraction)
                    print(f"Mean acceptance fraction: {acc_fr:.3f}")

                # compute the autocorrelation time so far
                # using tol=0 means that we'll always get an estimate even if
                # it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                # average over walkers
                autocorr[autocorrIdx] = np.nanmean(tau)
                autocorrIdx += 1

                # output current autocorrelation estimate
                if self.verbose:
                    print(f"Mean autocorr. time estimate: {np.nanmean(tau):.3f}")
                    list_gr = np.round(self.gelman_rubin(sampler.chain), 3)
                    print("Gelman-Rubin Test*: ", list_gr)

                # check convergence
                converged = np.all(tau*autocorreverynsteps < sampler.iteration)
                converged &= np.all(np.abs(tauold - tau) / tau < 0.01)
                converged &= np.all(self.gelman_rubin(sampler.chain) < 1.1)

                if converged:
                    break
                tauold = tau

        # Posterior diagnostics
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            tau = 5

        if all(np.isnan(tau)):
            tau = 5

        burnin = int(2*np.nanmax(tau))
        thin = int(0.5*np.nanmin(tau)) if int(0.5*np.nanmin(tau)) != 0 else 1
        finalsamples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        acc_fr = np.nanmean(sampler.acceptance_fraction)
        list_gr = np.round(self.gelman_rubin(sampler.chain[:, burnin:]), 3)

        # Print summary
        print('\n')
        print('-'*15 + 'Posterior diagnostics' + '-'*15)
        print(f"Mean auto-correlation time: {np.nanmean(tau):.3f}")
        print(f"Thin: {thin}")
        print(f"Burn-in: {burnin}")
        print(f"Flat chain shape: {finalsamples.shape}")
        print(f"Mean acceptance fraction*: {acc_fr:.3f}")
        print("Gelman-Rubin Test**: ", list_gr)

        print("\n* This value must lay between 0.234 and 0.5.")
        print("** These values must be smaller than 1.1.")
        print('-'*50)

        print(f"\n>>>> Bayesian inference with MCMC for {self.BayesOpts.name} "
              "successfully completed. <<<<<<\n")

        # Extract parameter names and their prior ranges
        par_names = self.BayesOpts.engine.ExpDesign.par_names

        if Discrepancy.opt_sigma != 'B':
            for i in range(len(Discrepancy.InputDisc.Marginals)):
                par_names.append(Discrepancy.InputDisc.Marginals[i].name)

        params_range = self.BayesOpts.engine.ExpDesign.bound_tuples

        # Plot traces
        if self.verbose and self.nsteps < 10000:
            pdf = PdfPages(output_dir+'/traceplots.pdf')
            fig = plt.figure()
            for parIdx in range(ndim):
                # Set up the axes with gridspec
                fig = plt.figure()
                grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
                main_ax = fig.add_subplot(grid[:-1, :3])
                y_hist = fig.add_subplot(grid[:-1, -1], xticklabels=[],
                                         sharey=main_ax)

                for i in range(self.nwalkers):
                    samples = sampler.chain[i, :, parIdx]
                    main_ax.plot(samples, '-')

                    # histogram on the attached axes
                    y_hist.hist(samples[burnin:], 40, histtype='stepfilled',
                                orientation='horizontal', color='gray')

                main_ax.set_ylim(params_range[parIdx])
                main_ax.set_title('traceplot for ' + par_names[parIdx])
                main_ax.set_xlabel('step number')

                # save the current figure
                pdf.savefig(fig, bbox_inches='tight')

                # Destroy the current plot
                plt.clf()

            pdf.close()

        # plot development of autocorrelation estimate
        if not self.mp:
            fig1 = plt.figure()
            steps = autocorreverynsteps*np.arange(1, autocorrIdx+1)
            taus = autocorr[:autocorrIdx]
            plt.plot(steps, steps / autocorreverynsteps, "--k")
            plt.plot(steps, taus)
            plt.xlim(0, steps.max())
            plt.ylim(0, np.nanmax(taus)+0.1*(np.nanmax(taus)-np.nanmin(taus)))
            plt.xlabel("number of steps")
            plt.ylabel(r"mean $\hat{\tau}$")
            fig1.savefig(f"{output_dir}/autocorrelation_time.pdf",
                         bbox_inches='tight')

        # logml_dict = self.marginal_llk_emcee(sampler, self.nburn, logp=None,
        # maxiter=5000)
        # print('\nThe Bridge Sampling Estimation is "
        #       f"{logml_dict['logml']:.5f}.')

        # # Posterior-based expectation of posterior probablity
        # postExpPostLikelihoods = np.mean(sampler.get_log_prob(flat=True)
        # [self.nburn*self.nwalkers:])

        # # Posterior-based expectation of prior densities
        # postExpPrior = np.mean(self.log_prior(emcee_trace.T))

        # # Posterior-based expectation of likelihoods
        # postExpLikelihoods_emcee = postExpPostLikelihoods - postExpPrior

        # # Calculate Kullback-Leibler Divergence
        # KLD_emcee = postExpLikelihoods_emcee - logml_dict['logml']
        # print("Kullback-Leibler divergence: %.5f"%KLD_emcee)

        # # Information Entropy based on Entropy paper Eq. 38
        # infEntropy_emcee = logml_dict['logml'] - postExpPrior -
        #                    postExpLikelihoods_emcee
        # print("Information Entropy: %.5f" %infEntropy_emcee)

        Posterior_df = pd.DataFrame(finalsamples, columns=par_names)

        return Posterior_df

    # -------------------------------------------------------------------------
    def log_prior(self, theta):
        """
        Calculates the log prior likelihood \\( p(\\theta)\\) for the given
        parameter set(s) \\( \\theta \\).

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter sets, i.e. proposals of MCMC chains.

        Returns
        -------
        logprior: float or array of shape n_samples
            Log prior likelihood. If theta has only one row, a single value is
            returned otherwise an array.

        """

        MetaModel = self.BayesOpts.MetaModel
        Discrepancy = self.BayesOpts.Discrepancy

        # Find the number of sigma2 parameters
        if Discrepancy.opt_sigma != 'B':
            disc_bound_tuples = Discrepancy.ExpDesign.bound_tuples
            disc_marginals = Discrepancy.ExpDesign.InputObj.Marginals
            disc_prior_space = Discrepancy.ExpDesign.prior_space
            n_sigma2 = len(disc_bound_tuples)
        else:
            n_sigma2 = -len(theta)
        prior_dist = self.BayesOpts.engine.ExpDesign.prior_space
        params_range = self.BayesOpts.engine.ExpDesign.bound_tuples
        theta = theta if theta.ndim != 1 else theta.reshape((1, -1))
        nsamples = theta.shape[0]
        logprior = -np.inf*np.ones(nsamples)

        for i in range(nsamples):
            # Check if the sample is within the parameters' range
            if self._check_ranges(theta[i], params_range):
                # Check if all dists are uniform, if yes priors are equal.
                if all(MetaModel.input_obj.Marginals[i].dist_type == 'uniform'
                       for i in range(MetaModel.n_params)):
                    logprior[i] = 0.0
                else:
                    logprior[i] = np.log(
                        prior_dist.pdf(theta[i, :-n_sigma2].T)
                        )

                # Check if bias term needs to be inferred
                if Discrepancy.opt_sigma != 'B':
                    if self._check_ranges(theta[i, -n_sigma2:],
                                          disc_bound_tuples):
                        if all('unif' in disc_marginals[i].dist_type for i in
                               range(Discrepancy.ExpDesign.ndim)):
                            logprior[i] = 0.0
                        else:
                            logprior[i] += np.log(
                                disc_prior_space.pdf(theta[i, -n_sigma2:])
                                )

        if nsamples == 1:
            return logprior[0]
        else:
            return logprior

    # -------------------------------------------------------------------------
    def log_likelihood(self, theta):
        """
        Computes likelihood \\( p(\\mathcal{Y}|\\theta)\\) of the performance
        of the (meta-)model in reproducing the observation data.

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter set, i.e. proposals of the MCMC chains.

        Returns
        -------
        log_like : array of shape (n_samples)
            Log likelihood.

        """

        BayesOpts = self.BayesOpts
        MetaModel = BayesOpts.MetaModel
        Discrepancy = self.BayesOpts.Discrepancy

        # Find the number of sigma2 parameters
        if Discrepancy.opt_sigma != 'B':
            disc_bound_tuples = Discrepancy.ExpDesign.bound_tuples
            n_sigma2 = len(disc_bound_tuples)
        else:
            n_sigma2 = -len(theta)
        # Check if bias term needs to be inferred
        if Discrepancy.opt_sigma != 'B':
            sigma2 = theta[:, -n_sigma2:]
            theta = theta[:, :-n_sigma2]
        else:
            sigma2 = None
        theta = theta if theta.ndim != 1 else theta.reshape((1, -1))

        # Evaluate Model/MetaModel at theta
        mean_pred, BayesOpts._std_pce_prior_pred = self.eval_model(theta)

        # Surrogate model's error using RMSE of test data
        surrError = MetaModel.rmse if hasattr(MetaModel, 'rmse') else None

        # Likelihood
        log_like = BayesOpts.normpdf(
            mean_pred, self.observation, self.total_sigma2, sigma2,
            std=surrError
            )
        return log_like

    # -------------------------------------------------------------------------
    def log_posterior(self, theta):
        """
        Computes the posterior likelihood \\(p(\\theta| \\mathcal{Y})\\) for
        the given parameterset.

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter set, i.e. proposals of the MCMC chains.

        Returns
        -------
        log_like : array of shape (n_samples)
            Log posterior likelihood.

        """

        nsamples = 1 if theta.ndim == 1 else theta.shape[0]

        if nsamples == 1:
            if self.log_prior(theta) == -np.inf:
                return -np.inf
            else:
                # Compute log prior
                log_prior = self.log_prior(theta)
                # Compute log Likelihood
                log_likelihood = self.log_likelihood(theta)

                return log_prior + log_likelihood
        else:
            # Compute log prior
            log_prior = self.log_prior(theta)

            # Initialize log_likelihood
            log_likelihood = -np.inf*np.ones(nsamples)

            # find the indices for -inf sets
            non_inf_idx = np.where(log_prior != -np.inf)[0]

            # Compute loLikelihoods
            if non_inf_idx.size != 0:
                log_likelihood[non_inf_idx] = self.log_likelihood(
                    theta[non_inf_idx]
                    )

            return log_prior + log_likelihood

    # -------------------------------------------------------------------------
    def eval_model(self, theta):
        """
        Evaluates the (meta-) model at the given theta.

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter set, i.e. proposals of the MCMC chains.

        Returns
        -------
        mean_pred : dict
            Mean model prediction.
        std_pred : dict
            Std of model prediction.

        """

        BayesObj = self.BayesOpts
        MetaModel = BayesObj.MetaModel
        Model = BayesObj.engine.Model

        if BayesObj.emulator:
            # Evaluate the MetaModel
            mean_pred, std_pred = MetaModel.eval_metamodel(samples=theta)
        else:
            # Evaluate the origModel
            mean_pred, std_pred = dict(), dict()

            model_outs, _ = Model.run_model_parallel(
                theta, prevRun_No=self.counter,
                key_str='_MCMC', mp=False, verbose=False)

            # Save outputs in respective dicts
            for varIdx, var in enumerate(Model.Output.names):
                mean_pred[var] = model_outs[var]
                std_pred[var] = np.zeros((mean_pred[var].shape))

            # Remove the folder
            if Model.link_type.lower() != 'function':
                shutil.rmtree(f"{Model.name}_MCMC_{self.counter+1}")

            # Add one to the counter
            self.counter += 1

        if hasattr(self, 'error_MetaModel') and BayesObj.error_model:
            meanPred, stdPred = self.error_MetaModel.eval_model_error(
                BayesObj.BiasInputs, mean_pred
                )

        return mean_pred, std_pred

    # -------------------------------------------------------------------------
    def train_error_model(self, sampler):
        """
        Trains an error model using a Gaussian Process Regression.

        Parameters
        ----------
        sampler : obj
            emcee sampler.

        Returns
        -------
        error_MetaModel : obj
            A error model.

        """
        BayesObj = self.BayesOpts
        MetaModel = BayesObj.MetaModel

        # Prepare the poster samples
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            tau = 5

        if all(np.isnan(tau)):
            tau = 5

        burnin = int(2*np.nanmax(tau))
        thin = int(0.5*np.nanmin(tau)) if int(0.5*np.nanmin(tau)) != 0 else 1
        finalsamples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        posterior = finalsamples[:, :MetaModel.n_params]

        # Select posterior mean as MAP
        map_theta = posterior.mean(axis=0).reshape((1, MetaModel.n_params))
        # MAP_theta = st.mode(Posterior_df,axis=0)[0]

        # Evaluate the (meta-)model at the MAP
        y_map, y_std_map = MetaModel.eval_metamodel(samples=map_theta)

        # Train a GPR meta-model using MAP
        error_MetaModel = MetaModel.create_model_error(
            BayesObj.BiasInputs, y_map, name='Calib')

        return error_MetaModel

    # -------------------------------------------------------------------------
    def gelman_rubin(self, chain, return_var=False):
        """
        The potential scale reduction factor (PSRF) defined by the variance
        within one chain, W, with the variance between chains B.
        Both variances are combined in a weighted sum to obtain an estimate of
        the variance of a parameter \\( \\theta \\).The square root of the
        ratio of this estimates variance to the within chain variance is called
        the potential scale reduction.
        For a well converged chain it should approach 1. Values greater than
        1.1 typically indicate that the chains have not yet fully converged.

        Source: http://joergdietrich.github.io/emcee-convergence.html

        https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py

        Parameters
        ----------
        chain : array (n_walkers, n_steps, n_params)
            The emcee ensamples.

        Returns
        -------
        R_hat : float
            The Gelman-Robin values.

        """
        m_chains, n_iters = chain.shape[:2]

        # Calculate between-chain variance
        θb = np.mean(chain, axis=1)
        θbb = np.mean(θb, axis=0)
        B_over_n = ((θbb - θb)**2).sum(axis=0)
        B_over_n /= (m_chains - 1)

        # Calculate within-chain variances
        ssq = np.var(chain, axis=1, ddof=1)
        W = np.mean(ssq, axis=0)

        # (over) estimate of variance
        var_θ = W * (n_iters - 1) / n_iters + B_over_n

        if return_var:
            return var_θ
        else:
            # The square root of the ratio of this estimates variance to the
            # within chain variance
            R_hat = np.sqrt(var_θ / W)
            return R_hat

    # -------------------------------------------------------------------------
    def marginal_llk_emcee(self, sampler, nburn=None, logp=None, maxiter=1000):
        """
        The Bridge Sampling Estimator of the Marginal Likelihood based on
        https://gist.github.com/junpenglao/4d2669d69ddfe1d788318264cdcf0583

        Parameters
        ----------
        sampler : TYPE
            MultiTrace, result of MCMC run.
        nburn : int, optional
            Number of burn-in step. The default is None.
        logp : TYPE, optional
            Model Log-probability function. The default is None.
        maxiter : int, optional
            Maximum number of iterations. The default is 1000.

        Returns
        -------
        marg_llk : dict
            Estimated Marginal log-Likelihood.

        """
        r0, tol1, tol2 = 0.5, 1e-10, 1e-4

        if logp is None:
            logp = sampler.log_prob_fn

        # Split the samples into two parts
        # Use the first 50% for fiting the proposal distribution
        # and the second 50% in the iterative scheme.
        if nburn is None:
            mtrace = sampler.chain
        else:
            mtrace = sampler.chain[:, nburn:, :]

        nchain, len_trace, nrofVars = mtrace.shape

        N1_ = len_trace // 2
        N1 = N1_*nchain
        N2 = len_trace*nchain - N1

        samples_4_fit = np.zeros((nrofVars, N1))
        samples_4_iter = np.zeros((nrofVars, N2))
        effective_n = np.zeros((nrofVars))

        # matrix with already transformed samples
        for var in range(nrofVars):

            # for fitting the proposal
            x = mtrace[:, :N1_, var]

            samples_4_fit[var, :] = x.flatten()
            # for the iterative scheme
            x2 = mtrace[:, N1_:, var]
            samples_4_iter[var, :] = x2.flatten()

            # effective sample size of samples_4_iter, scalar
            effective_n[var] = self._my_ESS(x2)

        # median effective sample size (scalar)
        neff = np.median(effective_n)

        # get mean & covariance matrix and generate samples from proposal
        m = np.mean(samples_4_fit, axis=1)
        V = np.cov(samples_4_fit)
        L = chol(V, lower=True)

        # Draw N2 samples from the proposal distribution
        gen_samples = m[:, None] + np.dot(
            L, st.norm.rvs(0, 1, size=samples_4_iter.shape)
            )

        # Evaluate proposal distribution for posterior & generated samples
        q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
        q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)

        # Evaluate unnormalized posterior for posterior & generated samples
        q11 = logp(samples_4_iter.T)
        q21 = logp(gen_samples.T)

        # Run iterative scheme:
        tmp = self._iterative_scheme(
            N1, N2, q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r'
            )
        if ~np.isfinite(tmp['logml']):
            warnings.warn(
                "Logml could not be estimated within maxiter, rerunning with "
                "adjusted starting value. Estimate might be more variable than"
                " usual.")
            # use geometric mean as starting value
            r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
            tmp = self._iterative_scheme(
                q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml'
                )

        marg_llk = dict(
            logml=tmp['logml'], niter=tmp['niter'], method="normal",
            q11=q11, q12=q12, q21=q21, q22=q22
            )
        return marg_llk

    # -------------------------------------------------------------------------
    def _iterative_scheme(self, N1, N2, q11, q12, q21, q22, r0, neff, tol,
                          maxiter, criterion):
        """
        Iterative scheme as proposed in Meng and Wong (1996) to estimate the
        marginal likelihood

        """
        l1 = q11 - q12
        l2 = q21 - q22
        # To increase numerical stability,
        # subtracting the median of l1 from l1 & l2 later
        lstar = np.median(l1)
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)
        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol

        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warnings.warn(
                    """Infinite value in iterative scheme, returning NaN.
                     Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion == 'r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion == 'logml':
                criterion_val = np.abs((logml - logmlold)/logml)

        if i >= maxiter:
            return dict(logml=np.NaN, niter=i, r_vals=np.asarray(r_vals))
        else:
            return dict(logml=logml, niter=i)

    # -------------------------------------------------------------------------
    def _my_ESS(self, x):
        """
        Compute the effective sample size of estimand of interest.
        Vectorised implementation.
        https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py


        Parameters
        ----------
        x : array of shape (n_walkers, n_steps)
            MCMC Samples.

        Returns
        -------
        int
            Effective sample size.

        """
        m_chains, n_iters = x.shape

        def variogram(t):
            variogram = ((x[:, t:] - x[:, :(n_iters - t)])**2).sum()
            variogram /= (m_chains * (n_iters - t))
            return variogram

        post_var = self.gelman_rubin(x, return_var=True)

        t = 1
        rho = np.ones(n_iters)
        negative_autocorr = False

        # Iterate until the sum of consecutive estimates of autocorrelation is
        # negative
        while not negative_autocorr and (t < n_iters):
            rho[t] = 1 - variogram(t) / (2 * post_var)

            if not t % 2:
                negative_autocorr = sum(rho[t-1:t+1]) < 0

            t += 1

        return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

    # -------------------------------------------------------------------------
    def _check_ranges(self, theta, ranges):
        """
        This function checks if theta lies in the given ranges.

        Parameters
        ----------
        theta : array
            Proposed parameter set.
        ranges : nested list
            List of the praremeter ranges.

        Returns
        -------
        c : bool
            If it lies in the given range, it return True else False.

        """
        c = True
        # traverse in the list1
        for i, bounds in enumerate(ranges):
            x = theta[i]
            # condition check
            if x < bounds[0] or x > bounds[1]:
                c = False
                return c
        return c
