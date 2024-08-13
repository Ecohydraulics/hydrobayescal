#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.stats as stats
from bayesvalidrox.surrogate_models.exp_designs import ExpDesigns


class Discrepancy:
    """
    Discrepancy class for Bayesian inference method.
    We define the reference or reality to be equal to what we can model and a
    descripancy term \\( \\epsilon \\). We consider the followin format:

    $$\\textbf{y}_{\\text{reality}} = \\mathcal{M}(\\theta) + \\epsilon,$$

    where \\( \\epsilon \\in R^{N_{out}} \\) represents the the effects of
    measurement error and model inaccuracy. For simplicity, it can be defined
    as an additive Gaussian disrepancy with zeromean and given covariance
    matrix \\( \\Sigma \\):

    $$\\epsilon \\sim \\mathcal{N}(\\epsilon|0, \\Sigma). $$

    In the context of model inversion or calibration, an observation point
    \\( \\textbf{y}_i \\in \\mathcal{y} \\) is a realization of a Gaussian
    distribution with mean value of \\(\\mathcal{M}(\\theta) \\) and covariance
    matrix of \\( \\Sigma \\).

    $$ p(\\textbf{y}|\\theta) = \\mathcal{N}(\\textbf{y}|\\mathcal{M}
                                             (\\theta))$$

    The following options are available:

    * Option A: With known redidual covariance matrix \\(\\Sigma\\) for
    independent measurements.

    * Option B: With unknown redidual covariance matrix \\(\\Sigma\\),
    paramethrized as \\(\\Sigma(\\theta_{\\epsilon})=\\sigma^2 \\textbf{I}_
    {N_{out}}\\) with unknown residual variances \\(\\sigma^2\\).
    This term will be jointly infer with the uncertain input parameters. For
    the inversion, you need to define a prior marginal via `Input` class. Note
    that \\(\\sigma^2\\) is only a single scalar multiplier for the diagonal
    entries of the covariance matrix \\(\\Sigma\\).

    Attributes
    ----------
    InputDisc : obj
        Input object. When the \\(\\sigma^2\\) is expected to be inferred
        jointly with the parameters (`Option B`).If multiple output groups are
        defined by `Model.Output.names`, each model output needs to have.
        a prior marginal using the `Input` class. The default is `''`.
    disc_type : str
        Type of the noise definition. `'Gaussian'` is only supported so far.
    parameters : dict or pandas.DataFrame
        Known residual variance \\(\\sigma^2\\), i.e. diagonal entry of the
        covariance matrix of the multivariate normal likelihood in case of
        `Option A`.

    """

    def __init__(self, InputDisc='', disc_type='Gaussian', parameters=None):
        self.InputDisc = InputDisc
        self.disc_type = disc_type
        self.parameters = parameters

    # -------------------------------------------------------------------------
    def get_sample(self, n_samples):
        """
        Generate samples for the \\(\\sigma^2\\), i.e. the diagonal entries of
        the variance-covariance matrix in the multivariate normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples (parameter sets).

        Returns
        -------
        sigma2_prior: array of shape (n_samples, n_params)
            \\(\\sigma^2\\) samples.

        """
        self.n_samples = n_samples # TODO: not used again in here - needed from the outside?
        
        if self.InputDisc == '':
            raise AttributeError('Cannot create new samples, please provide input distributions')
        
        # Create and store BoundTuples
        self.ExpDesign = ExpDesigns(self.InputDisc)
        self.ExpDesign.sampling_method = 'random'
        self.ExpDesign.generate_ED(
            n_samples, max_pce_deg=1
            )
        # TODO: need to recheck the following line
        # This used to simply be the return from the call above
        self.sigma2_prior = self.ExpDesign.X

        # Naive approach: Fit a gaussian kernel to the provided data
        self.ExpDesign.JDist = stats.gaussian_kde(self.ExpDesign.raw_data)

        # Save the names of sigmas
        if len(self.InputDisc.Marginals) != 0:
            self.name = []
            for Marginalidx in range(len(self.InputDisc.Marginals)):
                self.name.append(self.InputDisc.Marginals[Marginalidx].name)

        return self.sigma2_prior
