#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of metamodel as either PC, aPC or GPE
"""

import warnings
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from tqdm import tqdm
from sklearn.decomposition import PCA as sklearnPCA
import sklearn.linear_model as lm
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import os
from joblib import Parallel, delayed
import copy

from .input_space import InputSpace
from .glexindex import glexindex
from .eval_rec_rule import eval_univ_basis
from .reg_fast_ard import RegressionFastARD
from .reg_fast_laplace import RegressionFastLaplace
from .orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from .bayes_linear import VBLinearRegression, EBLinearRegression
from .apoly_construction import apoly_construction
warnings.filterwarnings("ignore")
# Load the mplstyle
plt.style.use(os.path.join(os.path.split(__file__)[0],
                           '../', 'bayesvalidrox.mplstyle'))


class MetaModel():
    """
    Meta (surrogate) model

    This class trains a surrogate model. It accepts an input object (input_obj)
    containing the specification of the distributions for uncertain parameters
    and a model object with instructions on how to run the computational model.

    Attributes
    ----------
    input_obj : obj
        Input object with the information on the model input parameters.
    meta_model_type : str
        Surrogate model types. Three surrogate model types are supported:
        polynomial chaos expansion (`PCE`), arbitrary PCE (`aPCE`) and
        Gaussian process regression (`GPE`). Default is PCE.
    pce_reg_method : str
        PCE regression method to compute the coefficients. The following
        regression methods are available:

        1. OLS: Ordinary Least Square method
        2. BRR: Bayesian Ridge Regression
        3. LARS: Least angle regression
        4. ARD: Bayesian ARD Regression
        5. FastARD: Fast Bayesian ARD Regression
        6. VBL: Variational Bayesian Learning
        7. EBL: Emperical Bayesian Learning
        Default is `OLS`.
    bootstrap_method : str
        Bootstraping method. Options are `'normal'` and `'fast'`. The default
        is `'fast'`. It means that in each iteration except the first one, only
        the coefficent are recalculated with the ordinary least square method.
    n_bootstrap_itrs : int
        Number of iterations for the bootstrap sampling. The default is `1`.
    pce_deg : int or list of int
        Polynomial degree(s). If a list is given, an adaptive algorithm is used
        to find the best degree with the lowest Leave-One-Out cross-validation
        (LOO) error (or the highest score=1-LOO). Default is `1`.
    pce_q_norm : float
        Hyperbolic (or q-norm) truncation for multi-indices of multivariate
        polynomials. Default is `1.0`.
    dim_red_method : str
        Dimensionality reduction method for the output space. The available
        method is based on principal component analysis (PCA). The Default is
        `'no'`. There are two ways to select number of components: use
        percentage of the explainable variance threshold (between 0 and 100)
        (Option A) or direct prescription of components' number (Option B):

            >>> MetaModelOpts.dim_red_method = 'PCA'
            >>> MetaModelOpts.var_pca_threshold = 99.999  # Option A
            >>> MetaModelOpts.n_pca_components = 12 # Option B
    apply_constraints : bool
        If set to true constraints will be applied during training. 
        In this case the training uses OLS. In this version the constraints 
        need to be set explicitly in this class.

    verbose : bool
        Prints summary of the regression results. Default is `False`.

    Note
    -------
    To define the sampling methods and the training set, an experimental design
    instance shall be defined. This can be done by:

    >>> MetaModelOpts.add_InputSpace()

    Two experimental design schemes are supported: one-shot (`normal`) and
    adaptive sequential (`sequential`) designs.
    For experimental design refer to `InputSpace`.

    """

    def __init__(self, input_obj, meta_model_type='PCE',
                 pce_reg_method='OLS', bootstrap_method='fast',
                 n_bootstrap_itrs=1, pce_deg=1, pce_q_norm=1.0,
                 dim_red_method='no', apply_constraints = False, 
                 verbose=False):

        self.input_obj = input_obj
        self.meta_model_type = meta_model_type
        self.pce_reg_method = pce_reg_method
        self.bootstrap_method = bootstrap_method
        self.n_bootstrap_itrs = n_bootstrap_itrs
        self.pce_deg = pce_deg
        self.pce_q_norm = pce_q_norm
        self.dim_red_method = dim_red_method
        self.apply_constraints = apply_constraints
        self.verbose = verbose
 
    def build_metamodel(self, n_init_samples = None) -> None:
        """
        Builds the parts for the metamodel (polynomes,...) that are neede before fitting.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        # Generate general warnings
        if self.apply_constraints or self.pce_reg_method.lower() == 'ols':
            print('There are no estimations of surrogate uncertainty available'
                  ' for the chosen regression options. This might lead to issues'
                  ' in later steps.')
        
        # Add InputSpace to MetaModel if it does not have any
        if not hasattr(self, 'InputSpace'):
            self.InputSpace = InputSpace(self.input_obj)
            self.InputSpace.n_init_samples = n_init_samples
            self.InputSpace.init_param_space(np.max(self.pce_deg))
            
        self.ndim = self.InputSpace.ndim
        
        if not hasattr(self, 'CollocationPoints'):
            raise AttributeError('Please provide samples to the metamodel before building it.')
            
        # Transform input samples
        # TODO: this is probably not yet correct! Make 'method' variable
        self.CollocationPoints = self.InputSpace.transform(self.CollocationPoints, method='user') 

        
        self.n_params = len(self.input_obj.Marginals)
        
        # Generate polynomials
        if self.meta_model_type.lower() != 'gpe':
            self.generate_polynomials(np.max(self.pce_deg))

        # Initialize the nested dictionaries
        if self.meta_model_type.lower() == 'gpe':
            self.gp_poly = self.auto_vivification()
            self.x_scaler = self.auto_vivification()
            self.LCerror = self.auto_vivification()
        else:
            self.deg_dict = self.auto_vivification()
            self.q_norm_dict = self.auto_vivification()
            self.coeffs_dict = self.auto_vivification()
            self.basis_dict = self.auto_vivification()
            self.score_dict = self.auto_vivification()
            self.clf_poly = self.auto_vivification()
            self.LCerror = self.auto_vivification()
        if self.dim_red_method.lower() == 'pca':
            self.pca = self.auto_vivification()

        # Define an array containing the degrees
        self.CollocationPoints = np.array(self.CollocationPoints)
        self.n_samples, ndim = self.CollocationPoints.shape
        if self.ndim != ndim:
            raise AttributeError('The given samples do not match the given number of priors. The samples should be a 2D array of size (#samples, #priors)')
            
        self.deg_array = self.__select_degree(ndim, self.n_samples)

        # Generate all basis indices
        self.allBasisIndices = self.auto_vivification()
        for deg in self.deg_array:
            keys = self.allBasisIndices.keys()
            if deg not in np.fromiter(keys, dtype=float):
                # Generate the polynomial basis indices
                for qidx, q in enumerate(self.pce_q_norm):
                    basis_indices = glexindex(start=0, stop=deg+1,
                                              dimensions=self.n_params,
                                              cross_truncation=q,
                                              reverse=False, graded=True)
                    self.allBasisIndices[str(deg)][str(q)] = basis_indices

        
        
    def fit(self, X, y, parallel = True, verbose = False):
        """
        Fits the surrogate to the given data (samples X, outputs y).
        Note here that the samples X should be the transformed samples provided
        by the experimental design if the transformation is used there.

        Parameters
        ----------
        X : 2D list or np.array of shape (#samples, #dim)
            The parameter value combinations that the model was evaluated at.
        y : dict of 2D lists or arrays of shape (#samples, #timesteps)
            The respective model evaluations.

        Returns
        -------
        None.

        """
        X = np.array(X)
        for key in y.keys():
            y_val = np.array(y[key])
            if y_val.ndim !=2:
                raise ValueError('The given outputs y should be 2D')
            y[key] = np.array(y[key])
        
        # Output names are the same as the keys in y
        self.out_names = list(y.keys())
        
        # Build the MetaModel on the static samples
        self.CollocationPoints = X
        
        # TODO: other option: rebuild every time
        if not hasattr(self, 'deg_array'):
            self.build_metamodel(n_init_samples = X.shape[1])
            
        # Evaluate the univariate polynomials on InputSpace
        if self.meta_model_type.lower() != 'gpe':
           self.univ_p_val = self.univ_basis_vals(self.CollocationPoints)
        
        # --- Loop through data points and fit the surrogate ---
        if verbose:
            print(f"\n>>>> Training the {self.meta_model_type} metamodel "
                  "started. <<<<<<\n")

        # --- Bootstrap sampling ---
        # Correct number of bootstrap if PCA transformation is required.
        if self.dim_red_method.lower() == 'pca' and self.n_bootstrap_itrs == 1:
            self.n_bootstrap_itrs = 100

        # Check if fast version (update coeffs with OLS) is selected.
        if self.bootstrap_method.lower() == 'fast':
            fast_bootstrap = True
            first_out = {}
            n_comp_dict = {}
        else:
            fast_bootstrap = False

        # Prepare tqdm iteration maessage
        if verbose and self.n_bootstrap_itrs > 1:
            enum_obj = tqdm(range(self.n_bootstrap_itrs),
                            total=self.n_bootstrap_itrs,
                            desc="Bootstrapping the metamodel",
                            ascii=True)
        else:
            enum_obj = range(self.n_bootstrap_itrs)

        # Loop over the bootstrap iterations
        for b_i in enum_obj:
            if b_i > 0:
                b_indices = np.random.randint(self.n_samples, size=self.n_samples)
            else:
                b_indices = np.arange(len(X))

            X_train_b = X[b_indices]

            if verbose and self.n_bootstrap_itrs == 1:
                items = tqdm(y.items(), desc="Fitting regression")
            else:
                items = y.items()

            # For loop over the components/outputs
            for key, Output in items:

                # Dimensionality reduction with PCA, if specified
                if self.dim_red_method.lower() == 'pca':

                    # Use the stored n_comp for fast bootsrtrapping
                    if fast_bootstrap and b_i > 0:
                        self.n_pca_components = n_comp_dict[key]

                    # Start transformation
                    pca, target, n_comp = self.pca_transformation(
                        Output[b_indices], verbose=False
                        )
                    self.pca[f'b_{b_i+1}'][key] = pca
                    # Store the number of components for fast bootsrtrapping
                    if fast_bootstrap and b_i == 0:
                        n_comp_dict[key] = n_comp
                else:
                    target = Output[b_indices]

                # Parallel fit regression
                if self.meta_model_type.lower() == 'gpe':
                    # Prepare the input matrix
                    scaler = MinMaxScaler()
                    X_S = scaler.fit_transform(X_train_b)

                    self.x_scaler[f'b_{b_i+1}'][key] = scaler
                    if parallel:
                        out = Parallel(n_jobs=-1, backend='multiprocessing')(
                            delayed(self.gaussian_process_emulator)(
                                X_S, target[:, idx]) for idx in
                            range(target.shape[1]))
                    else:
                        results = map(self.gaussian_process_emulator,
                                      [X_train_b]*target.shape[1],
                                      [target[:, idx] for idx in
                                       range(target.shape[1])]
                                      )
                        out = list(results)

                    for idx in range(target.shape[1]):
                        self.gp_poly[f'b_{b_i+1}'][key][f"y_{idx+1}"] = out[idx]

                else:
                    self.univ_p_val = self.univ_p_val[b_indices]
                    if parallel and (not fast_bootstrap or b_i == 0):
                        out = Parallel(n_jobs=-1, backend='multiprocessing')(
                            delayed(self.adaptive_regression)(X_train_b,
                                                              target[:, idx],
                                                              idx)
                            for idx in range(target.shape[1]))
                    elif not parallel and (not fast_bootstrap or b_i == 0):
                        results = map(self.adaptive_regression,
                                      [X_train_b]*target.shape[1],
                                      [target[:, idx] for idx in
                                       range(target.shape[1])],
                                      range(target.shape[1]))
                        out = list(results)

                    # Store the first out dictionary
                    if fast_bootstrap and b_i == 0:
                        first_out[key] = copy.deepcopy(out)

                    if b_i > 0 and fast_bootstrap:

                        # fast bootstrap
                        out = self.update_pce_coeffs(
                            X_train_b, target, first_out[key])

                    for i in range(target.shape[1]):
                        # Create a dict to pass the variables
                        self.deg_dict[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['degree']
                        self.q_norm_dict[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['qnorm']
                        self.coeffs_dict[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['coeffs']
                        self.basis_dict[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['multi_indices']
                        self.score_dict[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['LOOCVScore']
                        self.clf_poly[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['clf_poly']
                        #self.LCerror[f'b_{b_i+1}'][key][f"y_{i+1}"] = out[i]['LCerror']

        if verbose:
            print(f"\n>>>> Training the {self.meta_model_type} metamodel"
                  " sucessfully completed. <<<<<<\n")

    # -------------------------------------------------------------------------
    def update_pce_coeffs(self, X, y, out_dict = None):
        """
        Updates the PCE coefficents using only the ordinary least square method
        for the fast version of the bootstrapping.

        Parameters
        ----------
        X : array of shape (n_samples, n_params)
            Training set.
        y : array of shape (n_samples, n_outs)
            The (transformed) model responses.
        out_dict : dict
            The training output dictionary of the first iteration, i.e.
            the surrogate model for the original experimental design.

        Returns
        -------
        final_out_dict : dict
            The updated training output dictionary.

        """
        # Make a copy
        final_out_dict = copy.deepcopy(out_dict)

        # Loop over the points
        for i in range(y.shape[1]):

                    
            # Extract nonzero basis indices
            nnz_idx = np.nonzero(out_dict[i]['coeffs'])[0]
            if len(nnz_idx) != 0:
                basis_indices = out_dict[i]['multi_indices']

                # Evaluate the multivariate polynomials on CollocationPoints
                psi = self.create_psi(basis_indices, self.univ_p_val)

                # Calulate the cofficients of surrogate model
                updated_out = self.regression(
                    psi, y[:, i], basis_indices, reg_method='OLS',
                    sparsity=False
                    )

                # Update coeffs in out_dict
                final_out_dict[i]['coeffs'][nnz_idx] = updated_out['coeffs']

        return final_out_dict

    # -------------------------------------------------------------------------
    def add_InputSpace(self):
        """
        Instanciates experimental design object.

        Returns
        -------
        None.

        """
        self.InputSpace = InputSpace(self.input_obj,
                                    meta_Model_type=self.meta_model_type)

    # -------------------------------------------------------------------------
    def univ_basis_vals(self, samples, n_max=None):
        """
        Evaluates univariate regressors along input directions.

        Parameters
        ----------
        samples : array of shape (n_samples, n_params)
            Samples.
        n_max : int, optional
            Maximum polynomial degree. The default is `None`.

        Returns
        -------
        univ_basis: array of shape (n_samples, n_params, n_max+1)
            All univariate regressors up to n_max.
        """
        # Extract information
        poly_types = self.InputSpace.poly_types
        if samples.ndim != 2:
            samples = samples.reshape(1, len(samples))
        n_max = np.max(self.pce_deg) if n_max is None else n_max

        # Extract poly coeffs
        if self.InputSpace.input_data_given or self.InputSpace.apce:
            apolycoeffs = self.polycoeffs
        else:
            apolycoeffs = None

        # Evaluate univariate basis
        univ_basis = eval_univ_basis(samples, n_max, poly_types, apolycoeffs)

        return univ_basis

    # -------------------------------------------------------------------------
    def create_psi(self, basis_indices, univ_p_val):
        """
        This function assemble the design matrix Psi from the given basis index
        set INDICES and the univariate polynomial evaluations univ_p_val.

        Parameters
        ----------
        basis_indices : array of shape (n_terms, n_params)
            Multi-indices of multivariate polynomials.
        univ_p_val : array of (n_samples, n_params, n_max+1)
            All univariate regressors up to `n_max`.

        Raises
        ------
        ValueError
            n_terms in arguments do not match.

        Returns
        -------
        psi : array of shape (n_samples, n_terms)
            Multivariate regressors.

        """
        # Check if BasisIndices is a sparse matrix
        sparsity = sp.sparse.issparse(basis_indices)
        if sparsity:
            basis_indices = basis_indices.toarray()

        # Initialization and consistency checks
        # number of input variables
        n_params = univ_p_val.shape[1]

        # Size of the experimental design
        n_samples = univ_p_val.shape[0]

        # number of basis terms
        n_terms = basis_indices.shape[0]

        # check that the variables have consistent sizes
        if n_params != basis_indices.shape[1]:
            raise ValueError(
                f"The shapes of basis_indices ({basis_indices.shape[1]}) and "
                f"univ_p_val ({n_params}) don't match!!"
                )

        # Preallocate the Psi matrix for performance
        psi = np.ones((n_samples, n_terms))
        # Assemble the Psi matrix
        for m in range(basis_indices.shape[1]):
            aa = np.where(basis_indices[:, m] > 0)[0]
            try:
                basisIdx = basis_indices[aa, m]
                bb = univ_p_val[:, m, basisIdx].reshape(psi[:, aa].shape)
                psi[:, aa] = np.multiply(psi[:, aa], bb)
            except ValueError as err:
                raise err
        return psi

    # -------------------------------------------------------------------------
    def regression(self, X, y, basis_indices, reg_method=None, sparsity=True):
        """
        Fit regression using the regression method provided.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array of shape (n_samples,)
            Target values.
        basis_indices : array of shape (n_terms, n_params)
            Multi-indices of multivariate polynomials.
        reg_method : str, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        return_out_dict : Dict
            Fitted estimator, spareMulti-Index, sparseX and coefficients.

        """
        if reg_method is None:
            reg_method = self.pce_reg_method

        bias_term = self.dim_red_method.lower() != 'pca'

        compute_score = True if self.verbose else False

        #  inverse of the observed variance of the data
        if np.var(y) != 0:
            Lambda = 1 / np.var(y)
        else:
            Lambda = 1e-6

        # Bayes sparse adaptive aPCE
        if reg_method.lower() == 'ols':
            clf_poly = lm.LinearRegression(fit_intercept=False)
        elif reg_method.lower() == 'brr':
            clf_poly = lm.BayesianRidge(n_iter=1000, tol=1e-7,
                                        fit_intercept=False,
                                        #normalize=True,
                                        compute_score=compute_score,
                                        alpha_1=1e-04, alpha_2=1e-04,
                                        lambda_1=Lambda, lambda_2=Lambda)
            clf_poly.converged = True

        elif reg_method.lower() == 'ard':
            if X.shape[0]<2:
                raise ValueError('Regression with ARD can only be performed for more than 2 samples')
            clf_poly = lm.ARDRegression(fit_intercept=False,
                                        #normalize=True,
                                        compute_score=compute_score,
                                        n_iter=1000, tol=0.0001,
                                        alpha_1=1e-3, alpha_2=1e-3,
                                        lambda_1=Lambda, lambda_2=Lambda)

        elif reg_method.lower() == 'fastard':
            clf_poly = RegressionFastARD(fit_intercept=False,
                                         normalize=True,
                                         compute_score=compute_score,
                                         n_iter=300, tol=1e-10)

        elif reg_method.lower() == 'bcs':
            if X.shape[0]<10:
                raise ValueError('Regression with BCS can only be performed for more than 10 samples')
            clf_poly = RegressionFastLaplace(fit_intercept=False,
                                         bias_term=bias_term,
                                         n_iter=1000, tol=1e-7)

        elif reg_method.lower() == 'lars':
            if X.shape[0]<10:
                raise ValueError('Regression with LARS can only be performed for more than 5 samples')
            clf_poly = lm.LassoLarsCV(fit_intercept=False)

        elif reg_method.lower() == 'sgdr':
            clf_poly = lm.SGDRegressor(fit_intercept=False,
                                       max_iter=5000, tol=1e-7)

        elif reg_method.lower() == 'omp':
            clf_poly = OrthogonalMatchingPursuit(fit_intercept=False)

        elif reg_method.lower() == 'vbl':
            clf_poly = VBLinearRegression(fit_intercept=False)

        elif reg_method.lower() == 'ebl':
            clf_poly = EBLinearRegression(optimizer='em')
            
        
        # Training with constraints automatically uses L2
        if self.apply_constraints:       
            # TODO: set the constraints here
            # Define the nonlin. constraint     
            nlc = NonlinearConstraint(lambda x: np.matmul(X,x),-1,1.1)
            self.nlc = nlc
            
            fun = lambda x: (np.linalg.norm(np.matmul(X, x)-y, ord = 2))**2
            if self.init_type =='zeros':
                res = minimize(fun, np.zeros(X.shape[1]), method = 'trust-constr', constraints  = self.nlc) 
            if self.init_type == 'nonpi':
                clf_poly.fit(X, y)
                coeff = clf_poly.coef_
                res = minimize(fun, coeff, method = 'trust-constr', constraints  = self.nlc)
            
            coeff = np.array(res.x)
            clf_poly.coef_ = coeff
            clf_poly.X = X
            clf_poly.y = y
            clf_poly.intercept_ = 0
            
        # Training without constraints uses chosen regression method
        else:
            clf_poly.fit(X, y)

        # Select the nonzero entries of coefficients
        if sparsity:
            nnz_idx = np.nonzero(clf_poly.coef_)[0]
        else:
            nnz_idx = np.arange(clf_poly.coef_.shape[0])

        # This is for the case where all outputs are zero, thereby
        # all coefficients are zero
        if (y == 0).all():
            nnz_idx = np.insert(np.nonzero(clf_poly.coef_)[0], 0, 0)

        sparse_basis_indices = basis_indices[nnz_idx]
        sparse_X = X[:, nnz_idx]
        coeffs = clf_poly.coef_[nnz_idx]
        clf_poly.coef_ = coeffs

        # Create a dict to pass the outputs
        return_out_dict = dict()
        return_out_dict['clf_poly'] = clf_poly
        return_out_dict['spareMulti-Index'] = sparse_basis_indices
        return_out_dict['sparePsi'] = sparse_X
        return_out_dict['coeffs'] = coeffs
        return return_out_dict
    
    # -------------------------------------------------------------------------
    def create_psi(self, basis_indices, univ_p_val):
        """
        This function assemble the design matrix Psi from the given basis index
        set INDICES and the univariate polynomial evaluations univ_p_val.

        Parameters
        ----------
        basis_indices : array of shape (n_terms, n_params)
            Multi-indices of multivariate polynomials.
        univ_p_val : array of (n_samples, n_params, n_max+1)
            All univariate regressors up to `n_max`.

        Raises
        ------
        ValueError
            n_terms in arguments do not match.

        Returns
        -------
        psi : array of shape (n_samples, n_terms)
            Multivariate regressors.

        """
        # Check if BasisIndices is a sparse matrix
        sparsity = sp.sparse.issparse(basis_indices)
        if sparsity:
            basis_indices = basis_indices.toarray()

        # Initialization and consistency checks
        # number of input variables
        n_params = univ_p_val.shape[1]

        # Size of the experimental design
        n_samples = univ_p_val.shape[0]

        # number of basis terms
        n_terms = basis_indices.shape[0]

        # check that the variables have consistent sizes
        if n_params != basis_indices.shape[1]:
            raise ValueError(
                f"The shapes of basis_indices ({basis_indices.shape[1]}) and "
                f"univ_p_val ({n_params}) don't match!!"
                )

        # Preallocate the Psi matrix for performance
        psi = np.ones((n_samples, n_terms))
        # Assemble the Psi matrix
        for m in range(basis_indices.shape[1]):
            aa = np.where(basis_indices[:, m] > 0)[0]
            try:
                basisIdx = basis_indices[aa, m]
                bb = univ_p_val[:, m, basisIdx].reshape(psi[:, aa].shape)
                psi[:, aa] = np.multiply(psi[:, aa], bb)
            except ValueError as err:
                raise err
        return psi

    # --------------------------------------------------------------------------------------------------------
    def adaptive_regression(self, ED_X, ED_Y, varIdx, verbose=False):
        """
        Adaptively fits the PCE model by comparing the scores of different
        degrees and q-norm.

        Parameters
        ----------
        ED_X : array of shape (n_samples, n_params)
            Experimental design.
        ED_Y : array of shape (n_samples,)
            Target values, i.e. simulation results for the Experimental design.
        varIdx : int
            Index of the output.
        verbose : bool, optional
            Print out summary. The default is False.

        Returns
        -------
        returnVars : Dict
            Fitted estimator, best degree, best q-norm, LOOCVScore and
            coefficients.

        """

        n_samples, n_params = ED_X.shape
        # Initialization
        qAllCoeffs, AllCoeffs = {}, {}
        qAllIndices_Sparse, AllIndices_Sparse = {}, {}
        qAllclf_poly, Allclf_poly = {}, {}
        qAllnTerms, AllnTerms = {}, {}
        qAllLCerror, AllLCerror = {}, {}

        # Extract degree array and qnorm array
        deg_array = np.array([*self.allBasisIndices], dtype=int)
        qnorm = [*self.allBasisIndices[str(int(deg_array[0]))]]

        # Some options for EarlyStop
        errorIncreases = False
        # Stop degree, if LOO error does not decrease n_checks_degree times
        n_checks_degree = 3
        # Stop qNorm, if criterion isn't fulfilled n_checks_qNorm times
        n_checks_qNorm = 2
        nqnorms = len(qnorm)
        qNormEarlyStop = True
        if nqnorms < n_checks_qNorm+1:
            qNormEarlyStop = False

        # =====================================================================
        # basis adaptive polynomial chaos: repeat the calculation by increasing
        # polynomial degree until the highest accuracy is reached
        # =====================================================================
        # For each degree check all q-norms and choose the best one
        scores = -np.inf * np.ones(deg_array.shape[0])
        qNormScores = -np.inf * np.ones(nqnorms)

        for degIdx, deg in enumerate(deg_array):

            for qidx, q in enumerate(qnorm):

                # Extract the polynomial basis indices from the pool of
                # allBasisIndices
                BasisIndices = self.allBasisIndices[str(deg)][str(q)]

                # Assemble the Psi matrix
                Psi = self.create_psi(BasisIndices, self.univ_p_val)

                # Calulate the cofficients of the meta model
                outs = self.regression(Psi, ED_Y, BasisIndices)

                # Calculate and save the score of LOOCV
                score, LCerror = self.corr_loocv_error(outs['clf_poly'],
                                                       outs['sparePsi'],
                                                       outs['coeffs'],
                                                       ED_Y)

                # Check the convergence of noise for FastARD
                if self.pce_reg_method == 'FastARD' and \
                   outs['clf_poly'].alpha_ < np.finfo(np.float32).eps:
                    score = -np.inf

                qNormScores[qidx] = score
                qAllCoeffs[str(qidx+1)] = outs['coeffs']
                qAllIndices_Sparse[str(qidx+1)] = outs['spareMulti-Index']
                qAllclf_poly[str(qidx+1)] = outs['clf_poly']
                qAllnTerms[str(qidx+1)] = BasisIndices.shape[0]
                qAllLCerror[str(qidx+1)] = LCerror

                # EarlyStop check
                # if there are at least n_checks_qNorm entries after the
                # best one, we stop
                if qNormEarlyStop and \
                   sum(np.isfinite(qNormScores)) > n_checks_qNorm:
                    # If the error has increased the last two iterations, stop!
                    qNormScores_nonInf = qNormScores[np.isfinite(qNormScores)]
                    deltas = np.sign(np.diff(qNormScores_nonInf))
                    if sum(deltas[-n_checks_qNorm+1:]) == 2:
                        # stop the q-norm loop here
                        break
                if np.var(ED_Y) == 0:
                    break

            # Store the score in the scores list
            best_q = np.nanargmax(qNormScores)
            scores[degIdx] = qNormScores[best_q]

            AllCoeffs[str(degIdx+1)] = qAllCoeffs[str(best_q+1)]
            AllIndices_Sparse[str(degIdx+1)] = qAllIndices_Sparse[str(best_q+1)]
            Allclf_poly[str(degIdx+1)] = qAllclf_poly[str(best_q+1)]
            AllnTerms[str(degIdx+1)] = qAllnTerms[str(best_q+1)]
            AllLCerror[str(degIdx+1)] = qAllLCerror[str(best_q+1)]

            # Check the direction of the error (on average):
            # if it increases consistently stop the iterations
            if len(scores[scores != -np.inf]) > n_checks_degree:
                scores_nonInf = scores[scores != -np.inf]
                ss = np.sign(scores_nonInf - np.max(scores_nonInf))
                # ss<0 error decreasing
                errorIncreases = np.sum(np.sum(ss[-2:])) <= -1*n_checks_degree

            if errorIncreases:
                break

            # Check only one degree, if target matrix has zero variance
            if np.var(ED_Y) == 0:
                break

        # ------------------ Summary of results ------------------
        # Select the one with the best score and save the necessary outputs
        best_deg = np.nanargmax(scores)+1
        coeffs = AllCoeffs[str(best_deg)]
        basis_indices = AllIndices_Sparse[str(best_deg)]
        clf_poly = Allclf_poly[str(best_deg)]
        LOOCVScore = np.nanmax(scores)
        P = AllnTerms[str(best_deg)]
        LCerror = AllLCerror[str(best_deg)]
        degree = deg_array[np.nanargmax(scores)]
        qnorm = float(qnorm[best_q])

        # ------------------ Print out Summary of results ------------------
        if self.verbose:
            # Create PSI_Sparse by removing redundent terms
            nnz_idx = np.nonzero(coeffs)[0]
            BasisIndices_Sparse = basis_indices[nnz_idx]

            print(f'Output variable {varIdx+1}:')
            print('The estimation of PCE coefficients converged at polynomial '
                  f'degree {deg_array[best_deg-1]} with '
                  f'{len(BasisIndices_Sparse)} terms (Sparsity index = '
                  f'{round(len(BasisIndices_Sparse)/P, 3)}).')

            print(f'Final ModLOO error estimate: {1-max(scores):.3e}')
            print('\n'+'-'*50)

        if verbose:
            print('='*50)
            print(' '*10 + ' Summary of results ')
            print('='*50)

            print("Scores:\n", scores)
            print("Degree of best score:", self.deg_array[best_deg-1])
            print("No. of terms:", len(basis_indices))
            print("Sparsity index:", round(len(basis_indices)/P, 3))
            print("Best Indices:\n", basis_indices)

            if self.pce_reg_method in ['BRR', 'ARD']:
                fig, ax = plt.subplots(figsize=(12, 10))
                plt.title("Marginal log-likelihood")
                plt.plot(clf_poly.scores_, color='navy', linewidth=2)
                plt.ylabel("Score")
                plt.xlabel("Iterations")
                if self.pce_reg_method.lower() == 'bbr':
                    text = f"$\\alpha={clf_poly.alpha_:.1f}$\n"
                    f"$\\lambda={clf_poly.lambda_:.3f}$\n"
                    f"$L={clf_poly.scores_[-1]:.1f}$"
                else:
                    text = f"$\\alpha={clf_poly.alpha_:.1f}$\n$"
                    f"\\L={clf_poly.scores_[-1]:.1f}$"

                plt.text(0.75, 0.5, text, fontsize=18, transform=ax.transAxes)
                plt.show()
            print('='*80)

        # Create a dict to pass the outputs
        returnVars = dict()
        returnVars['clf_poly'] = clf_poly
        returnVars['degree'] = degree
        returnVars['qnorm'] = qnorm
        returnVars['coeffs'] = coeffs
        returnVars['multi_indices'] = basis_indices
        returnVars['LOOCVScore'] = LOOCVScore
        returnVars['LCerror'] = LCerror

        return returnVars

    # -------------------------------------------------------------------------
    def corr_loocv_error(self, clf, psi, coeffs, y):
        """
        Calculates the corrected LOO error for regression on regressor
        matrix `psi` that generated the coefficients based on [1] and [2].

        [1] Blatman, G., 2009. Adaptive sparse polynomial chaos expansions for
            uncertainty propagation and sensitivity analysis (Doctoral
            dissertation, Clermont-Ferrand 2).

        [2] Blatman, G. and Sudret, B., 2011. Adaptive sparse polynomial chaos
            expansion based on least angle regression. Journal of computational
            Physics, 230(6), pp.2345-2367.

        Parameters
        ----------
        clf : object
            Fitted estimator.
        psi : array of shape (n_samples, n_features)
            The multivariate orthogonal polynomials (regressor).
        coeffs : array-like of shape (n_features,)
            Estimated cofficients.
        y : array of shape (n_samples,)
            Target values.

        Returns
        -------
        R_2 : float
            LOOCV Validation score (1-LOOCV erro).
        residual : array of shape (n_samples,)
            Residual values (y - predicted targets).

        """
        psi = np.array(psi, dtype=float)

        # Create PSI_Sparse by removing redundent terms
        nnz_idx = np.nonzero(coeffs)[0]
        if len(nnz_idx) == 0:
            nnz_idx = [0]
        psi_sparse = psi[:, nnz_idx]

        # NrCoeffs of aPCEs
        P = len(nnz_idx)
        # NrEvaluation (Size of experimental design)
        N = psi.shape[0]

        # Build the projection matrix
        PsiTPsi = np.dot(psi_sparse.T, psi_sparse)

        if np.linalg.cond(PsiTPsi) > 1e-12: #and \
           # np.linalg.cond(PsiTPsi) < 1/sys.float_info.epsilon:
            # faster
            try:
                M = sp.linalg.solve(PsiTPsi,
                                sp.sparse.eye(PsiTPsi.shape[0]).toarray())
            except:
                raise AttributeError('There are too few samples for the corrected loo-cv error. Fit surrogate on at least as many samples as parameters to use this')
        else:
            # stabler
            M = np.linalg.pinv(PsiTPsi)

        # h factor (the full matrix is not calculated explicitly,
        # only the trace is, to save memory)
        PsiM = np.dot(psi_sparse, M)

        h = np.sum(np.multiply(PsiM, psi_sparse), axis=1, dtype=np.longdouble)#float128)

        # ------ Calculate Error Loocv for each measurement point ----
        # Residuals
        try:
            residual = clf.predict(psi) - y
        except:
            residual = np.dot(psi, coeffs) - y

        # Variance
        var_y = np.var(y)

        if var_y == 0:
            norm_emp_error = 0
            loo_error = 0
            LCerror = np.zeros((y.shape))
            return 1-loo_error, LCerror
        else:
            norm_emp_error = np.mean(residual**2)/var_y

            # LCerror = np.divide(residual, (1-h))
            LCerror = residual / (1-h)
            loo_error = np.mean(np.square(LCerror)) / var_y
            # if there are NaNs, just return an infinite LOO error (this
            # happens, e.g., when a strongly underdetermined problem is solved)
            if np.isnan(loo_error):
                loo_error = np.inf

        # Corrected Error for over-determined system
        tr_M = np.trace(M)
        if tr_M < 0 or abs(tr_M) > 1e6:
            tr_M = np.trace(np.linalg.pinv(np.dot(psi.T, psi)))

        # Over-determined system of Equation
        if N > P:
            T_factor = N/(N-P) * (1 + tr_M)

        # Under-determined system of Equation
        else:
            T_factor = np.inf

        corrected_loo_error = loo_error * T_factor

        R_2 = 1 - corrected_loo_error

        return R_2, LCerror

    # -------------------------------------------------------------------------
    def pca_transformation(self, target, verbose=False):
        """
        Transforms the targets (outputs) via Principal Component Analysis

        Parameters
        ----------
        target : array of shape (n_samples,)
            Target values.

        Returns
        -------
        pca : obj
            Fitted sklearnPCA object.
        OutputMatrix : array of shape (n_samples,)
            Transformed target values.
        n_pca_components : int
            Number of selected principal components.

        """
        # Transform via Principal Component Analysis
        if hasattr(self, 'var_pca_threshold'):
            var_pca_threshold = self.var_pca_threshold
        else:
            var_pca_threshold = 100.0
        n_samples, n_features = target.shape

        if hasattr(self, 'n_pca_components'):
            n_pca_components = self.n_pca_components
        else:
            # Instantiate and fit sklearnPCA object
            covar_matrix = sklearnPCA(n_components=None)
            covar_matrix.fit(target)
            var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_,
                                     decimals=5)*100)
            # Find the number of components to explain self.varPCAThreshold of
            # variance
            try:
                n_components = np.where(var >= var_pca_threshold)[0][0] + 1
            except IndexError:
                n_components = min(n_samples, n_features)

            n_pca_components = min(n_samples, n_features, n_components)

        # Print out a report
        if verbose:
            print()
            print('-' * 50)
            print(f"PCA transformation is performed with {n_pca_components}"
                  " components.")
            print('-' * 50)
            print()

        # Fit and transform with the selected number of components
        pca = sklearnPCA(n_components=n_pca_components, svd_solver='arpack')
        scaled_target = pca.fit_transform(target)

        return pca, scaled_target, n_pca_components

    # -------------------------------------------------------------------------
    def gaussian_process_emulator(self, X, y, nug_term=None, autoSelect=False,
                                  varIdx=None):
        """
        Fits a Gaussian Process Emulator to the target given the training
         points.

        Parameters
        ----------
        X : array of shape (n_samples, n_params)
            Training points.
        y : array of shape (n_samples,)
            Target values.
        nug_term : float, optional
            Nugget term. The default is None, i.e. variance of y.
        autoSelect : bool, optional
            Loop over some kernels and select the best. The default is False.
        varIdx : int, optional
            The index number. The default is None.

        Returns
        -------
        gp : object
            Fitted estimator.

        """

        nug_term = nug_term if nug_term else np.var(y)

        Kernels = [nug_term * kernels.RBF(length_scale=1.0,
                                          length_scale_bounds=(1e-25, 1e15)),
                   nug_term * kernels.RationalQuadratic(length_scale=0.2,
                                                        alpha=1.0),
                   nug_term * kernels.Matern(length_scale=1.0,
                                             length_scale_bounds=(1e-15, 1e5),
                                             nu=1.5)]

        # Automatic selection of the kernel
        if autoSelect:
            gp = {}
            BME = []
            for i, kernel in enumerate(Kernels):
                gp[i] = GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=3,
                                                 normalize_y=False)

                # Fit to data using Maximum Likelihood Estimation
                gp[i].fit(X, y)

                # Store the MLE as BME score
                BME.append(gp[i].log_marginal_likelihood())

            gp = gp[np.argmax(BME)]

        else:
            gp = GaussianProcessRegressor(kernel=Kernels[0],
                                          n_restarts_optimizer=3,
                                          normalize_y=False)
            gp.fit(X, y)

        # Compute score
        if varIdx is not None:
            Score = gp.score(X, y)
            print('-'*50)
            print(f'Output variable {varIdx}:')
            print('The estimation of GPE coefficients converged,')
            print(f'with the R^2 score: {Score:.3f}')
            print('-'*50)

        return gp

    # -------------------------------------------------------------------------
    def eval_metamodel(self, samples):
        """
        Evaluates meta-model at the requested samples. One can also generate
        nsamples.

        Parameters
        ----------
        samples : array of shape (n_samples, n_params), optional
            Samples to evaluate meta-model at. The default is None.
        nsamples : int, optional
            Number of samples to generate, if no `samples` is provided. The
            default is None.
        sampling_method : str, optional
            Type of sampling, if no `samples` is provided. The default is
            'random'.
        return_samples : bool, optional
            Retun samples, if no `samples` is provided. The default is False.

        Returns
        -------
        mean_pred : dict
            Mean of the predictions.
        std_pred : dict
            Standard deviatioon of the predictions.
        """
        # Transform into np array - can also be given as list
        samples = np.array(samples)
        
        # Transform samples to the independent space
        samples = self.InputSpace.transform(
            samples,
            method='user'
            )
        # Compute univariate bases for the given samples
        if self.meta_model_type.lower() != 'gpe':
            univ_p_val = self.univ_basis_vals(
                samples,
                n_max=np.max(self.pce_deg)
                )

        mean_pred_b = {}
        std_pred_b = {}
        # Loop over bootstrap iterations
        for b_i in range(self.n_bootstrap_itrs):

            # Extract model dictionary
            if self.meta_model_type.lower() == 'gpe':
                model_dict = self.gp_poly[f'b_{b_i+1}']
            else:
                model_dict = self.coeffs_dict[f'b_{b_i+1}']

            # Loop over outputs
            mean_pred = {}
            std_pred = {}
            for output, values in model_dict.items():

                mean = np.empty((len(samples), len(values)))
                std = np.empty((len(samples), len(values)))
                idx = 0
                for in_key, InIdxValues in values.items():

                    # Prediction with GPE
                    if self.meta_model_type.lower() == 'gpe':
                        X_T = self.x_scaler[f'b_{b_i+1}'][output].transform(samples)
                        gp = self.gp_poly[f'b_{b_i+1}'][output][in_key]
                        y_mean, y_std = gp.predict(X_T, return_std=True)

                    else:
                        # Prediction with PCE
                        # Assemble Psi matrix
                        basis = self.basis_dict[f'b_{b_i+1}'][output][in_key]
                        psi = self.create_psi(basis, univ_p_val)

                        # Prediction
                        if self.bootstrap_method != 'fast' or b_i == 0:
                            # with error bar, i.e. use clf_poly
                            clf_poly = self.clf_poly[f'b_{b_i+1}'][output][in_key]
                            try:
                                y_mean, y_std = clf_poly.predict(
                                    psi, return_std=True
                                    )
                            except TypeError:
                                y_mean = clf_poly.predict(psi)
                                y_std = np.zeros_like(y_mean)
                        else:
                            # without error bar
                            coeffs = self.coeffs_dict[f'b_{b_i+1}'][output][in_key]
                            y_mean = np.dot(psi, coeffs)
                            y_std = np.zeros_like(y_mean)

                    mean[:, idx] = y_mean
                    std[:, idx] = y_std
                    idx += 1

                # Save predictions for each output
                if self.dim_red_method.lower() == 'pca':
                    PCA = self.pca[f'b_{b_i+1}'][output]
                    mean_pred[output] = PCA.inverse_transform(mean)
                    std_pred[output] = np.zeros(mean.shape)
                else:
                    mean_pred[output] = mean
                    std_pred[output] = std

            # Save predictions for each bootstrap iteration
            mean_pred_b[b_i] = mean_pred
            std_pred_b[b_i] = std_pred

        # Change the order of nesting
        mean_pred_all = {}
        for i in sorted(mean_pred_b):
            for k, v in mean_pred_b[i].items():
                if k not in mean_pred_all:
                    mean_pred_all[k] = [None] * len(mean_pred_b)
                mean_pred_all[k][i] = v

        # Compute the moments of predictions over the predictions
        for output in self.out_names:
            # Only use bootstraps with finite values
            finite_rows = np.isfinite(
                mean_pred_all[output]).all(axis=2).all(axis=1)
            outs = np.asarray(mean_pred_all[output])[finite_rows]
            # Compute mean
            mean_pred[output] = np.mean(outs, axis=0)
            # Compute standard deviation
            if self.n_bootstrap_itrs > 1:
                std_pred[output] = np.std(outs, axis=0)
            else:
                std_pred[output] = std_pred_b[b_i][output]

        return mean_pred, std_pred

    # -------------------------------------------------------------------------
    def create_model_error(self, X, y, Model, name='Calib'):
        """
        Fits a GPE-based model error.

        Parameters
        ----------
        X : array of shape (n_outputs, n_inputs)
            Input array. It can contain any forcing inputs or coordinates of
             extracted data.
        y : array of shape (n_outputs,)
            The model response for the MAP parameter set.
        name : str, optional
            Calibration or validation. The default is `'Calib'`.

        Returns
        -------
        self: object
            Self object.

        """
        outputNames = self.out_names
        self.errorRegMethod = 'GPE'
        self.errorclf_poly = self.auto_vivification()
        self.errorScale = self.auto_vivification()

        # Read data
        # TODO: do this call outside the metamodel
        MeasuredData = Model.read_observation(case=name)

        # Fitting GPR based bias model
        for out in outputNames:
            nan_idx = ~np.isnan(MeasuredData[out])
            # Select data
            try:
                data = MeasuredData[out].values[nan_idx]
            except AttributeError:
                data = MeasuredData[out][nan_idx]

            # Prepare the input matrix
            scaler = MinMaxScaler()
            delta = data  # - y[out][0]
            BiasInputs = np.hstack((X[out], y[out].reshape(-1, 1)))
            X_S = scaler.fit_transform(BiasInputs)
            gp = self.gaussian_process_emulator(X_S, delta)

            self.errorScale[out]["y_1"] = scaler
            self.errorclf_poly[out]["y_1"] = gp

        return self

    # -------------------------------------------------------------------------
    def eval_model_error(self, X, y_pred):
        """
        Evaluates the error model.

        Parameters
        ----------
        X : array
            Inputs.
        y_pred : dict
            Predictions.

        Returns
        -------
        mean_pred : dict
            Mean predition of the GPE-based error model.
        std_pred : dict
            standard deviation of the GPE-based error model.

        """
        mean_pred = {}
        std_pred = {}

        for Outkey, ValuesDict in self.errorclf_poly.items():

            pred_mean = np.zeros_like(y_pred[Outkey])
            pred_std = np.zeros_like(y_pred[Outkey])

            for Inkey, InIdxValues in ValuesDict.items():

                gp = self.errorclf_poly[Outkey][Inkey]
                scaler = self.errorScale[Outkey][Inkey]

                # Transform Samples using scaler
                for j, pred in enumerate(y_pred[Outkey]):
                    BiasInputs = np.hstack((X[Outkey], pred.reshape(-1, 1)))
                    Samples_S = scaler.transform(BiasInputs)
                    y_hat, y_std = gp.predict(Samples_S, return_std=True)
                    pred_mean[j] = y_hat
                    pred_std[j] = y_std
                    # pred_mean[j] += pred

            mean_pred[Outkey] = pred_mean
            std_pred[Outkey] = pred_std

        return mean_pred, std_pred

    # -------------------------------------------------------------------------
    class auto_vivification(dict):
        """
        Implementation of perl's AutoVivification feature.

        Source: https://stackoverflow.com/a/651879/18082457
        """

        def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    # -------------------------------------------------------------------------
    def copy_meta_model_opts(self):
        """
        This method is a convinient function to copy the metamodel options.

        Returns
        -------
        new_MetaModelOpts : object
            The copied object.

        """
        # TODO: what properties should be moved to the new object?
        new_MetaModelOpts = copy.deepcopy(self)
        new_MetaModelOpts.input_obj = self.input_obj#InputObj
        new_MetaModelOpts.InputSpace = self.InputSpace
        #new_MetaModelOpts.InputSpace.meta_Model = 'aPCE'
        #new_MetaModelOpts.InputSpace.InputObj = self.input_obj
        #new_MetaModelOpts.InputSpace.ndim = len(self.input_obj.Marginals)
        new_MetaModelOpts.n_params = len(self.input_obj.Marginals)
        #new_MetaModelOpts.InputSpace.hdf5_file = None

        return new_MetaModelOpts

    # -------------------------------------------------------------------------
    def __select_degree(self, ndim, n_samples):
        """
        Selects degree based on the number of samples and parameters in the
        sequential design.

        Parameters
        ----------
        ndim : int
            Dimension of the parameter space.
        n_samples : int
            Number of samples.

        Returns
        -------
        deg_array: array
            Array containing the arrays.

        """
        # Define the deg_array
        max_deg = np.max(self.pce_deg)
        min_Deg = np.min(self.pce_deg)
        
        # TODO: remove the options for sequential?
        #nitr = n_samples - self.InputSpace.n_init_samples

        # Check q-norm
        if not np.isscalar(self.pce_q_norm):
            self.pce_q_norm = np.array(self.pce_q_norm)
        else:
            self.pce_q_norm = np.array([self.pce_q_norm])

        def M_uptoMax(maxDeg):
            n_combo = np.zeros(maxDeg)
            for i, d in enumerate(range(1, maxDeg+1)):
                n_combo[i] = math.factorial(ndim+d)
                n_combo[i] /= math.factorial(ndim) * math.factorial(d)
            return n_combo

        deg_new = max_deg
        #d = nitr if nitr != 0 and self.n_params > 5 else 1
        # d = 1
        # min_index = np.argmin(abs(M_uptoMax(max_deg)-ndim*n_samples*d))
        # deg_new = range(1, max_deg+1)[min_index]

        if deg_new > min_Deg and self.pce_reg_method.lower() != 'fastard':
            deg_array = np.arange(min_Deg, deg_new+1)
        else:
            deg_array = np.array([deg_new])

        return deg_array

    def generate_polynomials(self, max_deg=None):
        # Check for InputSpace
        if not hasattr(self, 'InputSpace'):
            raise AttributeError('Generate or add InputSpace before generating polynomials')
            
        ndim = self.InputSpace.ndim
        # Create orthogonal polynomial coefficients if necessary
        if (self.meta_model_type.lower()!='gpe') and max_deg is not None:# and self.input_obj.poly_coeffs_flag:
            self.polycoeffs = {}
            for parIdx in tqdm(range(ndim), ascii=True,
                               desc="Computing orth. polynomial coeffs"):
                poly_coeffs = apoly_construction(
                    self.InputSpace.raw_data[parIdx],
                    max_deg
                    )
                self.polycoeffs[f'p_{parIdx+1}'] = poly_coeffs
        else:
            raise AttributeError('MetaModel cannot generate polynomials in the given scenario!')

    # -------------------------------------------------------------------------
    def _compute_pce_moments(self):
        """
        Computes the first two moments using the PCE-based meta-model.

        Returns
        -------
        pce_means: dict
            The first moment (mean) of the surrogate.
        pce_stds: dict
            The second moment (standard deviation) of the surrogate.

        """
        
        # Check if its truly a pce-surrogate
        if self.meta_model_type.lower() == 'gpe':
            raise AttributeError('Moments can only be computed for pce-type surrogates')
        
        outputs = self.out_names
        pce_means_b = {}
        pce_stds_b = {}

        # Loop over bootstrap iterations
        for b_i in range(self.n_bootstrap_itrs):
            # Loop over the metamodels
            coeffs_dicts = self.coeffs_dict[f'b_{b_i+1}'].items()
            means = {}
            stds = {}
            for output, coef_dict in coeffs_dicts:

                pce_mean = np.zeros((len(coef_dict)))
                pce_var = np.zeros((len(coef_dict)))

                for index, values in coef_dict.items():
                    idx = int(index.split('_')[1]) - 1
                    coeffs = self.coeffs_dict[f'b_{b_i+1}'][output][index]

                    # Mean = c_0
                    if coeffs[0] != 0:
                        pce_mean[idx] = coeffs[0]
                    else:
                        clf_poly = self.clf_poly[f'b_{b_i+1}'][output]
                        pce_mean[idx] = clf_poly[index].intercept_
                    # Var = sum(coeffs[1:]**2)
                    pce_var[idx] = np.sum(np.square(coeffs[1:]))

                # Save predictions for each output
                if self.dim_red_method.lower() == 'pca':
                    PCA = self.pca[f'b_{b_i+1}'][output]
                    means[output] = PCA.inverse_transform(pce_mean)
                    stds[output] = PCA.inverse_transform(np.sqrt(pce_var))
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

        return pce_means, pce_stds
