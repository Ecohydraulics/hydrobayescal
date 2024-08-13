#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:41:45 2020

@author: farid
"""
import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import LinAlgError
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
import warnings
from sklearn.utils import check_X_y
from scipy.linalg import pinvh


def update_precisions(Q,S,q,s,A,active,tol,n_samples,clf_bias):
    '''
    Selects one feature to be added/recomputed/deleted to model based on
    effect it will have on value of log marginal likelihood.
    '''
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])

    # identify features that can be added , recomputed and deleted in model
    theta        =  q**2 - s
    add          =  (theta > 0) * (active == False)
    recompute    =  (theta > 0) * (active == True)
    delete       = ~(add + recompute)

    # compute sparsity & quality parameters corresponding to features in
    # three groups identified above
    Qadd,Sadd      = Q[add], S[add]
    Qrec,Srec,Arec = Q[recompute], S[recompute], A[recompute]
    Qdel,Sdel,Adel = Q[delete], S[delete], A[delete]

    # compute new alpha's (precision parameters) for features that are
    # currently in model and will be recomputed
    Anew           = s[recompute]**2/ ( theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha    = (1./Anew - 1./Arec)

    # compute change in log marginal likelihood
    deltaL[add]       = ( Qadd**2 - Sadd ) / Sadd + np.log(Sadd/Qadd**2 )
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec*delta_alpha)
    deltaL[delete]    = Qdel**2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    deltaL            = deltaL  / n_samples

    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)

    # no deletions or additions
    same_features  = np.sum( theta[~recompute] > 0) == 0

    # changes in precision for features already in model is below threshold
    no_delta       = np.sum( abs( Anew - Arec ) > tol ) == 0
    # if same_features: print(abs( Anew - Arec ))
    # print("same_features = {} no_delta = {}".format(same_features,no_delta))
    # check convergence: if no features to add or delete and small change in
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        return [A,converged]

    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = s[feature_index]**2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            # do not remove bias term in classification
            # (in regression it is factored in through centering)
            if not (feature_index == 0 and clf_bias):
                active[feature_index] = False
                A[feature_index]      = np.PINF

    return [A,converged]


class RegressionFastARD(LinearModel, RegressorMixin):
    '''
    Regression with Automatic Relevance Determination (Fast Version uses
    Sparse Bayesian Learning)
    https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations

    start: list, optional (DEFAULT = None)
        Initial selected features.

    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    verbose : boolean, optional (DEFAULT = FALSE)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    alpha_ : float
       estimated precision of the noise

    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization.

    References
    ----------
    [1] Fast marginal likelihood maximisation for sparse Bayesian models
    (Tipping & Faul 2003) (http://www.miketipping.com/papers/met-fastsbl.pdf)
    [2] Analysis of sparse Bayesian learning (Tipping & Faul 2001)
        (http://www.miketipping.com/abstracts.htm#Faul:NIPS01)
    '''

    def __init__(self, n_iter=300, start=None, tol=1e-3, fit_intercept=True,
                 normalize=False, copy_X=True, compute_score=False, verbose=False):
        self.n_iter          = n_iter
        self.start           = start
        self.tol             = tol
        self.scores_         = list()
        self.fit_intercept   = fit_intercept
        self.normalize       = normalize
        self.copy_X          = copy_X
        self.compute_score   = compute_score
        self.verbose         = verbose

    def _preprocess_data(self, X, y):
        """Center and scale data.
        Centers data to have mean zero along axis 0. If fit_intercept=False or
        if the X is a sparse matrix, no centering is done, but normalization
        can still be applied. The function returns the statistics necessary to
        reconstruct the input data, which are X_offset, y_offset, X_scale, such
        that the output
            X = (X - X_offset) / X_scale
        X_scale is the L2 norm of X - X_offset.
        """

        if self.copy_X:
            X = X.copy(order='K')

        y = np.asarray(y, dtype=X.dtype)

        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            X -= X_offset
            if self.normalize:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
                std = np.sqrt(np.sum(X**2, axis=0)/(len(X)-1))
                X_scale[std != 0] = std[std != 0]
                X /= X_scale
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = np.mean(y)
            y = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
            if y.ndim == 1:
                y_offset = X.dtype.type(0)
            else:
                y_offset = np.zeros(y.shape[1], dtype=X.dtype)

        return X, y, X_offset, y_offset, X_scale

    def fit(self, X, y):
        '''
        Fits ARD Regression with Sequential Sparse Bayes Algorithm.

        Parameters
        -----------
        X: {array-like, sparse matrix} of size (n_samples, n_features)
           Training data, matrix of explanatory variables

        y: array-like of size [n_samples, n_features]
           Target values

        Returns
        -------
        self : object
            Returns self.
        '''
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape

        X, y, X_mean, y_mean, X_std = self._preprocess_data(X, y)
        self._x_mean_ = X_mean
        self._y_mean = y_mean
        self._x_std = X_std

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors
        XY = np.dot(X.T, y)
        XX = np.dot(X.T, X)
        XXd = np.diag(XX)

        #  initialise precision of noise & and coefficients
        var_y = np.var(y)

        # check that variance is non zero !!!
        if var_y == 0:
            beta = 1e-2
            self.var_y = True
        else:
            beta = 1. / np.var(y)
            self.var_y = False

        A = np.PINF * np.ones(n_features)
        active = np.zeros(n_features, dtype=np.bool)

        if self.start is not None and not hasattr(self, 'active_'):
            start = self.start
            # start from a given start basis vector
            proj = XY**2 / XXd
            active[start] = True
            A[start] = XXd[start]/(proj[start] - var_y)

        else:
            # in case of almost perfect multicollinearity between some features
            # start from feature 0
            if np.sum(XXd - X_mean**2 < np.finfo(np.float32).eps) > 0:
                A[0] = np.finfo(np.float16).eps
                active[0] = True

            else:
                # start from a single basis vector with largest projection on
                # targets
                proj = XY**2 / XXd
                start = np.argmax(proj)
                active[start] = True
                A[start] = XXd[start]/(proj[start] - var_y +
                                       np.finfo(np.float32).eps)

        warning_flag = 0
        scores_ = []
        for i in range(self.n_iter):
            # Handle variance zero
            if self.var_y:
                A[0] = y_mean
                active[0] = True
                converged = True
                break

            XXa = XX[active, :][:, active]
            XYa = XY[active]
            Aa = A[active]

            # mean & covariance of posterior distribution
            Mn, Ri, cholesky = self._posterior_dist(Aa, beta, XXa, XYa)
            if cholesky:
                Sdiag = np.sum(Ri**2, 0)
            else:
                Sdiag = np.copy(np.diag(Ri))
                warning_flag += 1

            # raise warning in case cholesky fails
            if warning_flag == 1:
                warnings.warn(("Cholesky decomposition failed! Algorithm uses "
                               "pinvh, which is significantly slower. If you "
                               "use RVR it is advised to change parameters of "
                               "the kernel!"))

            # compute quality & sparsity parameters
            s, q, S, Q = self._sparsity_quality(XX, XXd, XY, XYa, Aa, Ri,
                                                active, beta, cholesky)

            # update precision parameter for noise distribution
            rss = np.sum((y - np.dot(X[:, active], Mn))**2)

            # if near perfect fit , then terminate
            if (rss / n_samples/var_y) < self.tol:
                warnings.warn('Early termination due to near perfect fit')
                converged = True
                break
            beta = n_samples - np.sum(active) + np.sum(Aa * Sdiag)
            beta /= rss
            # beta /= (rss + np.finfo(np.float32).eps)

            # update precision parameters of coefficients
            A, converged = update_precisions(Q, S, q, s, A, active, self.tol,
                                             n_samples, False)

            if self.compute_score:
                scores_.append(self.log_marginal_like(XXa, XYa, Aa, beta))

            if self.verbose:
                print(('Iteration: {0}, number of features '
                       'in the model: {1}').format(i, np.sum(active)))

            if converged or i == self.n_iter - 1:
                if converged and self.verbose:
                    print('Algorithm converged!')
                break

        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa, XYa, Aa = XX[active, :][:, active], XY[active], A[active]
        Mn, Sn, cholesky = self._posterior_dist(Aa, beta, XXa, XYa, True)
        self.coef_ = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_ = Sn
        self.active_ = active
        self.lambda_ = A
        self.alpha_ = beta
        self.converged = converged
        if self.compute_score:
            self.scores_ = np.array(scores_)

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.
        return self

    def log_marginal_like(self, XXa, XYa, Aa, beta):
        """Computes the log of the marginal likelihood."""
        N, M = XXa.shape
        A = np.diag(Aa)

        Mn, sigma_, cholesky = self._posterior_dist(Aa, beta, XXa, XYa,
                                                    full_covar=True)

        C = sigma_ + np.dot(np.dot(XXa.T, np.linalg.pinv(A)), XXa)

        score = np.dot(np.dot(XYa.T, np.linalg.pinv(C)), XYa) +\
            np.log(np.linalg.det(C)) + N * np.log(2 * np.pi)

        return -0.5 * score

    def predict(self, X, return_std=False):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and variance based on
        Ref.[1] Section 3.3.2.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        : list of length two [y_hat, var_hat]

             y_hat: numpy array of size (n_samples_test,)
                    Estimated values of targets on test set (i.e. mean of
                    predictive distribution)

                var_hat: numpy array of size (n_samples_test,)
                    Variance of predictive distribution
        References
        ----------
        [1] Bishop, C. M. (2006). Pattern recognition and machine learning.
        springer.
        '''

        y_hat = np.dot(X, self.coef_) + self.intercept_

        if return_std:
            # Handle the zero variance case
            if self.var_y:
                return y_hat, np.zeros_like(y_hat)

            if self.normalize:
                X -= self._x_mean_[self.active_]
                X /= self._x_std[self.active_]
            var_hat = 1./self.alpha_
            var_hat += np.sum(X.dot(self.sigma_) * X, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
        else:
            return y_hat

    def _posterior_dist(self, A, beta, XX, XY, full_covar=False):
        '''
        Calculates mean and covariance matrix of posterior distribution
        of coefficients.
        '''
        # compute precision matrix for active features
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + A)
        cholesky = True

        # try cholesky, if it fails go back to pinvh
        try:
            # find posterior mean : R*R.T*mean = beta*X.T*Y
            # solve(R*z = beta*X.T*Y) =>find z=> solve(R.T*mean = z)=>find mean
            R = np.linalg.cholesky(Sinv)
            Z = solve_triangular(R, beta*XY, check_finite=True, lower=True)
            Mn = solve_triangular(R.T, Z, check_finite=True, lower=False)

            # invert lower triangular matrix from cholesky decomposition
            Ri = solve_triangular(R, np.eye(A.shape[0]), check_finite=False,
                                  lower=True)
            if full_covar:
                Sn = np.dot(Ri.T, Ri)
                return Mn, Sn, cholesky
            else:
                return Mn, Ri, cholesky
        except LinAlgError:
            cholesky = False
            Sn = pinvh(Sinv)
            Mn = beta*np.dot(Sinv, XY)
            return Mn, Sn, cholesky

    def _sparsity_quality(self, XX, XXd, XY, XYa, Aa, Ri, active, beta, cholesky):
        '''
        Calculates sparsity and quality parameters for each feature

        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X * Sn * X'
        '''
        bxy = beta*XY
        bxx = beta*XXd
        if cholesky:
            # here Ri is inverse of lower triangular matrix obtained from
            # cholesky decomp
            xxr = np.dot(XX[:, active], Ri.T)
            rxy = np.dot(Ri, XYa)
            S = bxx - beta**2 * np.sum(xxr**2, axis=1)
            Q = bxy - beta**2 * np.dot(xxr, rxy)
        else:
            # here Ri is covariance matrix
            XXa = XX[:, active]
            XS = np.dot(XXa, Ri)
            S = bxx - beta**2 * np.sum(XS*XXa, 1)
            Q = bxy - beta**2 * np.dot(XS, XYa)
        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S)
        # so if A = np.PINF q = Q, s = S
        qi = np.copy(Q)
        si = np.copy(S)
        # If A is not np.PINF, then it should be 'active' feature => use (EQ 1)
        Qa, Sa = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa)
        si[active] = Aa * Sa / (Aa - Sa)

        return [si, qi, S, Q]
