#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:08:59 2022

@author: farid
"""
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_X_y


def corr(x, y):
    return abs(x.dot(y))/np.sqrt((x**2).sum())


class OrthogonalMatchingPursuit(LinearModel, RegressorMixin):
    '''
    Regression with Orthogonal Matching Pursuit [1].

    Parameters
    ----------
    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional (DEFAULT = FALSE)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    References
    ----------
    [1] Pati, Y., Rezaiifar, R., Krishnaprasad, P. (1993). Orthogonal matching
        pursuit: recursive function approximation with application to wavelet
        decomposition. Proceedings of 27th Asilomar Conference on Signals,
        Systems and Computers, 40-44.
    '''

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 verbose=False):
        self.fit_intercept   = fit_intercept
        self.normalize       = normalize
        self.copy_X          = copy_X
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
        Fits Regression with Orthogonal Matching Pursuit Algorithm.

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

        # Normalize columns of Psi, so that each column has norm = 1
        norm_X = np.linalg.norm(X, axis=0)
        X_norm = X/norm_X

        # Initialize residual vector to full model response and normalize
        R = y
        norm_y = np.sqrt(np.dot(y, y))
        r = y/norm_y

        # Check for constant regressors
        const_indices = np.where(~np.diff(X, axis=0).any(axis=0))[0]
        bool_const = not const_indices

        # Start regression using OPM algorithm
        precision = 0        # Set precision criterion to precision of program
        early_stop = True
        cond_early = True    # Initialize condition for early stop
        ind = []
        iindx = []           # index of selected columns
        indtot = np.arange(n_features)  # Full index set for remaining columns
        kmax = min(n_samples, n_features)  # Maximum number of iterations
        LOO = np.PINF * np.ones(kmax)  # Store LOO error at each iteration
        LOOmin = np.PINF               # Initialize minimum value of LOO
        coeff = np.zeros((n_features, kmax))
        count = 0
        k = 0.1                # Percentage of iteration history for early stop

        # Begin iteration over regressors set (Matrix X)
        while (np.linalg.norm(R) > precision) and (count <= kmax-1) and \
              ((cond_early or early_stop) ^ ~cond_early):

            # Update index set of columns yet to select
            if count != 0:
                indtot = np.delete(indtot, iindx)

            # Find column of X that is most correlated with residual
            h = abs(np.dot(r, X_norm))
            iindx = np.argmax(h[indtot])
            indx = indtot[iindx]

            # initialize with the constant regressor, if it exists in the basis
            if (count == 0) and bool_const:
                # overwrite values for iindx and indx
                iindx = const_indices[0]
                indx = indtot[iindx]

            # Invert the information matrix at the first iteration, later only
            # update its value on the basis of the previously inverted one,
            if count == 0:
                M = 1 / np.dot(X[:, indx], X[:, indx])
            else:
                x = np.dot(X[:, ind].T, X[:, indx])
                r = np.dot(X[:, indx], X[:, indx])
                M = self.blockwise_inverse(M, x, x.T, r)

            # Add newly found index to the selected indexes set
            ind.append(indx)

            # Select regressors subset (Projection subspace)
            Xpro = X[:, ind]

            # Obtain coefficient by performing OLS
            TT = np.dot(y, Xpro)
            beta = np.dot(M, TT)
            coeff[ind, count] = beta

            # Compute LOO error
            LOO[count] = self.loo_error(Xpro, M, y, beta)

            # Compute new residual due to new projection
            R = y - np.dot(Xpro, beta)

            # Normalize residual
            norm_R = np.sqrt(np.dot(R, R))
            r = R / norm_R

            # Update counters and early-stop criterions
            countinf = max(0, int(count-k*kmax))
            LOOmin = min(LOOmin, LOO[count])

            if count == 0:
                cond_early = (LOO[0] <= LOOmin)
            else:
                cond_early = (min(LOO[countinf:count+1]) <= LOOmin)

            if self.verbose:
                print(f'Iteration: {count+1}, mod. LOOCV error : '
                      f'{LOO[count]:.2e}')

            # Update counter
            count += 1

        # Select projection with smallest cross-validation error
        countmin = np.argmin(LOO[:-1])
        self.coef_ = coeff[:, countmin]
        self.active = coeff[:, countmin] != 0.0

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

        return self

    def predict(self, X):
        '''
        Computes predictive distribution for test set.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        y_hat: numpy array of size (n_samples_test,)
               Estimated values of targets on test set (i.e. mean of
               predictive distribution)
        '''

        y_hat = np.dot(X, self.coef_) + self.intercept_

        return y_hat

    def loo_error(self, psi, inv_inf_matrix, y, coeffs):
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
        psi : array of shape (n_samples, n_feature)
            Orthogonal bases evaluated at the samples.
        inv_inf_matrix : array
            Inverse of the information matrix.
        y : array of shape (n_samples, )
            Targets.
        coeffs : array
            Computed regresssor cofficients.

        Returns
        -------
        loo_error : float
            Modified LOOCV error.

        """

        # NrEvaluation (Size of experimental design)
        N, P = psi.shape

        # h factor (the full matrix is not calculated explicitly,
        # only the trace is, to save memory)
        PsiM = np.dot(psi, inv_inf_matrix)

        h = np.sum(np.multiply(PsiM, psi), axis=1, dtype=np.longdouble)

        # ------ Calculate Error Loocv for each measurement point ----
        # Residuals
        residual = np.dot(psi, coeffs) - y

        # Variance
        varY = np.var(y)

        if varY == 0:
            norm_emp_error = 0
            loo_error = 0
        else:
            norm_emp_error = np.mean(residual**2)/varY

            loo_error = np.mean(np.square(residual / (1-h))) / varY

            # if there are NaNs, just return an infinite LOO error (this
            # happens, e.g., when a strongly underdetermined problem is solved)
            if np.isnan(loo_error):
                loo_error = np.inf

        # Corrected Error for over-determined system
        tr_M = np.trace(np.atleast_2d(inv_inf_matrix))
        if tr_M < 0 or abs(tr_M) > 1e6:
            tr_M = np.trace(np.linalg.pinv(np.dot(psi.T, psi)))

        # Over-determined system of Equation
        if N > P:
            T_factor = N/(N-P) * (1 + tr_M)

        # Under-determined system of Equation
        else:
            T_factor = np.inf

        loo_error *= T_factor

        return loo_error

    def blockwise_inverse(self, Ainv, B, C, D):
        """
        non-singular square matrix M defined as M = [[A B]; [C D]] .
        B, C and D can have any dimension, provided their combination defines
        a square matrix M.

        Parameters
        ----------
        Ainv : float or array
            inverse of the square-submatrix A.
        B : float or array
            Information matrix with all new regressor.
        C : float or array
            Transpose of B.
        D : float or array
            Information matrix with all selected regressors.

        Returns
        -------
        M : array
            Inverse of the information matrix.

        """
        if np.isscalar(D):
            # Inverse of D
            Dinv = 1/D
            # Schur complement
            SCinv = 1/(D - np.dot(C, np.dot(Ainv, B[:, None])))[0]
        else:
            # Inverse of D
            Dinv = np.linalg.solve(D, np.eye(D.shape))
            # Schur complement
            SCinv = np.linalg.solve((D - C*Ainv*B), np.eye(D.shape))

        T1 = np.dot(Ainv, np.dot(B[:, None], SCinv))
        T2 = np.dot(C, Ainv)

        # Assemble the inverse matrix
        M = np.vstack((
            np.hstack((Ainv+T1*T2, -T1)),
            np.hstack((-(SCinv)*T2, SCinv))
            ))
        return M
