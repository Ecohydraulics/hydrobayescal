#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import as_float_array
from sklearn.model_selection import KFold


class RegressionFastLaplace():
    '''
    Sparse regression with Bayesian Compressive Sensing as described in Alg. 1
    (Fast Laplace) of Ref.[1], which updated formulas from [2].

    sigma2: noise precision (sigma^2)
    nu fixed to 0

    uqlab/lib/uq_regression/BCS/uq_bsc.m

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 1000)
        Maximum number of iterations

    tol: float, optional (DEFAULT = 1e-7)
        If absolute change in precision parameter for weights is below
        threshold algorithm terminates.

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

    alpha_ : float
       estimated precision of the noise

    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    References
    ----------
    [1] Babacan, S. D., Molina, R., & Katsaggelos, A. K. (2009). Bayesian
        compressive sensing using Laplace priors. IEEE Transactions on image
        processing, 19(1), 53-63.
    [2] Fast marginal likelihood maximisation for sparse Bayesian models
        (Tipping & Faul 2003).
        (http://www.miketipping.com/papers/met-fastsbl.pdf)
    '''

    def __init__(self, n_iter=1000, n_Kfold=10, tol=1e-7, fit_intercept=False,
                 bias_term=True, copy_X=True, verbose=False):
        self.n_iter = n_iter
        self.n_Kfold = n_Kfold
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.bias_term = bias_term
        self.copy_X = copy_X
        self.verbose = verbose

    def _center_data(self, X, y):
        ''' Centers data'''
        X = as_float_array(X, copy = self.copy_X)

        # normalisation should be done in preprocessing!
        X_std = np.ones(X.shape[1], dtype=X.dtype)
        if self.fit_intercept:
            X_mean = np.average(X, axis=0)
            y_mean = np.average(y, axis=0)
            X -= X_mean
            y -= y_mean
        else:
            X_mean = np.zeros(X.shape[1], dtype=X.dtype)
            y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
        return X, y, X_mean, y_mean, X_std

    def fit(self, X, y):

        k_fold = KFold(n_splits=self.n_Kfold)

        varY = np.var(y, ddof=1) if np.var(y, ddof=1) != 0 else 1.0
        sigma2s = len(y)*varY*(10**np.linspace(-16, -1, self.n_Kfold))

        errors = np.zeros((len(sigma2s), self.n_Kfold))
        for s, sigma2 in enumerate(sigma2s):
            for k, (train, test) in enumerate(k_fold.split(X, y)):
                self.fit_(X[train], y[train], sigma2)
                errors[s, k] = np.linalg.norm(
                    y[test] - self.predict(X[test])
                    )**2/len(test)

        KfCVerror = np.sum(errors, axis=1)/self.n_Kfold/varY
        i_minCV = np.argmin(KfCVerror)

        self.kfoldCVerror = np.min(KfCVerror)

        return self.fit_(X, y, sigma2s[i_minCV])

    def fit_(self, X, y, sigma2):

        N, P = X.shape
        # n_samples, n_features = X.shape

        X, y, X_mean, y_mean, X_std = self._center_data(X, y)
        self._x_mean_ = X_mean
        self._y_mean = y_mean
        self._x_std = X_std

        # check that variance is non zero !!!
        if np.var(y) == 0:
            self.var_y = True
        else:
            self.var_y = False
        beta = 1./sigma2

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors X=Psi
        PsiTY = np.dot(X.T, y)
        PsiTPsi = np.dot(X.T, X)
        XXd = np.diag(PsiTPsi)

        # initialize with constant regressor, or if that one does not exist,
        # with the one that has the largest correlation with Y
        ind_global_to_local = np.zeros(P, dtype=np.int32)

        # identify constant regressors
        constidx = np.where(~np.diff(X, axis=0).all(axis=0))[0]

        if self.bias_term and constidx.size != 0:
            ind_start = constidx[0]
            ind_global_to_local[ind_start] = True
        else:
            # start from a single basis vector with largest projection on
            # targets
            proj = np.divide(np.square(PsiTY), XXd)
            ind_start = np.argmax(proj)
            ind_global_to_local[ind_start] = True

        num_active = 1
        active_indices = [ind_start]
        deleted_indices = []
        bcs_path = [ind_start]
        gamma = np.zeros(P)
        # for the initial value of gamma(ind_start), use the RVM formula
        #   gamma = (q^2 - s) / (s^2)
        # and the fact that initially s = S = beta*Psi_i'*Psi_i and q = Q =
        # beta*Psi_i'*Y
        gamma[ind_start] = np.square(PsiTY[ind_start])
        gamma[ind_start] -= sigma2 * PsiTPsi[ind_start, ind_start]
        gamma[ind_start] /= np.square(PsiTPsi[ind_start, ind_start])

        Sigma = 1. / (beta * PsiTPsi[ind_start, ind_start]
                      + 1./gamma[ind_start])

        mu = Sigma * PsiTY[ind_start] * beta
        tmp1 = beta * PsiTPsi[ind_start]
        S = beta * np.diag(PsiTPsi).T - Sigma * np.square(tmp1)
        Q = beta * PsiTY.T - mu*(tmp1)

        tmp2 = np.ones(P)  # alternative computation for the initial s,q
        q0tilde = PsiTY[ind_start]
        s0tilde = PsiTPsi[ind_start, ind_start]
        tmp2[ind_start] = s0tilde / (q0tilde**2) / beta
        s = np.divide(S, tmp2)
        q = np.divide(Q, tmp2)
        Lambda = 2*(num_active - 1) / np.sum(gamma)

        Delta_L_max = []
        for i in range(self.n_iter):
            # Handle variance zero
            if self.var_y:
                mu = np.mean(y)
                break

            if self.verbose:
                print('    lambda = {0:.6e}\n'.format(Lambda))

            # Calculate the potential updated value of each gamma[i]
            if Lambda == 0.0:  # RVM
                gamma_potential = np.multiply((
                    (q**2 - s) > Lambda),
                    np.divide(q**2 - s, s**2)
                    )
            else:
                a = Lambda * s**2
                b = s**2 + 2*Lambda*s
                c = Lambda + s - q**2
                gamma_potential = np.multiply(
                    (c < 0), np.divide(
                        -b + np.sqrt(b**2 - 4*np.multiply(a, c)), 2*a)
                    )

            l_gamma = - np.log(np.absolute(1 + np.multiply(gamma, s)))
            l_gamma += np.divide(np.multiply(q**2, gamma),
                                 (1 + np.multiply(gamma, s)))
            l_gamma -= Lambda*gamma  # omitted the factor 1/2

            # Contribution of each updated gamma(i) to L(gamma)
            l_gamma_potential = - np.log(
                np.absolute(1 + np.multiply(gamma_potential, s))
                )
            l_gamma_potential += np.divide(
                np.multiply(q**2, gamma_potential),
                (1 + np.multiply(gamma_potential, s))
                )
            # omitted the factor 1/2
            l_gamma_potential -= Lambda*gamma_potential

            # Check how L(gamma) would change if we replaced gamma(i) by the
            # updated gamma_potential(i), for each i separately
            Delta_L_potential = l_gamma_potential - l_gamma

            # deleted indices should not be chosen again
            if len(deleted_indices) != 0:
                values = -np.inf * np.ones(len(deleted_indices))
                Delta_L_potential[deleted_indices] = values

            Delta_L_max.append(np.nanmax(Delta_L_potential))
            ind_L_max = np.nanargmax(Delta_L_potential)

            # in case there is only 1 regressor in the model and it would now
            # be deleted
            if len(active_indices) == 1 and ind_L_max == active_indices[0] \
               and gamma_potential[ind_L_max] == 0.0:
                Delta_L_potential[ind_L_max] = -np.inf
                Delta_L_max[i] = np.max(Delta_L_potential)
                ind_L_max = np.argmax(Delta_L_potential)

            # If L did not change significantly anymore, break
            if Delta_L_max[i] <= 0.0 or\
                    (i > 0 and all(np.absolute(Delta_L_max[i-1:])
                                   < sum(Delta_L_max)*self.tol)) or \
                    (i > 0 and all(np.diff(bcs_path)[i-1:] == 0.0)):
                if self.verbose:
                    print('Increase in L: {0:.6e} (eta = {1:.3e})\
                          -- break\n'.format(Delta_L_max[i], self.tol))
                break

            # Print information
            if self.verbose:
                print('    Delta L = {0:.6e} \n'.format(Delta_L_max[i]))

            what_changed = int(gamma[ind_L_max] == 0.0)
            what_changed -= int(gamma_potential[ind_L_max] == 0.0)

            # Print information
            if self.verbose:
                if what_changed < 0:
                    print(f'{i+1} - Remove regressor #{ind_L_max+1}..\n')
                elif what_changed == 0:
                    print(f'{i+1} - Recompute regressor #{ind_L_max+1}..\n')
                else:
                    print(f'{i+1} - Add regressor #{ind_L_max+1}..\n')

            # --- Update all quantities ----
            if what_changed == 1:
                # adding a regressor

                # update gamma
                gamma[ind_L_max] = gamma_potential[ind_L_max]

                Sigma_ii = 1.0 / (1.0/gamma[ind_L_max] + S[ind_L_max])
                try:
                    x_i = np.matmul(
                        Sigma, PsiTPsi[active_indices, ind_L_max].reshape(-1, 1)
                        )
                except ValueError:
                    x_i = Sigma * PsiTPsi[active_indices, ind_L_max]
                tmp_1 = - (beta * Sigma_ii) * x_i
                Sigma = np.vstack(
                    (np.hstack(((beta**2 * Sigma_ii) * np.dot(x_i, x_i.T)
                                + Sigma, tmp_1)), np.append(tmp_1.T, Sigma_ii))
                    )
                mu_i = Sigma_ii * Q[ind_L_max]
                mu = np.vstack((mu - (beta * mu_i) * x_i, mu_i))

                tmp2_1 = PsiTPsi[:, ind_L_max] - beta * np.squeeze(
                    np.matmul(PsiTPsi[:, active_indices], x_i)
                    )
                if i == 0:
                    tmp2_1[0] /= 2
                tmp2 = beta * tmp2_1.T
                S = S - Sigma_ii * np.square(tmp2)
                Q = Q - mu_i * tmp2

                num_active += 1
                ind_global_to_local[ind_L_max] = num_active
                active_indices.append(ind_L_max)
                bcs_path.append(ind_L_max)

            elif what_changed == 0:
                # recomputation
                # zero if regressor has not been chosen yet
                if not ind_global_to_local[ind_L_max]:
                    raise Exception('Cannot recompute index{0} -- not yet\
                                    part of the model!'.format(ind_L_max))
                Sigma = np.atleast_2d(Sigma)
                mu = np.atleast_2d(mu)
                gamma_i_new = gamma_potential[ind_L_max]
                gamma_i_old = gamma[ind_L_max]
                # update gamma
                gamma[ind_L_max] = gamma_potential[ind_L_max]

                # index of regressor in Sigma
                local_ind = ind_global_to_local[ind_L_max]-1

                kappa_i = (1.0/gamma_i_new - 1.0/gamma_i_old)
                kappa_i = 1.0 / kappa_i
                kappa_i += Sigma[local_ind, local_ind]
                kappa_i = 1 / kappa_i
                Sigma_i_col = Sigma[:, local_ind]

                Sigma = Sigma - kappa_i * (Sigma_i_col * Sigma_i_col.T)
                mu_i = mu[local_ind]
                mu = mu - (kappa_i * mu_i) * Sigma_i_col[:, None]

                tmp1 = beta * np.dot(
                    Sigma_i_col.reshape(1, -1), PsiTPsi[active_indices])[0]
                S = S + kappa_i * np.square(tmp1)
                Q = Q + (kappa_i * mu_i) * tmp1

                # no change in active_indices or ind_global_to_local
                bcs_path.append(ind_L_max + 0.1)

            elif what_changed == -1:
                gamma[ind_L_max] = 0

                # index of regressor in Sigma
                local_ind = ind_global_to_local[ind_L_max]-1

                Sigma_ii_inv = 1. / Sigma[local_ind, local_ind]
                Sigma_i_col = Sigma[:, local_ind]

                Sigma = Sigma - Sigma_ii_inv * (Sigma_i_col * Sigma_i_col.T)

                Sigma = np.delete(
                    np.delete(Sigma, local_ind, axis=0), local_ind, axis=1)

                mu = mu - (mu[local_ind] * Sigma_ii_inv) * Sigma_i_col[:, None]
                mu = np.delete(mu, local_ind, axis=0)

                tmp1 = beta * np.dot(Sigma_i_col, PsiTPsi[active_indices])
                S = S + Sigma_ii_inv * np.square(tmp1)
                Q = Q + (mu_i * Sigma_ii_inv) * tmp1

                num_active -= 1
                ind_global_to_local[ind_L_max] = 0.0
                v = ind_global_to_local[ind_global_to_local > local_ind] - 1
                ind_global_to_local[ind_global_to_local > local_ind] = v
                del active_indices[local_ind]
                deleted_indices.append(ind_L_max)
                # and therefore ineligible
                bcs_path.append(-ind_L_max)

            # same for all three cases
            tmp3 = 1 - np.multiply(gamma, S)
            s = np.divide(S, tmp3)
            q = np.divide(Q, tmp3)

            # Update lambda
            Lambda = 2*(num_active - 1) / np.sum(gamma)

        # Prepare the result object
        self.coef_ = np.zeros(P)
        self.coef_[active_indices] = np.squeeze(mu)
        self.sigma_ = Sigma
        self.active_ = active_indices
        self.gamma = gamma
        self.Lambda = Lambda
        self.beta = beta
        self.bcs_path = bcs_path

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

        return self

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

            var_hat = 1./self.beta
            var_hat += np.sum(X.dot(self.sigma_) * X, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
        else:
            return y_hat

# l2norm = 0.0
# for idx in range(10):
#     sigma2 = np.genfromtxt('./test/sigma2_{0}.csv'.format(idx+1), delimiter=',')
#     Psi_train = np.genfromtxt('./test/Psi_train_{0}.csv'.format(idx+1), delimiter=',')
#     Y_train = np.genfromtxt('./test/Y_train_{0}.csv'.format(idx+1))
#     Psi_test = np.genfromtxt('./test/Psi_test_{0}.csv'.format(idx+1), delimiter=',')
#     Y_test = np.genfromtxt('./test/Y_test_{0}.csv'.format(idx+1))

#     clf = RegressionFastLaplace(verbose=True)
#     clf.fit_(Psi_train, Y_train, sigma2)
#     coeffs_fold = np.genfromtxt('./test/coeffs_fold_{0}.csv'.format(idx+1))
#     print("coeffs error: {0:.4g}".format(np.linalg.norm(clf.coef_ - coeffs_fold)))
#     l2norm += np.linalg.norm(Y_test - clf.predict(Psi_test))**2/len(Y_test)
#     print("l2norm error: {0:.4g}".format(l2norm))
