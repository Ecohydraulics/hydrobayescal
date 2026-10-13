"""

A new class is generated, which inherits all attributes from the GaussianProcessRegressor class from Scikit learn. This
is done to manually set the "max_iter" and "gtol" values for the optimization of hyperparameters in the GPR kernel.

ToDo: Check GPyTorch+lbfgs to see if results can be improved by changing initial values or with Adam ?
ToDo: Save each gp (for each loc) in a list, to call it later to do BAL+MCMC methods with them.
"""

import numpy as np
import sys
import math
import sklearn
import scipy
from sklearn.utils.optimize import _check_optimize_result
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.gaussian_process.kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import copy
from joblib import Parallel, delayed
import torch
import gpytorch
from pathlib import Path

# The validation criteria live in validation_metrics.py, one implementation for both emulator libraries. They stay importable from this module, where scripts written for earlier versions look for them.
from hydroBayesCal.surrogate.validation_metrics import save_valid_criteria, validation_error  # noqa: F401


# General GPR class
class MyGeneralGPR:
    """
    Class assigns/creates the attributes which are constant for all GPR-library classes (such as SklTraining and
    GPyTraining)

    Parameters:
        collocation_points = np.array(number of TP, number of parameters per TP), with training points (parameter sets)
        model_evaluations = np.array(number of TP, number of points where the model is evaluated), with full-complexity
         model outputs in each location where the fcm was evaluated/in the locations being considered

         # xx
        prior_samples = np.array(MC, number of parameters per TP), with MC parameter sets in which to evaluate the
        trained GPE

    Attributes:
        self.n_obs = int, number of locations from the fcm where the GPE is to be trained. It is not necessarily the
         same as the number of true observations, since one could train the GPE in given locations (e.g. all grid
         points), where some locations coincide with the observation points.

        # xx
        self.surrogate_prediction = np.array(self.n_obs, self.prior_samples.shape[0]), where to save the mean for
         each GPE (n_obs) for each parameter set in prior_samples

        self.surrogate_std = np.array(self.n_obs, self.prior_samples.shape[0]), where to save the standard deviation for
         each GPE (n_obs) for each parameter set in prior_samples

        self.surrogate_up = np.array(self.n_obs, self.prior_samples.shape[0]), where to save the upper confidence level
         for each GPE (n_obs) for each parameter set in prior_samples

        self.surrogate_lc = np.array(self.n_obs, self.prior_samples.shape[0]), where to save the lower confidence level
         for each GPE (n_obs) for each parameter set in prior_samples

    """
    def __init__(self, collocation_points, model_evaluations):
        self.training_points = collocation_points
        self.model_evaluations = model_evaluations

        self.n_obs = self.model_evaluations.shape[1]
        self.n_params = collocation_points.shape[1]

        self.gp_list = []


# Scikit-Learn -----------------------------------------------------------------------------------------------------
class MySklGPR(GaussianProcessRegressor):
    def __init__(self, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                              options={'maxiter':self.max_iter, 'gtol': self.gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min


class SklTraining(MyGeneralGPR):
    """Train a single-output Gaussian Process Emulator with scikit-learn.

    Uses scikit-learn's GP regression to build a GPE for a forward model, from
    collocation points produced by that model. See the scikit-learn
    `GaussianProcessRegressor
    <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_
    for the underlying estimator.

    Parameters
    ----------
    collocation_points : numpy.ndarray
        Training points (parameter sets), shape ``[n_tp, n_params]``.
    model_evaluations : numpy.ndarray
        Full-complexity model outputs at each evaluated location, shape
        ``[n_tp, n_locations]``.
    kernel : object or list of objects
        ``sklearn.gaussian_process.kernels`` instance(s) used to train the GPE;
        converted internally to a list.
    alpha : float or list of float
        Value added to the diagonal to avoid numerical errors. A scalar is
        broadcast to a list.
    n_restarts : int
        Number of optimiser restarts used to find the kernel hyper-parameters
        (avoids local minima).
    noise : bool, optional
        ``True`` (default) to add a white-noise kernel to the input kernel.
    y_normalization : bool, optional
        ``True`` (default) to normalise model outputs before training.
    tp_normalization : bool, optional
        ``True`` to normalise training-point parameter values before training
        (default ``False``).
    optimizer : str, optional
        Name of the optimiser to use (default scikit-learn optimiser).
    parallelize : bool, optional
        ``True`` to parallelise surrogate training, ``False`` to train
        sequentially.

    Notes
    -----
    .. todo:: Accept the evaluation location as input and add a function that
       extracts the GPE predictions at the observation point (used in BAL).
    """
    def __init__(self, collocation_points, model_evaluations,  kernel,
                 alpha, n_restarts, noise=True,
                 y_normalization=True, y_log=False,
                 tp_normalization=False,
                 optimizer="fmin_l_bfgs_b", parallelize=False, n_jobs=-2):

        super(SklTraining, self).__init__(collocation_points=collocation_points, model_evaluations=model_evaluations)

        # Input for GPR library in sklearn:
        self.n_restart = n_restarts
        self.y_normalization_ = y_normalization
        self.y_log = y_log
        self.optimizer_ = optimizer
        self.noise = noise

        self.parallel = parallelize
        self.n_jobs = n_jobs

        # Options for GPR library:
        self.tp_norm = tp_normalization

        self._id_vectors(alpha, kernel)

    def _id_vectors(self, alpha, kernel):
        """
        Function checks if the inputs for alpha and kernel are a single variable or a list. If they are a single value,
        the function generates a list filled with the same value/object, so it can be properly read in the train_
        function.
        Args:
            alpha: <float> or <list of floats [n_obs]>
                with input alpha value(s). If list, there should be one value per observation.
            kernel: <object> or <list of objects [n_obs]>
                Scikit learn kernel objects, to send to the GPR training

        Returns:
        """
        if isinstance(alpha, list):
            self.alpha = np.array(alpha)
        elif isinstance(alpha, float):
            self.alpha = np.full((self.training_points.shape[0], self.n_obs), alpha)
        elif isinstance(alpha, np.ndarray):
            if alpha.shape != (self.training_points.shape[0], self.n_obs):
                print('Using an alpha of 0')
                self.alpha = np.full((self.training_points.shape[0], self.n_obs), 0.0000001)
            else:
                self.alpha = alpha
        else:
            self.alpha = np.full((self.training_points.shape[0], self.n_obs), 0.0000001)

        if isinstance(kernel, list):
            self.kernel = kernel
        else:
            self.kernel = np.full(self.n_obs, kernel)

    def train_(self):
        """
        ToDo: Use joblib to parallelize training
        Returns:

        """
        # train a surrogate for each output observation:
        if self.parallel and self.n_obs > 1:
            # out = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self._fit)(self.training_points,
            #                                                                         self.model_evaluations[:, i],
            #                                                                         self.kernel[i],
            #                                                                         self.alpha[i])
            #                                                      for i in range(self.n_obs))
            out = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._fit)(self.training_points,
                                                                              self.model_evaluations[:, i],
                                                                              self.kernel[i],
                                                                              self.alpha[:, i])
                                                           for i in range(self.n_obs))
            self.gp_list = out
        else:
            for i, model in enumerate(self.model_evaluations.T):
                out = self._fit(self.training_points, model, self.kernel[i], self.alpha[:, i])
                self.gp_list.append(out)

    def _fit(self, collocation_points, model_y, kernel, alpha):
        """
        Function trains the Scikit-Learn surrogate model for each training location
        Args:
            collocation_points: array[n_tp, n_param]
                with training parameter sets
            model_y: array[n_tp,]
                with simulator outputs in training points
            kernel: object
                base kernel object, with constant*RBF_kernel

        Returns: dict
            with trained gp object, hyperparameters and normalization parameters (if needed)

        """
        # Set Kernel:
        if self.noise:
            kernel = kernel + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=np.std(model_y)/np.sqrt(2))

        # 1.Initialize instance of sklearn GPR:
        gp = MySklGPR(kernel=kernel, alpha=alpha, normalize_y=self.y_normalization_,
                      n_restarts_optimizer=self.n_restart, optimizer=self.optimizer_)

        if self.y_log:
            model_y = np.log(model_y)

        if self.tp_norm:  # Normalize the training points (if the scales are very different)
            # scaler_x_train = MinMaxScaler()
            scaler_x_train = StandardScaler()
            scaler_x_train.fit(self.training_points)
            collocation_points_scaled = scaler_x_train.transform(self.training_points)

            # 2. Train GPR
            gp.fit(collocation_points_scaled, model_y)
            score = gp.score(collocation_points_scaled, model_y)

        else:  # KEEP TP as they are
            # 2. Train GPR
            gp.fit(collocation_points, model_y)
            score = gp.score(collocation_points, model_y)

        return_out_dic = dict()
        return_out_dic['gp'] = gp

        hp = np.exp(gp.kernel_.theta)

        return_out_dic['c_hp'] = hp[0]
        if hp.shape[0] < self.n_params:
            return_out_dic['cl_hp'] = hp[1]
        else:
            return_out_dic['cl_hp'] = hp[1:self.n_params + 1]

        if self.noise:
            return_out_dic['noise_hp'] = hp[-1]
        return_out_dic['R2'] = score
        if self.tp_norm:
            return_out_dic['normalizer'] = scaler_x_train

        return return_out_dic

    def predict_(self, input_sets, get_conf_int=False):
        """Evaluate the per-location surrogate models on all input sets.

        Parameters
        ----------
        input_sets : numpy.ndarray
            Parameter sets to evaluate the surrogate models on, shape
            ``[MC, n_params]``.
        get_conf_int : bool, optional
            ``True`` to also estimate the upper and lower confidence intervals.

        Returns
        -------
        dict
            Surrogate-model mean (``output``) and standard deviation (``std``)
            for each location, each of shape ``[n_obs, MC]``.
        """
        # surrogate_prediction = np.zeros((len(self.gp_list), input_sets.shape[0]))  # GPE mean, for each obs
        # surrogate_std = np.zeros((len(self.gp_list), input_sets.shape[0]))  # GPE mean, for each obs
        # if get_conf_int:
        #     upper_ci = np.zeros((len(self.gp_list), input_sets.shape[0]))  # GPE mean, for each obs
        #     lower_ci = np.zeros((len(self.gp_list), input_sets.shape[0]))  # GPE mean, for each obs
        surrogate_prediction = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
        surrogate_std = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
        if get_conf_int:
            upper_ci = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
            lower_ci = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
        for i in range(0, self.n_obs):
            # MySklGPR inherits scikit-learn's GaussianProcessRegressor, whose method is
            # predict(X, return_std=True); it has no predict_ of its own. Only this
            # class does, which is the one being called here.
            if self.tp_norm:
                input_scaled = self.gp_list[i]['normalizer'].transform(input_sets)

                surrogate_prediction[:, i], surrogate_std[:, i] = self.gp_list[i]['gp'].predict(
                    input_scaled, return_std=True)
            else:
                surrogate_prediction[:, i], surrogate_std[:, i] = self.gp_list[i]['gp'].predict(
                    input_sets, return_std=True)
            if get_conf_int:
                lower_ci[:, i] = surrogate_prediction[:, i] - (1.96 * surrogate_std[:, i])
                upper_ci[:, i] = surrogate_prediction[:, i] + (1.96 * surrogate_std[:, i])

        output_dic = dict()
        # if self.y_log:
        #     surrogate_prediction = np.exp(surrogate_prediction)
        #     surrogate_std = # TODO
        output_dic['output'] = surrogate_prediction
        output_dic['std'] = surrogate_std
        if get_conf_int:
            output_dic['upper_ci'] = upper_ci
            output_dic['lower_ci'] = lower_ci

        return output_dic


# ------------------------------------------------------------------------------------------------------------------
