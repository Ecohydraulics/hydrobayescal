"""
This module inherits from the PyTorch library for training a Gaussian Process Emulator (GPE).
The module supersede the ExactGP base class from GPyTorch and extend the functionality by customizing the mean function,likelihoods and kernel (covariance function)
The MultitaskGPModel class also extends the ExactGP base class to handle multitask (multiple outputs) learning scenarios. It is designed to model multiple related tasks simultaneously
especially if they have similarities by sharing information across them using a common GP framework. (https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html).
Author: Andres Heredia (2024)
"""
import numpy as np
import sys
import scipy
import sklearn
import copy
from joblib import Parallel, delayed
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel, LinearKernel, ScaleKernel, ProductKernel, AdditiveKernel, MaternKernel, PeriodicKernel
from scipy.optimize import dual_annealing, differential_evolution

from hydroBayesCal.surrogate.bal_functions import project_to_psd
from hydroBayesCal.utils.config_logging import logger, logger_warn

#: Set once the clipping of an indefinite posterior covariance has been reported (MultiGPyTraining._posterior_task_blocks).
_REPORTED_CLIPPED_COVARIANCE = False

#: Combinations of constant output columns that have been reported (_report_constant_outputs): one warning per combination and process.
_REPORTED_CONSTANT_OUTPUTS = set()


def _output_normalization(outputs):
    """Level and spread for the normalization of the training outputs, column by column.

    The emulators are trained on ``(outputs - level) / divisor`` and transform their predictions back with ``scale * prediction + level``. For an output that varies between the training runs, level and spread are its mean and its standard deviation. An output that is the same in every run, such as the water depth at a calibration point that stays dry, has no spread to divide by: the division gave NaN and stopped the training. Such a column is trained on zeros (``divisor`` 1) and transformed back with ``scale`` 0, so that the emulator returns the value of the training runs without any emulator uncertainty.

    Parameters
    ----------
    outputs : numpy.ndarray or torch.Tensor
        Shape ``[n_runs]`` or ``[n_runs, n_columns]``. For a tensor, the spread is ``torch.std`` (n - 1 in the denominator), as before.

    Returns
    -------
    level, divisor, scale, constant
        Arrays or tensors of shape ``[n_columns]`` (scalars for one column); ``constant`` flags the columns whose values are all equal.
    """
    if isinstance(outputs, torch.Tensor):
        constant = outputs.max(dim=0).values == outputs.min(dim=0).values
        spread = torch.std(outputs, dim=0)
        level = torch.where(constant, outputs[0], torch.mean(outputs, dim=0))
        return level, torch.where(constant, torch.ones_like(spread), spread), torch.where(constant, torch.zeros_like(spread), spread), constant
    outputs = np.asarray(outputs)
    constant = np.ptp(outputs, axis=0) == 0
    spread = np.std(outputs, axis=0)
    level = np.where(constant, outputs[0], np.mean(outputs, axis=0)).astype(spread.dtype)
    return level, np.where(constant, 1.0, spread).astype(spread.dtype), np.where(constant, 0.0, spread).astype(spread.dtype), constant


def _report_constant_outputs(columns, n_runs, n_quantities=None):
    """Warn once that some output columns are the same in all training runs.

    Parameters
    ----------
    columns : iterable of int
        Zero-based columns of ``model_evaluations``.
    n_runs : int
        Number of training runs.
    n_quantities : int, optional
        Number of calibration targets, to name calibration point and target of a column (quantities interleaved per point).
    """
    columns = tuple(sorted(int(column) for column in columns))
    if not columns or columns in _REPORTED_CONSTANT_OUTPUTS:
        return
    _REPORTED_CONSTANT_OUTPUTS.add(columns)
    if n_quantities:
        where = "; ".join(f"calibration point {column // n_quantities + 1}, target {column % n_quantities + 1}"
                          for column in columns)
    else:
        where = ", ".join(f"output column {column + 1}" for column in columns)
    logger_warn.warning(
        f"The model output is the same in all {n_runs} training runs at: {where}. The emulator returns that value "
        f"there, without emulator uncertainty. An output that does not respond to the calibration parameters, such as "
        f"the water depth at a point that stays dry, carries no information for the calibration; consider removing it "
        f"from the calibration points. This message is shown once per process.")


class MyExactGPyModel(gpytorch.models.ExactGP):
    """
    Instance of GPyTorch's "ExactGP" library, with custom likelihood, kernel, training points.

    The likelihood is kept constant: Gaussian Likelihood (https://docs.gpytorch.ai/en/latest/likelihoods.html)

    Parameters:
        :param train_x: <np.array[n_tp, n_p]> with parameter sets used to train GPR
        :param train_y: <np.array[n_tp, n_obs]> with forward model outputs used to train GPR
        :param kernel: <kernel instance> with kernel used in GPR
        :param likelihood <likelihood instance> to train noise in GPR
    """

    def __init__(
            self, train_x,
            train_y, kernel,
            likelihood
    ):
        super(MyExactGPyModel, self).__init__(train_inputs=train_x, train_targets=train_y, likelihood=likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        """
        Takes in the training data (x) and returns a multivariate normal distribution with mean and covariance (kernel)
        set in "__init__()"
        :param x: training data (parameter sets)
        :return:
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTraining:
    """Train a single-output Gaussian Process Emulator with GPyTorch.

    Uses GPyTorch's exact GP regression to build a GPE for a forward model, from
    collocation points produced by that model.

    Parameters
    ----------
    collocation_points : numpy.ndarray
        Training points (parameter sets), shape ``[n_tp, n_p]``.
    model_evaluations : numpy.ndarray
        Model outputs at each evaluated location, shape ``[n_tp, n_obs]``.
    kernel : gpytorch.kernels.Kernel
        Kernel used to train the GPE. May use default values or a user-defined
        anisotropy (``ard_num_dims = n_parameters``).
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Likelihood used to optimise the GPR. May use default
        constraints/initial values or be customised by the caller.
    training_iter : int
        Number of optimiser iterations used to train the GPE.
    optimizer : str, optional
        Optimiser to use: ``"adam"`` (default) or ``"lbfgs"``.
    loss : str, optional
        Loss function: ``"exact"`` or ``"loo"``.
    n_restarts : int, optional
        Number of optimisation restarts.
    tp_normalization : bool, optional
        ``True`` to normalise training-point parameter values before training
        (default ``False``).
    y_normalization : bool, optional
        ``True`` to normalise model outputs before training (predictions are
        de-normalised afterwards).
    parallelize : bool, optional
        ``True`` to parallelise surrogate training, ``False`` to train
        sequentially.

    Notes
    -----
    .. todo:: Accept the evaluation location as input and add a function that
       extracts the GPE predictions at the observation point (used in BAL).
    .. todo:: For GPyTorch, check the GPU settings and other
       ``gpytorch.settings`` for prediction.
    """

    def __init__(
            self,
            collocation_points,
            model_evaluations,
            kernel,
            training_iter,
            likelihood,
            y_normalization=True,
            tp_normalization=False,
            optimizer="adam",
            lr=0.5,
            loss='exact',
            n_restarts=1,
            weight_decay=0,
            gradient_free_start=False,
            verbose=True,
            parallelize=False
    ):

        # Basic attributed
        self.training_points = collocation_points
        self.model_evaluations = model_evaluations

        self.n_obs = self.model_evaluations.shape[1]
        self.n_params = collocation_points.shape[1]

        self.gp_list = []

        # Input for GPR library in GPyTorch:
        self.kernel = kernel
        self.optimizer_ = optimizer
        self.training_iter = training_iter
        self.loss = loss
        self.n_restarts = n_restarts
        self.gradient_free_start = gradient_free_start
        self.lr = lr
        self.weight_decay = weight_decay

        # self.likelihood = likelihood
        self.parallel = parallelize

        self.verbose = verbose

        # Options for GPR library:
        self.tp_norm = tp_normalization
        self.y_norm = y_normalization

        self._id_vectors(likelihood, kernel)

    def _id_vectors(self, likelihood, kernel):
        """
        Function checks if the inputs for alpha and kernel are a single variable or a list. If they are a single value,
        the function generates a list filled with the same value/object, so it can be properly read in the train_
        function.
        Args:
            likelihood: <object> or <list of objects [n_obs]>
                with input alpha value(s). If list, there should be one value per observation.
            kernel: <object> or <list of objects [n_obs]>
                Scikit learn kernel objects, to send to the GPR training

        Returns:
        """
        if isinstance(likelihood, list):
            self.likelihood = np.array(likelihood)
        else:
            self.likelihood = np.full(self.n_obs, likelihood)

    @staticmethod
    def convert_to_tensor(array):
        """
        Function to transform np.array to a tensor
        Args:
            array: <np.array> that you want to change to a tensor

        Returns: <tensor> data in np.array transformed to tensor format
        """
        transformed = torch.tensor(array).float()
        return transformed

    def normalize_tp(self, train_y):
        """
        Function to normalize training points outputs before training
        Args:
            train_y: <np.array[tp_size, n_obs]> with model output values to normalize

        Returns: <tensor> with normalized input values.
        """
        level, divisor, _, _ = _output_normalization(train_y)
        norm_y = (train_y - level) / divisor
        train_y = self.convert_to_tensor(norm_y)
        return train_y

    def train_(self):
        """
        Function trains the surrogate model using the GPyTorch library, using the given optimizer.
        Returns:
        ToDo: parallelize training
        """
        # Convert training points and prior parameter sets to tensor:
        train_x = self.convert_to_tensor(self.training_points)

        if self.y_norm:
            constant = np.ptp(np.asarray(self.model_evaluations), axis=0) == 0
            _report_constant_outputs(np.flatnonzero(constant), self.model_evaluations.shape[0])

        if self.parallel and self.n_obs > 1:
            out = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self._fit_adam)(train_x=train_x,
                                                                                         model_y=self.model_evaluations[:, i],
                                                                                         likelihood=self.likelihood[i],
                                                                                         kernel=self.kernel)
                                                                 for i in range(self.n_obs))
            self.gp_list = out
        else:
            for i, y_model in enumerate(self.model_evaluations.T):
                out = self._fit_adam(train_x=train_x, model_y=y_model, likelihood=self.likelihood[i],
                                     kernel=self.kernel)

                self.gp_list.append(out)

    @staticmethod
    def init_model_params(model):
        """
        Function to initalize model hyperparameters, for multi-start optimizations
        Args:
            model: GPyTorch instance
        Returns:
        """
        def initialize_tensor(param_):
            if len(param_.shape) < 2:
                # If the tensor has fewer than two dimensions, apply a different initialization method
                torch.nn.init.uniform_(param_)  # Example: Uniform initialization for tensors with fewer than 2 dimensions
            else:
                torch.nn.init.xavier_uniform_(param_)

        for name, param in model.named_parameters():
            initialize_tensor(param)

    def _fit_adam(self, train_x, model_y, kernel, likelihood):
        """
        Function trains the GPR for a given training location using the 'adam' optimizer from PyTorch
        Args:
            train_x: tensor [n_tp, n_param]
                with input parameter sets to use in training
            model_y: array[n_tp,]
                with simulator outputs in training points
        Returns: dict
            with trained gp object, trained likelihood object, hyperparameters
            and y_normalization parameters (if needed)

        """
        # Normalize, if needed, and transform model_evaluations at loc "i" to a tensor
        if self.y_norm:
            train_y = self.normalize_tp(model_y)
        else:
            train_y = self.convert_to_tensor(model_y)

        best_loss = float('inf')
        best_params = None

        for i in range(self.n_restarts):

            # Initialize kernel and likelihood:
            kernel_ = copy.deepcopy(kernel)
            likelihood_ = copy.deepcopy(likelihood)

            # Initialize instance of GPyTorch GPR:
            gp = MyExactGPyModel(train_x, train_y, kernel_, likelihood_)

            # Start training
            gp.train()
            likelihood_.train()

            if i > 1:
                self.init_model_params(gp)

            # Setup Optimizer
            if self.optimizer_ == 'adam':
                optimizer = torch.optim.Adam(gp.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer_ == 'lbfgs':
                optimizer = torch.optim.lbfgs(gp.parameters(), lr=self.lr)
            else:
                raise ValueError(f'There is no optimizer {self.optimizer_} available.')

            # Loss for GPs - log likelihood
            if 'exact' in self.loss.lower():
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_, gp)
            elif 'loo' in self.loss.lower():
                mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_, gp)

            # Change learning rate:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5,
                gamma=0.5
            )

            if i == 0 and self.gradient_free_start:
                def negative_log_likelihood(params):
                    """ Gradient-free optimizer from SciPy"""
                    # Set the model hyperparameters based on the optimization parameters
                    gp.covar_module.base_kernel.lengthscale = torch.tensor(params[0:self.n_params])
                    gp.covar_module.outputscale = torch.tensor(params[-2])
                    gp.likelihood.noise = torch.tensor(params[-1])

                    # Zero out the gradients
                    gp.zero_grad()

                    # Forward pass to compute the negative log likelihood
                    output = gp(train_x)
                    loss = -mll(output, train_y)

                    if self.verbose:
                        print(f'Gradient-free start - Loss: {loss.item()}',
                              f'   Outputscale: {gp.covar_module.outputscale.item()}',
                              f'   lengthscale: {gp.covar_module.base_kernel.lengthscale[0]}',
                              f'   noise: {gp.likelihood.noise.item()}')

                    # Return the negative log likelihood as a NumPy array
                    return loss.item()

                # Define the bounds for the optimization parameters (lengthscale and outputscale)
                value = np.empty((), dtype=object)
                value[()] = (1e-3, 1e2)
                bounds_1 = list(np.full(self.n_params, value, dtype=object))
                bounds_2 = [(1e-2, 1e1), (1e-6, 1)]
                bounds = bounds_1 + bounds_2

                # Perform the global optimization using differential evolution
                # t1 = time.time()
                result = differential_evolution(negative_log_likelihood, bounds, maxiter=10)

            # Optimize parameters:
            for j in range(self.training_iter):
                # a. Zero gradients from previous iteration
                optimizer.zero_grad()
                # b. Output from model
                output = gp(train_x)
                # c. Calculate loss and back propagation gradients
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                scheduler.step()   # Change learning rate
                if self.verbose:
                    print(f'Iter {j + 1}/{self.training_iter} - Loss: {loss.item()}',
                          f'   Outputscale: {gp.covar_module.outputscale.item()}',
                          f'   lengthscale: {gp.covar_module.base_kernel.lengthscale[0]}',
                          f'   noise: {gp.likelihood.noise.item()}')
            if loss < best_loss:
                best_loss = loss
                best_params = gp.state_dict()

        gp.load_state_dict(best_params)

        with torch.no_grad():
            return_out_dic = dict()
            return_out_dic['gp'] = gp
            return_out_dic['likelihood'] = likelihood_

            return_out_dic['c_hp'] = gp.covar_module.outputscale.item()
            return_out_dic['cl_hp'] = gp.covar_module.base_kernel.lengthscale.numpy()[0, :]
            return_out_dic['noise_hp'] = gp.likelihood.noise.item()
            if self.y_norm:
                level, _, scale, _ = _output_normalization(model_y)
                return_out_dic['y_norm'] = [level, scale]

        return return_out_dic

    def predict_(self, input_sets, get_conf_int=False):
        """
        TO DO: DESCRIPTION TO BE COMPLETED

        :param input_sets:
        :param get_conf_int:
        :return:
        """

        prior_x = self.convert_to_tensor(input_sets)

        surrogate_prediction = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
        surrogate_std = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
        if get_conf_int:
            upper_ci = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs
            lower_ci = np.zeros((input_sets.shape[0], len(self.gp_list)))  # GPE mean, for each obs

        for i in range(0, self.n_obs):
            # Extract data
            gp = copy.deepcopy(self.gp_list[i]['gp'])
            likelihood_ = copy.deepcopy(self.gp_list[i]['likelihood'])

            # 5.1 Go into eval mode:
            gp.eval()
            likelihood_.eval()

            with torch.no_grad():
                f_pred = likelihood_(gp(prior_x))
                prediction = f_pred.mean.numpy()
                std = f_pred.stddev.numpy()
                if self.y_norm:  # Back-transform
                    # normalized results
                    prediction = self.gp_list[i]['y_norm'][1] * prediction + self.gp_list[i]['y_norm'][0]
                    std = std * self.gp_list[i]['y_norm'][1]

                surrogate_prediction[:, i] = prediction
                surrogate_std[:, i] = std

                # Calculate 95% confidence intervals.
                if get_conf_int:
                    upper_ci[:, i] = surrogate_prediction[:, i] + 2 * surrogate_std[:, i]
                    lower_ci[:, i] = surrogate_prediction[:, i] - 2 * surrogate_std[:, i]

        output_dic = dict()
        output_dic['output'] = surrogate_prediction
        output_dic['std'] = surrogate_std
        if get_conf_int:
            output_dic['upper_ci'] = upper_ci
            output_dic['lower_ci'] = lower_ci

        return output_dic


def validation_error(true_y, sim_y, output_names, n_per_type):
    """Estimate validation criteria for a surrogate model per output location.

    Results for each output type are stored under separate keys in a dictionary.

    Parameters
    ----------
    true_y : numpy.ndarray
        Simulator outputs for the validation samples, shape
        ``[mc_valid, n_obs]``.
    sim_y : numpy.ndarray or dict
        Surrogate/emulator outputs for the validation samples, shape
        ``[mc_valid, n_obs]``. If a dict, it holds ``output`` and ``std`` keys.
    output_names : array-like of str
        Name of each output type, shape ``[n_types]``.
    n_per_type : int
        Number of observations per output type.

    Returns
    -------
    tuple
        Validation criteria for each output location and output type.

    Notes
    -----
    .. todo:: As in BayesValidRox, optionally estimate the surrogate predictions
       here by passing a surrogate object.
    .. todo:: Move into the GPR class and return a dictionary keyed by output
       type.
    """
    criteria_dict = {
        'rmse': dict(),
         'mse': dict(),
         'nse': dict(),
         'r2': dict(),
         'mean_error': dict(),
         'std_error': dict()
    }

    if isinstance(sim_y, dict):
        sm_out = sim_y['output']
        sm_std = sim_y['std']
        upper_ci = sim_y['upper_ci']
        lower_ci = sim_y['lower_ci']

        criteria_dict['norm_error'] = dict()
        criteria_dict['P95'] = dict()
    else:
        sm_out = sim_y

    # RMSE for each output location: not a dictionary (yet). [n_obs, ]
    rmse = sklearn.metrics.mean_squared_error(
        y_true=true_y,
        y_pred=sm_out,
        multioutput='raw_values',
        squared=False
    )

    c = 0
    for i, key in enumerate(output_names):
        # RMSE
        criteria_dict['rmse'][key] = sklearn.metrics.mean_squared_error(
            y_true=true_y[:, c:c + n_per_type],
            y_pred=sm_out[:, c:c + n_per_type],
            multioutput='raw_values',
            squared=False
        )
        # NSE
        criteria_dict['nse'][key] = sklearn.metrics.r2_score(
            y_true=true_y[:, c:c+n_per_type],
            y_pred=sm_out[:, c:c+n_per_type],
            multioutput='raw_values'
        )

        # NSE
        criteria_dict['nse'][key] = sklearn.metrics.r2_score(
            y_true=true_y[:, c:c + n_per_type],
            y_pred=sm_out[:, c:c + n_per_type],
            multioutput='raw_values'
        )
        criteria_dict['mse'][key] = sklearn.metrics.mean_squared_error(
            y_true=true_y[:, c:c + n_per_type],
            y_pred=sm_out[:, c:c + n_per_type],
            multioutput='raw_values',
            squared=True
        )

        # Mean errors
        criteria_dict['mean_error'][key] = np.abs(
            np.mean(true_y[:, c:c + n_per_type], axis=0) - np.mean(sm_out[:, c:c + n_per_type], axis=0)) / np.mean(
            true_y[:, c:c + n_per_type], axis=0)

        criteria_dict['std_error'][key] = np.abs(
            np.std(true_y[:, c:c + n_per_type], axis=0) - np.std(sm_out[:, c:c + n_per_type], axis=0)) / np.std(
            true_y[:, c:c + n_per_type], axis=0)

        # Norm error
        if isinstance(sim_y, dict):
            # Normalized error
            ind_val = np.divide(np.subtract(sm_out[:, c:c + n_per_type], true_y[:, c:c + n_per_type]),
                                sm_std[:, c:c + n_per_type])
            criteria_dict['norm_error'][key] = np.mean(ind_val ** 2, axis=0)

            # P95
            p95 = np.where((true_y[:, c:c + n_per_type] <= upper_ci[:, c:c + n_per_type]) & (
                        true_y[:, c:c + n_per_type] >= lower_ci[:, c:c + n_per_type]), 1, 0)
            criteria_dict['P95'][key] = np.mean(p95, axis=0)

        criteria_dict['r2'][key] = np.zeros(n_per_type)
        for j in range(n_per_type):
            criteria_dict['r2'][key][j] = np.corrcoef(true_y[:, j+c], sm_out[:, j+c])[0, 1]

        c = c + n_per_type

    return rmse, criteria_dict


def save_valid_criteria(new_dict, old_dict, n_tp):
    """Append the current iteration's validation criteria to a results dict.

    Stores the validation criteria for the current iteration (``n_tp``) in an
    existing dictionary, so the results for all iterations live in one file.
    Each validation criterion has a key per output type, holding a vector with
    one value per output location.

    Parameters
    ----------
    new_dict : dict
        Validation criteria for the current iteration.
    old_dict : dict
        Validation criteria for all previous iterations, including an ``N_tp``
        key that tracks the iteration number.
    n_tp : int
        Number of training points for the current BAL iteration.

    Returns
    -------
    dict
        The updated dictionary including the current iteration.
    """

    if len(old_dict) == 0:
        old_dict = dict(new_dict)
        old_dict['N_tp'] = [n_tp]
    else:
        for key in old_dict:
            if key == 'N_tp':
                old_dict[key].append(n_tp)
            else:
                for out_type in old_dict[key]:
                    old_dict[key][out_type] = np.vstack((old_dict[key][out_type], new_dict[key][out_type]))

    return old_dict


class MultiGPyTraining:
    """
    Class to train multiple Gaussian Process models using given collocation points and model evaluations.
    It uses the MultiGPyTraining class for multitask regression using GPyTorch.
    """

    def __init__(
            self,
            collocation_points,
            model_evaluations,
            kernel,
            training_iter,
            likelihood,
            optimizer="adam",
            lr=0.5,
            n_restarts=1,
            parallelize=False,
            number_quantities=2,
            noise_constraint=GreaterThan(1e-6)
    ):
        """
        Parameters
        ----------
        See original class docstring for parameter details.
        """
        # Basic attributes
        self.training_points = collocation_points
        self.model_evaluations = model_evaluations
        self.number_quantities = number_quantities
        self.n_obs = self.model_evaluations.shape[1]
        self.n_params = collocation_points.shape[1]
        self.gp_list = []
        self.normalization_params = []
        # Which train_tasks_* method produced ``gp_list``: "variables", "locations"
        # or "all". ``predict_`` dispatches on this instead of on len(gp_list),
        # which is ambiguous whenever nloc == number_quantities or nloc == 1.
        self.task_mode = None
        self._warned_locations_cov = False

        # Initialize likelihood and other hyperparameters
        self.likelihood = likelihood
        self.kernel = kernel
        self.optimizer_ = optimizer
        self.training_iter = training_iter
        self.n_restarts = n_restarts
        self.lr = lr
        self.parallel = parallelize
        self.noise_constraint = noise_constraint
        self.losses = []  # To store loss values for tracking convergence
        self.lengthscales = []
        self.noise_values = []  # To track noise values
        self.multitask_cov = [] # To store multitask covariance matrices (Kronecker product of task kernel and input kernel)

    def train_tasks_variables(self):#Training variables at each location separately
        """
        Train multitask Gaussian Process models using the provided collocation points and model evaluations.

        One multitask GP per location; the tasks are the calibration quantities, so
        the cross-quantity correlation at a location is modelled explicitly.
        """
        self.task_mode = "variables"
        X = torch.tensor(self.training_points, dtype=torch.float32)
        Y = torch.tensor(self.model_evaluations, dtype=torch.float32)
        # Number of locations
        num_locations = Y.shape[1] // self.number_quantities

        constant_columns = []

        # Iterate over each location
        for loc in range(num_locations):
            # Extract the columns corresponding to the current location
            Y_loc = Y[:, self.number_quantities * loc : self.number_quantities * (loc + 1)].numpy()

            # Normalize outputs. A target that is the same in every run at this location is trained on zeros and returned without emulator uncertainty, see _output_normalization.
            y_mean, divisor, y_std, constant = _output_normalization(Y_loc)
            Y_loc_norm = (Y_loc - y_mean) / divisor
            constant_columns.extend(self.number_quantities * loc + np.flatnonzero(constant))

            # Save normalization parameters
            self.normalization_params.append((y_mean, y_std))

            # Convert normalized Y_loc to torch
            Y_loc_norm = torch.tensor(Y_loc_norm, dtype=torch.float32)

            # Initialize multitask GP model. The likelihood and the kernel are
            # deep-copied per location: sharing one instance across the loop lets
            # every model overwrite the previous one's fitted noise and
            # lengthscales, so only the last location's hyperparameters survived.
            likelihood = copy.deepcopy(self.likelihood)
            kernel = copy.deepcopy(self.kernel)
            model = MultitaskGPModel(X, Y_loc_norm, likelihood, kernel, self.number_quantities)

            # Set model and likelihood to training mode
            model.train()
            likelihood.train()

            # Set optimizer
            if self.optimizer_ == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            else:
                raise ValueError(f"Optimizer '{self.optimizer_}' not supported.")

            # Set MLL objective
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            converged = False
            plateau_count = 0  # Counter for consecutive small changes in loss
            threshold = 1e-3  # Convergence threshold for loss
            required_plateau = 5  # Number of consecutive stable steps required for convergence

            # For length scale tracking and convergence check
            lengthscale_converged = False
            lengthscale_plateau_count = 0
            lengthscale_threshold = 1e-2  # Convergence threshold for length scale

            # Training loop
            for _ in range(self.training_iter):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, Y_loc_norm)
                loss.backward()
                optimizer.step()

                self.losses.append(loss.item())
                noise = likelihood.noise.detach().cpu().numpy()
                self.noise_values.append(noise)

                # Track lengthscale
                try:
                    lengthscale = model.covar_module.data_covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                except AttributeError:
                    lengthscale = None
                self.lengthscales.append(lengthscale)

                # Loss convergence check
                if len(self.losses) > 1:
                    loss_diff = abs(self.losses[-1] - self.losses[-2])
                    if loss_diff < threshold:
                        plateau_count += 1
                    else:
                        plateau_count = 0
                    if plateau_count >= required_plateau:
                        converged = True

                # Length scale convergence check
                if lengthscale is not None and len(self.lengthscales) > 1:
                    prev_ls = self.lengthscales[-2]
                    curr_ls = self.lengthscales[-1]
                    ls_diff = np.abs(curr_ls - prev_ls)
                    if np.all(ls_diff < lengthscale_threshold):
                        lengthscale_plateau_count += 1
                    else:
                        lengthscale_plateau_count = 0
                    if lengthscale_plateau_count >= required_plateau:
                        lengthscale_converged = True

            self.gp_list.append({'gp': model, 'y_norm': (y_mean, y_std),
                                 'likelihood': likelihood})

        _report_constant_outputs(constant_columns, Y.shape[0], self.number_quantities)

    def train_tasks_locations(self):
        """
        Train multitask Gaussian Process models using the provided collocation points and model evaluations.
        Trains one model per calibration quantity, with all locations as the tasks.

        Note that with this task layout the cross-quantity correlation at a given
        location is not modelled (the tasks are the locations), so ``predict_``
        can only return a diagonal per-location covariance.
        """
        self.task_mode = "locations"
        X = torch.tensor(self.training_points, dtype=torch.float32)
        Y = torch.tensor(self.model_evaluations, dtype=torch.float32)

        if self.n_obs % self.number_quantities:
            raise ValueError(
                f"Number of model outputs ({self.n_obs}) is not a multiple of the "
                f"number of calibration quantities ({self.number_quantities}).")

        # Number of locations
        num_locations = Y.shape[1] // self.number_quantities  # Total number of locations

        # The model outputs interleave the quantities per location, so quantity q
        # occupies columns q, q + n_quantities, q + 2*n_quantities, ...
        constant_columns = []
        for q in range(self.number_quantities):
            Y_output_var = Y[:, q::self.number_quantities]
            # A location where this quantity is the same in every run is trained on zeros and returned without emulator uncertainty, see _output_normalization.
            y_mean, divisor, y_std, constant = _output_normalization(Y_output_var)
            Y_output_var_norm = ((Y_output_var - y_mean) / divisor).to(torch.float32)
            constant_columns.extend(q + self.number_quantities * np.flatnonzero(self._as_numpy(constant)))
            self.normalization_params.append((f'output_{q + 1}', y_mean, y_std))

            # Initialize multitask GP model for the entire set of locations (tasks).
            # Deep copies keep each quantity's noise and lengthscales its own.
            likelihood = copy.deepcopy(self.likelihood)
            kernel = copy.deepcopy(self.kernel)
            model = MultitaskGPModel(X, Y_output_var_norm, likelihood, kernel,
                                     number_tasks=num_locations)  # num_tasks = num_locations

            # Set model and likelihood to training mode
            model.train()
            likelihood.train()

            # Set optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Set MLL objective
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Training loop for the entire set of N locations (tasks) for this variable
            for _ in range(self.training_iter):
                optimizer.zero_grad()
                output = model(X)

                # Ensure the output shape matches expected for MLL

                loss = -mll(output, Y_output_var_norm)
                loss.backward()
                optimizer.step()

            # Store trained model for the current calibration quantity
            self.gp_list.append({'gp': model, 'y_norm': (y_mean, y_std),
                                 'likelihood': likelihood})

        _report_constant_outputs(constant_columns, Y.shape[0], self.number_quantities)

    def train_tasks_all(self):
        """
        Train a single multitask Gaussian Process model using all outputs (water depth and velocity)
        at all locations simultaneously.
        """
        self.task_mode = "all"
        X = torch.tensor(self.training_points, dtype=torch.float32)
        Y = torch.tensor(self.model_evaluations, dtype=torch.float32)

        # Number of locations
        num_locations = Y.shape[1] // self.number_quantities  # Total number of locations

        # Normalize Y per (location, quantity). An output that is the same in every run is trained on zeros and returned without emulator uncertainty, see _output_normalization.
        y_mean, divisor, y_std, constant = _output_normalization(Y)
        Y_norm = (Y - y_mean) / divisor
        _report_constant_outputs(np.flatnonzero(self._as_numpy(constant)), Y.shape[0], self.number_quantities)

        # Save normalization parameters
        self.normalization_params = {'y_mean': y_mean, 'y_std': y_std}

        # Initialize multitask GP model. Copied for symmetry with the other two
        # task layouts, so that self.likelihood/self.kernel stay untrained templates.
        likelihood = copy.deepcopy(self.likelihood)
        kernel = copy.deepcopy(self.kernel)
        model = MultitaskGPModel(X, Y_norm, likelihood, kernel, number_tasks=self.n_obs)

        # Set model and likelihood to training mode
        model.train()
        likelihood.train()

        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Set MLL objective
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = model(X)

            # Compute negative marginal likelihood loss
            loss = -mll(output, Y_norm)
            loss.backward()
            optimizer.step()

        # Store trained model
        self.gp_list = [{'gp': model, 'y_norm': (y_mean, y_std),
                         'likelihood': likelihood}]

    @staticmethod
    def _as_numpy(value):
        """Return ``value`` as a numpy array, accepting torch tensors."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _model_likelihood(self, model_info):
        """Return the likelihood belonging to one trained sub-model.

        Surrogates pickled before the per-sub-model likelihood fix stored a single
        shared instance on the class, so fall back to the model's own attribute and
        then to the template.
        """
        likelihood = model_info.get('likelihood')
        if likelihood is None:
            likelihood = getattr(model_info['gp'], 'likelihood', None)
        if likelihood is None:
            likelihood = self.likelihood
        return likelihood

    def _resolve_task_mode(self, n_locations):
        """Return the task layout of the trained models.

        Uses ``self.task_mode`` (set by the ``train_tasks_*`` method that filled
        ``gp_list``). Surrogates pickled before that attribute existed fall back to
        the historical heuristic based on ``len(gp_list)``, which is ambiguous when
        ``n_locations`` equals the number of quantities or equals 1, so the fallback
        warns.
        """
        mode = getattr(self, "task_mode", None)
        if mode is not None:
            return mode

        n_models = len(self.gp_list)
        if n_models == 1:
            mode = "all"
        elif n_models == self.number_quantities:
            mode = "locations"
        elif n_models == n_locations:
            mode = "variables"
        else:
            raise ValueError(
                f"Mismatch in gp_list length ({n_models}). Expected {n_locations} "
                f"(locations) or {self.number_quantities} (quantities).")

        message = (
            f"MultiGPyTraining has no 'task_mode' attribute (surrogate trained with an "
            f"earlier version); assuming task layout '{mode}' from "
            f"len(gp_list)={n_models}.")
        if n_locations == self.number_quantities or n_locations == 1:
            message += (
                " This is ambiguous because the number of locations equals the number "
                "of calibration quantities (or is 1), so the output column order may "
                "be wrong. Retrain the surrogate to remove the ambiguity.")
        logger_warn.warning(message)
        return mode

    @staticmethod
    def _posterior_task_blocks(predictions, block_size, n_train=1, budget=2.0e7, scale=None):
        """Posterior predictive covariance between the tasks at every prediction point.

        Only the blocks the Bayesian active-learning utility consumes are evaluated: for each prediction point, the covariance within consecutive groups of ``block_size`` tasks. The covariance between different prediction points, and between tasks of different groups, is never formed. With the tasks spanning all outputs (``task_mode="all"``) that is the difference between ``n_locations * n_quantities**2`` and ``(n_locations * n_quantities)**2`` entries per point, and the latter does not fit into memory for a real calibration.

        Parameters
        ----------
        predictions : gpytorch.distributions.MultitaskMultivariateNormal
            Predictive distribution returned by ``likelihood(gp(input_sets))``.
        block_size : int
            Number of consecutive tasks per block, i.e. the number of calibration quantities.
        n_train : int
            Number of training points of the model. Only used to size the chunks.
        budget : float
            Upper bound on the entries handled per chunk of prediction points. Evaluating one covariance entry costs memory in proportion to ``n_train * n_tasks``, so the prediction points are processed in chunks.
        scale : array_like, optional
            Standard deviation of the output normalization of every task, shape ``[n_tasks]``. When given, the blocks are converted from the normalized output space of the model to physical units: with ``y = y_mean + D z``, ``Cov(y) = D Cov(z) D``.

        Returns
        -------
        numpy.ndarray
            Shape ``[n_samples, n_tasks // block_size, block_size, block_size]``, in physical units if ``scale`` is given and in the normalized output space of the model otherwise. Every block is symmetric positive semidefinite.
        """
        covariance = predictions.lazy_covariance_matrix
        n_points, n_tasks = predictions.mean.shape[-2:]
        n_blocks = n_tasks // block_size
        interleaved = getattr(predictions, "_interleaved", True)

        inner = torch.arange(block_size, device=covariance.device)
        first_task = torch.arange(n_blocks, device=covariance.device) * block_size
        task_rows = first_task.view(1, -1, 1, 1) + inner.view(1, 1, -1, 1)
        task_cols = first_task.view(1, -1, 1, 1) + inner.view(1, 1, 1, -1)

        entries_per_point = n_blocks * block_size ** 2
        chunk = max(1, int(budget // (entries_per_point * max(1, int(n_train)) * n_tasks)))

        blocks = np.empty((n_points, n_blocks, block_size, block_size))
        for start in range(0, n_points, chunk):
            points = torch.arange(start, min(start + chunk, n_points),
                                  device=covariance.device).view(-1, 1, 1, 1)
            if interleaved:
                rows, cols = points * n_tasks + task_rows, points * n_tasks + task_cols
            else:
                rows, cols = task_rows * n_points + points, task_cols * n_points + points
            rows, cols = torch.broadcast_tensors(rows, cols)
            values = covariance[rows, cols]
            if not isinstance(values, torch.Tensor):
                values = values.to_dense()
            blocks[start:start + chunk] = MultiGPyTraining._as_numpy(values)
        if scale is not None:
            # In double precision: a scaling rounded to single precision is not exactly of the form D B D and can itself tip a nearly singular block over.
            scale = np.asarray(MultiGPyTraining._as_numpy(scale), dtype=float).reshape(n_blocks, block_size)
            blocks = blocks * scale[None, :, :, None] * scale[None, :, None, :]
        # The blocks are evaluated in single precision as the difference between the prior covariance and the part explained by the training data. Where the emulator is nearly certain, the result is orders of magnitude smaller than both (six in the case that revealed this, an emulator with correlation lengths of ten times the parameter ranges, which is that certain at every candidate and not only next to a training run). The error of the difference then reaches the size of the result: the two triangles differ, and a block can come out with a negative eigenvalue. The sampler of the BAL utility refuses such a block, and the calibration would stop in its active-learning loop. The blocks are therefore symmetrized and the indefinite ones projected, as the last operation on them; a positive semidefinite block is returned as it is. The projection yields a valid covariance, it does not restore the lost digits; a double-precision evaluation of the model would.
        blocks, ratio = project_to_psd(blocks, return_ratio=True)
        global _REPORTED_CLIPPED_COVARIANCE
        if ratio < 0.0 and not _REPORTED_CLIPPED_COVARIANCE:
            logger_warn.warning(
                f"The emulator covariance between the calibration targets was not positive semidefinite at some prediction "
                f"points (smallest eigenvalue {ratio:.1e} times the largest in magnitude) and was clipped to a valid "
                f"covariance. The emulator is nearly certain there, and its variance is at the resolution of single "
                f"precision. The predicted means are not affected. This message is shown once per process.")
            _REPORTED_CLIPPED_COVARIANCE = True
        return blocks

    def predict_(self, input_sets, get_conf_int=False, multitask_cov=False):
        """
        Predict the outputs and their standard deviations for given input sets using the trained GP models.

        The task layout is taken from ``self.task_mode``, i.e. from the
        ``train_tasks_*`` method that produced ``self.gp_list``.

        Parameters
        ----------
        input_sets : array
            Parameter sets to predict at, shape ``[n_samples, n_params]``.
        get_conf_int : bool
            Additionally return the 2-sigma confidence bounds.
        multitask_cov : bool
            Additionally return the per-location task covariance consumed by the
            Bayesian active-learning utility, as a nested list
            ``[n_samples][n_locations]`` of ``[n_quantities, n_quantities]`` arrays.
            ``scipy.linalg.block_diag`` over one sample's entry reproduces the
            interleaved (location, quantity) column order of the model outputs.

        Returns
        -------
        dict
            ``output`` and ``std``, each ``[n_samples, nloc * n_quantities]`` in the
            interleaved (location, quantity) order, plus ``upper_ci``/``lower_ci``
            and ``multitask_cov`` when requested.

        Note
        ----
        The returned task covariance is the posterior predictive covariance between the calibration quantities at the prediction point, the same distribution the returned ``std`` is the marginal of. With ``task_mode="locations"`` the tasks are the locations, so the cross-quantity covariance is not part of the model at all and a diagonal covariance built from the marginal standard deviations is returned instead.
        """
        input_sets = torch.tensor(input_sets, dtype=torch.float32)
        n_samples = input_sets.shape[0]
        n_obs = self.n_obs
        n_quantities = self.number_quantities
        n_locations = n_obs // n_quantities

        # NaN-initialised so that any column left unwritten by a wrong dispatch
        # fails loudly at the check below instead of silently predicting zeros.
        surrogate_prediction = np.full((n_samples, n_obs), np.nan)
        surrogate_std = np.full((n_samples, n_obs), np.nan)
        multitask_cov_list = None

        if get_conf_int:
            upper_ci = np.full((n_samples, n_obs), np.nan)
            lower_ci = np.full((n_samples, n_obs), np.nan)

        if multitask_cov:
            multitask_cov_list = [[None] * n_locations for _ in range(n_samples)]

        mode = self._resolve_task_mode(n_locations)
        if mode == "all":
            model_info = self.gp_list[0]
            gp = model_info['gp']
            y_mean, y_std = model_info['y_norm']
            likelihood = self._model_likelihood(model_info)

            gp.eval()
            likelihood.eval()

            with torch.no_grad():
                predictions = likelihood(gp(input_sets))
                mean = predictions.mean  # PyTorch tensor of shape (n_samples, n_obs)
                std = predictions.stddev  # PyTorch tensor of shape (n_samples, n_obs)

                # Back-transform normalized predictions
                mean = y_std * mean + y_mean  # Undo normalization
                std = std * y_std

                # Convert to NumPy arrays and store in the respective matrices
                surrogate_prediction[:] = self._as_numpy(mean)
                surrogate_std[:] = self._as_numpy(std)

                if get_conf_int:
                    upper_ci[:] = surrogate_prediction + 2 * surrogate_std
                    lower_ci[:] = surrogate_prediction - 2 * surrogate_std

                if multitask_cov:
                    # The tasks span all outputs. BAL consumes one quantity x quantity block per location, so only those blocks are evaluated.
                    blocks = self._posterior_task_blocks(
                        predictions, n_quantities, n_train=self.training_points.shape[0], scale=y_std)
                    for j in range(n_samples):
                        for i in range(n_locations):
                            multitask_cov_list[j][i] = blocks[j, i]

        elif mode == "locations":
            for q, model_info in enumerate(self.gp_list):
                gp = model_info['gp']
                y_mean, y_std = model_info['y_norm']
                likelihood = self._model_likelihood(model_info)

                gp.eval()
                likelihood.eval()

                with torch.no_grad():
                    predictions = likelihood(gp(input_sets))
                    mean = predictions.mean  # This is a PyTorch tensor
                    std = predictions.stddev  # This is a PyTorch tensor

                    # Make sure the normalisation constants are tensors as well
                    if isinstance(y_std, np.ndarray):
                        y_std = torch.tensor(y_std, dtype=torch.float32)
                    if isinstance(y_mean, np.ndarray):
                        y_mean = torch.tensor(y_mean, dtype=torch.float32)

                    # Back-transform normalized predictions
                    mean = y_std * mean + y_mean
                    std = std * y_std

                    # Tasks are the locations, so calibration quantity q occupies
                    # every n_quantities-th output column starting at q.
                    surrogate_prediction[:, q::n_quantities] = self._as_numpy(mean)
                    surrogate_std[:, q::n_quantities] = self._as_numpy(std)
                    if get_conf_int:
                        upper_ci[:, q::n_quantities] = self._as_numpy(mean + 2 * std)
                        lower_ci[:, q::n_quantities] = self._as_numpy(mean - 2 * std)

            if multitask_cov:
                if not getattr(self, "_warned_locations_cov", False):
                    logger_warn.warning(
                        "multitask_selection='locations' models the locations as tasks, "
                        "so a cross-quantity covariance at a location does not exist in "
                        "this layout. Bayesian active learning falls back to a diagonal "
                        "per-location covariance built from the marginal standard "
                        "deviations. Use multitask_selection='variables' to model "
                        "cross-quantity correlation.")
                    self._warned_locations_cov = True
                for j in range(n_samples):
                    for i in range(n_locations):
                        sl = slice(i * n_quantities, (i + 1) * n_quantities)
                        multitask_cov_list[j][i] = np.diag(surrogate_std[j, sl] ** 2)

        elif mode == "variables":
            for i, model_info in enumerate(self.gp_list):
                gp = model_info['gp']
                y_mean, y_std = model_info['y_norm']
                likelihood = self._model_likelihood(model_info)

                gp.eval()
                likelihood.eval()

                with torch.no_grad():
                    predictions = likelihood(gp(input_sets))
                    mean = self._as_numpy(predictions.mean)
                    std = self._as_numpy(predictions.stddev)

                    # Back-transform normalized predictions
                    y_mean_np = self._as_numpy(y_mean)
                    y_std_np = self._as_numpy(y_std)
                    mean = y_std_np * mean + y_mean_np
                    std = std * y_std_np

                    # Tasks are the quantities, so this location's quantities occupy
                    # one contiguous block of output columns.
                    sl = slice(i * n_quantities, (i + 1) * n_quantities)
                    surrogate_prediction[:, sl] = mean
                    surrogate_std[:, sl] = std
                    if get_conf_int:
                        upper_ci[:, sl] = mean + 2 * std
                        lower_ci[:, sl] = mean - 2 * std

                    if multitask_cov:
                        # Posterior predictive covariance between the calibration quantities at this location, for every prediction point.
                        blocks = self._posterior_task_blocks(
                            predictions, n_quantities, n_train=self.training_points.shape[0], scale=y_std_np)[:, 0]
                        for j in range(n_samples):
                            multitask_cov_list[j][i] = blocks[j]
        else:
            raise ValueError(
                f"Unknown task_mode '{mode}'. Expected 'variables', 'locations' or 'all'.")

        n_unwritten = int(np.isnan(surrogate_prediction).sum() + np.isnan(surrogate_std).sum())
        if n_unwritten:
            raise RuntimeError(
                f"Surrogate prediction left {n_unwritten} output entries unwritten with "
                f"task_mode='{mode}' and len(gp_list)={len(self.gp_list)}. The trained "
                f"task layout does not match the expected output shape "
                f"({n_samples}, {n_obs}).")

        # Prepare output dictionary
        if multitask_cov:
            output_dic = {
                'output': surrogate_prediction,
                'std': surrogate_std,
                'multitask_cov': multitask_cov_list
            }
        else:
            output_dic = {
                'output': surrogate_prediction,
                'std': surrogate_std
            }
        if get_conf_int:
            output_dic['upper_ci'] = upper_ci
            output_dic['lower_ci'] = lower_ci

        return output_dic


class MultitaskGPModel(ExactGP):
    """
    Gaussian Process model for multitask regression using the GPyTorch library. This model handles multiple tasks (or quantities) simultaneously by using a multitask kernel and multitask mean
    function.
    """

    def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            kernel,
            number_tasks
    ):
        """
        :param train_x: torch.Tensor
            The input training data. A tensor of shape (n_samples, n_params) where `n_samples` is the number of samples and
            `n_params` is the number of input model parameters.

        :param train_y: torch.Tensor
            The output training data. A tensor of shape (n_samples, n_tasks) where `n_samples` is the number of samples and
            `n_tasks` is the number of tasks or quantities. The output is typically organized so that each column corresponds
            to a different task.

        :param likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood
            A multitask likelihood function used with the GP model.

        :param kernel: tuple(gpytorch.kernels.Kernel, gpytorch.kernels.Kernel)
            A tuple of kernel components to be used in the GP model. The tuple should contain two kernel components.
        """
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=number_tasks)
        self.covar_module = MultitaskKernel(
                            kernel,
            num_tasks=number_tasks, rank=2
        )

    def forward(self, x):
        """
         Computes the mean and covariance of the Gaussian Process given the input data `x`.

        :param x: torch.Tensor
            A tensor containing the training data for the model.
        :return: gpytorch.distributions.MultitaskMultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class SOSequentialGPyTraining:
    """Container for independent quantity-specific ``GPyTraining`` objects.

    Each contained GPE predicts one calibration quantity at every measurement
    location.  This adapter exposes the ``training_points`` and ``predict_``
    interface expected by ``SequentialDesign`` while reconstructing predictions
    in the original location-major/interleaved ordering::

        [Q1_P1, Q2_P1, ..., Qn_P1, Q1_P2, Q2_P2, ..., Qn_P2, ...]

    The quantity GPEs remain statistically independent; therefore no multitask
    cross-covariance is constructed.
    """

    def __init__(self, surrogate_models, calibration_quantities):
        self.models = dict(surrogate_models)
        self.calibration_quantities = list(calibration_quantities)
        self.num_quantities = len(self.calibration_quantities)

        if self.num_quantities == 0:
            raise ValueError("calibration_quantities cannot be empty.")

        missing = [
            var for var in self.calibration_quantities
            if var not in self.models
        ]
        if missing:
            raise KeyError(
                f"Missing SO surrogate model(s) for calibration quantities: {missing}"
            )

        first_model = self.models[self.calibration_quantities[0]]
        self.training_points = np.asarray(first_model.training_points)
        self.nloc = int(first_model.n_obs)
        self.n_obs = self.nloc * self.num_quantities
        self.n_params = self.training_points.shape[1]
        self.exp_design = None

        # Keep a combined copy of the training outputs in the exact same ordering
        # used by the complex model, observations and Bayesian likelihood.
        self.model_evaluations = np.empty(
            (self.training_points.shape[0], self.n_obs), dtype=float
        )

        for quantity_idx, var in enumerate(self.calibration_quantities):
            model = self.models[var]

            if int(model.n_obs) != self.nloc:
                raise ValueError(
                    f"All SO GPEs must contain the same number of locations. "
                    f"'{var}' has {model.n_obs}; expected {self.nloc}."
                )

            model_tp = np.asarray(model.training_points)
            if (
                model_tp.shape != self.training_points.shape
                or not np.allclose(model_tp, self.training_points)
            ):
                raise ValueError(
                    "All SO GPEs must use exactly the same collocation points."
                )

            expected_shape = (self.training_points.shape[0], self.nloc)
            if np.asarray(model.model_evaluations).shape != expected_shape:
                raise ValueError(
                    f"Unexpected training-output shape for '{var}': "
                    f"{np.asarray(model.model_evaluations).shape}; "
                    f"expected {expected_shape}."
                )

            self.model_evaluations[
                :, quantity_idx::self.num_quantities
            ] = model.model_evaluations

    @classmethod
    def from_training_data(cls, collocation_points, model_outputs, calibration_quantities,
                           training_iter=150, lr=0.07, verbose=False):
        """Build the container from the training data of a calibration.

        One single-output GPE is set up per calibration target, with the kernel, likelihood and optimizer settings the drivers use for a calibration on one target, so a target is emulated the same way whether it is calibrated alone or together with others. The models are not trained yet, call :meth:`train_`.

        Parameters
        ----------
        collocation_points : numpy.ndarray
            Parameter sets of the training runs, shape ``[n_runs, n_params]``.
        model_outputs : numpy.ndarray
            Model outputs in the interleaved (location, quantity) order, shape ``[n_runs, n_locations * n_quantities]``.
        calibration_quantities : list of str
            Names of the calibration targets, in the order of the output columns of one location.
        training_iter : int
            Number of optimizer iterations per GPE.
        lr : float
            Learning rate of the optimizer.
        verbose : bool
            Passed on to :class:`GPyTraining`.

        Returns
        -------
        SOSequentialGPyTraining
        """
        collocation_points = np.atleast_2d(np.asarray(collocation_points, dtype=float))
        model_outputs = np.atleast_2d(np.asarray(model_outputs, dtype=float))
        calibration_quantities = list(calibration_quantities)
        n_quantities = len(calibration_quantities)
        if n_quantities == 0:
            raise ValueError("calibration_quantities cannot be empty.")
        if model_outputs.shape[1] % n_quantities:
            raise ValueError(
                f"The model outputs have {model_outputs.shape[1]} columns, which is not a multiple of the "
                f"{n_quantities} calibration quantities {calibration_quantities}.")

        surrogate_models = {}
        for quantity_idx, quantity in enumerate(calibration_quantities):
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=collocation_points.shape[1]))
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
            likelihood.noise = 1e-5
            # Interleaved ordering: quantity q occupies every n_quantities-th column, starting at q.
            surrogate_models[quantity] = GPyTraining(
                collocation_points=collocation_points,
                model_evaluations=model_outputs[:, quantity_idx::n_quantities],
                likelihood=likelihood, kernel=kernel,
                training_iter=training_iter,
                optimizer="adam", lr=lr,
                verbose=verbose)
        return cls(surrogate_models=surrogate_models, calibration_quantities=calibration_quantities)

    def __getitem__(self, key):
        """Allow ``surrogate_object[var]`` access to an individual SO GPE."""
        return self.models[key]

    def train_(self):
        """Train all quantity-specific GPEs independently."""
        for var in self.calibration_quantities:
            self.models[var].train_()

    def predict_(self, input_sets, get_conf_int=False, multitask_cov=False):
        """Predict all quantities and restore the original interleaved ordering."""
        if multitask_cov:
            raise ValueError(
                "SO_sequential uses independent GPEs and therefore has no "
                "multitask cross-covariance. Use multitask=False in SequentialDesign."
            )

        input_sets = np.asarray(input_sets)
        n_samples = input_sets.shape[0]
        full_shape = (n_samples, self.n_obs)

        output = np.empty(full_shape, dtype=float)
        std = np.empty(full_shape, dtype=float)

        if get_conf_int:
            upper_ci = np.empty(full_shape, dtype=float)
            lower_ci = np.empty(full_shape, dtype=float)

        for quantity_idx, var in enumerate(self.calibration_quantities):
            quantity_prediction = self.models[var].predict_(
                input_sets=input_sets,
                get_conf_int=get_conf_int,
            )

            expected_shape = (n_samples, self.nloc)
            mean_q = np.asarray(quantity_prediction["output"])
            std_q = np.asarray(quantity_prediction["std"])

            if mean_q.shape != expected_shape or std_q.shape != expected_shape:
                raise ValueError(
                    f"Unexpected prediction shape for '{var}'. "
                    f"Mean={mean_q.shape}, std={std_q.shape}; "
                    f"expected {expected_shape}."
                )

            # Restore location-major/interleaved ordering:
            # Q1_P1, Q2_P1, ..., Qn_P1, Q1_P2, ...
            output[:, quantity_idx::self.num_quantities] = mean_q
            std[:, quantity_idx::self.num_quantities] = std_q

            if get_conf_int:
                if "upper_ci" in quantity_prediction and "lower_ci" in quantity_prediction:
                    upper_q = np.asarray(quantity_prediction["upper_ci"])
                    lower_q = np.asarray(quantity_prediction["lower_ci"])
                else:
                    upper_q = mean_q + 2.0 * std_q
                    lower_q = mean_q - 2.0 * std_q

                upper_ci[:, quantity_idx::self.num_quantities] = upper_q
                lower_ci[:, quantity_idx::self.num_quantities] = lower_q

        result = {
            "output": output,
            "std": std,
        }

        if get_conf_int:
            result["upper_ci"] = upper_ci
            result["lower_ci"] = lower_ci

        return result
