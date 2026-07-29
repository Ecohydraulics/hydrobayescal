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

from hydroBayesCal.utils.config_logging import logger, logger_warn


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
        norm_y = (train_y - np.mean(train_y))/(np.std(train_y))
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
                return_out_dic['y_norm'] = [np.mean(model_y), np.std(model_y)]

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

        # Iterate over each location
        for loc in range(num_locations):
            # Extract the columns corresponding to the current location
            Y_loc = Y[:, self.number_quantities * loc : self.number_quantities * (loc + 1)].numpy()

            # Normalize outputs
            y_mean = np.mean(Y_loc, axis=0)
            y_std = np.std(Y_loc, axis=0)
            Y_loc_norm = (Y_loc - y_mean) / y_std

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
        for q in range(self.number_quantities):
            Y_output_var = Y[:, q::self.number_quantities]
            y_mean = torch.mean(Y_output_var, dim=0)
            y_std = torch.std(Y_output_var, dim=0)
            Y_output_var_norm = ((Y_output_var - y_mean) / y_std).to(torch.float32)
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

        # Normalize Y
        y_mean = torch.mean(Y, dim=0)  # Compute mean per (location, quantity)
        y_std = torch.std(Y, dim=0)  # Compute std per (location, quantity)
        Y_norm = (Y - y_mean) / y_std  # Normalize Y

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

    @staticmethod
    def _task_kernel(gp, n_tasks):
        """Task covariance matrix of a multitask GP, shape ``[n_tasks, n_tasks]``."""
        task_indices = torch.arange(n_tasks)
        task_kernel = gp.covar_module.task_covar_module(task_indices).evaluate()
        return task_kernel.detach().cpu().numpy()

    @staticmethod
    def _input_kernel_diag(gp, input_sets):
        """k(x, x) at every prediction point, shape ``[n_samples]``."""
        base_kernel = gp.covar_module.data_covar_module.base_kernel
        try:
            diag = base_kernel(input_sets, input_sets, diag=True)
            return MultiGPyTraining._as_numpy(diag).reshape(-1)
        except (TypeError, RuntimeError):
            # Fall back to the per-point evaluation used before vectorisation.
            values = np.empty(input_sets.shape[0])
            for j in range(input_sets.shape[0]):
                x = input_sets[j].unsqueeze(0)
                values[j] = float(base_kernel(x, x).evaluate().squeeze())
            return values

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
        The returned task covariance is the prior Kronecker term (task kernel times
        the input kernel at the prediction point), not the posterior predictive
        covariance. With ``task_mode="locations"`` the tasks are the locations, so
        the cross-quantity covariance is not part of the model at all and a diagonal
        covariance built from the marginal standard deviations is returned instead.
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
                    # The single model's tasks span all outputs; the BAL utility
                    # consumes only the per-location (quantity x quantity) diagonal
                    # blocks, which are sample-independent up to the input kernel.
                    scale = self._input_kernel_diag(gp, input_sets)
                    task_kernel = self._task_kernel(gp, n_obs)
                    y_std_np = self._as_numpy(y_std)
                    blocks = []
                    for i in range(n_locations):
                        sl = slice(i * n_quantities, (i + 1) * n_quantities)
                        scaling = np.diag(y_std_np[sl])
                        blocks.append(scaling @ task_kernel[sl, sl] @ scaling)
                    for j in range(n_samples):
                        for i in range(n_locations):
                            multitask_cov_list[j][i] = scale[j] * blocks[i]

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
                        # The (n_quantities x n_quantities) task covariance is
                        # genuinely available in this layout. It does not depend on
                        # the sample, so build it once per location and only scale it
                        # by the input kernel at each prediction point.
                        scaling = np.diag(y_std_np)
                        task_kernel = self._task_kernel(gp, n_quantities)
                        block = scaling @ task_kernel @ scaling
                        scale = self._input_kernel_diag(gp, input_sets)
                        for j in range(n_samples):
                            multitask_cov_list[j][i] = scale[j] * block

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

