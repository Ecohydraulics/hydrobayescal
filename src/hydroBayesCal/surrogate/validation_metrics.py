"""Validation criteria of an emulator against full-complexity runs.

One implementation for both emulator libraries: :mod:`~hydroBayesCal.surrogate.gpe_gpytorch` and :mod:`~hydroBayesCal.surrogate.gpe_skl` import these functions and keep them available under their old names.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def validation_error(true_y, sim_y, output_names, n_per_type):
    """Estimate validation criteria for a surrogate model per output location.

    Results for each output type are stored under separate keys in a dictionary.

    Parameters
    ----------
    true_y : numpy.ndarray
        Simulator outputs for the validation samples, shape ``[mc_valid, n_obs]``.
    sim_y : numpy.ndarray or dict
        Surrogate/emulator outputs for the validation samples, shape ``[mc_valid, n_obs]``, or the dictionary returned by ``predict_``. The dictionary holds ``output`` and, optionally, ``std``, ``upper_ci`` and ``lower_ci``. The normalized error needs ``std``; the coverage ``P95`` needs both bounds, which ``predict_`` returns with ``get_conf_int=True``. A criterion whose input is missing is left out.
    output_names : array-like of str
        Name of each output type, shape ``[n_types]``.
    n_per_type : int
        Number of observations per output type. The columns of ``true_y`` are read in blocks of this size, one block per output type.

    Returns
    -------
    rmse : numpy.ndarray
        Root mean square error of every output column, shape ``[n_obs]``.
    criteria_dict : dict
        For every criterion (``rmse``, ``mse``, ``nse``, ``r2``, ``mean_error``, ``std_error`` and, if available, ``norm_error`` and ``P95``) a dictionary with one array of shape ``[n_per_type]`` per output type.

    Notes
    -----
    .. todo:: As in BayesValidRox, optionally estimate the surrogate predictions here by passing a surrogate object.
    """
    criteria_dict = {'rmse': dict(), 'mse': dict(), 'nse': dict(), 'r2': dict(), 'mean_error': dict(), 'std_error': dict()}

    sm_std = upper_ci = lower_ci = None
    if isinstance(sim_y, dict):
        sm_out = np.asarray(sim_y['output'])
        sm_std = sim_y.get('std')
        upper_ci, lower_ci = sim_y.get('upper_ci'), sim_y.get('lower_ci')
        if sm_std is not None:
            criteria_dict['norm_error'] = dict()
        if upper_ci is not None and lower_ci is not None:
            criteria_dict['P95'] = dict()
    else:
        sm_out = np.asarray(sim_y)
    true_y = np.asarray(true_y)

    # Root of the mean square error. The keyword squared=False of mean_squared_error did this until scikit-learn 1.6 removed it.
    rmse = np.sqrt(mean_squared_error(y_true=true_y, y_pred=sm_out, multioutput='raw_values'))

    c = 0
    for key in output_names:
        block = slice(c, c + n_per_type)
        criteria_dict['mse'][key] = mean_squared_error(y_true=true_y[:, block], y_pred=sm_out[:, block], multioutput='raw_values')
        criteria_dict['rmse'][key] = np.sqrt(criteria_dict['mse'][key])
        # Nash-Sutcliffe efficiency
        criteria_dict['nse'][key] = r2_score(y_true=true_y[:, block], y_pred=sm_out[:, block], multioutput='raw_values')

        # Relative errors of mean and standard deviation
        criteria_dict['mean_error'][key] = np.abs(
            np.mean(true_y[:, block], axis=0) - np.mean(sm_out[:, block], axis=0)) / np.mean(true_y[:, block], axis=0)
        criteria_dict['std_error'][key] = np.abs(
            np.std(true_y[:, block], axis=0) - np.std(sm_out[:, block], axis=0)) / np.std(true_y[:, block], axis=0)

        if 'norm_error' in criteria_dict:
            # Error normalized by the emulator standard deviation
            ind_val = np.divide(np.subtract(sm_out[:, block], true_y[:, block]), np.asarray(sm_std)[:, block])
            criteria_dict['norm_error'][key] = np.mean(ind_val ** 2, axis=0)

        if 'P95' in criteria_dict:
            # Share of the validation runs inside the confidence bounds
            p95 = np.where((true_y[:, block] <= np.asarray(upper_ci)[:, block])
                           & (true_y[:, block] >= np.asarray(lower_ci)[:, block]), 1, 0)
            criteria_dict['P95'][key] = np.mean(p95, axis=0)

        criteria_dict['r2'][key] = np.zeros(n_per_type)
        for j in range(n_per_type):
            criteria_dict['r2'][key][j] = np.corrcoef(true_y[:, j + c], sm_out[:, j + c])[0, 1]

        c = c + n_per_type

    return rmse, criteria_dict


def save_valid_criteria(new_dict, old_dict, n_tp):
    """Append the current iteration's validation criteria to a results dict.

    Stores the validation criteria for the current iteration (``n_tp``) in an existing dictionary, so the results for all iterations live in one file. Each validation criterion has a key per output type, holding a vector with one value per output location.

    Parameters
    ----------
    new_dict : dict
        Validation criteria for the current iteration.
    old_dict : dict
        Validation criteria for all previous iterations, including an ``N_tp`` key that tracks the iteration number.
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
