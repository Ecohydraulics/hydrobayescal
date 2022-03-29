"""
Auxiliary functions for the stochastic calibration of model using Surrogate-Assisted Bayesian
inversion
"""

import numpy as _np  # use underscore import to avoid double-numpy imports with wildcards (import *)


def compute_likelihood(prediction, observations, error_variance, normalize=False):
    """
    Calculates the multivariate Gaussian likelihood between model predictions and
    measured/observed data taking independent errors (diagonal covariance matrix)

    Input
    ----------
    prediction : array [MC, n_points]
        predicted / modeled values
    observations: array [1, n_points]
        observed / measured values
    error variance : array [n_points]]
        error of the observations
    normalize : bool
        Default False sets the constant before the exponent in eq. 3.14 (p. 27 in Acuna thesis)
        to 1.0. This setting is recommended when
        (1) the computational mesh does not vary, and
        (2) many measurement points are available with small measurement error.
        However, for enabling the comparison of multiple models or in the context of BME/information
        entropy calculations, this value should be set to True.

    Returns
    -------
    likelihood: array [MC]
       likelihood value

    Notes
    -----
    MC is the total number of model runs and n_points is the number of points considered for the comparison
    """

    cov_mat = _np.diag(error_variance)  # covariance matrix
    inv_r = _np.linalg.inv(cov_mat)

    # Calculate constants
    if normalize:
        n_points = observations.shape[1]
        det_R = _np.linalg.det(cov_mat)
        const_mvn = pow(2 * _np.pi, -n_points / 2) / _np.sqrt(det_R)
    else:
        const_mvn = 1.0

    # Vectorize means
    means_vect = observations[:, _np.newaxis]

    # Calculate differences and convert to 4D array (and its transpose)
    diff = means_vect - prediction  # Shape: # means
    diff_4d = diff[:, :, _np.newaxis]
    transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

    # Calculate values inside the exponent
    inside_1 = _np.einsum("abcd, dd->abcd", diff_4d, inv_r)
    inside_2 = _np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
    total_inside_exponent = inside_2.transpose(2, 1, 0)
    total_inside_exponent = _np.reshape(
        total_inside_exponent,
        (total_inside_exponent.shape[1], total_inside_exponent.shape[2])
    )

    likelihood = const_mvn * _np.exp(-0.5 * total_inside_exponent)
    if likelihood.shape[1] == 1:
        return likelihood[:, 0]
    else:
        return likelihood


def compute_bayesian_scores(prediction, observations, error_variance, enthropy_normalization="bayesian"):
    """
    Compute Bayesian Model Evidence (BME) and Relative Entropy (RE)

    Input
    ----------
    prediction : array [MC, n_points]
        predicted / modelled values
    observations: array [1, n_points]
        observed / measured values
    error_variance : array [n_points]]
        error of the observations
    enthropy_normalization : string
        Method for entrhopy cross normalization. The default is "bayesian" for Bayesian weighting.
        Other option is "rejection" for rejection sampling.

    Returns
    -------
    BME: float
        bayesian model evidence
    RE: float
        relative entropy

    Notes
    -----
    MC is the total number of model runs and n_points is the number of points considered for the comparison
    """

    # get likelihood
    likelihood = compute_likelihood(prediction, observations, error_variance).reshape(1, prediction.shape[0])

    # BME calculation
    BME = _np.mean(likelihood)

    if not(BME <= 0):
        # For cases in which the prediction of the surrogate is not too bad
        if not ("bay" in enthropy_normalization.lower()):
            # Non normalized cross entropy with rejection sampling
            accepted = likelihood / _np.amax(likelihood) >= _np.random.rand(1, prediction.shape[0])
            exp_log_pred = _np.mean(_np.log(likelihood[accepted]))
        else:
            # Non normalized cross entropy with bayesian weighting
            non_zero_likel = likelihood[_np.where(likelihood != 0)]
            post_weigth = non_zero_likel / _np.sum(non_zero_likel)
            exp_log_pred = _np.sum(post_weigth * _np.log(non_zero_likel))

        # Relative entropy calculation
        RE = exp_log_pred - _np.log(BME)
    else:
        # For cases in which the BME is zero (point selected from the prior gives bad results),
        # or the surrogate is giving bad results.
        BME = 0
        RE = 0
    return BME, RE


def BAL_selection_criteria(al_strategy, al_BME, al_RE):
    """
    Calculate the best value of the selected bayesian score and the index of the associated parameter combination

    Input
    ----------
    al_strategy : string
        strategy for active learning, selected bayesian score
    al_BME : array [d_size_AL,1]
        bayesian model evidence of active learning sets
    al_RE: array [d_size_AL,1]
        relative entropy of active learning sets

    Returns
    -------
    al_value: float
        best value of the selected bayesian score
    al_value_index: int
        index of the associated parameter combination

    Notes
    -----
    d_size_AL is the number of active learning sets (sets from the prior for the active learning procedure)
    """

    if al_strategy == "BME":
        al_value = _np.amax(al_BME)
        al_value_index = _np.argmax(al_BME)

        if _np.amax(al_BME) == 0:
            print("WARNING: Active Learning -- all values of Bayesian model evidences are 0")
            print("Active Learning Action: training points were selected randomly")

    elif al_strategy == "RE":
        al_value = _np.amax(al_RE)
        al_value_index = _np.argmax(al_RE)

        if _np.amax(al_RE) == 0 and _np.amax(al_BME) != 0:
            al_value = _np.amax(al_BME)
            al_value_index = _np.argmax(al_BME)
            print("WARNING: Active Learning -- all values of Relative entropies are 0")
            print("Active Learning Action: training points were selected according to Bayesian model evidences")
        elif _np.amax(al_RE) == 0 and _np.amax(al_BME) == 0:
            al_value = _np.amax(al_BME)
            al_value_index = _np.argmax(al_BME)
            print("WARNING: Active Learning -- all values of Relative entropies are 0")
            print("Warning Active Learning: all values of Bayesian model evidences are also 0")
            print("Active Learning Action: training points were selected randomly")
    try:
        return al_value, al_value_index
    except NameError:
        print("ERROR: failed to calculate selection (unknown active learning strategy %s provided)" % str(al_strategy))
        return _np.nan, _np.nan
