"""
Sizing, sampling and vetting of the initial design of a surrogate-assisted calibration.

Bayesian active learning refines a Gaussian process emulator that it inherits from the
initial design. If that emulator is wrong in whole regions of the parameter space, the
likelihood surface BAL exploits has peaks the full complexity model does not have, and
the loop happily converges onto one of them: BAL adds training points where its own
error looks most informative, so a region it never saw stays unseen. Too many initial
runs are just as bad in practice, because every initial run is a full TELEMAC, OpenFOAM
or Delft3D simulation and the budget spent there is a budget not spent on BAL.

This module answers the two questions that follow from that, without ever guessing:

* **How many initial runs are needed?** :func:`recommended_init_runs` sizes the design
  from the number of calibration parameters and rounds to a power of two, which is where
  a Sobol sequence is balanced.
* **Were they enough?** :func:`initial_design_sufficiency` measures it on the runs that
  have actually been carried out: emulator predictivity by leave-one-out
  cross-validation, whether the emulator's own error bars are honest, whether the
  implied posterior is resolved by enough accepted samples, whether its shape is driven
  by the data rather than by emulator uncertainty, and whether it stopped moving between
  two successive blocks of runs.

The two combine into a **staged Sobol ladder**: run a block, measure, and only extend if
the measurement says so. Extending is free of waste because a Sobol sequence of order
``2n`` has the order-``n`` sequence as its exact prefix, so the second block continues
the first one instead of replacing it (:func:`sobol_block`).

Following the convention of
:func:`~hydroBayesCal.function_pool.diagnose_roughness_identifiability` and of
:mod:`~hydroBayesCal.surrogate.posterior_analysis`, every function here is report-only:
it returns a plain dictionary carrying a ``verdict``, a ``message`` and a
``recommendation``, mutates no state, and leaves the logging decision to
:func:`log_initial_design`. Nothing in this module launches a simulation or decides a
training point.
"""
import json
import math
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from threadpoolctl import threadpool_limits

from hydroBayesCal.utils.config_logging import logger, logger_warn

__all__ = [
    "recommended_init_runs",
    "initial_design_ladder",
    "validate_sampling_method",
    "sobol_block",
    "loo_predictivity",
    "initial_design_sufficiency",
    "log_initial_design",
    "run_staged_initial_design",
    "write_initial_design_record",
    "DEFAULT_THRESHOLDS",
    "SAMPLING_METHODS",
]

#: Runs per calibration parameter for the a-priori sizing rule. Ten is the established
#: rule of thumb for an initial design of a computer experiment that a Gaussian process
#: is to be fitted to (Loeppky, Sacks and Welch 2009).
SAMPLES_PER_DIMENSION = 10

#: Absolute floor. Below this a Gaussian process over more than one dimension cannot be
#: fitted to anything worth calling a surrogate, whatever the number of parameters.
MINIMUM_INIT_RUNS = 16

#: Sampling methods that reach chaospy intact through ``bayesvalidrox``. The strings
#: ``chebyshev(FT)`` and ``grid(FT)``, still advertised in older configurations, are not
#: among chaospy's rule names and are mapped below instead of being passed on to fail
#: several minutes into a run.
SAMPLING_METHODS = (
    "random",
    "latin_hypercube",
    "sobol",
    "halton",
    "hammersley",
    "chebyshev",
    "nested_chebyshev",
    "grid",
    "nested_grid",
    "korobov",
    "additive_recursion",
    "user",
)

_SAMPLING_METHOD_ALIASES = {
    "chebyshev(ft)": "chebyshev",
    "grid(ft)": "grid",
    "lhs": "latin_hypercube",
    "latin-hypercube": "latin_hypercube",
    "mc": "random",
}

#: Thresholds of the sufficiency gate. Every one of them is a property of the emulator
#: and of the posterior it implies, not of the solver, so the same numbers apply to a
#: TELEMAC, an OpenFOAM and a Delft3D calibration.
DEFAULT_THRESHOLDS = {
    # Leave-one-out predictivity Q2 = 1 - SS_loo / SS_total, per output column.
    "q2_median": 0.90,
    "q2_min": 0.70,
    # Fraction of standardised leave-one-out residuals inside the 95 % interval.
    "coverage_low": 0.85,
    # Accepted posterior samples, absolute floor and per calibration parameter.
    "post_size_absolute": 200,
    "post_size_per_dimension": 25,
    # Median emulator standard deviation relative to the observation standard
    # deviation, over the accepted samples.
    "error_ratio": 0.50,
    # Movement of the posterior between two blocks, in posterior standard deviations,
    # and change of the log evidence. Measured on the posterior mean rather than on its
    # maximum, which at a few hundred accepted samples wanders by a third of a standard
    # deviation between two rejection samplings of the same emulator.
    "posterior_shift": 0.25,
    "delta_log_bme": 1.0,
}

#: Largest prior sample the gate will draw while chasing the accepted-sample floor.
#: A sharp posterior in a wide prior is accepted rarely, and the resolution criterion
#: has to measure the design rather than the size of the prior sample.
PRIOR_SAMPLE_CAP = 60000


# ---------------------------------------------------------------------------
# a-priori sizing
# ---------------------------------------------------------------------------
def _next_power_of_two(value):
    """Smallest power of two greater than or equal to ``value``."""
    value = max(math.ceil(value), 1)
    return int(2 ** math.ceil(math.log2(value)))


def recommended_init_runs(
        ndim,
        init_runs=None,
        max_runs=None,
        samples_per_dimension=SAMPLES_PER_DIMENSION,
        minimum=MINIMUM_INIT_RUNS,
):
    """How many initial runs a calibration with ``ndim`` parameters needs.

    The rule is ``samples_per_dimension * ndim``, floored at ``minimum`` and rounded up
    to the next power of two. The rounding is not cosmetic: chaospy generates an
    unscrambled Sobol sequence, whose equidistribution properties hold at
    :math:`n = 2^m`. A design of 100 Sobol points is measurably less uniform than one of
    128, and the extra 28 runs buy more than the 28 BAL iterations they would otherwise
    pay for, because BAL cannot repair a region the emulator has never seen.

    Nothing here is enforced. The result is a recommendation, compared against the
    configured ``init_runs`` so a run that is going to be undersized says so in its log
    before the first simulation starts rather than after the last one.

    Parameters
    ----------
    ndim : int
        Number of calibration parameters.
    init_runs : int, optional
        The configured number of initial runs, for the comparison.
    max_runs : int, optional
        The configured total budget, to check that a usable BAL budget is left.
    samples_per_dimension : int
        Runs per calibration parameter before rounding. Default 10.
    minimum : int
        Absolute floor before rounding. Default 16.

    Returns
    -------
    dict
        ``recommended``, ``floor``, ``ladder``, ``configured``, ``bal_budget``,
        ``verdict`` (``adequate`` | ``generous`` | ``undersized`` | ``no_bal_budget``),
        ``message`` and ``recommendation``.
    """
    ndim = int(ndim)
    if ndim < 1:
        raise ValueError(f"A calibration needs at least one parameter, got ndim={ndim}.")

    floor = max(int(samples_per_dimension) * ndim, int(minimum))
    recommended = _next_power_of_two(floor)
    ceiling = int(init_runs) if init_runs else recommended
    ladder = initial_design_ladder(ceiling, ndim)
    bal_budget = (int(max_runs) - ceiling) if max_runs else None

    if init_runs is None:
        verdict = "adequate"
        message = (f"{ndim} calibration parameters need about {samples_per_dimension} "
                   f"runs each, so {floor} runs, rounded up to {recommended} for a "
                   f"balanced Sobol design.")
        recommendation = f"Set sampling['init_runs'] = {recommended}."
    elif bal_budget is not None and bal_budget < 1:
        verdict = "no_bal_budget"
        message = (f"init_runs = {ceiling} leaves {bal_budget} runs for Bayesian active "
                   f"learning out of max_runs = {max_runs}, so the calibration would be "
                   f"an initial design and nothing else.")
        recommendation = (f"max_runs must exceed init_runs. Budget at least {ndim} BAL "
                          f"iterations, i.e. max_runs >= {ceiling + ndim}.")
    elif ceiling < floor:
        verdict = "undersized"
        message = (
            f"init_runs = {ceiling} is below the {floor} runs that {ndim} calibration "
            f"parameters need ({samples_per_dimension} per parameter). An emulator "
            f"trained on that few points is unreliable away from its training points, "
            f"and Bayesian active learning will refine whatever maximum it inherits, "
            f"including a spurious one.")
        recommendation = (
            f"Raise sampling['init_runs'] to {recommended}. If the simulation budget "
            f"does not allow it, reduce the number of calibration parameters instead of "
            f"the number of initial runs.")
    elif ceiling >= 4 * recommended:
        verdict = "generous"
        message = (f"init_runs = {ceiling} is well above the recommended {recommended} "
                   f"for {ndim} parameters. The extra runs are not wasted, but the same "
                   f"budget spent on BAL iterations would target the posterior instead "
                   f"of covering the whole prior.")
        recommendation = ("Consider sampling['adaptive_init_runs'] = True, which stops "
                          "the initial design as soon as it is sufficient and spends "
                          "the remainder on BAL.")
    else:
        verdict = "adequate"
        message = (f"init_runs = {ceiling} covers the {floor} runs that {ndim} "
                   f"calibration parameters need.")
        recommendation = ""

    if verdict not in ("no_bal_budget",) and ceiling != _next_power_of_two(ceiling):
        recommendation = (recommendation + " " if recommendation else "") + (
            f"A Sobol design is balanced at a power of two; {_next_power_of_two(ceiling)} "
            f"is the nearest one above init_runs = {ceiling}.")

    return {
        "recommended": recommended,
        "floor": floor,
        "ladder": ladder,
        "configured": int(init_runs) if init_runs else None,
        "bal_budget": bal_budget,
        "samples_per_dimension": int(samples_per_dimension),
        "verdict": verdict,
        "message": message,
        "recommendation": recommendation,
    }


def initial_design_ladder(ceiling, ndim, first_block=None):
    """Cumulative design sizes of the staged Sobol ladder, up to ``ceiling``.

    The first block is large enough for a Gaussian process to be fitted at all
    (``4 * ndim``, floored at 16, rounded to a power of two); every further block doubles
    the design, which is exactly the refinement at which a Sobol sequence stays balanced.
    The ceiling is always the last entry, so the ladder never runs more simulations than
    ``init_runs`` authorises.

    Parameters
    ----------
    ceiling : int
        Maximum total number of initial runs, i.e. the configured ``init_runs``.
    ndim : int
        Number of calibration parameters.
    first_block : int, optional
        Size of the first block. Default ``2 ** ceil(log2(max(4 * ndim, 16)))``.

    Returns
    -------
    list of int
        Strictly increasing cumulative design sizes, ending at ``ceiling``.
    """
    ceiling = int(ceiling)
    if ceiling < 1:
        raise ValueError(f"The initial design needs at least one run, got {ceiling}.")

    start = int(first_block) if first_block else _next_power_of_two(
        max(4 * int(ndim), MINIMUM_INIT_RUNS))
    start = min(start, ceiling)

    ladder = []
    size = start
    while size < ceiling:
        ladder.append(size)
        size *= 2
    ladder.append(ceiling)
    return ladder


# ---------------------------------------------------------------------------
# sampling method and Sobol blocks
# ---------------------------------------------------------------------------
def validate_sampling_method(name):
    """Canonical chaospy rule name for a configured sampling method.

    Called before the first simulation is launched, because an invalid rule currently
    surfaces as a ``RuntimeError`` from inside ``bayesvalidrox`` only once the model
    object has been built.

    Parameters
    ----------
    name : str
        Value of ``sampling['parameter_sampling_method']``.

    Returns
    -------
    str
        The canonical name, with the historical ``chebyshev(FT)`` and ``grid(FT)``
        spellings mapped to the rules chaospy actually knows.

    Raises
    ------
    ValueError
        If the method is not one chaospy can generate.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"parameter_sampling_method must be a string, got {type(name).__name__}. "
            f"Valid options: {', '.join(SAMPLING_METHODS)}.")

    key = name.strip().lower()
    canonical = _SAMPLING_METHOD_ALIASES.get(key, key)
    if canonical != key:
        logger_warn.warning(
            f"parameter_sampling_method '{name}' is not a chaospy rule; using "
            f"'{canonical}' instead. Update the configuration to '{canonical}'.")
    if canonical not in SAMPLING_METHODS:
        raise ValueError(
            f"Unknown parameter_sampling_method '{name}'. Valid options: "
            f"{', '.join(SAMPLING_METHODS)}.")
    return canonical


def sobol_block(exp_design, n_from, n_to, existing=None, sampling_method="sobol",
                tolerance=1e-8):
    """Rows ``n_from`` to ``n_to`` of the design, continuing the existing sequence.

    A Sobol sequence of order ``n_to`` starts with the order-``n_from`` sequence, so a
    block generated this way extends the runs already carried out instead of replacing
    them: no simulation is ever wasted by growing the design. The same holds for the
    other low-discrepancy rules (``halton``, ``hammersley``); it does not hold for
    ``random`` or ``latin_hypercube``, where a fresh block is simply drawn.

    The prefix property is verified rather than assumed. If the regenerated prefix does
    not reproduce the points that were actually simulated, the function warns and falls
    back to a Latin hypercube block, because a silently reordered design would attach
    every stored model output to the wrong parameter set.

    Parameters
    ----------
    exp_design : object
        ``bayesvalidrox`` ``ExpDesigns`` instance, used for its ``generate_samples``.
    n_from, n_to : int
        Number of runs already carried out, and the target total.
    existing : array, optional
        The ``[n_from, ndim]`` design already simulated, for the prefix check.
    sampling_method : str
        Rule to continue. Default ``'sobol'``.
    tolerance : float
        Absolute tolerance of the prefix check.

    Returns
    -------
    array
        The ``[n_to - n_from, ndim]`` block of new parameter sets.
    """
    n_from, n_to = int(n_from), int(n_to)
    if n_to <= n_from:
        raise ValueError(f"An initial-design block needs n_to > n_from, got "
                         f"n_from={n_from}, n_to={n_to}.")

    method = validate_sampling_method(sampling_method)
    extensible = method in ("sobol", "halton", "hammersley", "korobov",
                            "additive_recursion")

    if n_from == 0:
        return np.atleast_2d(exp_design.generate_samples(n_to, method))

    if extensible:
        full = np.atleast_2d(exp_design.generate_samples(n_to, method))
        prefix_ok = full.shape[0] == n_to
        if prefix_ok and existing is not None:
            existing = np.atleast_2d(np.asarray(existing, dtype=float))
            prefix_ok = (existing.shape[0] <= full.shape[0]
                         and np.allclose(full[:existing.shape[0]], existing,
                                         atol=tolerance, rtol=0.0))
        if prefix_ok:
            return full[n_from:]
        logger_warn.warning(
            f"The '{method}' sequence did not reproduce the {n_from} parameter sets that "
            f"were already simulated, so the design cannot be extended along it. Falling "
            f"back to a Latin hypercube block of {n_to - n_from} points; the design is "
            f"still valid, only less uniform than a pure {method} design.")

    return np.atleast_2d(exp_design.generate_samples(n_to - n_from, "latin_hypercube"))


# ---------------------------------------------------------------------------
# gate emulator
# ---------------------------------------------------------------------------
def _scale_inputs(collocation_points, parameter_ranges):
    """Map the parameter sets onto the unit hypercube spanned by the prior."""
    points = np.atleast_2d(np.asarray(collocation_points, dtype=float))
    if parameter_ranges is None:
        lower = points.min(axis=0)
        upper = points.max(axis=0)
    else:
        bounds = np.asarray(parameter_ranges, dtype=float)
        lower, upper = bounds[:, 0], bounds[:, 1]
    span = np.where(np.abs(upper - lower) > 0, upper - lower, 1.0)
    return (points - lower) / span, lower, span


def _select_columns(model_outputs, max_columns, seed=0):
    """Output columns the gate is evaluated on, most informative first.

    The gate must not become more expensive than the simulations it is protecting, and
    its cost is one Gaussian process per output column. Where a calibration has hundreds
    of columns, the ones that vary most across the design carry most of the information
    about the parameters, so they are kept first and the remainder is filled with a
    seeded random draw for coverage.
    """
    outputs = np.atleast_2d(np.asarray(model_outputs, dtype=float))
    n_columns = outputs.shape[1]
    if max_columns is None or n_columns <= max_columns:
        return np.arange(n_columns)

    variance = np.nanvar(outputs, axis=0)
    ranked = np.argsort(variance)[::-1]
    n_top = max_columns // 2
    selected = list(ranked[:n_top])
    remaining = [index for index in ranked[n_top:] if np.isfinite(variance[index])]
    rng = np.random.default_rng(seed)
    if remaining:
        extra = rng.choice(np.asarray(remaining), size=min(max_columns - n_top,
                                                           len(remaining)),
                           replace=False)
        selected.extend(np.atleast_1d(extra).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def _fit_gate_gps(scaled_points, outputs):
    """One anisotropic Matern 5/2 Gaussian process per output column.

    Deliberately independent of ``sampling['gp_library']``: the gate measures whether the
    *design* carries enough information, which must not depend on which emulator
    implementation the calibration happens to use. Hyper-parameters are fitted once,
    without restarts, because the gate needs a fair predictivity estimate and not the
    best achievable emulator.

    Inputs live on the unit hypercube and outputs are standardised, so the kernel bounds
    can be tight. That matters for more than tidiness: with the bounds wide open the
    optimiser wanders into length scales far larger than the box, where the likelihood is
    flat, and takes an order of magnitude longer to fit each column.

    The BLAS thread pool is pinned to one thread for the duration. The matrices here are
    tens of rows across, so spawning one thread per core costs far more than the
    arithmetic it parallelises: on a 32-core machine that is the difference between 30
    milliseconds and 1.5 seconds per column.
    """
    ndim = scaled_points.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.full(ndim, 0.5),
                       length_scale_bounds=(5e-2, 1e1), nu=2.5)
              + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e0)))

    fitted = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        # A hyper-parameter resting on a bound is normal for a deterministic solver
        # (the noise term goes to zero) and says nothing the gate does not measure.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for column in range(outputs.shape[1]):
            values = outputs[:, column]
            centre, spread = float(np.mean(values)), float(np.std(values))
            standardised = (values - centre) / (spread if spread > 0 else 1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=False,
                                          n_restarts_optimizer=0)
            gp.fit(scaled_points, standardised)
            fitted.append({"gp": gp, "centre": centre,
                           "spread": spread if spread > 0 else 1.0,
                           "standardised": standardised})
    return fitted


def _predict_gate(fitted, scaled_points):
    """Mean and standard deviation of the gate emulators, in physical units."""
    means, deviations = [], []
    with threadpool_limits(limits=1):
        for entry in fitted:
            mean, deviation = entry["gp"].predict(scaled_points, return_std=True)
            means.append(mean * entry["spread"] + entry["centre"])
            deviations.append(deviation * entry["spread"])
    return np.column_stack(means), np.column_stack(deviations)


def loo_predictivity(collocation_points, model_outputs, parameter_ranges=None,
                     max_columns=60, seed=0, fitted=None, columns=None):
    """Leave-one-out predictivity and error-bar calibration of the initial design.

    Both quantities come from the closed-form leave-one-out identities for a Gaussian
    process (Rasmussen and Williams 2006, eq. 5.12), which need one inversion of the
    already-fitted covariance matrix rather than one refit per training point. Refitting
    would cost ``n`` fits per output column and buy nothing here, since the
    hyper-parameters barely move when one of several dozen points is removed.

    ``Q2 = 1 - SS_loo / SS_total`` is the fraction of the output variance the emulator
    predicts at points it has not seen. It is the honest counterpart of the training fit,
    which a Gaussian process interpolator reproduces perfectly by construction and which
    therefore says nothing at all.

    Returns
    -------
    dict
        ``q2`` (per column), ``q2_median``, ``q2_min``, ``coverage``, ``columns`` and
        ``n_runs``.
    """
    outputs = np.atleast_2d(np.asarray(model_outputs, dtype=float))
    if columns is None:
        columns = _select_columns(outputs, max_columns, seed=seed)
    outputs = outputs[:, columns]
    scaled, _, _ = _scale_inputs(collocation_points, parameter_ranges)
    if fitted is None:
        fitted = _fit_gate_gps(scaled, outputs)

    q2_values, inside = [], []
    for entry in fitted:
        standardised = entry["standardised"]
        covariance = entry["gp"].kernel_(scaled)
        covariance[np.diag_indices_from(covariance)] += entry["gp"].alpha
        try:
            inverse = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(covariance)
        diagonal = np.diag(inverse)
        # Guard a numerically singular covariance: a zero diagonal would divide by zero
        # and report a perfect emulator where there is none.
        diagonal = np.where(np.abs(diagonal) > 1e-12, diagonal, np.nan)
        weights = inverse @ standardised
        loo_mean = standardised - weights / diagonal
        loo_variance = 1.0 / diagonal

        total = float(np.sum((standardised - standardised.mean()) ** 2))
        residual = float(np.nansum((standardised - loo_mean) ** 2))
        q2_values.append(1.0 - residual / total if total > 0 else np.nan)
        with np.errstate(invalid="ignore"):
            standard_residual = (standardised - loo_mean) / np.sqrt(loo_variance)
        inside.append(np.nanmean(np.abs(standard_residual) <= 1.96))

    q2_values = np.asarray(q2_values, dtype=float)
    return {
        "q2": q2_values,
        "q2_median": float(np.nanmedian(q2_values)) if q2_values.size else np.nan,
        "q2_min": float(np.nanmin(q2_values)) if q2_values.size else np.nan,
        "coverage": float(np.nanmean(inside)) if inside else np.nan,
        "columns": np.asarray(columns, dtype=int),
        "n_runs": int(np.atleast_2d(collocation_points).shape[0]),
    }


# ---------------------------------------------------------------------------
# the sufficiency gate
# ---------------------------------------------------------------------------
def initial_design_sufficiency(
        collocation_points,
        model_outputs,
        observations,
        variances,
        parameter_ranges=None,
        prior=None,
        previous=None,
        thresholds=None,
        max_columns=60,
        prior_samples=5000,
        seed=0,
):
    """Is the initial design good enough to start Bayesian active learning on?

    Five properties are measured, each of which breaks BAL in a different way when it
    fails:

    1. **Predictivity** (``q2_median``, ``q2_min``). An emulator that cannot predict a
       point it has not seen gives BAL a likelihood surface with maxima the solver does
       not have.
    2. **Error-bar calibration** (``coverage``). BAL's utility is an expectation over the
       emulator's predictive distribution, so an overconfident emulator makes it explore
       the wrong places for the right reason.
    3. **Posterior resolution** (``post_size``). Everything downstream, the maximum
       included, is estimated from the accepted rejection sample. A handful of accepted
       samples is noise, and its maximum is a random draw.
    4. **Data-driven posterior shape** (``error_ratio``). If the emulator standard
       deviation dominates the observation error, the posterior is a picture of what the
       emulator does not know rather than of what the measurements say.
    5. **Stability** (``mean_shift``, ``delta_log_bme``). The one criterion that needs
       two blocks: if the posterior and the evidence stop moving when runs are added, the
       design has converged; if they still jump, it has not, whatever the other four say.
       Measured on the posterior mean rather than on its maximum, see :func:`_stability`.

    The function never raises. It runs between blocks of full-complexity simulations, and
    a diagnostic failure must not abort a calibration that has already cost days.

    Parameters
    ----------
    collocation_points : array
        The design carried out so far, ``[n_runs, ndim]``.
    model_outputs : array
        Its outputs, ``[n_runs, nloc * n_quantities]``.
    observations : array
        Measured values, ``[1, n_obs]``, as held by the model instance.
    variances : array
        Total observation variances, ``[n_obs]``, as held by the model instance.
    parameter_ranges : array, optional
        ``[[min, max], ...]`` per calibration parameter.
    prior : array or callable, optional
        Prior sample to run the inference on, or a callable ``n -> [n, ndim]`` such as
        ``ExpDesigns.generate_samples``. A callable lets the gate enlarge the sample
        until the posterior is resolved; a fixed array is used as given. Drawn uniformly
        from ``parameter_ranges`` when omitted.
    previous : dict, optional
        The report of the preceding block, for the stability criterion.
    thresholds : dict, optional
        Overrides for :data:`DEFAULT_THRESHOLDS`.
    max_columns : int
        Cap on the number of output columns the gate emulator is fitted to.
    prior_samples : int
        Size of the prior sample when ``prior`` is not given.
    seed : int
        Seed of the column selection and of the prior draw, so the gate is reproducible.

    Returns
    -------
    dict
        ``verdict`` (``sufficient`` | ``marginal`` | ``insufficient`` | ``unavailable``),
        ``criteria`` (per-criterion value, threshold and pass flag), the measured
        quantities themselves, ``joint_map``, ``log_bme``, ``message`` and
        ``recommendation``.
    """
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)

    points = np.atleast_2d(np.asarray(collocation_points, dtype=float))
    n_runs = points.shape[0]
    ndim = points.shape[1]

    try:
        outputs = np.atleast_2d(np.asarray(model_outputs, dtype=float))
        if outputs.shape[0] != n_runs:
            raise ValueError(
                f"The initial design has {n_runs} runs but {outputs.shape[0]} rows of "
                f"model outputs.")

        columns = _select_columns(outputs, max_columns, seed=seed)
        scaled, lower, span = _scale_inputs(points, parameter_ranges)
        fitted = _fit_gate_gps(scaled, outputs[:, columns])

        predictivity = loo_predictivity(points, outputs, parameter_ranges,
                                        fitted=fitted, columns=columns)

        observed = np.atleast_2d(np.asarray(observations, dtype=float))[:, columns]
        observation_variance = np.asarray(variances, dtype=float).ravel()[columns]
        post_floor = max(limits["post_size_absolute"],
                         limits["post_size_per_dimension"] * ndim)

        inference, deviation, prior_used = _resolve_posterior(
            fitted, prior, observed, observation_variance, post_floor,
            lower, span, ndim, prior_samples, seed)
        posterior = inference["posterior"]
        post_size = posterior.shape[0]

        error_ratio = np.nan
        joint_map = None
        if post_size:
            accepted = np.take(deviation, inference["indices"], axis=0)
            error_ratio = float(np.median(accepted)
                                / np.median(np.sqrt(observation_variance)))
        if post_size >= max(4, ndim + 1):
            joint_map = _joint_map(posterior)

        stability = _stability(joint_map, inference["log_bme"], posterior, previous)

        criteria = {
            "predictivity": _criterion(
                predictivity["q2_median"], limits["q2_median"], "at least",
                "median leave-one-out predictivity Q2 of the gate emulator"),
            "worst_column": _criterion(
                predictivity["q2_min"], limits["q2_min"], "at least",
                "predictivity Q2 of the worst-predicted output column"),
            "error_bars": _criterion(
                predictivity["coverage"], limits["coverage_low"], "at least",
                "fraction of leave-one-out residuals inside the 95 % interval"),
            "posterior_resolution": _criterion(
                post_size, post_floor, "at least", "accepted posterior samples"),
            "data_driven": _criterion(
                error_ratio, limits["error_ratio"], "at most",
                "median emulator standard deviation over observation standard "
                "deviation, at the accepted samples"),
            "stability": _criterion(
                stability["mean_shift"], limits["posterior_shift"], "at most",
                "movement of the posterior since the previous block, in posterior "
                "standard deviations"),
            "evidence_stability": _criterion(
                stability["delta_log_bme"], limits["delta_log_bme"], "at most",
                "change of the log evidence since the previous block"),
        }

        core = ("predictivity", "worst_column", "error_bars", "posterior_resolution")
        core_passed = all(criteria[name]["passed"] for name in core)
        refinement = ("data_driven", "stability", "evidence_stability")
        # An unmeasurable criterion (the first block has nothing to compare against)
        # must not be read as a pass: it is simply not yet decidable.
        refinement_passed = all(criteria[name]["passed"] is True for name in refinement)

        if core_passed and refinement_passed:
            verdict = "sufficient"
        elif core_passed:
            verdict = "marginal"
        else:
            verdict = "insufficient"

        failed = [name for name, entry in criteria.items() if entry["passed"] is False]
        undecided = [name for name, entry in criteria.items() if entry["passed"] is None]
        message = _sufficiency_message(verdict, n_runs, criteria, failed, undecided)
        recommendation = _sufficiency_recommendation(verdict, n_runs, failed)

        return {
            "n_runs": n_runs,
            "ndim": ndim,
            "verdict": verdict,
            "criteria": criteria,
            "failed": failed,
            "undecided": undecided,
            "q2_median": predictivity["q2_median"],
            "q2_min": predictivity["q2_min"],
            "coverage": predictivity["coverage"],
            "columns_used": predictivity["columns"],
            "post_size": post_size,
            "prior_size": int(prior_used),
            "error_ratio": error_ratio,
            "joint_map": joint_map,
            "posterior_std": (posterior.std(axis=0) if post_size else None),
            "log_bme": inference["log_bme"],
            "posterior_mean": (posterior.mean(axis=0) if post_size else None),
            "mean_shift": stability["mean_shift"],
            "map_shift": stability["map_shift"],
            "delta_log_bme": stability["delta_log_bme"],
            "next_block": 2 * n_runs,
            "message": message,
            "recommendation": recommendation,
        }

    except Exception as exception:  # never abort a running calibration
        logger_warn.warning(f"Initial-design sufficiency check skipped: {exception}")
        return {
            "n_runs": n_runs,
            "ndim": ndim,
            "verdict": "unavailable",
            "criteria": {},
            "failed": [],
            "undecided": [],
            "q2_median": np.nan, "q2_min": np.nan, "coverage": np.nan,
            "columns_used": np.asarray([], dtype=int),
            "post_size": 0, "prior_size": 0, "error_ratio": np.nan,
            "joint_map": None, "posterior_std": None, "posterior_mean": None,
            "log_bme": np.nan,
            "mean_shift": np.nan, "map_shift": np.nan, "delta_log_bme": np.nan,
            "next_block": 2 * n_runs,
            "message": (f"The sufficiency of the {n_runs}-run initial design could not "
                        f"be measured: {exception}"),
            "recommendation": ("Treat the design as unvetted and judge it from the "
                               "surrogate validation plots after the calibration."),
        }


def _resolve_posterior(fitted, prior, observations, variances, post_floor,
                       lower, span, ndim, prior_samples, seed):
    """Posterior of the gate emulator, on a prior sample large enough to resolve it.

    Rejection sampling accepts a fraction of the prior that depends on how sharp the
    posterior is relative to the prior volume, not on how good the design is. Measuring
    the accepted-sample count on a fixed prior sample would therefore turn the resolution
    criterion into a criterion on ``prior_samples``, and a calibration with a sharp
    posterior would be told to add solver runs when what it needs is a denser prior.

    So the prior sample is grown instead: the observed acceptance rate says how many
    draws the floor needs, and the gate redraws once at that size, capped at
    :data:`PRIOR_SAMPLE_CAP`. Failure after that is a genuine finding, reported as such,
    and it applies to the calibration itself, whose ``prior_samples`` faces exactly the
    same acceptance rate.

    ``prior`` may be an array (used as given, and never grown, because a caller who
    supplies the calibration's own prior wants the gate measured on it) or a callable
    ``n -> [n, ndim]``, which is what ``ExpDesigns.generate_samples`` is.
    """
    def draw(size):
        if callable(prior):
            return np.atleast_2d(np.asarray(prior(int(size)), dtype=float))
        rng = np.random.default_rng(seed)
        return lower + span * rng.random((int(size), ndim))

    fixed = prior is not None and not callable(prior)
    sample = (np.atleast_2d(np.asarray(prior, dtype=float)) if fixed
              else draw(prior_samples))

    def run(sample):
        prediction, deviation = _predict_gate(fitted, (sample - lower) / span)
        inference = _run_inference(prediction, observations, variances, deviation, sample)
        return inference, deviation

    inference, deviation = run(sample)
    post_size = inference["posterior"].shape[0]

    if fixed or post_size >= post_floor or post_size == 0:
        return inference, deviation, sample.shape[0]

    acceptance = post_size / sample.shape[0]
    needed = int(min(PRIOR_SAMPLE_CAP, math.ceil(1.5 * post_floor / acceptance)))
    if needed <= sample.shape[0]:
        return inference, deviation, sample.shape[0]

    logger.info(
        f"Initial design: {post_size} of {sample.shape[0]} prior samples were accepted, "
        f"below the {post_floor} needed to resolve the posterior. Redrawing "
        f"{needed} prior samples, so the check measures the design and not the size of "
        f"the prior sample.")
    sample = draw(needed)
    inference, deviation = run(sample)
    return inference, deviation, sample.shape[0]


def _run_inference(prediction, observations, variances, deviation, prior):
    """Rejection-sampled posterior of the gate emulator.

    Reuses :class:`~hydroBayesCal.surrogate.bal_functions.BayesianInference` so the gate
    measures the very likelihood the calibration itself will use, including the emulator
    standard deviation as ``model_error``. Imported lazily to keep this module importable
    without the surrogate stack.
    """
    from hydroBayesCal.surrogate.bal_functions import BayesianInference

    inference = BayesianInference(
        model_predictions=prediction,
        observations=observations,
        error=variances,
        model_error=deviation,
        sampling_method="rejection_sampling",
        prior=prior,
    )
    inference.estimate_bme()
    posterior = (np.atleast_2d(inference.posterior)
                 if inference.posterior is not None and len(inference.posterior)
                 else np.empty((0, prior.shape[1])))
    indices = (np.asarray(inference.post_index, dtype=int)
               if inference.post_index is not None else np.asarray([], dtype=int))
    return {"posterior": posterior, "indices": indices,
            "log_bme": float(inference.log_BME) if inference.log_BME is not None
            else np.nan}


def _joint_map(posterior):
    """Maximum of the joint posterior density of an accepted sample."""
    from hydroBayesCal.surrogate.posterior_analysis import joint_optimum

    return np.asarray(joint_optimum(posterior, method="knn")["vector"], dtype=float)


def _stability(joint_map, log_bme, posterior, previous):
    """How far the posterior moved since the previous block.

    Measured on the posterior **mean**, not on its maximum, even though the maximum is
    the quantity the calibration ultimately reports. The maximum of a finite accepted
    sample is the noisiest summary of that sample: with a few hundred accepted draws it
    wanders by a third to a half of a posterior standard deviation between two rejection
    samplings of the *same* emulator, which is the size of the movement the criterion is
    supposed to detect. Testing convergence on it would test the random number generator.
    The mean moves only when the posterior itself moves, so it answers the intended
    question, and the maximum is still reported alongside it.
    """
    if previous is None or previous.get("posterior_mean") is None or posterior.shape[0] == 0:
        return {"mean_shift": None, "map_shift": None, "delta_log_bme": None}

    spread = posterior.std(axis=0)
    previous_spread = previous.get("posterior_std")
    if previous_spread is not None:
        spread = np.maximum(spread, np.asarray(previous_spread, dtype=float))
    spread = np.where(spread > 0, spread, np.nan)

    shift = (np.abs(posterior.mean(axis=0)
                    - np.asarray(previous["posterior_mean"], dtype=float)) / spread)
    if joint_map is None or previous.get("joint_map") is None:
        map_shift = None
    else:
        map_shift = float(np.nanmax(
            np.abs(joint_map - np.asarray(previous["joint_map"], dtype=float)) / spread))

    previous_log_bme = previous.get("log_bme", np.nan)
    delta = (abs(float(log_bme) - float(previous_log_bme))
             if np.isfinite(log_bme) and np.isfinite(previous_log_bme) else None)
    return {
        "mean_shift": float(np.nanmax(shift)) if np.any(np.isfinite(shift)) else None,
        "map_shift": map_shift,
        "delta_log_bme": delta,
    }


def _criterion(value, threshold, direction, description):
    """One entry of the sufficiency report.

    ``passed`` is ``None`` where the criterion cannot be decided yet, which is different
    from failing it: the stability criteria need a preceding block to compare against.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        passed = None
    elif direction == "at least":
        passed = bool(value >= threshold)
    else:
        passed = bool(value <= threshold)
    return {"value": value, "threshold": threshold, "direction": direction,
            "passed": passed, "description": description}


_CRITERION_CONSEQUENCE = {
    "predictivity": ("the emulator cannot predict parameter sets it has not been trained "
                     "on, so the likelihood surface Bayesian active learning refines is "
                     "not the solver's"),
    "worst_column": ("at least one calibration point is predicted badly, and that point "
                     "enters the joint likelihood with the same weight as every other"),
    "error_bars": ("the emulator is overconfident, and the active-learning utility is an "
                   "expectation over exactly those error bars"),
    "posterior_resolution": ("too few prior samples are accepted for the posterior, and "
                             "its maximum, to be more than noise"),
    "data_driven": ("the posterior shape is dominated by what the emulator does not know "
                    "rather than by what the measurements say"),
    "stability": ("the posterior still moves when runs are added, so neither it nor its "
                  "maximum has converged"),
    "evidence_stability": ("the model evidence still moves when runs are added, so the "
                           "design has not converged"),
}


def _sufficiency_message(verdict, n_runs, criteria, failed, undecided):
    """Prose statement of what the measurement found."""
    numbers = ", ".join(
        f"{name} = {entry['value']:.3g}" if isinstance(entry["value"], (int, float))
        and entry["value"] is not None and np.isfinite(entry["value"])
        else f"{name} = n/a"
        for name, entry in criteria.items())

    if verdict == "sufficient":
        return (f"The {n_runs}-run initial design is sufficient: every criterion is met "
                f"({numbers}). Bayesian active learning starts from an emulator that is "
                f"predictive over the whole prior, so the maximum it refines is the "
                f"solver's and not the emulator's.")
    if verdict == "marginal":
        pending = ", ".join(undecided) if undecided else "none"
        return (f"The {n_runs}-run initial design is usable but not settled: the emulator "
                f"and the posterior resolution are adequate, while {', '.join(failed) or 'no'} "
                f"criteria fail and {pending} cannot be decided yet ({numbers}).")
    reasons = "; ".join(f"{name}, i.e. {_CRITERION_CONSEQUENCE.get(name, 'see the report')}"
                        for name in failed)
    return (f"The {n_runs}-run initial design is insufficient: {reasons} ({numbers}).")


def _sufficiency_recommendation(verdict, n_runs, failed):
    """What to do about it."""
    if verdict == "sufficient":
        return ("Start Bayesian active learning. Any further initial runs would cover the "
                "prior where the posterior is not.")
    if "posterior_resolution" in failed:
        return ("Raise sampling['prior_samples']: the posterior was not resolved even "
                "after the check enlarged its own prior sample, so the calibration, which "
                "draws the same way, will not resolve it either. If the accepted sample "
                f"stays small, extend the design to {2 * n_runs} runs, since a sharper "
                f"emulator narrows the likelihood far less than a wrong one does.")
    if "data_driven" in failed:
        return (f"Extend the design to {2 * n_runs} runs. If the ratio does not fall, the "
                f"measurements themselves are too uncertain to constrain these parameters, "
                f"and more runs will not change that.")
    return (f"Extend the design to {2 * n_runs} runs. With adaptive_init_runs the ladder "
            f"does this on its own up to sampling['init_runs'].")


def log_initial_design(report, logger_obj=None):
    """Log a sizing or sufficiency report, warning where the design is not adequate.

    Returns ``report`` so calls can be chained.
    """
    info = logger_obj or logger
    verdict = report.get("verdict", "unavailable")
    message = report.get("message", "")
    recommendation = report.get("recommendation", "")

    if verdict in ("undersized", "no_bal_budget", "insufficient"):
        logger_warn.warning(f"Initial design [{verdict}]: {message}")
        if recommendation:
            logger_warn.warning(f"Initial design: {recommendation}")
    else:
        info.info(f"Initial design [{verdict}]: {message}")
        if recommendation:
            info.info(f"Initial design: {recommendation}")
    return report


# ---------------------------------------------------------------------------
# the staged ladder
# ---------------------------------------------------------------------------
def run_staged_initial_design(complex_model, experiment_design, adaptive=True,
                              init_runs_min=None, **run_kwargs):
    """Run the initial design, growing it in blocks until it is sufficient.

    The initial design decides whether Bayesian active learning can find the global
    maximum of the posterior at all. BAL refines the emulator it inherits and looks for
    training points where *that* emulator says information is to be gained; an emulator
    that has never seen a region cannot report that anything is missing there. An
    undersized initial design therefore does not announce itself, it just produces a
    confident calibration around the wrong maximum.

    How many runs that takes cannot be known in advance, only bounded, so the design is
    run in blocks and measured in between (:func:`initial_design_sufficiency`). Each
    block continues the same Sobol sequence instead of replacing it
    (:func:`sobol_block`), so nothing already simulated is ever wasted, and the ladder
    stops at the configured ``init_runs`` at the latest. Runs saved by stopping early are
    not lost either: ``max_runs`` is the total budget, so they go to BAL iterations,
    which target the posterior rather than covering the whole prior.

    Solver-agnostic. It uses only the parts of the
    :class:`~hydroBayesCal.hysim.HydroSimulations` contract every binding implements, so
    the TELEMAC, OpenFOAM, Delft3D and multiflow drivers share one implementation of the
    ladder instead of three copies of a loop whose bookkeeping has to stay exact.

    Parameters
    ----------
    complex_model : obj
        Model instance. Its ``init_runs`` is set to the size of the design actually run.
    experiment_design : obj
        ``bayesvalidrox`` ``ExpDesigns``; its ``n_init_samples`` and ``x`` are updated to
        the design actually run, so the BAL budget follows from what happened rather than
        from what was configured.
    adaptive : bool
        ``True`` (default) to run the ladder. ``False`` runs all ``init_runs`` in one
        block, i.e. the behaviour of earlier versions.
    init_runs_min : int, optional
        Size of the first block. Default ``2 ** ceil(log2(4 * ndim))``.
    **run_kwargs
        Passed through to ``run_multiple_simulations`` (e.g. TELEMAC's
        ``output_extraction_time`` and ``n``).

    Returns
    -------
    tuple
        ``(collocation_points, model_outputs)`` of the design that was run.
    """
    ceiling = int(complex_model.init_runs)
    sampling_method = getattr(experiment_design, "sampling_method", "sobol")

    if not adaptive or ceiling <= 1:
        logger.info(f"Sampling {ceiling} collocation points for the selected calibration "
                    f"parameters with {sampling_method} sampling method.")
        collocation_points = experiment_design.x
        complex_model.run_multiple_simulations(
            collocation_points=collocation_points,
            complete_bal_mode=complex_model.complete_bal_mode,
            validation=complex_model.validation,
            **run_kwargs)
        return collocation_points, complex_model.model_evaluations

    ladder = initial_design_ladder(ceiling, complex_model.ndim, first_block=init_runs_min)
    logger.info(f"Staged initial design with {sampling_method} sampling: blocks "
                f"{ladder} (ceiling init_runs = {ceiling}). The ladder stops as soon as "
                f"the design is sufficient, and the runs it saves go to BAL.")

    collocation_points = None
    model_outputs = None
    report = None
    stages = []
    n_from = 0

    for n_to in ladder:
        block = sobol_block(experiment_design, n_from, n_to, existing=collocation_points,
                            sampling_method=sampling_method)
        collocation_points = (block if collocation_points is None
                              else np.vstack((collocation_points, block)))
        # The binding reads init_runs as the size of the design it is running, and
        # start_index tells it which rows of that design are new.
        complex_model.init_runs = n_to
        logger.info(f"Initial design block {n_from + 1}-{n_to} of at most {ceiling} runs.")
        complex_model.run_multiple_simulations(
            collocation_points=collocation_points,
            complete_bal_mode=complex_model.complete_bal_mode,
            validation=complex_model.validation,
            start_index=n_from,
            **run_kwargs)
        model_outputs = complex_model.model_evaluations
        n_from = n_to

        report = initial_design_sufficiency(
            collocation_points=collocation_points,
            model_outputs=model_outputs,
            observations=complex_model.observations,
            variances=complex_model.variances,
            parameter_ranges=complex_model.param_values,
            prior=experiment_design.generate_samples,
            previous=report)
        log_initial_design(report)
        stages.append({"n_runs": n_to, "verdict": report["verdict"],
                       "q2_median": report["q2_median"], "coverage": report["coverage"],
                       "post_size": report["post_size"],
                       "failed": list(report["failed"])})
        if report["verdict"] == "sufficient":
            break

    achieved = int(collocation_points.shape[0])
    if report is not None and report["verdict"] != "sufficient":
        logger_warn.warning(
            f"The initial design reached its ceiling of {ceiling} runs while still "
            f"'{report['verdict']}'. Bayesian active learning will start from an emulator "
            f"that is not yet trustworthy over the whole prior, so treat the posterior "
            f"maximum as provisional and check the surrogate validation plots.")
    complex_model.init_runs = achieved
    experiment_design.n_init_samples = achieved
    experiment_design.x = collocation_points
    write_initial_design_record(complex_model, achieved, ceiling, ladder, stages,
                                sampling_method)
    return collocation_points, model_outputs


def write_initial_design_record(complex_model, achieved, ceiling, ladder, stages,
                                sampling_method):
    """Record what the ladder actually ran, next to the restart files.

    A restart with ``only_bal_mode`` reads ``init_runs`` rows from
    ``initial-collocation-points.csv``, i.e. the *configured* value, not the achieved
    one. When the ladder stops early those differ, and the restart would ask for runs
    that were never carried out, so the achieved count is both written here and logged.
    """
    record = {
        "achieved_init_runs": int(achieved),
        "configured_init_runs": int(ceiling),
        "sampling_method": sampling_method,
        "ladder": [int(step) for step in ladder],
        "stages": stages,
    }
    try:
        with open(os.path.join(complex_model.restart_data_folder,
                               "initial-design.json"), "w") as record_file:
            json.dump(record, record_file, indent=2, default=str)
    except Exception as exception:
        logger_warn.warning(f"Could not write initial-design.json: {exception}")

    if achieved != ceiling:
        logger.info(
            f"The initial design stopped at {achieved} of the {ceiling} authorised runs, "
            f"so {ceiling - achieved} simulations go to Bayesian active learning instead. "
            f"To restart this calibration with only_bal_mode, set "
            f"sampling['init_runs'] = {achieved}.")
