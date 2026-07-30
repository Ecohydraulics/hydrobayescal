"""
Derivation of calibrated parameter sets from a Bayesian Active Learning posterior.

The BAL loop stores, for every iteration, a *joint* posterior sample in
``BAL_dictionary.pkl`` (``bayesian_dict['posterior'][it]``): the rows of the prior
sample accepted by rejection sampling against the joint likelihood over all
calibration points and calibration targets. This module turns that sample into
the results a modeller actually needs at the end of a calibration:

* the **per-parameter marginal optimum**, i.e. the peak of each parameter's own
  posterior marginal, together with credible intervals and identifiability flags;
* the **joint posterior optimum**, i.e. the single parameter vector of highest
  joint posterior density;
* an **equifinality diagnostic** that states whether the vector assembled from the
  independent marginal peaks is jointly plausible at all, or whether parameter
  correlation makes that combination a point of near-zero posterior density;
* the **distinct posterior modes**, i.e. genuinely different parameter combinations
  that explain the observations comparably well.

Nothing here selects Bayesian active-learning training points. Those are, and must
remain, information-gain choices made by
:class:`~hydroBayesCal.surrogate.bal_functions.SequentialDesign`: the parameter set
that most reduces uncertainty about the posterior is not the parameter set that best
fits the data, and the last training point of a calibration is therefore not a
calibration result.

The module is deliberately solver-agnostic and depends only on the numerical stack
(numpy/scipy/scikit-learn), so it works on any archived ``BAL_dictionary.pkl`` from
a TELEMAC, OpenFOAM or Delft3D calibration, with no surrogate and no model instance
required.

Following the convention of
:func:`~hydroBayesCal.function_pool.diagnose_roughness_identifiability`, every
analysis function is report-only: it returns a plain dictionary carrying a
``verdict``, a ``message`` and a ``recommendation``, mutates no state, and leaves
the logging decision to a separate ``log_*`` function.
"""
import csv
import os
import shutil

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from hydroBayesCal.utils.config_logging import logger, logger_warn

__all__ = [
    "select_posterior_iteration",
    "marginal_optima",
    "joint_optimum",
    "equifinality_diagnostic",
    "detect_posterior_modes",
    "assemble_candidates",
    "analyze_posterior",
    "log_posterior_analysis",
    "track_iteration",
    "record_iteration",
    "write_user_collocation_points",
    "write_candidate_report",
    "ITERATION_KEYS",
]

#: ``bayesian_dict`` keys written by :func:`record_iteration`. They are additive:
#: every consumer reads them with ``.get`` and can rebuild them from
#: ``bayesian_dict['posterior']`` for result files written before they existed.
ITERATION_KEYS = (
    "marginal_optima",
    "marginal_hdi",
    "variance_reduction",
    "identifiability_flags",
    "marginal_joint_gap",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_2d_array(posterior, name="posterior"):
    """Return ``posterior`` as a float 2D array, or raise a helpful error."""
    if posterior is None:
        raise ValueError(f"{name} is None (no accepted samples for this iteration).")
    array = np.asarray(posterior, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"{name} must be a 2D array [n_samples, n_parameters], got shape "
            f"{np.shape(posterior)}.")
    return array


def _parameter_names(parameter_names, ndim):
    """Fall back to positional names when the caller has none."""
    if parameter_names is None:
        return [f"parameter_{i + 1}" for i in range(ndim)]
    names = [str(name) for name in parameter_names]
    if len(names) != ndim:
        raise ValueError(
            f"Got {len(names)} parameter names for {ndim} parameters. The names must "
            f"be in the same order as the calibration parameters.")
    return names


def _resolve_prior_bounds(prior_bounds, prior, posterior):
    """Return prior bounds as an ``[ndim, 2]`` array, inferring them if needed.

    ``prior_bounds`` are the calibration ranges (``config.calibration['param_values']``
    / ``complex_model.param_values``). When they are missing they are inferred from
    the prior sample, which is exact enough for the shipped uniform prior, and as a
    last resort from the posterior sample itself.
    """
    ndim = posterior.shape[1]
    if prior_bounds is not None:
        bounds = np.asarray(prior_bounds, dtype=float)
        if bounds.shape != (ndim, 2):
            raise ValueError(
                f"prior_bounds must have shape [{ndim}, 2], got {bounds.shape}.")
        return bounds
    if prior is not None and np.size(prior):
        prior = _as_2d_array(prior, "prior")
        logger_warn.warning(
            "No prior bounds given; inferring them from the prior sample. Pass "
            "param_values to make the identifiability flags exact.")
        return np.column_stack([prior.min(axis=0), prior.max(axis=0)])
    logger_warn.warning(
        "No prior bounds and no prior sample given; inferring bounds from the "
        "posterior sample. Bound-pinning flags and variance reduction will be "
        "unreliable.")
    return np.column_stack([posterior.min(axis=0), posterior.max(axis=0)])


def _standardize(samples, reference=None):
    """Standardise columns to zero mean and unit variance.

    Parameters here span very different magnitudes (critical Shields parameters on
    ``[0.047, 0.070]`` against friction zones on ``[0.002, 0.6]``), so any distance,
    density or clustering computation has to run on standardised coordinates or the
    widest parameter dominates it.
    """
    reference = samples if reference is None else reference
    center = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (samples - center) / scale, center, scale


def _hdi(samples, credible_mass=0.95):
    """Shortest interval containing ``credible_mass`` of the samples.

    The shortest-interval (highest-density) definition is used rather than the
    symmetric quantile interval because posterior marginals here are routinely
    skewed or piled against a prior bound, where the quantile interval misleads.
    """
    ordered = np.sort(np.asarray(samples, dtype=float))
    n = ordered.size
    if n < 2:
        return float(ordered[0]), float(ordered[0])
    n_in = int(np.floor(credible_mass * n))
    n_in = max(1, min(n_in, n - 1))
    widths = ordered[n_in:] - ordered[:n - n_in]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + n_in])


def marginal_bin_count(column, samples_per_bin=25, min_bins=8):
    """Number of histogram bins for locating the peak of one posterior marginal.

    The rule is the Freedman-Diaconis width ``2 * IQR / n**(1/3)`` over the range the
    samples actually occupy, with a Sturges floor for small samples (this pair is
    NumPy's ``bins="auto"``), then capped so that a bin holds ``samples_per_bin``
    samples on average.

    The cap is what makes the peak stable. Freedman-Diaconis minimises the error of
    the *whole* density curve, but locating its argmax is a different problem: once
    bins get so narrow that the Poisson noise on a bin count exceeds the density
    variation between neighbouring bins, the modal bin starts wandering. Requiring
    roughly 25 samples per bin keeps that noise near 20 % of a bin count.

    The values were chosen by measuring the located peak against known modes for
    narrow, broad, skewed and bound-pinned marginals at 400, 2000 and 20000 accepted
    samples; this rule was best or tied-best at every sample size, with a typical
    error of 0.2-2 % of the parameter range. A fixed bin count, in contrast,
    quantises the reported optimum to one bin width regardless of how much
    information the posterior actually carries: at 10 bins that is a tenth of the
    range, and roughly three times worse than this rule for large samples.

    Parameters
    ----------
    column : array
        Posterior samples of one calibration parameter.
    samples_per_bin : int
        Target average occupancy. Lower values resolve the peak more finely but make
        the modal bin noisier.
    min_bins : int
        Absolute floor, so that a very small posterior sample still gives a histogram.

    Returns
    -------
    int
        Number of bins.
    """
    column = np.asarray(column, dtype=float)
    n = column.size
    sturges = int(np.ceil(np.log2(max(n, 2)) + 1))
    span = float(np.ptp(column))
    iqr = float(np.subtract(*np.percentile(column, [75, 25])))

    if iqr <= 0 or span <= 0:
        count = sturges
    else:
        width = 2.0 * iqr / n ** (1.0 / 3.0)
        count = max(sturges, int(np.ceil(span / width)))

    return int(np.clip(count, min_bins, max(min_bins, n // samples_per_bin)))


def _marginal_histogram(column, samples_per_bin=25, min_bins=8):
    """Histogram of one posterior marginal over the range its samples occupy.

    Binning over the occupied range rather than the full calibration range keeps the
    resolution proportional to the posterior width, so a sharply constrained
    parameter inside a wide prior range is not reduced to one or two bins.

    Returns ``(counts, edges)``, or ``(None, None)`` for a degenerate column.
    """
    column = np.asarray(column, dtype=float)
    if column.size < 5 or np.ptp(column) <= 0:
        return None, None
    n_bins = marginal_bin_count(column, samples_per_bin=samples_per_bin,
                                min_bins=min_bins)
    counts, edges = np.histogram(column, bins=n_bins,
                                 range=(float(column.min()), float(column.max())))
    return counts, edges


def _count_marginal_modes(counts, valley_fraction=0.3, min_mode_height=0.10):
    """Number of separate modes in a marginal histogram.

    A second peak counts as a separate mode only when the histogram between it and
    every already-accepted mode drops below ``valley_fraction`` of the lower of the
    two peak counts, i.e. only when there is a real valley between them. This is the
    same criterion used for the joint posterior modes in
    :func:`detect_posterior_modes`, applied in one dimension.

    Ranking peaks by prominence instead does not work here: two bins that happen to
    tie at the maximum are each assigned nearly the full height as their prominence,
    because neither is higher than the other, and a perfectly unimodal marginal is
    then reported as bimodal. Bin counts tie often enough for that to be the common
    case rather than a corner case.

    The counts are padded before the local-maximum search because a peak in the first
    or last bin has no neighbour on one side and would otherwise be invisible. That
    is not a corner case either: it is what a parameter piled against a prior bound
    looks like.

    Measured against unimodal, flat, skewed, bound-pinned and bimodal marginals at
    400, 2000 and 20000 accepted samples, this rule returned the correct mode count in
    every case at the default thresholds.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or counts.max() <= 0:
        return 1

    padded = np.concatenate([[-1.0], counts, [-1.0]])
    candidates = set((find_peaks(padded)[0] - 1).tolist())
    candidates.add(int(np.argmax(counts)))
    ordered = sorted(candidates, key=lambda index: counts[index], reverse=True)

    threshold = min_mode_height * counts.max()
    accepted = []
    for candidate in ordered:
        if counts[candidate] < threshold:
            continue
        separate = all(
            counts[min(other, candidate):max(other, candidate) + 1].min()
            < valley_fraction * min(counts[other], counts[candidate])
            for other in accepted)
        if separate:
            accepted.append(candidate)
    return max(1, len(accepted))


def _histogram_peak(column, counts, edges):
    """Peak of a marginal: the modal bin, refined to the mean of its samples.

    Taking the mean of the samples inside the modal bin instead of the bin centre
    gives sub-bin resolution, so the reported optimum is not quantised to the bin
    grid.
    """
    modal = int(np.argmax(counts))
    inside = column[(column >= edges[modal]) & (column <= edges[modal + 1])]
    if inside.size:
        return float(np.mean(inside))
    return float(0.5 * (edges[modal] + edges[modal + 1]))


class _DensityEstimator:
    """Joint posterior density, evaluable at arbitrary parameter vectors.

    The density is estimated from the distance to the k-th nearest neighbour among
    the accepted posterior samples, on standardised coordinates. No smoothing kernel
    and no bandwidth are involved: the estimate is a direct statement about how
    densely the posterior samples themselves crowd around a point.

    A histogram, which is what the per-parameter marginals use, is not an option
    here. This density lives in the full calibration-parameter space, where a
    regular grid of bins is empty almost everywhere: seven parameters at ten bins
    each would be ten million cells for a few thousand accepted samples. The
    nearest-neighbour distance adapts to wherever the samples actually are.

    Being evaluable away from the sample matters: the vector assembled from the
    per-parameter marginal peaks is in general not one of the posterior samples, and
    the whole point of the equifinality diagnostic is to score exactly that
    off-sample point.
    """

    def __init__(self, posterior, method="auto", k_neighbors=None):
        self.samples = _as_2d_array(posterior)
        self.n_samples, self.ndim = self.samples.shape
        self._z, self._center, self._scale = _standardize(self.samples)

        if method in ("auto", "knn"):
            self.method = "knn"
        else:
            raise ValueError(
                f"Unknown joint density method '{method}'. Use 'auto'/'knn' for the "
                f"sample-based estimate, or 'likelihood' in joint_optimum() to score "
                f"the posterior samples with a trained surrogate.")

        if k_neighbors is None:
            # ~sqrt(n) neighbours averages out sampling noise without smoothing over
            # the structure the diagnostic is meant to detect. The cap keeps the
            # neighbour query cheap enough to run inside the BAL loop: beyond a few
            # dozen neighbours the density rank barely moves, while the query cost
            # keeps growing with the accepted sample size.
            k_neighbors = int(np.clip(round(np.sqrt(max(self.n_samples, 1))), 5, 50))
        self.k_neighbors = int(np.clip(k_neighbors, 1, max(self.n_samples - 1, 1)))
        self._tree = cKDTree(self._z)

    def log_density(self, points):
        """Log density at ``points`` (``[n_points, ndim]``), up to an additive constant."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        z = (points - self._center) / self._scale

        # Nearest-neighbour density ~ 1 / d_k**ndim. Query one extra neighbour and
        # drop a zero distance, so evaluating at a sample point does not return the
        # point itself and an infinite density.
        k = min(self.k_neighbors + 1, self.n_samples)
        distances, _ = self._tree.query(z, k=k)
        distances = np.atleast_2d(distances)
        log_d = np.empty(distances.shape[0])
        for row_index, row in enumerate(distances):
            row = row[np.isfinite(row)]
            non_self = row[row > 0]
            d_k = non_self[-1] if non_self.size else np.finfo(float).tiny
            log_d[row_index] = -self.ndim * np.log(max(d_k, np.finfo(float).tiny))
        return log_d

    def percentile_of(self, point, sample_log_density=None):
        """Percentile rank (0-100) of the density at ``point`` among the samples."""
        if sample_log_density is None:
            sample_log_density = self.log_density(self.samples)
        value = float(self.log_density(np.atleast_2d(point))[0])
        finite = sample_log_density[np.isfinite(sample_log_density)]
        if finite.size == 0:
            return float("nan")
        return float(100.0 * np.mean(finite <= value))


# ---------------------------------------------------------------------------
# iteration selection
# ---------------------------------------------------------------------------
def select_posterior_iteration(bayesian_dict, iteration=-1):
    """Pick one iteration's posterior sample out of a BAL dictionary.

    Rejection sampling can accept nothing in an early iteration, so
    ``bayesian_dict['posterior']`` legitimately contains ``None`` entries. This
    mirrors the guard that ``templates/plot_posteriors.py`` applies before plotting.

    Parameters
    ----------
    bayesian_dict : dict
        Loaded ``BAL_dictionary.pkl``.
    iteration : int
        Index into ``bayesian_dict['posterior']``. ``-1`` (default) selects the last
        iteration that has a non-empty posterior. Any other index falls back to that
        same last valid iteration when the requested one is empty.

    Returns
    -------
    tuple
        ``(posterior, iteration_index)``.
    """
    posteriors = bayesian_dict.get("posterior")
    if posteriors is None:
        raise ValueError("BAL dictionary has no 'posterior' entry.")

    valid = [i for i, p in enumerate(posteriors)
             if p is not None and np.asarray(p).size > 0]
    if not valid:
        raise ValueError(
            "No valid posterior found in the BAL dictionary: rejection sampling "
            "accepted no samples in any iteration.")

    if iteration == -1 or iteration not in valid:
        if iteration != -1:
            logger_warn.warning(
                f"Iteration {iteration} has no accepted posterior samples; using the "
                f"last valid iteration {valid[-1]} instead.")
        iteration = valid[-1]
    return _as_2d_array(posteriors[iteration]), iteration


# ---------------------------------------------------------------------------
# per-parameter marginal optima
# ---------------------------------------------------------------------------
def marginal_optima(
        posterior,
        prior_bounds=None,
        parameter_names=None,
        prior=None,
        credible_mass=0.95,
        samples_per_bin=25,
        min_bins=8,
        bound_tol=0.05,
        var_reduction_tol=0.10,
        mode_valley_fraction=0.3,
        min_mode_height=0.10,
        min_post_size=200,
):
    """Per-parameter marginal optimum and identifiability from a joint posterior.

    Each parameter's optimum is the peak of that parameter's own posterior marginal,
    read directly off a histogram of the accepted posterior samples: the most
    populated bin, refined to the mean of the samples inside it. The bin count comes
    from :func:`marginal_bin_count`, so it follows from the sample size and the
    spread of the posterior instead of being fixed in advance.

    The estimate is deliberately made on the samples themselves, with no smoothing.
    A smoothed density estimate returns the peak of the smoothed curve, not of the
    posterior, and the discrepancy is largest exactly where it matters here: against
    a prior bound, where a symmetric kernel spreads mass across the bound and pulls
    the apparent peak inward, so a parameter pinned at its calibration limit no
    longer looks pinned.

    Reporting the marginal peaks alone is not sufficient: a peak is only meaningful
    if the data actually constrain that parameter, and the peaks of different
    parameters only form a usable parameter *set* if the posterior does not couple
    them. The flags below cover the first question and
    :func:`equifinality_diagnostic` covers the second.

    Parameters
    ----------
    posterior : array
        Joint posterior sample, shape ``[n_samples, n_parameters]``.
    prior_bounds : list of [min, max], optional
        Calibration ranges in parameter order (``param_values``). Inferred from
        ``prior`` or ``posterior`` when omitted.
    parameter_names : list of str, optional
        Calibration parameter names, in the same order.
    prior : array, optional
        Prior sample, used for the reference variance of the identifiability check.
        Without it a uniform prior over ``prior_bounds`` is assumed.
    credible_mass : float
        Probability mass of the reported highest-density interval.
    samples_per_bin : int
        Target average occupancy of a histogram bin, see :func:`marginal_bin_count`.
    min_bins : int
        Absolute floor on the number of histogram bins.
    bound_tol : float
        A peak within this fraction of the prior range of a bound counts as pinned.
    var_reduction_tol : float
        Below this posterior-to-prior variance reduction a parameter counts as
        non-identifiable.
    mode_valley_fraction : float
        How deep the histogram must dip between two peaks, as a fraction of the lower
        peak, for them to count as separate marginal modes. See
        :func:`_count_marginal_modes`.
    min_mode_height : float
        Minimum height of a secondary peak, as a fraction of the largest bin count,
        for it to be considered a mode at all.
    min_post_size : int
        Below this number of accepted samples the marginal estimates are reported
        but flagged as untrustworthy.

    Returns
    -------
    dict
        ``parameter_names``, ``peak``, ``mean``, ``median``, ``hdi_low``,
        ``hdi_high``, ``std_post``, ``std_prior``, ``variance_reduction``,
        ``n_marginal_modes``, ``n_bins``, ``flags`` (list of lists), ``post_size``,
        ``verdict``, ``message`` and ``recommendation``.
    """
    posterior = _as_2d_array(posterior)
    n_post, ndim = posterior.shape
    names = _parameter_names(parameter_names, ndim)
    bounds = _resolve_prior_bounds(prior_bounds, prior, posterior)

    if prior is not None and np.size(prior):
        std_prior = _as_2d_array(prior, "prior").std(axis=0)
    else:
        # Uniform prior over the calibration range.
        std_prior = (bounds[:, 1] - bounds[:, 0]) / np.sqrt(12.0)

    peak = np.full(ndim, np.nan)
    mean = posterior.mean(axis=0)
    median = np.median(posterior, axis=0)
    std_post = posterior.std(axis=0)
    hdi_low = np.full(ndim, np.nan)
    hdi_high = np.full(ndim, np.nan)
    n_modes = np.ones(ndim, dtype=int)
    n_bins = np.zeros(ndim, dtype=int)
    flags = [[] for _ in range(ndim)]

    with np.errstate(divide="ignore", invalid="ignore"):
        variance_reduction = 1.0 - (std_post ** 2) / np.where(
            std_prior ** 2 > 0, std_prior ** 2, np.nan)
    variance_reduction = np.clip(np.nan_to_num(variance_reduction, nan=0.0), 0.0, 1.0)

    for i in range(ndim):
        column = posterior[:, i]
        low, high = bounds[i]
        span = high - low

        counts, edges = _marginal_histogram(column, samples_per_bin=samples_per_bin,
                                            min_bins=min_bins)
        if counts is None:
            peak[i] = median[i]
            flags[i].append("degenerate_marginal")
        else:
            n_bins[i] = counts.size
            peak[i] = _histogram_peak(column, counts, edges)
            n_modes[i] = _count_marginal_modes(
                counts, valley_fraction=mode_valley_fraction,
                min_mode_height=min_mode_height)
            if n_modes[i] > 1:
                flags[i].append("multimodal_marginal")

        hdi_low[i], hdi_high[i] = _hdi(column, credible_mass)

        if variance_reduction[i] < var_reduction_tol:
            flags[i].append("non_identifiable")
        if span > 0:
            if (peak[i] - low) < bound_tol * span:
                flags[i].append("pinned_at_lower_bound")
            if (high - peak[i]) < bound_tol * span:
                flags[i].append("pinned_at_upper_bound")

    pinned = [names[i] for i in range(ndim)
              if any(f.startswith("pinned") for f in flags[i])]
    unidentified = [names[i] for i in range(ndim) if "non_identifiable" in flags[i]]
    multimodal = [names[i] for i in range(ndim) if "multimodal_marginal" in flags[i]]

    if n_post < min_post_size:
        verdict = "low_posterior_sample"
        message = (
            f"Only {n_post} accepted posterior samples (below {min_post_size}). The "
            f"marginal peaks are reported but are not a reliable density estimate.")
        recommendation = (
            "Increase prior_samples, or widen the observation variances, so that "
            "rejection sampling accepts more samples before reading any optimum off "
            "this posterior.")
    elif not unidentified and not pinned:
        verdict = "well_identified"
        message = (
            f"All {ndim} calibration parameters are identified by the data "
            f"(posterior variance reduced by "
            f"{100 * float(np.min(variance_reduction)):.0f}% or more).")
        recommendation = (
            "Check the equifinality diagnostic before combining the per-parameter "
            "peaks into one calibrated parameter set.")
    elif len(unidentified) == ndim:
        verdict = "non_identifiable"
        message = (
            "No calibration parameter is constrained by the data: every posterior "
            "marginal is essentially the prior.")
        recommendation = (
            "The calibration targets carry no information about these parameters. "
            "Revisit the choice of calibration targets, the measurement errors, "
            "or the parameter ranges before interpreting any optimum.")
    else:
        verdict = "partially_identified"
        parts = []
        if unidentified:
            parts.append(f"not constrained by the data: {', '.join(unidentified)}")
        if pinned:
            parts.append(f"peaking at a prior bound: {', '.join(pinned)}")
        if multimodal:
            parts.append(f"multimodal marginal: {', '.join(multimodal)}")
        message = "Some parameters are problematic (" + "; ".join(parts) + ")."
        recommendation = (
            "A parameter pinned at a prior bound indicates the range is too narrow or "
            "that this parameter alone cannot compensate the model error; widen the "
            "range or add a second calibration parameter. Do not report an optimum "
            "for a parameter flagged non_identifiable.")

    return {
        "parameter_names": names,
        "peak": peak,
        "mean": mean,
        "median": median,
        "hdi_low": hdi_low,
        "hdi_high": hdi_high,
        "credible_mass": credible_mass,
        "std_post": std_post,
        "std_prior": np.asarray(std_prior, dtype=float),
        "variance_reduction": variance_reduction,
        "n_marginal_modes": n_modes,
        "n_bins": n_bins,
        "flags": flags,
        "prior_bounds": bounds,
        "post_size": int(n_post),
        "verdict": verdict,
        "message": message,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# joint optimum
# ---------------------------------------------------------------------------
def joint_optimum(
        posterior,
        method="auto",
        surrogate=None,
        observations=None,
        error=None,
        k_neighbors=None,
        n_top=1,
        refine_with_surrogate=False,
        density=None,
):
    """Parameter vector of highest *joint* posterior density.

    Unlike the per-parameter marginal peaks, this is by construction a jointly
    plausible parameter set: it is an actual posterior sample, so it respects every
    correlation the calibration induced between the parameters.

    Parameters
    ----------
    posterior : array
        Joint posterior sample, shape ``[n_samples, n_parameters]``.
    method : {'auto', 'knn', 'likelihood'}
        ``'auto'`` and ``'knn'`` estimate the density of the accepted sample itself
        from nearest-neighbour distances and need nothing but ``BAL_dictionary.pkl``.
        ``'likelihood'`` re-evaluates the surrogate at the posterior samples and takes
        the maximum joint likelihood, which requires ``surrogate``, ``observations``
        and ``error``.
    surrogate : object, optional
        Trained GPE exposing ``predict_(input_sets=...)``, for the likelihood method.
    observations, error : array, optional
        Measured values ``[1, n_obs]`` and observation variances ``[n_obs]``, as held
        by the model instance, for the likelihood method.
    n_top : int
        Also return this many highest-density candidates.
    refine_with_surrogate : bool
        Re-rank the ``n_top`` density candidates by surrogate likelihood. Cheap
        (``n_top`` predictions) and combines the robustness of the density estimate
        with the exactness of the likelihood.
    density : _DensityEstimator, optional
        Reuse an existing estimator instead of building a new one.

    Returns
    -------
    dict
        ``vector``, ``index``, ``log_density``, ``top_vectors``, ``method_used``,
        ``density`` (the estimator, reusable by :func:`equifinality_diagnostic`) and
        ``message``.

    Note
    ----
    Because the shipped prior is uniform, the density mode of the accepted sample
    and the maximum-likelihood sample estimate the same point. For a non-uniform
    prior the ``'likelihood'`` method returns the maximum *likelihood* rather than
    the maximum *posterior* vector, since the prior density is not added.
    """
    posterior = _as_2d_array(posterior)
    n_post = posterior.shape[0]
    n_top = int(np.clip(n_top, 1, n_post))
    message = ""

    if method == "likelihood":
        if surrogate is None or observations is None or error is None:
            raise ValueError(
                "method='likelihood' needs surrogate, observations and error. Use "
                "method='auto' to score the posterior sample without a surrogate.")
        log_density = _surrogate_log_likelihood(surrogate, posterior, observations, error)
        method_used = "likelihood"
        message = ("Joint optimum from the maximum surrogate likelihood over the "
                   "posterior samples.")
    else:
        if density is None:
            density = _DensityEstimator(posterior, method=method, k_neighbors=k_neighbors)
        log_density = density.log_density(posterior)
        method_used = density.method
        message = (f"Joint optimum from the highest joint posterior density "
                   f"('{method_used}' estimate over {n_post} accepted samples).")

        if refine_with_surrogate:
            if surrogate is None or observations is None or error is None:
                logger_warn.warning(
                    "refine_with_surrogate requested without a surrogate, observations "
                    "and error; keeping the density ranking.")
            else:
                top_idx = np.argsort(log_density)[::-1][:n_top]
                top_ll = _surrogate_log_likelihood(
                    surrogate, posterior[top_idx], observations, error)
                best = int(top_idx[int(np.argmax(top_ll))])
                order = np.argsort(log_density)[::-1][:n_top]
                return {
                    "vector": posterior[best],
                    "index": best,
                    "log_density": log_density,
                    "top_vectors": posterior[order],
                    "method_used": f"{method_used}+likelihood",
                    "density": density,
                    "message": (f"Joint optimum from the {n_top} highest-density "
                                f"candidates re-ranked by surrogate likelihood."),
                }

    order = np.argsort(log_density)[::-1]
    best = int(order[0])
    return {
        "vector": posterior[best],
        "index": best,
        "log_density": log_density,
        "top_vectors": posterior[order[:n_top]],
        "method_used": method_used,
        "density": density if method != "likelihood" else None,
        "message": message,
    }


def _surrogate_log_likelihood(surrogate, parameter_sets, observations, error):
    """Joint log-likelihood of surrogate predictions at ``parameter_sets``.

    Reuses :class:`~hydroBayesCal.surrogate.bal_functions.BayesianInference` so the
    likelihood is exactly the one the calibration itself used. Imported lazily to
    keep this module importable without the surrogate stack.
    """
    from hydroBayesCal.surrogate.bal_functions import BayesianInference

    predictions = surrogate.predict_(input_sets=np.atleast_2d(parameter_sets))["output"]
    inference = BayesianInference(
        model_predictions=predictions,
        observations=np.atleast_2d(observations),
        error=np.asarray(error, dtype=float).ravel(),
    )
    inference.calculate_likelihood_manual()
    return np.asarray(inference.log_likelihood, dtype=float).ravel()


# ---------------------------------------------------------------------------
# equifinality
# ---------------------------------------------------------------------------
def equifinality_diagnostic(
        posterior,
        marginal_peak_vector,
        joint_optimum_vector=None,
        density=None,
        sample_log_density=None,
        parameter_names=None,
        corr_warn=0.30,
        corr_alarm=0.60,
        density_percentile_alarm=10.0,
        mahalanobis_percentile_alarm=95.0,
):
    """Is the vector of independent marginal peaks a jointly plausible parameter set?

    Each calibration parameter has its own posterior marginal and therefore its own
    optimum, but stacking those optima into one vector implicitly assumes the
    parameters are independent. Under equifinality they are not: a friction zone and
    a critical Shields parameter can trade off along a ridge, so that the combination
    of their individual peaks falls in the empty middle of that ridge, a parameter
    set the posterior considers implausible even though each component is
    individually optimal.

    This function measures that directly, by the posterior correlation, by the
    Mahalanobis distance of the marginal-peak vector under the posterior covariance,
    and by the joint posterior density at that vector relative to the accepted
    samples.

    Returns
    -------
    dict
        ``correlation_matrix``, ``max_abs_correlation``, ``most_correlated_pair``,
        ``mahalanobis``, ``mahalanobis_percentile``, ``density_percentile``,
        ``gap_to_joint_optimum``, ``gap_normalised``, ``verdict``
        (``consistent`` | ``acceptable`` | ``coupled`` | ``inconsistent``),
        ``message`` and ``recommendation``.
    """
    posterior = _as_2d_array(posterior)
    if posterior.shape[0] < 2:
        raise ValueError(
            "The equifinality diagnostic needs at least two accepted posterior "
            f"samples, got {posterior.shape[0]}.")
    ndim = posterior.shape[1]
    names = _parameter_names(parameter_names, ndim)
    peak_vector = np.asarray(marginal_peak_vector, dtype=float).ravel()

    # Correlation, ignoring parameters that never moved.
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(posterior, rowvar=False)
    correlation = np.atleast_2d(np.nan_to_num(correlation, nan=0.0))
    off_diagonal = correlation - np.diag(np.diag(correlation))
    max_abs_corr = float(np.max(np.abs(off_diagonal))) if ndim > 1 else 0.0
    if ndim > 1:
        flat = int(np.argmax(np.abs(off_diagonal)))
        pair = (flat // ndim, flat % ndim)
    else:
        pair = (0, 0)

    # Mahalanobis distance under the posterior covariance, and its rank among the
    # posterior samples' own distances.
    center = posterior.mean(axis=0)
    covariance = np.atleast_2d(np.cov(posterior, rowvar=False))
    inverse = np.linalg.pinv(covariance)

    def _mahalanobis(points):
        delta = np.atleast_2d(points) - center
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inverse, delta), 0.0))

    mahalanobis = float(_mahalanobis(peak_vector)[0])
    sample_distances = _mahalanobis(posterior)
    mahalanobis_percentile = float(100.0 * np.mean(sample_distances <= mahalanobis))

    if density is None:
        density = _DensityEstimator(posterior)
    if sample_log_density is None:
        sample_log_density = density.log_density(posterior)
    density_percentile = density.percentile_of(peak_vector, sample_log_density)

    std_post = posterior.std(axis=0)
    if joint_optimum_vector is None:
        gap = np.full(ndim, np.nan)
        gap_normalised = np.full(ndim, np.nan)
    else:
        gap = peak_vector - np.asarray(joint_optimum_vector, dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            gap_normalised = np.where(std_post > 0, gap / std_post, np.nan)

    pair_text = (f"'{names[pair[0]]}' and '{names[pair[1]]}' correlate at "
                 f"r = {correlation[pair]:+.2f} under the posterior")

    in_low_density = (density_percentile < density_percentile_alarm
                      or mahalanobis_percentile >= mahalanobis_percentile_alarm)

    if in_low_density:
        verdict = "inconsistent"
        message = (
            f"The parameters are coupled by the calibration ({pair_text}), and the "
            f"vector assembled from their independent marginal peaks sits at the "
            f"{density_percentile:.0f}th percentile of joint posterior density "
            f"(Mahalanobis {mahalanobis:.2f}, above {mahalanobis_percentile:.0f}% of "
            f"the posterior samples). Combining independent marginal peaks is not a "
            f"valid joint parameter set for this calibration.")
        recommendation = (
            "Report the marginal peaks as a per-parameter summary only. Run the full "
            "complexity model at the joint optimum and at the per-mode "
            "representatives instead, and let the model outputs arbitrate.")
    elif max_abs_corr >= corr_alarm:
        verdict = "coupled"
        message = (
            f"The marginal-peak vector does lie in the posterior bulk "
            f"({density_percentile:.0f}th percentile of joint posterior density), but "
            f"the calibration couples the parameters tightly ({pair_text}). Along such "
            f"a trade-off each parameter's marginal peak is compatible with a whole "
            f"range of values of the other, so the marginals do not determine which "
            f"combination is right, and that the two happen to be compatible here is "
            f"not something the marginals guarantee.")
        recommendation = (
            "Prefer the joint optimum as the calibrated parameter set, and run the "
            "mode representatives to see how far the coupled parameters can trade off "
            "against each other without degrading the fit.")
    elif (max_abs_corr < corr_warn and density_percentile >= 50.0
            and mahalanobis_percentile < 90.0):
        verdict = "consistent"
        message = (
            f"The calibration parameters are effectively independent under the "
            f"posterior (largest |r| = {max_abs_corr:.2f}), and the marginal-peak "
            f"vector lies at the {density_percentile:.0f}th percentile of joint "
            f"posterior density. The per-parameter optima do form a valid joint "
            f"parameter set.")
        recommendation = (
            "The marginal-peak vector can be used as the calibrated parameter set.")
    else:
        verdict = "acceptable"
        message = (
            f"The calibration parameters are mildly coupled (largest |r| = "
            f"{max_abs_corr:.2f}, {pair_text}). The marginal-peak vector is inside the "
            f"posterior bulk ({density_percentile:.0f}th density percentile) but is "
            f"not the joint optimum.")
        recommendation = (
            "Run the full complexity model at both the marginal-peak vector and the "
            "joint optimum and compare them against the observations before picking "
            "one as the calibrated parameter set.")

    return {
        "parameter_names": names,
        "correlation_matrix": correlation,
        "max_abs_correlation": max_abs_corr,
        "most_correlated_pair": pair,
        "most_correlated_pair_names": (names[pair[0]], names[pair[1]]),
        "mahalanobis": mahalanobis,
        "mahalanobis_percentile": mahalanobis_percentile,
        "density_percentile": density_percentile,
        "gap_to_joint_optimum": gap,
        "gap_normalised": gap_normalised,
        "verdict": verdict,
        "message": message,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# posterior modes
# ---------------------------------------------------------------------------
def detect_posterior_modes(
        posterior,
        max_modes=5,
        min_weight=0.05,
        min_separation=0.20,
        prior_bounds=None,
        prior=None,
        random_state=0,
        min_samples=100,
        relevant_mask=None,
        merge_connected=True,
        valley_fraction=0.5,
        n_neighbors=10,
):
    """Distinct joint posterior modes, i.e. the concrete expression of equifinality.

    Several genuinely different parameter combinations can explain the same
    measurements comparably well. Where that happens, no single optimum is a
    defensible answer and the honest deliverable is a small set of representative
    parameter vectors with their posterior weights.

    A Gaussian mixture is fitted on standardised coordinates for one to
    ``max_modes`` components and selected by BIC, then pruned: components lighter
    than ``min_weight`` are dropped, components closer than ``min_separation`` of the
    prior range on every parameter are merged, and components that are **connected**
    are merged. Without that pruning, BIC readily splits a single curved posterior
    into several components and reports a false equifinality alarm, because a mixture
    of Gaussians needs several components to follow a curved ridge.

    The connectivity test separates the two situations a modeller has to tell apart.
    Along a **ridge**, parameters trade off continuously and every intermediate
    combination is about as good, so the two candidates are joined by a chain of
    posterior samples of undiminished density; they describe one connected family of
    solutions and are merged. Between **genuinely distinct** solutions the density
    collapses in between and the components are kept apart.

    Connectivity is tested on the level set: the posterior samples denser than
    ``valley_fraction`` of the lower of the two representatives' densities are linked
    into a nearest-neighbour graph, and the two candidates count as one mode when
    they fall in the same connected component. Following the samples rather than the
    straight line between the candidates is what makes this work for a *curved*
    ridge, where the straight chord leaves the ridge and crosses empty space.

    The representative of a mode is the highest-density posterior *sample* assigned
    to it, never the component mean: a mixture mean can average two lobes and land in
    a low-density void, which is the very failure this module warns about elsewhere.

    Parameters
    ----------
    relevant_mask : array of bool, optional
        Which parameters to cluster on. Non-identifiable parameters keep their prior
        spread in the posterior, and a mixture will happily cut that spread in half
        and report two modes that differ in nothing the measurements constrain. Pass
        the identified parameters here (:func:`analyze_posterior` does so
        automatically) to keep the mode count meaningful.
    merge_connected : bool
        Merge candidate modes that are joined by a path of undiminished posterior
        density, i.e. that are two points on one ridge rather than two solutions.
    valley_fraction : float
        How far the density between the candidates may drop, as a fraction of the
        lower of their two densities, before they count as separate modes.
    n_neighbors : int
        Neighbours per node in the connectivity graph.

    Returns
    -------
    dict
        ``n_modes``, ``weights``, ``representatives``, ``means``, ``labels``,
        ``bic``, ``verdict``, ``message`` and ``recommendation``.
    """
    posterior = _as_2d_array(posterior)
    n_post, ndim = posterior.shape
    bounds = _resolve_prior_bounds(prior_bounds, prior, posterior)
    span = np.where(bounds[:, 1] - bounds[:, 0] > 0, bounds[:, 1] - bounds[:, 0], 1.0)

    unimodal = {
        "n_modes": 1,
        "weights": np.array([1.0]),
        "representatives": posterior.mean(axis=0).reshape(1, -1),
        "means": posterior.mean(axis=0).reshape(1, -1),
        "labels": np.zeros(n_post, dtype=int),
        "bic": np.array([np.nan]),
        "verdict": "unimodal",
        "message": "",
        "recommendation": "",
    }

    if n_post < min_samples:
        unimodal["message"] = (
            f"Only {n_post} accepted posterior samples (below {min_samples}); mode "
            f"detection was skipped and the posterior is treated as unimodal.")
        unimodal["recommendation"] = (
            "Increase prior_samples so that rejection sampling accepts more samples "
            "if you need the equifinality structure resolved.")
        return unimodal

    from sklearn.mixture import GaussianMixture

    z, _, _ = _standardize(posterior)

    # Cluster only on the parameters the data actually constrain. A parameter that
    # kept its prior spread carries no mode structure, and letting the mixture split
    # along it manufactures modes that differ in nothing meaningful.
    if relevant_mask is None:
        mask = np.ones(ndim, dtype=bool)
    else:
        mask = np.asarray(relevant_mask, dtype=bool).reshape(ndim)
        if not mask.any():
            logger_warn.warning(
                "No calibration parameter is identified by the data; mode detection "
                "falls back to using all parameters.")
            mask = np.ones(ndim, dtype=bool)
    z_fit = z[:, mask]

    max_modes = int(np.clip(max_modes, 1, max(1, n_post // 10)))

    models, bic = [], []
    for k in range(1, max_modes + 1):
        try:
            model = GaussianMixture(n_components=k, covariance_type="full",
                                    random_state=random_state).fit(z_fit)
        except ValueError:
            break
        models.append(model)
        bic.append(model.bic(z_fit))

    if not models:
        unimodal["message"] = "Gaussian-mixture fitting failed; treating the posterior as unimodal."
        return unimodal

    best = models[int(np.argmin(bic))]
    labels = best.predict(z_fit)
    log_density = best.score_samples(z_fit)
    weights = np.asarray(best.weights_, dtype=float)

    # Prune: drop light components, then merge components that are not actually
    # separated in parameter space.
    keep = [k for k in range(best.n_components)
            if weights[k] >= min_weight and np.any(labels == k)]
    if not keep:
        keep = [int(np.argmax(weights))]

    if merge_connected:
        density = _DensityEstimator(posterior)
        sample_log_density = density.log_density(posterior)
        z_all, _, _ = _standardize(posterior)

    def _connected(first, second):
        """True when a chain of dense posterior samples joins the two candidates."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        endpoints = density.log_density(np.vstack([first, second]))
        if not np.all(np.isfinite(endpoints)):
            return False
        level = float(np.min(endpoints)) + np.log(valley_fraction)

        keep = np.where(sample_log_density >= level)[0]
        if keep.size < 3:
            return False

        z_keep = z_all[keep]
        tree = cKDTree(z_keep)
        k = int(min(n_neighbors + 1, z_keep.shape[0]))
        _, neighbours = tree.query(z_keep, k=k)
        neighbours = np.atleast_2d(neighbours)

        rows = np.repeat(np.arange(z_keep.shape[0]), neighbours.shape[1] - 1)
        cols = neighbours[:, 1:].ravel()
        graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                           shape=(z_keep.shape[0],) * 2)
        _, component = connected_components(graph, directed=False)

        # Attach each candidate to its nearest sample inside the level set.
        z_pair = (np.vstack([first, second]) - density._center) / density._scale
        _, nearest = tree.query(z_pair, k=1)
        return component[int(nearest[0])] == component[int(nearest[1])]

    representatives, kept_weights, kept_means = [], [], []
    for k in keep:
        member = np.where(labels == k)[0]
        representative = posterior[member[int(np.argmax(log_density[member]))]]
        mean = posterior[member].mean(axis=0)

        merged = False
        for existing_index, existing in enumerate(representatives):
            separation = np.abs(existing - representative) / span
            close = np.all(separation[mask] < min_separation)
            if close or (merge_connected and _connected(existing, representative)):
                kept_weights[existing_index] += weights[k]
                merged = True
                break
        if not merged:
            representatives.append(representative)
            kept_weights.append(float(weights[k]))
            kept_means.append(mean)

    weights_out = np.asarray(kept_weights, dtype=float)
    weights_out = weights_out / weights_out.sum()
    order = np.argsort(weights_out)[::-1]
    representatives = np.asarray(representatives)[order]
    kept_means = np.asarray(kept_means)[order]
    weights_out = weights_out[order]
    n_modes = len(weights_out)

    if n_modes > 1:
        verdict = "multimodal_equifinality"
        message = (
            f"The joint posterior has {n_modes} distinct modes carrying "
            f"{', '.join(f'{100 * w:.0f}%' for w in weights_out)} of the posterior "
            f"mass: several different parameter combinations explain the measurements "
            f"comparably well.")
        recommendation = (
            "Do not report a single calibrated parameter set. Run the full complexity "
            "model at each mode representative and compare the outputs against the "
            "observations; if they are indistinguishable, the calibration is "
            "equifinal and the remaining choice is a physical one.")
    else:
        verdict = "unimodal"
        message = "The joint posterior has a single mode."
        recommendation = ""

    return {
        "n_modes": n_modes,
        "weights": weights_out,
        "representatives": representatives,
        "means": kept_means,
        "labels": labels,
        "bic": np.asarray(bic, dtype=float),
        "verdict": verdict,
        "message": message,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# candidate assembly
# ---------------------------------------------------------------------------
def assemble_candidates(
        marginal,
        joint,
        modes,
        posterior,
        parameter_names=None,
        equifinality=None,
        include=("marginal_peak", "joint_map", "posterior_mean", "modes"),
        prior_bounds=None,
        clip_to_bounds=True,
        dedupe_tol=1e-9,
):
    """Build the labelled table of candidate calibrated parameter sets.

    The candidates are deliberately plural. Which one deserves to be called *the*
    calibrated parameter set is a question the surrogate cannot settle on its own,
    and the cheapest way to settle it is to run the full complexity model at each
    candidate and compare the outputs against the measurements.

    Returns
    -------
    dict
        ``labels``, ``kinds``, ``vectors`` (``[n_candidates, n_parameters]``),
        ``weights``, ``notes`` and ``parameter_names``.
    """
    posterior = _as_2d_array(posterior)
    ndim = posterior.shape[1]
    names = _parameter_names(parameter_names, ndim)
    bounds = _resolve_prior_bounds(prior_bounds, None, posterior)

    labels, kinds, vectors, weights, notes = [], [], [], [], []

    def _add(label, kind, vector, weight, note):
        vector = np.asarray(vector, dtype=float).ravel()
        for existing in vectors:
            if np.allclose(existing, vector, atol=dedupe_tol, rtol=0.0):
                return
        labels.append(label)
        kinds.append(kind)
        vectors.append(vector)
        weights.append(weight)
        notes.append(note)

    if "marginal_peak" in include:
        note = ("Peak of each parameter's own posterior marginal, combined "
                "component-wise.")
        if equifinality is not None:
            note += (f" Equifinality check: {equifinality['verdict']} "
                     f"({equifinality['density_percentile']:.0f}th percentile of joint "
                     f"posterior density).")
        _add("marginal_peak", "per-parameter marginal optimum", marginal["peak"],
             float("nan"), note)

    if "joint_map" in include and joint is not None:
        _add("joint_map", "joint posterior optimum", joint["vector"], float("nan"),
             f"Highest joint posterior density ({joint['method_used']} estimate); "
             f"an accepted posterior sample, so jointly consistent by construction.")

    if "posterior_mean" in include:
        _add("posterior_mean", "posterior mean", posterior.mean(axis=0), float("nan"),
             "Mean of the joint posterior sample; misleading for a multimodal "
             "posterior.")

    if "modes" in include and modes is not None and modes["n_modes"] > 1:
        for index, (weight, representative) in enumerate(
                zip(modes["weights"], modes["representatives"]), start=1):
            _add(f"mode_{index}_w{weight:.2f}", "posterior mode representative",
                 representative, float(weight),
                 f"Representative of posterior mode {index}, carrying "
                 f"{100 * weight:.0f}% of the posterior mass.")

    vectors = np.asarray(vectors, dtype=float).reshape(-1, ndim)
    if clip_to_bounds and vectors.size:
        clipped = np.clip(vectors, bounds[:, 0], bounds[:, 1])
        if not np.array_equal(clipped, vectors):
            logger_warn.warning(
                "Some candidate parameter values fell outside the calibration ranges "
                "and were clipped to the bounds.")
        vectors = clipped

    return {
        "labels": labels,
        "kinds": kinds,
        "vectors": vectors,
        "weights": np.asarray(weights, dtype=float),
        "notes": notes,
        "parameter_names": names,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def analyze_posterior(
        bayesian_dict=None,
        posterior=None,
        prior=None,
        parameter_names=None,
        prior_bounds=None,
        iteration=-1,
        joint_method="auto",
        surrogate=None,
        observations=None,
        error=None,
        include=("marginal_peak", "joint_map", "posterior_mean", "modes"),
        max_modes=5,
        **kwargs
):
    """Run the full posterior analysis and return every part of it.

    Accepts either a loaded ``BAL_dictionary.pkl`` or raw arrays.

    Returns
    -------
    dict
        ``iteration``, ``posterior``, ``marginal``, ``joint``, ``equifinality``,
        ``modes`` and ``candidates``.
    """
    if bayesian_dict is not None:
        posterior, iteration = select_posterior_iteration(bayesian_dict, iteration)
        if prior is None:
            prior = bayesian_dict.get("prior")
        if parameter_names is None:
            parameter_names = bayesian_dict.get("calibration_parameters")
        if prior_bounds is None:
            prior_bounds = bayesian_dict.get("param_values")
    posterior = _as_2d_array(posterior)

    marginal = marginal_optima(
        posterior, prior_bounds=prior_bounds, parameter_names=parameter_names,
        prior=prior, **kwargs)

    joint = joint_optimum(
        posterior, method=joint_method, surrogate=surrogate,
        observations=observations, error=error)

    equifinality = equifinality_diagnostic(
        posterior, marginal["peak"], joint_optimum_vector=joint["vector"],
        density=joint.get("density"), sample_log_density=(
            joint["log_density"] if joint["method_used"] != "likelihood" else None),
        parameter_names=marginal["parameter_names"])

    modes = detect_posterior_modes(
        posterior, max_modes=max_modes, prior_bounds=marginal["prior_bounds"],
        relevant_mask=np.array([
            "non_identifiable" not in flags for flags in marginal["flags"]]))

    candidates = assemble_candidates(
        marginal, joint, modes, posterior,
        parameter_names=marginal["parameter_names"], equifinality=equifinality,
        include=include, prior_bounds=marginal["prior_bounds"])

    return {
        "iteration": iteration,
        "posterior": posterior,
        "marginal": marginal,
        "joint": joint,
        "equifinality": equifinality,
        "modes": modes,
        "candidates": candidates,
    }


def log_posterior_analysis(analysis, logger_obj=None):
    """Log a posterior analysis, warning where the result needs care.

    Returns ``analysis`` so calls can be chained.
    """
    info = logger_obj or logger
    marginal = analysis["marginal"]
    joint = analysis["joint"]
    equifinality = analysis["equifinality"]
    modes = analysis["modes"]
    names = marginal["parameter_names"]

    info.info("=" * 78)
    info.info(f"POSTERIOR ANALYSIS (BAL iteration {analysis['iteration']}, "
              f"{marginal['post_size']} accepted samples)")
    info.info("=" * 78)

    info.info("Per-parameter marginal optima "
              f"({100 * marginal['credible_mass']:.0f}% credible interval):")
    for i, name in enumerate(names):
        flags = ", ".join(marginal["flags"][i]) or "-"
        info.info(
            f"  {name:<40s} peak={marginal['peak'][i]:.5g}  "
            f"[{marginal['hdi_low'][i]:.5g}, {marginal['hdi_high'][i]:.5g}]  "
            f"variance reduction={100 * marginal['variance_reduction'][i]:5.1f}%  "
            f"flags: {flags}")

    if marginal["verdict"] in ("non_identifiable", "low_posterior_sample",
                               "partially_identified"):
        logger_warn.warning(f"Identifiability [{marginal['verdict']}]: {marginal['message']}")
        logger_warn.warning(f"  -> {marginal['recommendation']}")
    else:
        info.info(f"Identifiability [{marginal['verdict']}]: {marginal['message']}")

    info.info(f"Joint posterior optimum ({joint['method_used']}):")
    for i, name in enumerate(names):
        info.info(f"  {name:<40s} {joint['vector'][i]:.5g}")

    if equifinality["verdict"] in ("inconsistent", "coupled"):
        logger_warn.warning(f"Equifinality [{equifinality['verdict']}]: {equifinality['message']}")
        logger_warn.warning(f"  -> {equifinality['recommendation']}")
    else:
        info.info(f"Equifinality [{equifinality['verdict']}]: {equifinality['message']}")
        info.info(f"  -> {equifinality['recommendation']}")

    if modes["verdict"] == "multimodal_equifinality":
        logger_warn.warning(f"Posterior modes: {modes['message']}")
        logger_warn.warning(f"  -> {modes['recommendation']}")
    elif modes["message"]:
        info.info(f"Posterior modes: {modes['message']}")

    candidates = analysis["candidates"]
    info.info(f"Candidate calibrated parameter sets ({len(candidates['labels'])}):")
    for label, vector in zip(candidates["labels"], candidates["vectors"]):
        info.info(f"  {label:<24s} " + "  ".join(f"{value:.5g}" for value in vector))
    info.info("=" * 78)
    return analysis


# ---------------------------------------------------------------------------
# per-BAL-iteration tracking
# ---------------------------------------------------------------------------
def track_iteration(posterior, prior=None, parameter_names=None, prior_bounds=None,
                    joint_method="auto"):
    """Cheap per-iteration posterior summary for use inside the BAL loop.

    Records where each parameter's own optimum currently sits, how well the data
    constrain it, and how far the combination of those optima is from a jointly
    plausible parameter set. Uses the nearest-neighbour density and no mixture fit,
    so the cost stays negligible next to one full-complexity simulation.

    This function never raises. It runs inside a loop whose iterations each cost a
    solver run, so a diagnostic failure must not abort a multi-day calibration; on
    any error it warns and returns NaN-filled fields.

    Returns
    -------
    dict
        ``peak``, ``hdi``, ``variance_reduction``, ``flags``, ``gap`` (dict with
        ``density_percentile``, ``mahalanobis_percentile``, ``max_abs_correlation``
        and ``verdict``) and ``post_size``.
    """
    ndim = None
    try:
        posterior = _as_2d_array(posterior)
        ndim = posterior.shape[1]

        marginal = marginal_optima(
            posterior, prior_bounds=prior_bounds, parameter_names=parameter_names,
            prior=prior)
        joint = joint_optimum(posterior, method=joint_method)
        equifinality = equifinality_diagnostic(
            posterior, marginal["peak"], joint_optimum_vector=joint["vector"],
            density=joint.get("density"), sample_log_density=joint["log_density"],
            parameter_names=marginal["parameter_names"])

        return {
            "peak": marginal["peak"],
            "hdi": np.column_stack([marginal["hdi_low"], marginal["hdi_high"]]),
            "variance_reduction": marginal["variance_reduction"],
            "flags": marginal["flags"],
            "gap": {
                "density_percentile": equifinality["density_percentile"],
                "mahalanobis_percentile": equifinality["mahalanobis_percentile"],
                "max_abs_correlation": equifinality["max_abs_correlation"],
                "verdict": equifinality["verdict"],
            },
            "post_size": marginal["post_size"],
        }
    except Exception as exception:  # never abort a running calibration
        logger_warn.warning(f"Per-iteration posterior diagnostic skipped: {exception}")
        if ndim is None:
            ndim = len(parameter_names) if parameter_names else 0
        return {
            "peak": np.full(ndim, np.nan),
            "hdi": np.full((ndim, 2), np.nan),
            "variance_reduction": np.full(ndim, np.nan),
            "flags": [[] for _ in range(ndim)],
            "gap": {"density_percentile": np.nan, "mahalanobis_percentile": np.nan,
                    "max_abs_correlation": np.nan, "verdict": "unavailable"},
            "post_size": 0,
        }


def record_iteration(bayesian_dict, iteration, posterior, prior=None,
                     parameter_names=None, prior_bounds=None, log=True):
    """Write one iteration's posterior diagnostic into ``bayesian_dict``.

    Called from the BAL loop of the driver scripts right after the posterior of the
    iteration has been stored. Creates the :data:`ITERATION_KEYS` entries on first
    use, so it also works on a dictionary built before those keys existed.

    Returns
    -------
    dict
        The summary from :func:`track_iteration`.
    """
    summary = track_iteration(posterior, prior=prior, parameter_names=parameter_names,
                              prior_bounds=prior_bounds)

    n_entries = len(bayesian_dict.get("posterior", [])) or (iteration + 1)
    for key in ITERATION_KEYS:
        if key not in bayesian_dict:
            bayesian_dict[key] = [None] * n_entries
        while len(bayesian_dict[key]) <= iteration:
            bayesian_dict[key].append(None)

    bayesian_dict["marginal_optima"][iteration] = summary["peak"]
    bayesian_dict["marginal_hdi"][iteration] = summary["hdi"]
    bayesian_dict["variance_reduction"][iteration] = summary["variance_reduction"]
    bayesian_dict["identifiability_flags"][iteration] = summary["flags"]
    bayesian_dict["marginal_joint_gap"][iteration] = summary["gap"]

    # Make the stored dictionary self-describing: without these, no consumer can
    # recover the parameter names or the calibration ranges from the result file.
    if parameter_names is not None:
        bayesian_dict.setdefault("calibration_parameters", list(parameter_names))
    if prior_bounds is not None:
        bayesian_dict.setdefault("param_values", prior_bounds)

    if log and summary["post_size"]:
        names = parameter_names or [f"parameter_{i + 1}" for i in range(len(summary["peak"]))]
        peaks = ", ".join(f"{name}={value:.4g}"
                          for name, value in zip(names, summary["peak"]))
        logger.info(f"Marginal optima after iteration {iteration}: {peaks}")
        if summary["gap"]["verdict"] == "inconsistent":
            logger_warn.warning(
                f"Iteration {iteration}: the per-parameter marginal peaks do not form a "
                f"jointly plausible parameter set (joint posterior density percentile "
                f"{summary['gap']['density_percentile']:.0f}, largest |r| = "
                f"{summary['gap']['max_abs_correlation']:.2f}).")
    return summary


# ---------------------------------------------------------------------------
# CSV export into the existing user_param_values run path
# ---------------------------------------------------------------------------
def write_user_collocation_points(candidates, parameter_names, restart_data_folder,
                                  filename="user-collocation-points.csv",
                                  backup_existing=True, fmt="%.10g"):
    """Write candidate parameter sets where the model driver expects to read them.

    The file format is fixed by ``HydroSimulations.__init__``, which reads it with
    ``np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)``: exactly one header line
    followed by numeric rows with one column per calibration parameter, in the order
    of ``calibration['parameters']``.

    Running the model at these points is then the existing manual workflow: set
    ``execution['user_param_values'] = True``, ``execution['complete_bal_mode'] =
    False``, ``execution['only_bal_mode'] = False`` and ``sampling['init_runs']`` to
    the number of candidates, because the run loop is bounded by ``init_runs`` and
    not by the number of rows in this file.

    Returns
    -------
    str
        Path of the written file.
    """
    vectors = np.atleast_2d(np.asarray(candidates["vectors"], dtype=float))
    if vectors.shape[1] != len(parameter_names):
        raise ValueError(
            f"Candidate vectors have {vectors.shape[1]} columns but "
            f"{len(parameter_names)} calibration parameters were given.")

    os.makedirs(restart_data_folder, exist_ok=True)
    path = os.path.join(restart_data_folder, filename)

    if backup_existing and os.path.isfile(path):
        backup = path + ".bak"
        shutil.copyfile(path, backup)
        logger_warn.warning(f"Existing {filename} backed up to {os.path.basename(backup)}.")

    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list(parameter_names))
        for row in vectors:
            writer.writerow([fmt % value for value in row])

    logger.info(f"Wrote {vectors.shape[0]} candidate parameter sets to {path}")
    return path


def write_candidate_report(analysis, output_folder,
                           candidates_filename="calibrated-parameter-candidates.csv",
                           diagnostics_filename="calibrated-parameter-diagnostics.csv"):
    """Write the labelled candidate table and the per-parameter diagnostics.

    These sidecars carry the labels and the verdicts that cannot live in the numeric
    ``user-collocation-points.csv``.

    Returns
    -------
    tuple
        Paths of the two written files.
    """
    os.makedirs(output_folder, exist_ok=True)
    candidates = analysis["candidates"]
    marginal = analysis["marginal"]
    equifinality = analysis["equifinality"]
    names = marginal["parameter_names"]

    candidates_path = os.path.join(output_folder, candidates_filename)
    with open(candidates_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label"] + list(names)
                        + ["kind", "posterior_weight", "equifinality_verdict",
                           "density_percentile", "mahalanobis", "note"])
        for index, label in enumerate(candidates["labels"]):
            row = [label] + [f"{value:.10g}" for value in candidates["vectors"][index]]
            row += [candidates["kinds"][index],
                    "" if np.isnan(candidates["weights"][index])
                    else f"{candidates['weights'][index]:.4f}"]
            if label == "marginal_peak":
                row += [equifinality["verdict"],
                        f"{equifinality['density_percentile']:.2f}",
                        f"{equifinality['mahalanobis']:.4f}"]
            else:
                row += ["", "", ""]
            row.append(candidates["notes"][index])
            writer.writerow(row)

    diagnostics_path = os.path.join(output_folder, diagnostics_filename)
    with open(diagnostics_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "parameter", "marginal_peak", "posterior_mean", "posterior_median",
            "hdi_low", "hdi_high", "std_posterior", "std_prior",
            "variance_reduction", "n_marginal_modes", "prior_min", "prior_max",
            "joint_optimum", "gap_to_joint_optimum_in_posterior_std", "flags"])
        for i, name in enumerate(names):
            writer.writerow([
                name,
                f"{marginal['peak'][i]:.10g}",
                f"{marginal['mean'][i]:.10g}",
                f"{marginal['median'][i]:.10g}",
                f"{marginal['hdi_low'][i]:.10g}",
                f"{marginal['hdi_high'][i]:.10g}",
                f"{marginal['std_post'][i]:.10g}",
                f"{marginal['std_prior'][i]:.10g}",
                f"{marginal['variance_reduction'][i]:.4f}",
                int(marginal["n_marginal_modes"][i]),
                f"{marginal['prior_bounds'][i, 0]:.10g}",
                f"{marginal['prior_bounds'][i, 1]:.10g}",
                f"{analysis['joint']['vector'][i]:.10g}",
                f"{equifinality['gap_normalised'][i]:.4f}",
                "|".join(marginal["flags"][i]),
            ])

    logger.info(f"Wrote candidate report to {candidates_path}")
    logger.info(f"Wrote per-parameter diagnostics to {diagnostics_path}")
    return candidates_path, diagnostics_path
