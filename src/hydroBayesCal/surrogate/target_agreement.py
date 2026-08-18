"""
Agreement between modeled and measured calibration targets, before and after calibration.

A finished Bayesian calibration reports parameter optima, evidence and relative entropy,
all of which live in *parameter* space. None of them answers the question a modeller asks
first: does the calibrated model reproduce the measurements, and if it does not, is the
mismatch a systematic offset or scatter? This module answers that in *observation* space,
for the two states a calibration passes through:

* **before calibration** - the initial-design ensemble of full-complexity runs, i.e. the
  model as it behaves across the prior range of the calibration parameters;
* **after calibration** - the posterior predictive sample, i.e. the surrogate predictions
  at the parameter sets that rejection sampling accepted.

Both states are summarised per calibration point by the ensemble median and compared with
the measured value. A **calibration target** is a measured variable that the calibration
fits (``calibration_quantities`` in the configuration); a **calibration parameter** is a
model parameter that the calibration adjusts. This module only ever looks at targets.

The verdict per target separates the two mismatch types that call for different actions:

* a **systematic** over- or underestimation (mean offset resolvable against the
  measurement uncertainty, and most calibration points on the same side of the 1:1 line)
  is a bias that a calibration parameter can still remove, roughness being the classic
  one - see
  :func:`~hydroBayesCal.function_pool.diagnose_roughness_identifiability`, which is
  re-run here on both states so that the plot states whether roughness was the knob and
  whether calibration actually moved it;
* **scatter** (an offset whose sign changes from point to point) cannot be removed by any
  global parameter value, and points at the model structure, the mesh or the
  measurements instead.

Following the convention of the roughness diagnostic and of
:mod:`~hydroBayesCal.surrogate.posterior_analysis`, every analysis function here is
report-only: it returns a plain dictionary carrying a ``verdict``, a ``message`` and a
``recommendation``, mutates no state and leaves the logging to a separate ``log_*``
function. Nothing in this module depends on a solver binding, so it also runs on an
archived ``BAL_dictionary.pkl`` from a TELEMAC, OpenFOAM or Delft3D calibration.
"""
import os
import pickle

import numpy as np

from hydroBayesCal.utils.config_logging import logger, logger_warn

__all__ = [
    "AGREEMENT_KEY",
    "measurement_error_bars",
    "target_agreement_data",
    "diagnose_target_agreement",
    "log_target_agreement",
    "plot_target_agreement",
    "finalize_target_agreement",
]

#: Key under which :func:`finalize_target_agreement` stores its result in the
#: ``bayesian_dict`` / ``BAL_dictionary.pkl``. Additive: consumers read it with ``.get``,
#: and result files written before it existed simply do not carry it.
AGREEMENT_KEY = "target_agreement"

#: Labels of the two calibration states, in plotting order.
STATE_LABELS = {
    "pre": "before calibration",
    "post": "after calibration",
}


def _quantity_columns(values, n_quantities, index):
    """Columns of calibration target ``index`` from an interleaved array.

    The data contract shared by ``model_evaluations``, ``observations`` and the
    surrogate output is ``column i * nq + j = location i, calibration target j``.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[index::n_quantities]
    return array[:, index::n_quantities]


def measurement_error_bars(complex_model):
    """Per-observation measurement uncertainty for the error bars of the agreement plot.

    ``HydroSimulations.measurement_errors`` holds only the relative term
    (``measurement_error`` times the absolute measured value). The measured fluctuations
    of a campaign, e.g. the standard deviation of a velocity record, come from the
    ``<TARGET>_ERROR`` columns of the calibration-points CSV and are in physical units.
    Both describe the measurement rather than the model, so the error bar is their
    quadratic sum, while ``variances`` (which the likelihood uses) additionally carries
    the model-side ``gpe_error`` and ``model_structural_error`` terms and is therefore
    the wrong quantity to draw against a measured value.

    Parameters
    ----------
    complex_model : object
        A model instance exposing ``measurement_errors``, ``calibration_quantities`` and
        ``calibration_pts_df``.

    Returns
    -------
    numpy.ndarray
        Standard deviations of shape ``[nloc * nq]``, interleaved like ``observations``.
        Falls back to ``measurement_errors`` if the ``_ERROR`` columns cannot be read.
    """
    relative = np.asarray(getattr(complex_model, "measurement_errors", []), dtype=float)
    relative = relative.ravel()
    dataframe = getattr(complex_model, "calibration_pts_df", None)
    quantities = list(getattr(complex_model, "calibration_quantities", []) or [])
    if dataframe is None or not quantities:
        return relative

    lookup = {str(column).lower(): column for column in dataframe.columns}
    columns = [lookup.get(f"{quantity}_ERROR".lower()) for quantity in quantities]
    if any(column is None for column in columns):
        return relative

    site_specific = dataframe[columns].to_numpy(dtype=float).flatten()
    if site_specific.shape != relative.shape:
        # A mismatch means the CSV and the stored errors disagree about the layout;
        # the stored vector is the one the likelihood used, so it wins.
        return relative
    return np.sqrt(relative ** 2 + site_specific ** 2)


def target_agreement_data(
        observations,
        calibration_quantities,
        errors=None,
        initial_model_outputs=None,
        posterior_predictions=None,
        credible_mass=0.95,
):
    """Assemble the modeled-vs-measured series of the two calibration states.

    Parameters
    ----------
    observations : array-like, shape ``[1, nloc * nq]`` or ``[nloc * nq]``
        Measured values of the calibration targets, interleaved by target.
    calibration_quantities : list of str
        Calibration target names in column order ``j``.
    errors : array-like, optional
        Measurement standard deviations of shape ``[nloc * nq]``, see
        :func:`measurement_error_bars`.
    initial_model_outputs : array-like, optional
        Full-complexity outputs of the initial design, shape ``[n_runs, nloc * nq]``.
        Summarised into the ``pre`` state.
    posterior_predictions : array-like, optional
        Surrogate predictions at the accepted posterior samples
        (``BayesianInference.posterior_output``), shape ``[n_accepted, nloc * nq]``.
        Summarised into the ``post`` state.
    credible_mass : float
        Central mass of the ensemble spread reported per point. Default 0.95.

    Returns
    -------
    dict
        ``{"quantities", "n_quantities", "n_locations", "measured", "errors",
        "states": {"pre"/"post": {"modeled", "lower", "upper", "ensemble",
        "n_members", "label"}}}``. States whose input was ``None`` or empty are absent.
    """
    quantities = [str(quantity) for quantity in (calibration_quantities or [])]
    n_quantities = len(quantities)
    if n_quantities == 0:
        raise ValueError("target_agreement_data needs at least one calibration target.")

    measured = np.asarray(observations, dtype=float).ravel()
    if measured.size % n_quantities:
        raise ValueError(
            f"{measured.size} observations do not split into {n_quantities} calibration "
            f"targets; the interleaved layout [nloc * nq] is assumed.")

    if errors is None:
        error_vector = np.zeros_like(measured)
    else:
        error_vector = np.asarray(errors, dtype=float).ravel()
        if error_vector.size != measured.size:
            raise ValueError(
                f"{error_vector.size} error values for {measured.size} observations.")

    lower_percentile = 50.0 * (1.0 - credible_mass)
    upper_percentile = 100.0 - lower_percentile

    data = {
        "quantities": quantities,
        "n_quantities": n_quantities,
        "n_locations": measured.size // n_quantities,
        "measured": measured,
        "errors": error_vector,
        "credible_mass": float(credible_mass),
        "states": {},
    }

    for key, ensemble in (("pre", initial_model_outputs),
                          ("post", posterior_predictions)):
        if ensemble is None:
            continue
        members = np.atleast_2d(np.asarray(ensemble, dtype=float))
        if members.size == 0 or members.shape[1] != measured.size:
            logger_warn.warning(
                f"Calibration-target agreement: the '{key}' ensemble has shape "
                f"{members.shape} and does not match the {measured.size} observations; "
                f"this state is skipped.")
            continue
        data["states"][key] = {
            "modeled": np.nanmedian(members, axis=0),
            "lower": np.nanpercentile(members, lower_percentile, axis=0),
            "upper": np.nanpercentile(members, upper_percentile, axis=0),
            "ensemble": members,
            "n_members": int(members.shape[0]),
            "label": STATE_LABELS[key],
        }

    if not data["states"]:
        raise ValueError(
            "target_agreement_data needs the initial-design outputs, the posterior "
            "predictions, or both.")
    return data


def _target_statistics(modeled, measured, errors, deadband, sign_threshold):
    """Bias, spread and verdict for one calibration target."""
    modeled = np.asarray(modeled, dtype=float)
    measured = np.asarray(measured, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = np.isfinite(modeled) & np.isfinite(measured)
    statistics = {
        "n_points": int(valid.sum()),
        "bias": np.nan, "relative_bias": np.nan, "bias_in_sigma": np.nan,
        "rmse": np.nan, "mae": np.nan, "coverage": np.nan,
        "sign_consistency": np.nan, "verdict": "unavailable", "systematic": None,
        "message": "",
    }
    if statistics["n_points"] == 0:
        statistics["message"] = "no finite modeled/measured pairs."
        return statistics

    residual = modeled[valid] - measured[valid]        # + => modeled too high
    reference = measured[valid]
    point_errors = errors[valid] if errors.size == valid.size else np.zeros_like(residual)

    scale = float(np.mean(np.abs(reference))) or 1.0
    bias = float(np.mean(residual))
    mean_error = float(np.mean(np.abs(point_errors)))

    statistics.update(
        bias=bias,
        relative_bias=bias / scale,
        rmse=float(np.sqrt(np.mean(residual ** 2))),
        mae=float(np.mean(np.abs(residual))),
        bias_in_sigma=(bias / mean_error) if mean_error > 0 else np.nan,
        coverage=(float(np.mean(np.abs(residual) <= point_errors))
                  if mean_error > 0 else np.nan),
    )

    positive = float(np.mean(residual > 0))
    negative = float(np.mean(residual < 0))
    statistics["sign_consistency"] = max(positive, negative)

    # An offset counts as resolvable only above the relative deadband *and* above the
    # standard error of the mean measurement uncertainty: below either, the data cannot
    # tell the offset from noise and calling it a bias would send the modeller chasing
    # a parameter that is already right.
    standard_error = mean_error / np.sqrt(statistics["n_points"]) if mean_error > 0 else 0.0
    resolvable = abs(bias) > max(deadband * scale, standard_error)
    consistent = statistics["sign_consistency"] >= sign_threshold
    # Residuals that alternate in sign average to nearly zero, so a vanishing mean is
    # not by itself a good fit: agreement additionally requires the residuals to be
    # small. Judged on the same two scales as the bias.
    small = statistics["rmse"] <= max(deadband * scale, mean_error)

    percent = statistics["relative_bias"] * 100.0
    if resolvable and consistent and bias > 0:
        statistics.update(
            verdict="overestimation", systematic=True,
            message=(f"systematic overestimation: modeled values exceed the measured "
                     f"ones by {bias:+.3g} on average ({percent:+.1f}%), with "
                     f"{statistics['sign_consistency'] * 100:.0f}% of the calibration "
                     f"points above the 1:1 line."))
    elif resolvable and consistent:
        statistics.update(
            verdict="underestimation", systematic=True,
            message=(f"systematic underestimation: modeled values fall short of the "
                     f"measured ones by {bias:+.3g} on average ({percent:+.1f}%), with "
                     f"{statistics['sign_consistency'] * 100:.0f}% of the calibration "
                     f"points below the 1:1 line."))
    elif not small:
        statistics.update(
            verdict="scatter", systematic=False,
            message=(f"scatter rather than bias: RMSE {statistics['rmse']:.3g} against a "
                     f"mean residual of only {bias:+.3g} ({percent:+.1f}% of the mean "
                     f"measured value), with "
                     f"{statistics['sign_consistency'] * 100:.0f}% of the calibration "
                     f"points on the same side of the 1:1 line. No single calibration "
                     f"parameter value can remove a mismatch that changes sign across "
                     f"the calibration points."))
    else:
        statistics.update(
            verdict="agreement", systematic=False,
            message=(f"agreement: mean residual {bias:+.3g} ({percent:+.1f}% of the "
                     f"mean measured value) and RMSE {statistics['rmse']:.3g}, both "
                     f"within the {deadband * 100:.0f}% deadband or the measurement "
                     f"uncertainty."))
    return statistics


def _roughness_reading(state, data):
    """Run the roughness-identifiability diagnostic on one calibration state.

    Imported lazily: :mod:`hydroBayesCal.function_pool` pulls in the mesh and VTK stack,
    which this module has no other reason to need.
    """
    try:
        from hydroBayesCal.function_pool import diagnose_roughness_identifiability
    except Exception as error:                                    # pragma: no cover
        logger_warn.warning(f"Roughness reading skipped, function_pool unavailable: "
                            f"{error}")
        return None
    try:
        # The stored block drops the ensembles, so a re-diagnosis of an archived result
        # reads the median series instead: the diagnostic takes the median over the runs
        # anyway, and the median of a single row is that row.
        return diagnose_roughness_identifiability(
            state.get("ensemble", state["modeled"]), data["measured"],
            data["quantities"])
    except Exception as error:                                    # pragma: no cover
        logger_warn.warning(f"Roughness reading skipped: {error}")
        return None


def diagnose_target_agreement(data, deadband=0.02, sign_threshold=0.6):
    """Judge modeled-vs-measured agreement per calibration target and state.

    Parameters
    ----------
    data : dict
        Output of :func:`target_agreement_data`.
    deadband : float
        A mean residual below this fraction of the mean absolute measured value counts as
        agreement. Default 0.02, the value the roughness diagnostic uses.
    sign_threshold : float
        Fraction of calibration points that must share the sign of the mean residual for
        the offset to count as systematic rather than scatter. Default 0.6.

    Returns
    -------
    dict
        Keys ``states`` (per state: per-target statistics under the target name, plus the
        ``roughness`` reading), ``verdict`` (``calibrated``, ``systematic_deviation``,
        ``scatter_dominated`` or ``unavailable``), ``improved`` (bool or ``None``,
        whether calibration reduced the total RMSE), ``message`` and ``recommendation``.
    """
    quantities = data["quantities"]
    n_quantities = data["n_quantities"]
    diagnosis = {"states": {}, "targets": list(quantities), "verdict": "unavailable",
                 "improved": None, "message": "", "recommendation": ""}

    for key, state in data["states"].items():
        per_target = {}
        for index, quantity in enumerate(quantities):
            per_target[quantity] = _target_statistics(
                _quantity_columns(state["modeled"], n_quantities, index),
                _quantity_columns(data["measured"], n_quantities, index),
                _quantity_columns(data["errors"], n_quantities, index),
                deadband=deadband,
                sign_threshold=sign_threshold,
            )
        diagnosis["states"][key] = {
            "label": state["label"],
            "n_members": state["n_members"],
            "targets": per_target,
            "roughness": _roughness_reading(state, data),
        }

    final = diagnosis["states"].get("post") or diagnosis["states"].get("pre")
    final_key = "post" if "post" in diagnosis["states"] else "pre"
    statistics = final["targets"]

    systematic = [name for name, target in statistics.items() if target["systematic"]]
    scattered = [name for name, target in statistics.items()
                 if target["verdict"] == "scatter"]

    if "pre" in diagnosis["states"] and "post" in diagnosis["states"]:
        rmse_pre = np.nanmean([target["rmse"]
                               for target in diagnosis["states"]["pre"]["targets"].values()])
        rmse_post = np.nanmean([target["rmse"] for target in statistics.values()])
        diagnosis["improved"] = bool(np.isfinite(rmse_pre) and np.isfinite(rmse_post)
                                     and rmse_post < rmse_pre)
        diagnosis["rmse_pre"] = float(rmse_pre)
        diagnosis["rmse_post"] = float(rmse_post)

    state_name = STATE_LABELS[final_key]
    if systematic:
        directions = ", ".join(f"{name} ({statistics[name]['verdict']}, "
                               f"{statistics[name]['relative_bias'] * 100:+.1f}%)"
                               for name in systematic)
        diagnosis.update(
            verdict="systematic_deviation",
            message=(f"{state_name}, the model deviates systematically for "
                     f"{len(systematic)} of {len(statistics)} calibration targets: "
                     f"{directions}."),
            recommendation=("a systematic offset is still a calibration-parameter "
                            "problem: check the roughness reading below and, if "
                            "roughness is identifiable, whether its posterior optimum "
                            "pins at a prior bound (widen the bound) rather than "
                            "sitting inside the range"))
    elif scattered:
        diagnosis.update(
            verdict="scatter_dominated",
            message=(f"{state_name}, no calibration target shows a systematic offset, "
                     f"but the residuals of {', '.join(scattered)} change sign across "
                     f"the calibration points."),
            recommendation=("the remaining mismatch is not reachable by any global "
                            "calibration-parameter value; look at the mesh, the "
                            "boundary conditions, the model structure or the "
                            "measurements at the deviating points"))
    else:
        diagnosis.update(
            verdict="calibrated",
            message=(f"{state_name}, every calibration target agrees with the "
                     f"measurements within the deadband and the measurement "
                     f"uncertainty."),
            recommendation="no further parameter adjustment is indicated by the residuals")

    roughness = final.get("roughness") or {}
    if roughness.get("message"):
        diagnosis["message"] += f" Roughness reading {state_name}: {roughness['message']}"
    return diagnosis


def log_target_agreement(diagnosis, logger_obj=None):
    """Log a :func:`diagnose_target_agreement` result at the right level.

    A ``systematic_deviation`` verdict is a WARNING (the calibrated model is biased
    against its own calibration targets); everything else is INFO. Returns the diagnosis
    so it can be chained."""
    log = logger_obj or logger
    header = "CALIBRATION-TARGET AGREEMENT: "
    write = log.warning if diagnosis.get("verdict") == "systematic_deviation" else log.info
    write(header + (diagnosis.get("message") or "no verdict"))

    for key in ("pre", "post"):
        state = diagnosis.get("states", {}).get(key)
        if not state:
            continue
        log.info(f"  {state['label']} ({state['n_members']} ensemble members):")
        for name, target in state["targets"].items():
            log.info(f"    {name}: {target['message']}")

    if diagnosis.get("improved") is not None:
        if diagnosis["improved"]:
            log.info(f"  -> calibration reduced the mean RMSE over the calibration "
                     f"targets from {diagnosis['rmse_pre']:.3g} to "
                     f"{diagnosis['rmse_post']:.3g}.")
        else:
            log.warning(f"  -> calibration did NOT reduce the mean RMSE over the "
                        f"calibration targets ({diagnosis['rmse_pre']:.3g} before, "
                        f"{diagnosis['rmse_post']:.3g} after). The calibration "
                        f"parameters cannot reach the measurements from within their "
                        f"prior ranges.")
    if diagnosis.get("recommendation"):
        write(f"  -> recommendation: {diagnosis['recommendation']}")
    return diagnosis


def plot_target_agreement(data, diagnosis=None, results_folder_path=None,
                          variable_name="", **kwargs):
    """Draw the modeled-vs-measured scatter.

    Thin wrapper that keeps matplotlib out of the import path of this module; see
    :meth:`~hydroBayesCal.visualize.agreement_plots.AgreementPlots.plot_target_agreement`
    for the parameters. The same method is available on
    :class:`~hydroBayesCal.visualize.BayesianPlotter`.
    """
    from hydroBayesCal.visualize.agreement_plots import AgreementPlotter

    plotter = AgreementPlotter(results_folder_path=results_folder_path or "",
                               variable_name=variable_name)
    return plotter.plot_target_agreement(data, diagnosis=diagnosis, **kwargs)


def finalize_target_agreement(
        complex_model,
        bayesian_dict=None,
        initial_model_outputs=None,
        posterior_predictions=None,
        errors=None,
        make_plot=True,
        deadband=0.02,
        sign_threshold=0.6,
):
    """Post-processing step at the end of a calibration: diagnose, log, store and plot.

    Called by the ``bal_*`` drivers once the BAL loop has finished. It is report-only and
    never raises: a calibration that has just spent hours of solver time must not be lost
    to a plotting or file-system error, so every failure is logged as a warning and the
    driver returns normally.

    Parameters
    ----------
    complex_model : object
        The model instance, read for ``observations``, ``calibration_quantities``,
        ``asr_dir`` and ``calibration_folder``.
    bayesian_dict : dict, optional
        The BAL dictionary. When given, the result is stored under :data:`AGREEMENT_KEY`
        and ``BAL_dictionary.pkl`` is rewritten, so the plot can be reproduced later from
        the archived results alone.
    initial_model_outputs : array-like, optional
        Full-complexity outputs of the initial design, ``[n_runs, nloc * nq]``, i.e. the
        state *before* calibration.
    posterior_predictions : array-like, optional
        ``BayesianInference.posterior_output`` of the last BAL iteration, i.e. the
        surrogate predictions at the accepted parameter sets, the state *after*
        calibration.
    errors : array-like, optional
        Measurement standard deviations for the error bars. Defaults to
        :func:`measurement_error_bars` of ``complex_model``.
    make_plot : bool
        Whether to write the figure. Default ``True``.
    deadband, sign_threshold : float
        Passed to :func:`diagnose_target_agreement`.

    Returns
    -------
    dict or None
        The diagnosis, or ``None`` if it could not be produced.
    """
    try:
        quantities = list(getattr(complex_model, "calibration_quantities", []) or [])
        data = target_agreement_data(
            observations=complex_model.observations,
            calibration_quantities=quantities,
            errors=(measurement_error_bars(complex_model) if errors is None else errors),
            initial_model_outputs=initial_model_outputs,
            posterior_predictions=posterior_predictions,
        )
        diagnosis = log_target_agreement(
            diagnose_target_agreement(data, deadband=deadband,
                                      sign_threshold=sign_threshold))
    except Exception as error:
        logger_warn.warning(f"Calibration-target agreement skipped: {error}")
        return None

    if bayesian_dict is not None:
        try:
            # The ensembles are the bulky part and are already stored elsewhere (model
            # outputs as JSON/CSV, posteriors in the dictionary); only the series the
            # plot needs are kept here.
            bayesian_dict[AGREEMENT_KEY] = {
                "data": {
                    **{key: value for key, value in data.items() if key != "states"},
                    "states": {key: {name: value for name, value in state.items()
                                     if name != "ensemble"}
                               for key, state in data["states"].items()},
                },
                "diagnosis": diagnosis,
            }
            folder = getattr(complex_model, "calibration_folder", None)
            if folder:
                with open(os.path.join(folder, "BAL_dictionary.pkl"), "wb") as archive:
                    pickle.dump(bayesian_dict, archive)
        except Exception as error:
            logger_warn.warning(f"Calibration-target agreement not stored in "
                                f"BAL_dictionary.pkl: {error}")

    if make_plot:
        try:
            paths = plot_target_agreement(
                data, diagnosis=diagnosis,
                results_folder_path=getattr(complex_model, "asr_dir", ""),
                variable_name="_".join(quantities))
            for path in paths:
                logger.info(f"Calibration-target agreement plot written to {path}")
        except Exception as error:
            logger_warn.warning(f"Calibration-target agreement plot skipped: {error}")
    return diagnosis
