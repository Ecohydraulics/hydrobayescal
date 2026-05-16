"""
Despiking utilities for Nortek Vectrino / ADV velocity records.

This module applies an iterative phase-space threshold despiking method
in the spirit of Goring and Nikora (2002). It is intended to be applied
to scalar velocity time series before computing means, variances, TKE,
Reynolds stresses, or plotting cleaned instantaneous velocities.

Author: Andres Heredia / post-processing helper
"""

# from __future__ import annotations

import numpy as np
import pandas as pd


def _find_time_column(df: pd.DataFrame) -> str | None:
    """
    Try to identify a reasonable time column automatically.

    If no known time column is found, the despiking algorithm falls back to
    sample index spacing, which is acceptable for evenly sampled ADV records.
    """

    candidate_columns = [
        "Time (s)",
        "time (s)",
        "TimeStamp (s)",
        "Timestamp (s)",
        "Profiles_TimeStamp (s)",
        "VelocityHeader_TimeStamp (s)",
        "HostTime_start (s)",
        "Profiles_HostTime_start (s)",
    ]

    for col in candidate_columns:
        if col in df.columns:
            return col

    return None


def _to_float_array(values) -> np.ndarray:
    """
    Convert a pandas Series or array-like object to a 1D float array.
    Non-numeric values are converted to NaN.
    """

    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _linear_interpolate_nan(y: np.ndarray) -> np.ndarray:
    """
    Replace NaN values using linear interpolation.

    Edge NaNs are replaced by the nearest finite value. If fewer than two
    finite values exist, the input is returned unchanged.
    """

    y = np.asarray(y, dtype=float).copy()
    x = np.arange(len(y))

    valid = np.isfinite(y)

    if np.sum(valid) == 0:
        return y

    if np.sum(valid) == 1:
        y[~valid] = y[valid][0]
        return y

    y[~valid] = np.interp(x[~valid], x[valid], y[valid])

    return y


def _compute_derivatives(
    y: np.ndarray,
    time: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute first and second derivatives of a scalar time series.

    If a valid time vector is supplied, derivatives are computed with respect
    to time. Otherwise, unit sample spacing is assumed.
    """

    if time is None:
        dy = np.gradient(y)
        ddy = np.gradient(dy)
        return dy, ddy

    time = np.asarray(time, dtype=float)

    if len(time) != len(y):
        dy = np.gradient(y)
        ddy = np.gradient(dy)
        return dy, ddy

    if not np.all(np.isfinite(time)):
        dy = np.gradient(y)
        ddy = np.gradient(dy)
        return dy, ddy

    if np.any(np.diff(time) <= 0):
        dy = np.gradient(y)
        ddy = np.gradient(dy)
        return dy, ddy

    dy = np.gradient(y, time)
    ddy = np.gradient(dy, time)

    return dy, ddy


def phase_space_despike_series(
    values,
    time=None,
    max_iterations: int = 20,
    min_samples: int = 20,
    threshold_factor: float = 1.0,
    replacement: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Despike one scalar velocity time series.

    Parameters
    ----------
    values : array-like
        Velocity time series.

    time : array-like or None, optional
        Time vector in seconds. If None, unit sample spacing is used.

    max_iterations : int, optional
        Maximum number of despiking iterations.

    min_samples : int, optional
        Minimum number of finite samples required for despiking.

    threshold_factor : float, optional
        Multiplier applied to the universal threshold sqrt(2 ln N).
        Use values > 1.0 for less aggressive despiking and values < 1.0
        for more aggressive despiking.

    replacement : {"linear", "nan"}, optional
        How detected spikes are treated.
        "linear" replaces spikes by linear interpolation.
        "nan" replaces spikes by NaN.

    Returns
    -------
    cleaned_values : np.ndarray
        Despiked velocity time series.

    spike_mask : np.ndarray of bool
        Boolean mask where True means the original sample was identified
        as a spike.
    """

    y_original = _to_float_array(values)

    if time is not None:
        time_array = _to_float_array(time)
    else:
        time_array = None

    n = len(y_original)

    if n == 0:
        return y_original, np.zeros(0, dtype=bool)

    finite_original = np.isfinite(y_original)

    if np.sum(finite_original) < min_samples:
        return y_original.copy(), np.zeros(n, dtype=bool)

    if replacement not in {"linear", "nan"}:
        raise ValueError(
            "ERROR: replacement must be either 'linear' or 'nan'."
        )

    spike_mask = np.zeros(n, dtype=bool)

    # Universal threshold used in phase-space despiking.
    # This is a common threshold form for ADV despiking applications.
    n_valid = np.sum(finite_original)
    universal_threshold = threshold_factor * np.sqrt(2.0 * np.log(n_valid))

    if not np.isfinite(universal_threshold) or universal_threshold <= 0:
        return y_original.copy(), spike_mask

    for _ in range(max_iterations):

        # Temporarily remove currently detected spikes and interpolate
        # so derivatives can be computed without spike contamination.
        y_work = y_original.copy()
        y_work[spike_mask] = np.nan
        y_filled = _linear_interpolate_nan(y_work)

        if not np.all(np.isfinite(y_filled)):
            break

        accepted = finite_original & (~spike_mask)

        if np.sum(accepted) < min_samples:
            break

        # Velocity fluctuation.
        y_mean = np.mean(y_filled[accepted])
        fluct = y_filled - y_mean

        # First and second derivatives.
        d1, d2 = _compute_derivatives(fluct, time_array)

        phase_space = np.column_stack([fluct, d1, d2])

        valid_phase = (
            finite_original
            & np.all(np.isfinite(phase_space), axis=1)
        )

        accepted_phase = valid_phase & (~spike_mask)

        if np.sum(accepted_phase) < min_samples:
            break

        phase_accepted = phase_space[accepted_phase, :]

        center = np.mean(phase_accepted, axis=0)
        covariance = np.cov(phase_accepted, rowvar=False)

        if covariance.shape != (3, 3):
            break

        # Regularize covariance to avoid singular matrix problems.
        covariance = covariance + np.eye(3) * 1.0e-12

        try:
            inverse_covariance = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inverse_covariance = np.linalg.pinv(covariance)

        difference = phase_space - center

        mahalanobis_distance_squared = np.einsum(
            "ij,jk,ik->i",
            difference,
            inverse_covariance,
            difference,
        )

        new_spikes = (
            valid_phase
            & (mahalanobis_distance_squared > universal_threshold ** 2)
        )

        updated_spike_mask = spike_mask | new_spikes

        if np.array_equal(updated_spike_mask, spike_mask):
            break

        spike_mask = updated_spike_mask

    cleaned = y_original.copy()

    if replacement == "linear":
        cleaned[spike_mask] = np.nan
        cleaned = _linear_interpolate_nan(cleaned)
    elif replacement == "nan":
        cleaned[spike_mask] = np.nan

    return cleaned, spike_mask


def despike_velocity_dataframe(
    df: pd.DataFrame,
    velocity_columns: list[str] | tuple[str, ...],
    time_column: str | None = None,
    max_iterations: int = 20,
    min_samples: int = 20,
    threshold_factor: float = 1.0,
    replacement: str = "linear",
    keep_original: bool = True,
    add_flag_columns: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Despike selected velocity columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    velocity_columns : list or tuple of str
        Velocity columns to despike, for example:
        ["u (m/s)", "v (m/s)", "w1 (m/s)", "w2 (m/s)"]

    time_column : str or None, optional
        Time column in seconds. If None, the function tries to detect one.
        If no suitable time column exists, unit sample spacing is used.

    max_iterations : int, optional
        Maximum number of phase-space despiking iterations.

    min_samples : int, optional
        Minimum number of finite samples required in each velocity series.

    threshold_factor : float, optional
        Multiplier for the universal threshold sqrt(2 ln N).
        Larger values are less aggressive.
        Smaller values are more aggressive.

    replacement : {"linear", "nan"}, optional
        Spike replacement method.

    keep_original : bool, optional
        If True, stores original columns as "<column> raw" before overwriting.

    add_flag_columns : bool, optional
        If True, adds Boolean spike flag columns named "<column> despike flag".

    inplace : bool, optional
        If True, modifies the input DataFrame directly.

    Returns
    -------
    pandas.DataFrame
        DataFrame with despiked velocity columns.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("ERROR: df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("ERROR: Input DataFrame is empty. Cannot despike.")

    if len(velocity_columns) == 0:
        raise ValueError("ERROR: velocity_columns is empty.")

    missing_columns = [col for col in velocity_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "ERROR: Cannot despike because velocity columns are missing.\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    if inplace:
        out = df
    else:
        out = df.copy()

    if time_column is None:
        time_column = _find_time_column(out)

    if time_column is not None and time_column in out.columns:
        time_values = _to_float_array(out[time_column])
        print(f"   - despiking uses time column: {time_column}")
    else:
        time_values = None
        print("   - despiking uses unit sample spacing")

    total_spikes = 0

    for col in velocity_columns:

        original_values = _to_float_array(out[col])

        cleaned_values, spike_mask = phase_space_despike_series(
            original_values,
            time=time_values,
            max_iterations=max_iterations,
            min_samples=min_samples,
            threshold_factor=threshold_factor,
            replacement=replacement,
        )

        n_spikes = int(np.sum(spike_mask))
        total_spikes += n_spikes

        if keep_original:
            raw_col = f"{col} raw"
            if raw_col not in out.columns:
                out[raw_col] = original_values

        out[col] = cleaned_values

        if add_flag_columns:
            flag_col = f"{col} despike flag"
            out[flag_col] = spike_mask

        n_total = len(out)
        percentage = 100.0 * n_spikes / n_total if n_total > 0 else 0.0

        print(
            f"   - despiked {col}: "
            f"{n_spikes} spikes / {n_total} samples "
            f"({percentage:.2f}%)"
        )

    print(f"   - total despiked samples across columns: {total_spikes}")

    return out