"""
Correct a fixed Vectrino probe pitch in the streamwise-vertical plane.

The supplied pitch angle is the measured velocity angle:

    alpha = atan2(mean_w, mean_u)

The inverse coordinate rotation is:

    u_corrected = u*cos(alpha) + w*sin(alpha)
    w_corrected = -u*sin(alpha) + w*cos(alpha)

Author: Andres Heredia
"""

import numpy as np
import pandas as pd


def estimate_probe_pitch_angle(
    velocity_dataframe,
    u_column="u (m/s)",
    vertical_columns=("w1 (m/s)", "w2 (m/s)"),
):
    """
    Estimate the measured pitch angle from mean u and mean w.

    Use only a reference measurement where the physical mean vertical
    velocity is expected to be approximately zero.

    Returns
    -------
    float
        Measured pitch angle in degrees.
    """
    required_columns = [
        u_column,
        *vertical_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in velocity_dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing velocity columns: {missing_columns}"
        )

    u = velocity_dataframe[
        u_column
    ].to_numpy(dtype=float)

    w_components = velocity_dataframe[
        list(vertical_columns)
    ].to_numpy(dtype=float)

    w = np.nanmean(
        w_components,
        axis=1,
    )

    mean_u = np.nanmean(u)
    mean_w = np.nanmean(w)

    if not np.isfinite(mean_u) or not np.isfinite(mean_w):
        raise ValueError(
            "Mean velocity components are not finite."
        )

    if np.isclose(mean_u, 0.0):
        raise ValueError(
            "Mean streamwise velocity is approximately zero. "
            "The pitch angle cannot be estimated reliably."
        )

    pitch_angle_rad = np.arctan2(
        mean_w,
        mean_u,
    )

    return float(
        np.rad2deg(pitch_angle_rad)
    )


def correct_probe_tilt_dataframe(
    velocity_dataframe,
    pitch_angle_deg,
    u_column="u (m/s)",
    vertical_columns=("w1 (m/s)", "w2 (m/s)"),
    corrected_vertical_column="w (m/s)",
    keep_original=True,
):
    """
    Correct a fixed probe pitch in the streamwise-vertical plane.

    Parameters
    ----------
    velocity_dataframe : pandas.DataFrame
        Data after Vectrino coordinate transformation and despiking.

    pitch_angle_deg : float
        Measured velocity angle:

            pitch_angle_deg = degrees(atan2(mean_w, mean_u))

        Do not reverse the sign before passing it to this function.

    u_column : str
        Streamwise velocity column.

    vertical_columns : tuple of str
        Redundant vertical velocity columns.

    corrected_vertical_column : str
        Output column containing the mean corrected vertical velocity.

    keep_original : bool
        Preserve the uncorrected velocity columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing corrected velocity components.
    """
    if not isinstance(velocity_dataframe, pd.DataFrame):
        raise TypeError(
            "velocity_dataframe must be a pandas DataFrame."
        )

    try:
        pitch_angle_deg = float(pitch_angle_deg)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "pitch_angle_deg must be numeric."
        ) from error

    if not np.isfinite(pitch_angle_deg):
        raise ValueError(
            "pitch_angle_deg must be finite."
        )

    required_columns = [
        u_column,
        *vertical_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in velocity_dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing velocity columns: {missing_columns}"
        )

    corrected_data = velocity_dataframe.copy()

    # Original instantaneous velocities.
    u_original = corrected_data[
        u_column
    ].to_numpy(dtype=float)

    w_original = corrected_data[
        list(vertical_columns)
    ].to_numpy(dtype=float)

    w_original_mean = np.nanmean(
        w_original,
        axis=1,
    )

    if keep_original:
        corrected_data[
            "u_before_tilt_correction (m/s)"
        ] = u_original

        corrected_data[
            "w_before_tilt_correction (m/s)"
        ] = w_original_mean

        for index, column in enumerate(vertical_columns):
            original_column = column.replace(
                " (m/s)",
                "_before_tilt_correction (m/s)",
            )

            corrected_data[
                original_column
            ] = w_original[:, index]

    alpha = np.deg2rad(
        pitch_angle_deg
    )

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Rotate each u-w pair independently.
    u_corrected_components = (
        u_original[:, np.newaxis] * cos_alpha
        + w_original * sin_alpha
    )

    w_corrected_components = (
        -u_original[:, np.newaxis] * sin_alpha
        + w_original * cos_alpha
    )

    # Average the two estimates produced from w1 and w2.
    u_corrected = np.nanmean(
        u_corrected_components,
        axis=1,
    )

    w_corrected = np.nanmean(
        w_corrected_components,
        axis=1,
    )

    corrected_data[
        u_column
    ] = u_corrected

    for index, column in enumerate(vertical_columns):
        corrected_data[
            column
        ] = w_corrected_components[:, index]

    corrected_data[
        corrected_vertical_column
    ] = w_corrected

    return corrected_data