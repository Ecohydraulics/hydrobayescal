"""
Vertical profile plots and log-law fitting for Vectrino summary data.

This module reads summary_mean_velocities_and_tke.csv, groups measurements
with the same root name but different depth suffixes, plots Ux, Uy, and Uz
as vertical profiles, and fits a rough-wall logarithmic velocity profile
to Ux using a fixed ks and fitted shear velocity u_star.

Velocity processing and log-law fitting are performed in m/s.
Velocities are converted to cm/s only for plotting.

The plotting vertical axis can be:
    - z above bed in cm
    - normalized z/h, if total_water_depth_m is provided

Author: Andres Heredia
"""

import os
import re

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ======================================================================
# PLOTTING CONSTANTS
# ======================================================================

# Convert velocities from m/s to cm/s only when generating figures.
VELOCITY_PLOT_FACTOR = 100.0
VELOCITY_PLOT_UNIT = "cm/s"

# Fixed plotting range for the transverse and vertical velocity components.
# Ux remains automatically scaled because its magnitude is substantially larger.
UY_UZ_X_LIMITS_CM_PER_S = (-3.0, 3.0)
UY_UZ_X_TICKS_CM_PER_S = np.linspace(-3.0, 3.0, 5)


# ======================================================================
# FILE-NAME AND PROFILE UTILITIES
# ======================================================================


def _extract_depth_and_profile_id(case_name):
    """
    Extract the measurement elevation and root profile ID.

    Examples
    --------
    xyz_+05_2cm_CL
        profile_id = xyz_+05_CL
        depth_cm = 2

    xyz_+05_10cm_CL
        profile_id = xyz_+05_CL
        depth_cm = 10

    xyz_+1_4cm_offCL20
        profile_id = xyz_+1_offCL20
        depth_cm = 4

    Parameters
    ----------
    case_name : str
        Case name containing a depth suffix such as "_2cm".

    Returns
    -------
    tuple
        profile_id : str
            Case name without the depth suffix.
        depth_cm : float
            Extracted elevation or depth in centimeters.
    """
    text = str(case_name)

    match = re.search(
        r"_(?P<depth_cm>\d+(?:\.\d+)?)cm(?=_|$)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return text, np.nan

    depth_cm = float(match.group("depth_cm"))

    # Remove only the "_2cm", "_4cm", "_10cm", etc. section.
    profile_id = text[:match.start()] + text[match.end():]

    # Remove accidental duplicate underscores.
    profile_id = re.sub(
        r"__+",
        "_",
        profile_id,
    ).strip("_")

    return profile_id, depth_cm


def _sanitize_filename(text):
    """
    Convert a profile name into a safe output filename.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Sanitized filename component.
    """
    return re.sub(
        r"[^A-Za-z0-9_.+-]+",
        "_",
        str(text),
    )


# ======================================================================
# READ AND PREPARE VERTICAL-PROFILE DATA
# ======================================================================


def _read_vertical_profile_table(
    summary_csv_file,
    depth_suffix_is_z_above_bed=True,
    water_depth_m=None,
    total_water_depth_m=None,
):
    """
    Read the summary CSV and add vertical-profile coordinates.

    The returned table contains:
        - profile_id
        - depth_cm
        - z_m
        - z_cm
        - z_over_h

    Parameters
    ----------
    summary_csv_file : str
        Full path to summary_mean_velocities_and_tke.csv.

    depth_suffix_is_z_above_bed : bool, optional
        If True, a suffix such as 2cm is interpreted as an elevation
        z = 0.02 m above the bed.

        If False, the suffix is interpreted as a depth below the water
        surface, and z is calculated as:

            z = water_depth_m - depth

    water_depth_m : float or None, optional
        Required only when depth_suffix_is_z_above_bed=False.

    total_water_depth_m : float or None, optional
        Total flow depth h in meters. If provided, z/h is calculated.

    Returns
    -------
    pandas.DataFrame
        Processed table with vertical-coordinate columns.
    """
    if not os.path.exists(summary_csv_file):
        raise FileNotFoundError(
            f"Summary CSV file does not exist: {summary_csv_file}"
        )

    df = pd.read_csv(
        summary_csv_file,
        sep=None,
        engine="python",
    )

    required_columns = [
        "case_name",
        "u_mean (m/s)",
        "v_mean (m/s)",
        "w_mean (m/s)",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The summary CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    parsed_names = df["case_name"].apply(
        _extract_depth_and_profile_id
    )

    df["profile_id"] = parsed_names.apply(
        lambda item: item[0]
    )

    df["depth_cm"] = parsed_names.apply(
        lambda item: item[1]
    )

    numeric_columns = [
        "depth_cm",
        "u_mean (m/s)",
        "v_mean (m/s)",
        "w_mean (m/s)",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "profile_id",
            "depth_cm",
            "u_mean (m/s)",
            "v_mean (m/s)",
            "w_mean (m/s)",
        ]
    ).copy()

    if depth_suffix_is_z_above_bed:
        # Measurement names such as 2cm, 4cm, and 6cm are interpreted
        # as elevations above the bed.
        df["z_m"] = df["depth_cm"] / 100.0

    else:
        # Measurement names are interpreted as depths below the
        # water surface.
        if water_depth_m is None:
            raise ValueError(
                "water_depth_m must be provided when "
                "depth_suffix_is_z_above_bed=False."
            )

        if water_depth_m <= 0.0:
            raise ValueError(
                "water_depth_m must be positive."
            )

        df["z_m"] = (
            water_depth_m
            - df["depth_cm"] / 100.0
        )

    df["z_cm"] = df["z_m"] * 100.0

    if total_water_depth_m is not None:
        if total_water_depth_m <= 0.0:
            raise ValueError(
                "total_water_depth_m must be positive."
            )

        df["z_over_h"] = (
            df["z_m"]
            / total_water_depth_m
        )

    else:
        df["z_over_h"] = np.nan

    # Remove invalid points located at or below z = 0.
    df = df[
        df["z_m"] > 0.0
    ].copy()

    df = df.sort_values(
        [
            "profile_id",
            "z_m",
        ]
    ).reset_index(drop=True)

    return df


# ======================================================================
# LOG-LAW FITTING
# ======================================================================


def _fit_log_law_fixed_ks(
    profile_df,
    initial_ks_m,
    kappa=0.41,
    fit_depth_limits_cm=None,
    excluded_depths_cm=None,
):
    """
    Fit shear velocity u_star using a fixed equivalent roughness ks.

    Rough-wall logarithmic law:

        U(z) = u_star / kappa * ln(30 z / ks)

    The equivalent sand roughness ks is fixed. Only u_star is fitted
    using least squares.

    Important
    ---------
    The fitting is performed using:
        - z in meters
        - velocity in m/s
        - ks in meters

    Conversion to cm/s is performed only later when plotting.

    Parameters
    ----------
    profile_df : pandas.DataFrame
        Data for one vertical profile.

    initial_ks_m : float
        Fixed equivalent sand roughness height in meters.

    kappa : float, optional
        von Karman constant.

    fit_depth_limits_cm : tuple or None, optional
        Optional fitting interval:

            (z_min_cm, z_max_cm)

        Example:

            (2, 8)

        Use None to fit all available elevations.

    excluded_depths_cm : list or None, optional
        Optional elevations to exclude from fitting.

        Example:

            [10]

    Returns
    -------
    dict
        Fitting results.
    """
    if initial_ks_m <= 0.0:
        raise ValueError(
            "initial_ks_m must be positive."
        )

    if kappa <= 0.0:
        raise ValueError(
            "kappa must be positive."
        )

    if fit_depth_limits_cm is None:
        fit_depth_limits_cm = (
            None,
            None,
        )

    if excluded_depths_cm is None:
        excluded_depths_cm = []

    z_m = profile_df[
        "z_m"
    ].to_numpy(dtype=float)

    z_cm = profile_df[
        "z_cm"
    ].to_numpy(dtype=float)

    ux = profile_df[
        "u_mean (m/s)"
    ].to_numpy(dtype=float)

    fit_mask = (
        np.isfinite(z_m)
        & np.isfinite(ux)
        & (z_m > 0.0)
    )

    lower_cm, upper_cm = fit_depth_limits_cm

    if lower_cm is not None:
        fit_mask &= z_cm >= lower_cm

    if upper_cm is not None:
        fit_mask &= z_cm <= upper_cm

    for depth_cm in excluded_depths_cm:
        fit_mask &= ~np.isclose(
            z_cm,
            depth_cm,
        )

    log_argument = (
        30.0
        * z_m
        / initial_ks_m
    )

    # Require a positive and hydraulically meaningful logarithm.
    fit_mask &= log_argument > 1.0

    n_fit = int(
        np.sum(fit_mask)
    )

    empty_result = {
        "success": False,
        "message": "",
        "u_star_m_per_s": np.nan,
        "rmse_m_per_s": np.nan,
        "r2": np.nan,
        "n_fit": n_fit,
        "z_smooth_m": np.array([]),
        "z_smooth_cm": np.array([]),
        "u_log_smooth": np.array([]),
    }

    if n_fit < 1:
        empty_result["message"] = (
            "No valid points available for log-law fitting."
        )

        return empty_result

    phi = (
        np.log(log_argument[fit_mask])
        / kappa
    )

    denominator = np.sum(
        phi ** 2
    )

    if denominator <= 0.0:
        empty_result["message"] = (
            "Degenerate log-law basis."
        )

        return empty_result

    # Least-squares fitting:
    #
    # Ux = u_star * phi
    u_star = (
        np.sum(
            phi * ux[fit_mask]
        )
        / denominator
    )

    # Report positive shear velocity for normal open-channel flow.
    u_star = max(
        float(u_star),
        0.0,
    )

    ux_predicted = (
        u_star
        * phi
    )

    residuals = (
        ux[fit_mask]
        - ux_predicted
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    ss_res = float(
        np.sum(
            residuals ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                ux[fit_mask]
                - np.mean(ux[fit_mask])
            ) ** 2
        )
    )

    if ss_tot > 0.0:
        r2 = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r2 = np.nan

    z_min = np.min(
        z_m[fit_mask]
    )

    z_max = np.max(
        z_m[fit_mask]
    )

    z_smooth_m = np.linspace(
        z_min,
        z_max,
        200,
    )

    u_log_smooth = (
        u_star
        / kappa
        * np.log(
            30.0
            * z_smooth_m
            / initial_ks_m
        )
    )

    return {
        "success": True,
        "message": "OK",
        "u_star_m_per_s": u_star,
        "rmse_m_per_s": rmse,
        "r2": r2,
        "n_fit": n_fit,
        "z_smooth_m": z_smooth_m,
        "z_smooth_cm": z_smooth_m * 100.0,
        "u_log_smooth": u_log_smooth,
    }


# ======================================================================
# VERTICAL-PROFILE PLOTTING
# ======================================================================


def extract_vertical_plots(
    summary_csv_file,
    results_directory,
    initial_ks_m,
    kappa=0.41,
    depth_suffix_is_z_above_bed=True,
    water_depth_m=None,
    total_water_depth_m=None,
    fit_depth_limits_cm=None,
    excluded_depths_cm=None,
    show_plot=False,
):
    """
    Plot Ux, Uy, and Uz vertical profiles and fit a log law to Ux.

    The velocity data are read and fitted in m/s. They are multiplied by
    100 only for plotting, so figure velocities are displayed in cm/s.

    Parameters
    ----------
    summary_csv_file : str
        Full path to summary_mean_velocities_and_tke.csv.

    results_directory : str
        Main results directory. A folder named "vertical_profiles"
        is created inside this directory.

    initial_ks_m : float
        Fixed equivalent sand roughness height ks in meters.

    kappa : float, optional
        von Karman constant. Default is 0.41.

    depth_suffix_is_z_above_bed : bool, optional
        True if suffixes such as 2cm, 4cm, and 6cm represent elevations
        above the bed.

        False if they represent depths below the water surface.

    water_depth_m : float or None, optional
        Required only when depth_suffix_is_z_above_bed=False.

    total_water_depth_m : float or None, optional
        Total flow depth h in meters.

        If provided, the vertical axis is z/h.

        If None, the vertical axis is z above the bed in centimeters.

    fit_depth_limits_cm : tuple or None, optional
        Optional log-law fitting limits:

            (z_min_cm, z_max_cm)

        Example:

            (2, 8)

    excluded_depths_cm : list or None, optional
        Optional elevations excluded from the log-law fit.

        Example:

            [10]

    show_plot : bool, optional
        If True, display figures interactively.

        If False, figures are saved without being displayed.

    Returns
    -------
    pandas.DataFrame
        Table containing fitted u_star, RMSE, R2, number of fitting
        points, fitting status, and figure path for each profile.
    """
    if fit_depth_limits_cm is None:
        fit_depth_limits_cm = (
            None,
            None,
        )

    if excluded_depths_cm is None:
        excluded_depths_cm = []

    profile_df = _read_vertical_profile_table(
        summary_csv_file=summary_csv_file,
        depth_suffix_is_z_above_bed=depth_suffix_is_z_above_bed,
        water_depth_m=water_depth_m,
        total_water_depth_m=total_water_depth_m,
    )

    vertical_output_directory = os.path.join(
        results_directory,
        "vertical_profiles",
    )

    os.makedirs(
        vertical_output_directory,
        exist_ok=True,
    )

    if total_water_depth_m is not None:
        y_column = "z_over_h"
        y_label = r"$z/h$ (-)"
        y_axis_name_for_table = "z_over_h"

    else:
        y_column = "z_cm"
        y_label = "z above bed (cm)"
        y_axis_name_for_table = "z_cm"

    fit_rows = []

    for profile_id, group_df in profile_df.groupby(
        "profile_id"
    ):
        group_df = group_df.sort_values(
            "z_m"
        ).copy()

        fit_result = _fit_log_law_fixed_ks(
            profile_df=group_df,
            initial_ks_m=initial_ks_m,
            kappa=kappa,
            fit_depth_limits_cm=fit_depth_limits_cm,
            excluded_depths_cm=excluded_depths_cm,
        )

        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(13.0, 5.5),
            sharey=True,
        )

        plot_definitions = [
            (
                "u_mean (m/s)",
                "Ux",
            ),
            (
                "v_mean (m/s)",
                "Uy",
            ),
            (
                "w_mean (m/s)",
                "Uz",
            ),
        ]

        for ax, (
            velocity_column,
            velocity_label,
        ) in zip(
            axes,
            plot_definitions,
        ):
            # Convert measured velocities from m/s to cm/s
            # only for the figure.
            velocity_plot_values = (
                group_df[velocity_column]
                * VELOCITY_PLOT_FACTOR
            )

            ax.plot(
                velocity_plot_values,
                group_df[y_column],
                marker="o",
                markersize=5.5,
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
                label=f"{velocity_label} measured",
            )

            ax.axvline(
                0.0,
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
            )

            ax.set_xlabel(
                f"{velocity_label} ({VELOCITY_PLOT_UNIT})"
            )

            ax.grid(
                True,
                alpha=0.35,
                linewidth=0.45,
            )

            # Use the same fixed symmetric range for Uy and Uz.
            if velocity_label in {"Uy", "Uz"}:
                ax.set_xlim(
                    UY_UZ_X_LIMITS_CM_PER_S
                )
                ax.set_xticks(
                    UY_UZ_X_TICKS_CM_PER_S
                )

            # Add log-law fit only to the streamwise velocity plot.
            if (
                velocity_label == "Ux"
                and fit_result["success"]
            ):
                if total_water_depth_m is not None:
                    z_log_plot = (
                        fit_result["z_smooth_m"]
                        / total_water_depth_m
                    )

                else:
                    z_log_plot = (
                        fit_result["z_smooth_cm"]
                    )

                # Convert fitted velocity from m/s to cm/s
                # only for plotting.
                u_log_plot = (
                    fit_result["u_log_smooth"]
                    * VELOCITY_PLOT_FACTOR
                )

                u_star_cm_per_s = (
                    fit_result["u_star_m_per_s"]
                    * VELOCITY_PLOT_FACTOR
                )

                ax.plot(
                    u_log_plot,
                    z_log_plot,
                    linestyle=":",
                    linewidth=2.0,
                    label=(
                        "log-law fit\n"
                        f"$u_*$ = "
                        f"{u_star_cm_per_s:.2f} cm/s"
                    ),
                )

            ax.legend(
                frameon=False,
                fontsize=9,
            )

        axes[0].set_ylabel(
            y_label
        )

        if total_water_depth_m is not None:
            axes[0].set_ylim(
                bottom=0.0,
                top=max(
                    1.0,
                    (
                        group_df["z_over_h"].max()
                        * 1.05
                    ),
                ),
            )

        if fit_result["success"]:
            u_star_cm_per_s = (
                fit_result["u_star_m_per_s"]
                * VELOCITY_PLOT_FACTOR
            )

            rmse_cm_per_s = (
                fit_result["rmse_m_per_s"]
                * VELOCITY_PLOT_FACTOR
            )

            title = (
                f"{profile_id}\n"
                f"fixed $k_s$ = {initial_ks_m:.4g} m, "
                f"fitted $u_*$ = "
                f"{u_star_cm_per_s:.2f} cm/s, "
                f"RMSE = "
                f"{rmse_cm_per_s:.2f} cm/s, "
                f"$R^2$ = {fit_result['r2']:.3f}"
            )

        else:
            title = (
                f"{profile_id}\n"
                f"Log-law fit failed: "
                f"{fit_result['message']}"
            )

        fig.suptitle(
            title
        )

        fig.tight_layout()

        figure_file = os.path.join(
            vertical_output_directory,
            (
                "vertical_profile_loglaw_"
                f"{_sanitize_filename(profile_id)}.svg"
            ),
        )

        fig.savefig(
            figure_file,
            format="svg",
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()

        plt.close(fig)

        # Keep original SI results in the CSV.
        fit_rows.append({
            "profile_id": profile_id,
            "initial_ks_m": initial_ks_m,
            "total_water_depth_m": total_water_depth_m,
            "vertical_axis": y_axis_name_for_table,
            "u_star_m_per_s": fit_result[
                "u_star_m_per_s"
            ],
            "rmse_m_per_s": fit_result[
                "rmse_m_per_s"
            ],
            "r2": fit_result["r2"],
            "n_points_fit": fit_result["n_fit"],
            "fit_status": fit_result["message"],
            "figure_file": figure_file,
        })

        print(
            f"   - saved vertical profile plot: "
            f"{figure_file}"
        )

    fit_df = pd.DataFrame(
        fit_rows
    )

    fit_output_file = os.path.join(
        vertical_output_directory,
        "loglaw_fitted_shear_velocity.csv",
    )

    fit_df.to_csv(
        fit_output_file,
        index=False,
    )

    print(
        "\n============================================================"
    )
    print(
        " Vertical profile plots and log-law fitting finished"
    )
    print(
        f" Output directory: {vertical_output_directory}"
    )
    print(
        f" Fit table       : {fit_output_file}"
    )
    print(
        " Figure velocity units: cm/s"
    )
    print(
        " Fit-table velocity units: m/s"
    )
    print(
        "============================================================"
    )

    return fit_df