"""
Vertical profile plots and log-law fitting for Vectrino summary data.

This module reads summary_mean_velocities_and_tke.csv, groups measurements
with the same root name but different depth suffixes, plots Ux, Uy, and Uz
as vertical profiles, and fits a rough-wall logarithmic velocity profile
to Ux using a fixed ks and fitted shear velocity u_star.

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


def _extract_depth_and_profile_id(case_name):
    """
    Extract the measurement depth/elevation and root profile ID.

    Examples
    --------
    xyz_+05_2cm_CL      -> profile_id = xyz_+05_CL, depth_cm = 2
    xyz_+05_10cm_CL     -> profile_id = xyz_+05_CL, depth_cm = 10
    xyz_+1_4cm_offCL20  -> profile_id = xyz_+1_offCL20, depth_cm = 4
    """
    text = str(case_name)

    match = re.search(
        r"_(?P<depth_cm>\d+(?:\.\d+)?)cm(?=_|$)",
        text,
        flags=re.IGNORECASE
    )

    if match is None:
        return text, np.nan

    depth_cm = float(match.group("depth_cm"))

    # Remove only the "_2cm", "_4cm", "_10cm", etc. part.
    profile_id = text[:match.start()] + text[match.end():]
    profile_id = re.sub(r"__+", "_", profile_id).strip("_")

    return profile_id, depth_cm


def _sanitize_filename(text):
    """Create safe filenames for saved figures."""
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text))


def _read_vertical_profile_table(
    summary_csv_file,
    depth_suffix_is_z_above_bed=True,
    water_depth_m=None,
    total_water_depth_m=None
):
    """
    Read the summary CSV and add profile_id, depth_cm, z_m, z_cm, and z_over_h.

    Parameters
    ----------
    summary_csv_file : str
        Full path to summary_mean_velocities_and_tke.csv.

    depth_suffix_is_z_above_bed : bool
        If True, a suffix like 2cm is interpreted as z = 0.02 m above the bed.
        If False, the suffix is interpreted as depth below the water surface,
        and z is computed as water_depth_m - depth.

    water_depth_m : float or None
        Required only if depth_suffix_is_z_above_bed=False.

    total_water_depth_m : float or None
        Total flow depth h [m]. If provided, z/h is computed.

    Returns
    -------
    pandas.DataFrame
        Processed table with vertical-coordinate columns.
    """
    if not os.path.exists(summary_csv_file):
        raise FileNotFoundError(
            f"Summary CSV file does not exist: {summary_csv_file}"
        )

    df = pd.read_csv(summary_csv_file, sep=None, engine="python")

    required_columns = [
        "case_name",
        "u_mean (m/s)",
        "v_mean (m/s)",
        "w_mean (m/s)",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "The summary CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    parsed = df["case_name"].apply(_extract_depth_and_profile_id)

    df["profile_id"] = parsed.apply(lambda item: item[0])
    df["depth_cm"] = parsed.apply(lambda item: item[1])

    numeric_columns = [
        "depth_cm",
        "u_mean (m/s)",
        "v_mean (m/s)",
        "w_mean (m/s)",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        # Case used in your setup:
        # 2cm, 4cm, 6cm, ... are elevations above the bed.
        df["z_m"] = df["depth_cm"] / 100.0

    else:
        # Use this only if 2cm, 4cm, ... are distances below the free surface.
        if water_depth_m is None:
            raise ValueError(
                "water_depth_m must be provided when "
                "depth_suffix_is_z_above_bed=False."
            )

        if water_depth_m <= 0.0:
            raise ValueError("water_depth_m must be positive.")

        df["z_m"] = water_depth_m - df["depth_cm"] / 100.0

    df["z_cm"] = df["z_m"] * 100.0

    if total_water_depth_m is not None:
        if total_water_depth_m <= 0.0:
            raise ValueError("total_water_depth_m must be positive.")

        df["z_over_h"] = df["z_m"] / total_water_depth_m

    else:
        df["z_over_h"] = np.nan

    df = df[df["z_m"] > 0.0].copy()
    df = df.sort_values(["profile_id", "z_m"]).reset_index(drop=True)

    return df


def _fit_log_law_fixed_ks(
    profile_df,
    initial_ks_m,
    kappa=0.41,
    fit_depth_limits_cm=None,
    excluded_depths_cm=None
):
    """
    Fit shear velocity u_star using a fixed equivalent sand roughness ks.

    Rough-wall log law:

        U(z) = u_star / kappa * ln(30 z / ks)

    Here ks is fixed and only u_star is fitted by least squares.

    Important
    ---------
    The fitting is done using physical z_m, not z/h.
    This is correct because the log law is dimensional in z and ks.

    Parameters
    ----------
    profile_df : pandas.DataFrame
        One grouped vertical profile.

    initial_ks_m : float
        Fixed equivalent sand roughness height [m].

    kappa : float
        von Karman constant.

    fit_depth_limits_cm : tuple or None
        Optional fitting range as (z_min_cm, z_max_cm).
        Example: (2, 8). Use None to fit all available depths.

    excluded_depths_cm : list or None
        Optional depths to exclude from fitting.
        Example: [10].

    Returns
    -------
    dict
        Fitting result.
    """
    if initial_ks_m <= 0.0:
        raise ValueError("initial_ks_m must be positive.")

    if fit_depth_limits_cm is None:
        fit_depth_limits_cm = (None, None)

    if excluded_depths_cm is None:
        excluded_depths_cm = []

    z_m = profile_df["z_m"].to_numpy(dtype=float)
    z_cm = profile_df["z_cm"].to_numpy(dtype=float)
    ux = profile_df["u_mean (m/s)"].to_numpy(dtype=float)

    fit_mask = np.isfinite(z_m) & np.isfinite(ux) & (z_m > 0.0)

    lower_cm, upper_cm = fit_depth_limits_cm

    if lower_cm is not None:
        fit_mask &= z_cm >= lower_cm

    if upper_cm is not None:
        fit_mask &= z_cm <= upper_cm

    for depth_cm in excluded_depths_cm:
        fit_mask &= ~np.isclose(z_cm, depth_cm)

    log_argument = 30.0 * z_m / initial_ks_m

    # Need positive, hydraulically meaningful logarithm argument.
    fit_mask &= log_argument > 1.0

    n_fit = int(np.sum(fit_mask))

    if n_fit < 1:
        return {
            "success": False,
            "message": "No valid points available for log-law fitting.",
            "u_star_m_per_s": np.nan,
            "rmse_m_per_s": np.nan,
            "r2": np.nan,
            "n_fit": n_fit,
            "z_smooth_cm": np.array([]),
            "z_smooth_over_h": np.array([]),
            "u_log_smooth": np.array([]),
        }

    phi = np.log(log_argument[fit_mask]) / kappa

    denominator = np.sum(phi ** 2)

    if denominator <= 0.0:
        return {
            "success": False,
            "message": "Degenerate log-law basis.",
            "u_star_m_per_s": np.nan,
            "rmse_m_per_s": np.nan,
            "r2": np.nan,
            "n_fit": n_fit,
            "z_smooth_cm": np.array([]),
            "z_smooth_over_h": np.array([]),
            "u_log_smooth": np.array([]),
        }

    # Least-squares fit:
    # Ux = u_star * phi
    u_star = np.sum(phi * ux[fit_mask]) / denominator

    # For normal open-channel flow, u_star is reported as a positive value.
    # Remove this line only if you intentionally want signed u_star.
    u_star = max(float(u_star), 0.0)

    ux_predicted = u_star * phi
    residuals = ux[fit_mask] - ux_predicted

    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ux[fit_mask] - np.mean(ux[fit_mask])) ** 2))

    if ss_tot > 0.0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    z_min = np.min(z_m[fit_mask])
    z_max = np.max(z_m[fit_mask])

    z_smooth_m = np.linspace(z_min, z_max, 200)

    u_log_smooth = (
        u_star / kappa
        * np.log(30.0 * z_smooth_m / initial_ks_m)
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
    show_plot=False
):
    """
    Plot Ux, Uy, and Uz vertical profiles and fit a log law to Ux.

    Parameters
    ----------
    summary_csv_file : str
        Full path to summary_mean_velocities_and_tke.csv.

    results_directory : str
        Main results directory. A folder named vertical_profiles will be created.

    initial_ks_m : float
        Fixed equivalent sand roughness height ks [m].

    kappa : float, optional
        von Karman constant. Default is 0.41.

    depth_suffix_is_z_above_bed : bool, optional
        True if suffixes like 2cm, 4cm, 6cm are elevations above the bed.
        False if they are depths below the free surface.

    water_depth_m : float or None, optional
        Required only when depth_suffix_is_z_above_bed=False.

    total_water_depth_m : float or None, optional
        Total flow depth h [m]. If provided, the plot vertical axis is z/h.
        Example for your case:
            total_water_depth_m = 0.15

    fit_depth_limits_cm : tuple or None, optional
        Optional fitting limits as (z_min_cm, z_max_cm).
        Example:
            fit_depth_limits_cm = (2, 8)

    excluded_depths_cm : list or None, optional
        Optional list of depths to exclude from the log-law fit.
        Example:
            excluded_depths_cm = [10]

    show_plot : bool, optional
        If True, show figures interactively. If False, only save figures.

    Returns
    -------
    pandas.DataFrame
        Table with fitted u_star, RMSE, R2, and figure path for each profile.
    """
    if fit_depth_limits_cm is None:
        fit_depth_limits_cm = (None, None)

    if excluded_depths_cm is None:
        excluded_depths_cm = []

    profile_df = _read_vertical_profile_table(
        summary_csv_file=summary_csv_file,
        depth_suffix_is_z_above_bed=depth_suffix_is_z_above_bed,
        water_depth_m=water_depth_m,
        total_water_depth_m=total_water_depth_m
    )

    vertical_output_directory = os.path.join(
        results_directory,
        "vertical_profiles"
    )

    os.makedirs(vertical_output_directory, exist_ok=True)

    if total_water_depth_m is not None:
        y_column = "z_over_h"
        y_label = r"$z/h$ (-)"
        y_axis_name_for_table = "z_over_h"
    else:
        y_column = "z_cm"
        y_label = "z above bed (cm)"
        y_axis_name_for_table = "z_cm"

    fit_rows = []

    for profile_id, group_df in profile_df.groupby("profile_id"):
        group_df = group_df.sort_values("z_m").copy()

        fit_result = _fit_log_law_fixed_ks(
            profile_df=group_df,
            initial_ks_m=initial_ks_m,
            kappa=kappa,
            fit_depth_limits_cm=fit_depth_limits_cm,
            excluded_depths_cm=excluded_depths_cm
        )

        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(13.0, 5.5),
            sharey=True
        )

        plot_definitions = [
            ("u_mean (m/s)", "Ux"),
            ("v_mean (m/s)", "Uy"),
            ("w_mean (m/s)", "Uz"),
        ]

        for ax, (column, velocity_label) in zip(axes, plot_definitions):
            ax.plot(
                group_df[column],
                group_df[y_column],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                label=f"{velocity_label} measured"
            )

            ax.axvline(
                0.0,
                linestyle=":",
                linewidth=0.8
            )

            ax.set_xlabel(f"{velocity_label} (m/s)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            if velocity_label == "Ux" and fit_result["success"]:

                if total_water_depth_m is not None:
                    z_log_plot = (
                        fit_result["z_smooth_m"] / total_water_depth_m
                    )
                else:
                    z_log_plot = fit_result["z_smooth_cm"]

                ax.plot(
                    fit_result["u_log_smooth"],
                    z_log_plot,
                    linestyle="--",
                    linewidth=2.0,
                    label=(
                        "log-law fit\n"
                        f"$u_*$ = "
                        f"{fit_result['u_star_m_per_s']:.4f} m/s"
                    )
                )

                ax.legend()

        axes[0].set_ylabel(y_label)

        if total_water_depth_m is not None:
            axes[0].set_ylim(
                bottom=0.0,
                top=max(1.0, group_df["z_over_h"].max() * 1.05)
            )

        if fit_result["success"]:
            title = (
                f"{profile_id}\n"
                f"fixed $k_s$ = {initial_ks_m:.4g} m, "
                f"fitted $u_*$ = "
                f"{fit_result['u_star_m_per_s']:.4f} m/s, "
                f"RMSE = {fit_result['rmse_m_per_s']:.4f} m/s, "
                f"$R^2$ = {fit_result['r2']:.3f}"
            )
        else:
            title = (
                f"{profile_id}\n"
                f"Log-law fit failed: {fit_result['message']}"
            )

        fig.suptitle(title)
        fig.tight_layout()

        figure_file = os.path.join(
            vertical_output_directory,
            f"vertical_profile_loglaw_{_sanitize_filename(profile_id)}.png"
        )

        fig.savefig(
            figure_file,
            dpi=300,
            bbox_inches="tight"
        )

        if show_plot:
            plt.show()

        plt.close(fig)

        fit_rows.append({
            "profile_id": profile_id,
            "initial_ks_m": initial_ks_m,
            "total_water_depth_m": total_water_depth_m,
            "vertical_axis": y_axis_name_for_table,
            "u_star_m_per_s": fit_result["u_star_m_per_s"],
            "rmse_m_per_s": fit_result["rmse_m_per_s"],
            "r2": fit_result["r2"],
            "n_points_fit": fit_result["n_fit"],
            "fit_status": fit_result["message"],
            "figure_file": figure_file,
        })

        print(f"   - saved vertical profile plot: {figure_file}")

    fit_df = pd.DataFrame(fit_rows)

    fit_output_file = os.path.join(
        vertical_output_directory,
        "loglaw_fitted_shear_velocity.csv"
    )

    fit_df.to_csv(fit_output_file, index=False)

    print("\n============================================================")
    print(" Vertical profile plots and log-law fitting finished")
    print(f" Output directory: {vertical_output_directory}")
    print(f" Fit table       : {fit_output_file}")
    print("============================================================")

    return fit_df