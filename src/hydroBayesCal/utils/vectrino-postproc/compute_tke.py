"""
Compute time-binned mean velocities and turbulent kinetic energy (TKE)
from processed Vectrino velocity data.

Input DataFrame must contain:
    Time (s)
    u (m/s)
    v (m/s)
    w1 (m/s)
    w2 (m/s)

Output table contains one row per averaging window.
"""

import os
import numpy as np
import pandas as pd


def compute_tke_data(
        df,
        output_name=None,
        output_directory="tke",
        averaging_window=0.05,
        save_csv=True,
):
    """
    Compute mean velocities and TKE every averaging_window seconds.

    TKE definition:
        TKE = 0.5 * (mean(u'^2) + mean(v'^2) + mean(w'^2))

    where:
        u' = u - mean(u)
        v' = v - mean(v)
        w' = w - mean(w)

    Means and fluctuations are computed independently inside each time bin.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed Vectrino data.

    output_name : str or None
        Base name for saved CSV.

    output_directory : str
        Folder where output CSV is saved.

    averaging_window : float
        Time-bin size in seconds.

    save_csv : bool
        If True, save output table as CSV.

    Returns
    -------
    tke_df : pandas.DataFrame
        Table with time-bin mean velocities, variances, and TKE.
    """

    required_columns = [
        "Time (s)",
        "u (m/s)",
        "v (m/s)",
        "w1 (m/s)",
        "w2 (m/s)",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"ERROR: Missing required column: {col}\n"
                f"Available columns are: {list(df.columns)}"
            )

    df = df.copy()

    # Relative time starting at zero
    df["t_rel (s)"] = df["Time (s)"] - df["Time (s)"].iloc[0]

    # Single vertical velocity estimate
    df["w (m/s)"] = 0.5 * (df["w1 (m/s)"] + df["w2 (m/s)"])

    # Create time bins: 0.00, 0.05, 0.10, ...
    df["time_bin_start (s)"] = (
        np.floor(df["t_rel (s)"] / averaging_window) * averaging_window
    )

    rows = []

    for time_bin, group in df.groupby("time_bin_start (s)"):

        n_samples = len(group)

        if n_samples < 2:
            # Variance/TKE from one sample is not meaningful.
            u_mean = group["u (m/s)"].mean()
            v_mean = group["v (m/s)"].mean()
            w_mean = group["w (m/s)"].mean()

            rows.append({
                "time_start (s)": time_bin,
                "time_end (s)": time_bin + averaging_window,
                "time_center (s)": time_bin + 0.5 * averaging_window,
                "n_samples": n_samples,

                "u_mean (m/s)": u_mean,
                "v_mean (m/s)": v_mean,
                "w_mean (m/s)": w_mean,

                "u_var (m2/s2)": np.nan,
                "v_var (m2/s2)": np.nan,
                "w_var (m2/s2)": np.nan,

                "TKE (m2/s2)": np.nan,
            })

            continue

        u_mean = group["u (m/s)"].mean()
        v_mean = group["v (m/s)"].mean()
        w_mean = group["w (m/s)"].mean()

        u_fluc = group["u (m/s)"] - u_mean
        v_fluc = group["v (m/s)"] - v_mean
        w_fluc = group["w (m/s)"] - w_mean

        # Population variance inside the bin
        u_var = np.mean(u_fluc ** 2)
        v_var = np.mean(v_fluc ** 2)
        w_var = np.mean(w_fluc ** 2)

        tke = 0.5 * (u_var + v_var + w_var)

        rows.append({
            "time_start (s)": time_bin,
            "time_end (s)": time_bin + averaging_window,
            "time_center (s)": time_bin + 0.5 * averaging_window,
            "n_samples": n_samples,

            "u_mean (m/s)": u_mean,
            "v_mean (m/s)": v_mean,
            "w_mean (m/s)": w_mean,

            "u_var (m2/s2)": u_var,
            "v_var (m2/s2)": v_var,
            "w_var (m2/s2)": w_var,

            "TKE (m2/s2)": tke,
        })

    tke_df = pd.DataFrame(rows)

    if save_csv:
        if output_name is None:
            raise ValueError("ERROR: output_name must be provided when save_csv=True.")

        os.makedirs(output_directory, exist_ok=True)

        output_file = os.path.join(
            output_directory,
            f"{output_name}_tke_{averaging_window:.3f}s.csv"
        )

        tke_df.to_csv(output_file, index=False)

        print(f"   - saved TKE table: {output_file}")

    return tke_df