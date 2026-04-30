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

    Adds a final row with the mean values across all valid time bins.
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

    df["t_rel (s)"] = df["Time (s)"] - df["Time (s)"].iloc[0]

    df["w (m/s)"] = 0.5 * (df["w1 (m/s)"] + df["w2 (m/s)"])

    df["time_bin_start (s)"] = (
        np.floor(df["t_rel (s)"] / averaging_window) * averaging_window
    )

    rows = []

    for time_bin, group in df.groupby("time_bin_start (s)"):

        n_samples = len(group)

        u_mean = group["u (m/s)"].mean()
        v_mean = group["v (m/s)"].mean()
        w_mean = group["w (m/s)"].mean()

        if n_samples < 2:
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

        u_fluc = group["u (m/s)"] - u_mean
        v_fluc = group["v (m/s)"] - v_mean
        w_fluc = group["w (m/s)"] - w_mean

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

    # ----------------------------------------------------------
    # Add final mean row
    # ----------------------------------------------------------
    mean_row = {
        "time_start (s)": "MEAN",
        "time_end (s)": np.nan,
        "time_center (s)": np.nan,
        "n_samples": tke_df["n_samples"].sum(),

        "u_mean (m/s)": tke_df["u_mean (m/s)"].mean(skipna=True),
        "v_mean (m/s)": tke_df["v_mean (m/s)"].mean(skipna=True),
        "w_mean (m/s)": tke_df["w_mean (m/s)"].mean(skipna=True),

        "u_var (m2/s2)": tke_df["u_var (m2/s2)"].mean(skipna=True),
        "v_var (m2/s2)": tke_df["v_var (m2/s2)"].mean(skipna=True),
        "w_var (m2/s2)": tke_df["w_var (m2/s2)"].mean(skipna=True),

        "TKE (m2/s2)": tke_df["TKE (m2/s2)"].mean(skipna=True),
    }

    tke_df = pd.concat(
        [tke_df, pd.DataFrame([mean_row])],
        ignore_index=True
    )

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