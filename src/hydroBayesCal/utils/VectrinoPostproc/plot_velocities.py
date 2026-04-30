"""
Plot instantaneous Vectrino velocities with moving/block mean velocity estimates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_instantaneous_velocities(
        df,
        output_name,
        output_directory="plots",
        averaging_window=0.05,
        show_plot=False,
):
    """
    Plot instantaneous u, v, w velocities and moving mean velocity lines.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed Vectrino DataFrame containing:
        Time (s), u (m/s), v (m/s), w1 (m/s), w2 (m/s)

    output_name : str
        Base name for saved plot file.

    output_directory : str
        Folder where plots are saved.

    averaging_window : float
        Averaging window in seconds.
        Example: 0.05 means one mean value every 0.05 s.

    show_plot : bool
        If True, show plot window.
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

    os.makedirs(output_directory, exist_ok=True)

    df = df.copy()

    # Relative time starting from zero
    df["t_rel (s)"] = df["Time (s)"] - df["Time (s)"].iloc[0]

    # Single vertical velocity estimate
    df["w (m/s)"] = 0.5 * (df["w1 (m/s)"] + df["w2 (m/s)"])

    # Create 0.05 s time bins
    df["time_bin"] = np.floor(df["t_rel (s)"] / averaging_window) * averaging_window

    velocity_columns = [
        ("u (m/s)", "u"),
        ("v (m/s)", "v"),
        ("w (m/s)", "w"),
    ]

    for column, label in velocity_columns:

        # Mean every averaging_window seconds
        mean_df = (
            df.groupby("time_bin", as_index=False)[column]
            .mean()
            .rename(columns={column: f"{label}_mean"})
        )

        plt.figure(figsize=(10, 4))

        # Instantaneous velocity
        plt.plot(
            df["t_rel (s)"],
            df[column],
            linewidth=0.7,
            label=f"{label}(t)"
        )

        # Block mean every 0.05 s
        plt.step(
            mean_df["time_bin"],
            mean_df[f"{label}_mean"],
            where="post",
            linewidth=1.8,
            label=f"{averaging_window:.3f} s mean {label}"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title(
            f"Instantaneous {label}-velocity with {averaging_window:.3f} s mean: {output_name}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        figure_path = os.path.join(
            output_directory,
            f"{output_name}_{label}_instantaneous_velocity_{averaging_window:.3f}s_mean.png"
        )

        plt.savefig(figure_path, dpi=300)

        if show_plot:
            plt.show()

        plt.close()

        print(f"   - saved plot: {figure_path}")