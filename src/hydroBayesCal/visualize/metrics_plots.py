"""
Surrogate performance metrics: evolution over training points, per-location
metric exports, and metric heatmaps.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MetricsPlots:
    def compute_evolution_metrics(self, surrogate_outputs, complex_model_outputs, sm_ci_upper, sm_ci_lower,
                                  selected_locations):
        """
        Computes overall and per-location metrics between surrogate and complex models.

        Returns
        -------
        overall_mse : float
        overall_rmse : float
        overall_mae : float
        overall_corr : float
        ci_range_mean : float
        location_metrics : numpy.ndarray, shape (n_locations, 5)
            Each row: [mse, rmse, mae, correlation, p_value]
        ci_range_per_location : numpy.ndarray, shape (n_locations,)
        """
        def compute_metrics(cm_values, sm_values):
            mse = mean_squared_error(cm_values, sm_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(cm_values, sm_values)
            correlation, p_value = spearmanr(cm_values, sm_values)
            return mse, rmse, mae, correlation, p_value

        all_cm_values = []
        all_sm_values = []

        ci_range_evolution_location = []  # List of CI ranges per location
        location_metrics = []  # List of [mse, rmse, mae, corr, p] per location

        for loc in selected_locations:
            sm_values = surrogate_outputs[:, loc]
            cm_values = complex_model_outputs[:, loc]
            ci_upper = sm_ci_upper[:, loc]
            ci_lower = sm_ci_lower[:, loc]

            all_cm_values.append(cm_values)
            all_sm_values.append(sm_values)

            # Compute metrics for this location
            mse, rmse, mae, corr, p_val = compute_metrics(cm_values, sm_values)
            location_metrics.append([mse, rmse, mae, corr, p_val])

            # CI range for this location
            ci_range = np.mean(ci_upper - ci_lower)
            ci_range_evolution_location.append(ci_range)

        # Convert to arrays
        all_cm = np.concatenate(all_cm_values)
        all_sm = np.concatenate(all_sm_values)
        locations_metrics = np.array(location_metrics)
        ci_range_evolution = np.array(ci_range_evolution_location)

        # Compute overall metrics
        overall_mse, overall_rmse, overall_mae, overall_corr, p_val = compute_metrics(all_cm, all_sm)
        print("\nOverall Metrics across all selected locations:")
        print(f" - MSE       : {overall_mse:.4e}")
        print(f" - RMSE      : {overall_rmse:.4e}")
        print(f" - MAE       : {overall_mae:.4e}")
        print(f" - Correlation: {overall_corr:.2f}")
        print(f" - Confidence Interval Range (mean): {np.mean(ci_range_evolution_location):.4e}")

        return overall_mse, overall_rmse, overall_mae, overall_corr, np.mean(
            ci_range_evolution), locations_metrics, ci_range_evolution_location
        # plt.tight_layout()
        # plt.show()

    def location_metrics(
            self,
            surrogate_metrics,
            coordinates_df,
    ):
        """
        Export per-location validation metrics.

        Two files are generated:

        1. metrics-locations-tp-raw.csv
           Individual metrics from every validation set.

        2. metrics-locations-tp-mean-std.csv
           Mean and sample standard deviation across validation sets,
           independently for each:
               - SurrogateType
               - Quantity
               - TrainPoints
               - Location
               - Metric
        """

        save_folder = self.save_folder

        entries = surrogate_metrics.get("metrics_per_location", [])

        if not entries:
            print("No metrics_per_location found.")
            return

        coordinates_df = coordinates_df.reset_index(drop=True)
        n_locations = len(coordinates_df)

        # ---------------------------------------------------------
        # Validation-set and surrogate information
        # ---------------------------------------------------------
        n_entries = len(entries)

        surrogate_types = np.asarray(
            surrogate_metrics.get(
                "SurrogateType",
                ["Unknown"] * n_entries
            )
        )

        validation_sets = np.asarray(
            surrogate_metrics.get(
                "ValidationSet",
                np.ones(n_entries, dtype=int)
            )
        )

        if len(surrogate_types) != n_entries:
            raise ValueError(
                "Length mismatch: SurrogateType does not match "
                "metrics_per_location."
            )

        if len(validation_sets) != n_entries:
            raise ValueError(
                "Length mismatch: ValidationSet does not match "
                "metrics_per_location."
            )

        # ---------------------------------------------------------
        # Determine metric keys
        # ---------------------------------------------------------
        sample = entries[0]

        non_metric_keys = {
            "Quantity",
            "TrainPoints",
            "SurrogateType",
            "ValidationSet",
        }

        metric_keys = [
            k for k in sample
            if k not in non_metric_keys
        ]

        # ---------------------------------------------------------
        # Build RAW dataframe
        # One row = one validation set × one location
        # ---------------------------------------------------------
        rows = []

        for entry_idx, entry in enumerate(entries):

            quantity = entry["Quantity"]
            train_point = entry["TrainPoints"]

            surrogate_type = surrogate_types[entry_idx]
            validation_set = validation_sets[entry_idx]

            for loc_idx in range(n_locations):

                row = {
                    "SurrogateType": surrogate_type,
                    "ValidationSet": validation_set,
                    "Quantity": quantity,
                    "TrainPoints": train_point,
                    "LocationIndex": loc_idx + 1,
                    "X": coordinates_df.loc[loc_idx, "x"],
                    "Y": coordinates_df.loc[loc_idx, "y"],
                }

                for key in metric_keys:
                    value_list = entry.get(key, [])

                    row[key] = (
                        value_list[loc_idx]
                        if loc_idx < len(value_list)
                        else np.nan
                    )

                rows.append(row)

        raw_df = pd.DataFrame(rows)

        # ---------------------------------------------------------
        # Save RAW results
        # ---------------------------------------------------------
        raw_path = os.path.join(
            save_folder,
            "metrics-locations-tp-raw.csv"
        )

        raw_df.to_csv(
            raw_path,
            index=False
        )

        print(
            f"Raw per-location metrics saved to: {raw_path}"
        )

        # =========================================================
        # MEAN + STD ACROSS VALIDATION SETS
        # =========================================================

        group_columns = [
            "SurrogateType",
            "Quantity",
            "TrainPoints",
            "LocationIndex",
            "X",
            "Y",
        ]

        # Only aggregate actual metrics
        metric_columns = [
            m for m in metric_keys
            if m in raw_df.columns
        ]

        # ---------------------------------------------------------
        # Mean
        # ---------------------------------------------------------
        mean_df = (
            raw_df
            .groupby(group_columns, as_index=False)[metric_columns]
            .mean()
        )

        mean_df = mean_df.rename(
            columns={
                metric: f"{metric}_Mean"
                for metric in metric_columns
            }
        )

        # ---------------------------------------------------------
        # Sample SD across validation sets
        # pandas std() uses ddof=1 by default
        # ---------------------------------------------------------
        std_df = (
            raw_df
            .groupby(group_columns, as_index=False)[metric_columns]
            .std()
        )

        std_df = std_df.rename(
            columns={
                metric: f"{metric}_Std"
                for metric in metric_columns
            }
        )

        # ---------------------------------------------------------
        # Number of validation sets
        # ---------------------------------------------------------
        count_df = (
            raw_df
            .groupby(group_columns, as_index=False)
            ["ValidationSet"]
            .nunique()
            .rename(
                columns={
                    "ValidationSet": "NValidationSets"
                }
            )
        )

        # ---------------------------------------------------------
        # Combine mean + SD + number of validation sets
        # ---------------------------------------------------------
        summary_df = mean_df.merge(
            std_df,
            on=group_columns,
            how="left"
        )

        summary_df = summary_df.merge(
            count_df,
            on=group_columns,
            how="left"
        )

        # Match behavior of summarize_validation_metrics():
        # if only one validation set exists, SD = 0
        std_columns = [
            c for c in summary_df.columns
            if c.endswith("_Std")
        ]

        summary_df[std_columns] = (
            summary_df[std_columns].fillna(0.0)
        )

        # ---------------------------------------------------------
        # Save summary
        # ---------------------------------------------------------
        summary_path = os.path.join(
            save_folder,
            "metrics-locations-tp-mean-std.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False
        )

        print(
            f"Mean ± SD per-location metrics saved to: "
            f"{summary_path}"
        )

        return summary_df

    def location_metric_heatmap(
            self,
            surrogate_metrics,
            quantities=("SCALAR VELOCITY", "WATER DEPTH", "CUMUL BED EVOL"),
            metric="RMSE",
            ci_key="CI",
            cmap="viridis",
            vmax_metric=None,
            max_xticks=10,
            max_yticks=15
    ):
        """
        Plots heatmaps showing a given metric and confidence interval (CI) per location, across training points,
        for each quantity.

        Parameters
        ----------
        surrogate_metrics : dict
            Dictionary storing 'metrics_per_location' with per-training-point evaluation results.
        quantities : tuple of str
            Quantities to visualize (e.g., "SCALAR VELOCITY", "WATER DEPTH").
        metric : str
            Metric to visualize (e.g., "RMSE", "MAE").
        ci_key : str
            Key for CI metric (default: "CI").
        cmap : str
            Colormap for the metric heatmap.
        vmax_metric : float or None
            Optional upper bound for color scale of metric.
        max_xticks : int
            Max number of x-axis ticks (Training Points).
        max_yticks : int
            Max number of y-axis ticks (Locations).
        """

        n_quantities = len(quantities)
        fig, axes = plt.subplots(n_quantities, 2, figsize=(15, 4 * n_quantities), constrained_layout=True)

        if n_quantities == 1:
            axes = np.array([axes])

        for i, quantity in enumerate(quantities):
            # Filter for this quantity
            entries = [e for e in surrogate_metrics["metrics_per_location"] if e["Quantity"] == quantity]

            if not entries:
                print(f"No data found for quantity: {quantity}")
                continue

            # Sort entries by training points
            entries.sort(key=lambda e: e["TrainPoints"])
            train_points = [e["TrainPoints"] for e in entries]

            # Build matrices: shape (n_training_pts, n_locations)
            metric_matrix = np.array([e[metric] for e in entries])  # Each e[metric] is a list per location
            ci_matrix = np.array([e[ci_key] for e in entries])

            if metric_matrix.ndim != 2 or ci_matrix.ndim != 2:
                print(f"Invalid shape for metric/CI data in {quantity}")
                continue

            # === Plot Metric ===
            ax_metric = axes[i, 0]
            sns.heatmap(
                metric_matrix.T,
                ax=ax_metric,
                cmap=cmap,
                vmin=0,
                vmax=vmax_metric or np.percentile(metric_matrix, 98),
                cbar_kws={'label': metric},
                xticklabels=train_points if len(train_points) <= max_xticks else False,
                yticklabels=np.arange(1, metric_matrix.shape[1] + 1) if metric_matrix.shape[1] <= max_yticks else False,
            )
            ax_metric.set_title(f"{metric} per Location ({quantity})", fontsize=14)
            ax_metric.set_xlabel("Training Points")
            ax_metric.set_ylabel("Location Index")

            # === Plot CI ===
            ax_ci = axes[i, 1]
            sns.heatmap(
                ci_matrix.T,
                ax=ax_ci,
                cmap="magma",
                cbar_kws={'label': 'CI Range'},
                xticklabels=train_points if len(train_points) <= max_xticks else False,
                yticklabels=np.arange(ci_matrix.shape[1]) if ci_matrix.shape[1] <= max_yticks else False,
            )
            ax_ci.set_title(f"CI Range per Location ({quantity})", fontsize=14)
            ax_ci.set_xlabel("Training Points")
            ax_ci.set_ylabel("Location Index")

        plt.suptitle("Metric and CI Evolution Across Locations", fontsize=16)
        plt.show()

    def summarize_validation_metrics(
            self,
            surrogate_metrics,
            metrics=None,
    ):
        """
        Aggregate independently computed validation metrics across
        validation sets.

        The metric is first computed separately on each validation
        set. This function then reports the mean and sample standard
        deviation across validation sets.

        For a single validation set, SD is defined as 0.
        """

        if metrics is None:
            metrics = ["MSE", "RMSE", "MAE", "Correlation", "CI"]

        train_points = np.asarray(
            surrogate_metrics["TrainPoints"]
        )

        quantities = np.asarray(
            surrogate_metrics["Quantity"]
        )

        surrogate_types = np.asarray(
            surrogate_metrics["SurrogateType"]
        )

        rows = []

        for quantity in np.unique(quantities):

            for surrogate_type in np.unique(surrogate_types):

                mask_base = (
                        (quantities == quantity)
                        & (surrogate_types == surrogate_type)
                )

                tps = np.unique(train_points[mask_base])

                for tp in np.sort(tps):

                    mask = (
                            mask_base
                            & (train_points == tp)
                    )

                    for metric in metrics:

                        values = np.asarray(
                            surrogate_metrics[metric],
                            dtype=float,
                        )[mask]

                        valid = np.isfinite(values)
                        values = values[valid]

                        n_sets = len(values)

                        if n_sets == 0:
                            mean_value = np.nan
                            std_value = np.nan

                        elif n_sets == 1:
                            mean_value = float(values[0])
                            std_value = 0.0

                        else:
                            mean_value = float(
                                np.mean(values)
                            )

                            # Sample SD across independent
                            # validation replicates.
                            std_value = float(
                                np.std(values, ddof=1)
                            )

                        rows.append(
                            {
                                "TrainPoints": int(tp),
                                "Quantity": quantity,
                                "SurrogateType":
                                    surrogate_type,
                                "Metric": metric,
                                "Mean": mean_value,
                                "Std": std_value,
                                "NValidationSets": n_sets,
                            }
                        )

        summary_df = pd.DataFrame(rows)

        summary_path = os.path.join(
            self.save_folder,
            "validation_metrics_mean_std.csv",
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        print(
            f"Validation metric summary saved to: "
            f"{summary_path}"
        )

        return summary_df
    def plot_metric_comparison(
            self,
            surrogate_metrics: dict,
            quantities: list,
            metrics: list = None,
            metric_labels: list = None,
            spine_linewidth: float = 0.8,  # controls external subplot border thickness
    ):

        save_folder = self.save_folder

        if metrics is None:
            metrics = ["RMSE", "Correlation", "CI"]

        if metric_labels is None:
            metric_labels = []
            for m in metrics:
                if m.lower() in ["correlation", "spearman", "spearmanr"]:
                    metric_labels.append(r"Spearman $\rho$")
                else:
                    metric_labels.append(m)

        assert len(metric_labels) == len(metrics)

        summary_df = self.summarize_validation_metrics(
            surrogate_metrics,
            metrics=metrics,
        )


        num_metrics = len(metrics)
        num_quantities = len(quantities)

        fig, axs = plt.subplots(
            nrows=num_metrics,
            ncols=num_quantities,
            figsize=(8 * num_quantities, 4 * num_metrics),
            sharex='col',
            squeeze=False
        )

        fig.subplots_adjust(left=0.12)

        all_handles = []
        all_labels = []

        # ---------------------------------------------------
        # GLOBAL Y LIMITS PER METRIC (used only as default)
        # ---------------------------------------------------
        row_limits = {}

        for metric in metrics:
            metric_data = summary_df[
                summary_df["Metric"] == metric
                ]

            lower = (
                    metric_data["Mean"]
                    - metric_data["Std"]
            ).to_numpy()

            upper = (
                    metric_data["Mean"]
                    + metric_data["Std"]
            ).to_numpy()

            ymin = np.nanmin(lower)
            ymax = np.nanmax(upper)

            pad = (
                0.05 * (ymax - ymin)
                if ymax > ymin
                else 0.01
            )

            row_limits[metric] = (
                ymin - pad,
                ymax + pad
            )
        # ---------------------------------------------------

        for q_idx, quantity in enumerate(quantities):
            for r_idx, metric in enumerate(metrics):
                ax = axs[r_idx, q_idx]

                mo_data = summary_df[
                    (summary_df["Quantity"] == quantity)
                    & (summary_df["Metric"] == metric)
                    & (summary_df["SurrogateType"] == "MO")
                    ].sort_values("TrainPoints")

                so_data = summary_df[
                    (summary_df["Quantity"] == quantity)
                    & (summary_df["Metric"] == metric)
                    & (summary_df["SurrogateType"] == "SO")
                    ].sort_values("TrainPoints")

                mo_tp = mo_data["TrainPoints"].to_numpy()
                mo_mean = mo_data["Mean"].to_numpy()
                mo_std = mo_data["Std"].to_numpy()

                so_tp = so_data["TrainPoints"].to_numpy()
                so_mean = so_data["Mean"].to_numpy()
                so_std = so_data["Std"].to_numpy()

                line_mo, = ax.plot(
                    mo_tp,
                    mo_mean,
                    marker='o',
                    linestyle='-',
                    color='black',
                    linewidth=1.5,
                    markersize=4,
                    label='MO (Multi-output GP)'
                )

                band_mo = ax.fill_between(
                    mo_tp,
                    mo_mean - mo_std,
                    mo_mean + mo_std,
                    color='royalblue',
                    alpha=0.40,
                    linewidth=0,
                    zorder=1,
                    label='MO ±1 SD'
                )

                line_so, = ax.plot(
                    so_tp,
                    so_mean,
                    marker='s',
                    linestyle='--',
                    color='slategray',
                    linewidth=1.5,
                    markersize=4,
                    label='SO (Single-output GP)'
                )

                band_so = ax.fill_between(
                    so_tp,
                    so_mean - so_std,
                    so_mean + so_std,
                    color='darkorange',
                    alpha=0.40,
                    linewidth=0,
                    zorder=1,
                    label='SO ±1 SD'
                )

                if q_idx == 0 and r_idx == 0:
                    all_handles = [
                        line_mo,
                        band_mo,
                        line_so,
                        band_so,
                    ]

                    all_labels = [
                        h.get_label()
                        for h in all_handles
                    ]

                ax.axvline(x=30, color='lightgray', linestyle='--', linewidth=1)

                # ---------------------------------------------------
                # X TICKS WITH EXTRA EMPTY LAST TICK
                # ---------------------------------------------------
                all_tp = np.concatenate(
                    [mo_tp, so_tp]
                )

                min_tp = all_tp.min()
                max_tp = all_tp.max()
                xticks = np.arange(min_tp, max_tp + 11, 10)
                ax.set_xticks(xticks)

                xtick_labels = [str(x) for x in xticks]
                xtick_labels[-1] = ""
                ax.set_xticklabels(xtick_labels)

                # ---------------------------------------------------
                # CUSTOM Y LIMITS PER SUBPLOT
                # ---------------------------------------------------
                ymin, ymax = row_limits[metric]

                if q_idx in [0, 2]:  # first and third columns
                    if r_idx == 0:
                        ymin, ymax = 0.0, 0.005
                    elif r_idx == 1:
                        ymin, ymax = 0.0, 0.10

                elif q_idx == 1:  # second column
                    if r_idx == 0:
                        ymin, ymax = 0.0, 0.015
                    elif r_idx == 1:
                        ymin, ymax = 0.0, 0.25

                ax.set_ylim(ymin, ymax)

                # ---------------------------------------------------
                # 6 Y TICKS + FORMAT TO 3 DECIMALS
                # ---------------------------------------------------
                yticks = np.linspace(ymin, ymax, 6)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{y:.3f}" for y in yticks])

                ax.tick_params(axis='both', which='major', labelsize=28)
                ax.grid(True, linestyle=':', alpha=0.7)

                # ---------------------------------------------------
                # THINNER SUBPLOT BORDER LINES
                # ---------------------------------------------------
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_linewidth)

                if r_idx == num_metrics - 1:
                    ax.set_xlabel("Training Points", fontsize=28)

        # ---------------------------------------------------
        # ROW LABELS
        # ---------------------------------------------------
        for r_idx, row_label in enumerate(metric_labels):
            pos = axs[r_idx, 0].get_position()
            y_center = (pos.y0 + pos.y1) / 2

            fig.text(
                0.06,
                y_center,
                row_label,
                ha='center',
                va='center',
                rotation='vertical',
                fontsize=30
            )

        fig.legend(
            all_handles,
            all_labels,
            loc='upper center',
            ncol=2,
            fontsize=20,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(
            save_folder / "combined_metrics_all_quantities.svg",
            format="svg",
            bbox_inches="tight"
        )

        plt.close(fig)
