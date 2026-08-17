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

    def location_metrics(self,
            surrogate_metrics,
            coordinates_df,
    ):
        """
        Export per-location metrics (for all training points and quantities) to a long-format CSV,
        using user-provided coordinate DataFrame.

        Parameters
        ----------
        surrogate_metrics : dict
            Dictionary with 'metrics_per_location' entries.
        coordinates_df : pd.DataFrame
            DataFrame containing at least "X" and "Y" columns (one row per location).
        output_csv_path : str
            Path to save the CSV file.
        """
        save_folder = self.save_folder

        entries = surrogate_metrics.get("metrics_per_location", [])
        if not entries:
            print(" No metrics_per_location found.")
            return


        coordinates_df = coordinates_df.reset_index(drop=True)
        n_locations = len(coordinates_df)

        # Determine all metric keys (excluding non-metric fields)
        sample = entries[0]
        metric_keys = [k for k in sample if k not in {"Quantity", "TrainPoints"}]

        rows = []
        for entry in entries:
            quantity = entry["Quantity"]
            train_point = entry["TrainPoints"]

            for loc_idx in range(n_locations):
                row = {
                    "Quantity": quantity,
                    "TrainPoints": train_point,
                    "LocationIndex": loc_idx,
                    "X": coordinates_df.loc[loc_idx, "x"],
                    "Y": coordinates_df.loc[loc_idx, "y"],
                }
                for key in metric_keys:
                    value_list = entry.get(key, [])
                    row[key] = value_list[loc_idx] if loc_idx < len(value_list) else None
                rows.append(row)

        df = pd.DataFrame(rows)
        file_path = os.path.join(save_folder, "metrics-locations-tp.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Exported metrics to CSV: {file_path} ({len(df)} rows)")

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

        train_points = np.asarray(surrogate_metrics["TrainPoints"])
        quantity_names = np.asarray(surrogate_metrics["Quantity"])
        surrogate_type = np.asarray(surrogate_metrics["SurrogateType"])
        metric_values = {m: np.asarray(surrogate_metrics[m]) for m in metrics}

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
            vals_all = []
            for quantity in quantities:
                mask_q = quantity_names == quantity
                vals_all.append(metric_values[metric][mask_q])

            vals_all = np.concatenate(vals_all)
            ymin = vals_all.min()
            ymax = vals_all.max()
            pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01
            row_limits[metric] = (ymin - pad, ymax + pad)
        # ---------------------------------------------------

        for q_idx, quantity in enumerate(quantities):
            for r_idx, metric in enumerate(metrics):
                ax = axs[r_idx, q_idx]

                mask_q = quantity_names == quantity
                mask_so = mask_q & (surrogate_type == "SO")
                mask_mo = mask_q & (surrogate_type == "MO")

                so_tp = train_points[mask_so]
                mo_tp = train_points[mask_mo]

                so_val = metric_values[metric][mask_so]
                mo_val = metric_values[metric][mask_mo]

                line_mo, = ax.plot(
                    mo_tp,
                    mo_val,
                    marker='o',
                    linestyle='-',
                    color='black',
                    linewidth=1.5,
                    markersize=6,
                    label='MO (Multi-output GP)'
                )

                line_so, = ax.plot(
                    so_tp,
                    so_val,
                    marker='s',
                    linestyle='--',
                    color='slategray',
                    linewidth=1.5,
                    markersize=6,
                    label='SO (Single-output GP)'
                )

                if q_idx == 0 and r_idx == 0:
                    all_handles = [line_mo, line_so]
                    all_labels = [h.get_label() for h in all_handles]

                ax.axvline(x=30, color='lightgray', linestyle='--', linewidth=1)

                # ---------------------------------------------------
                # X TICKS WITH EXTRA EMPTY LAST TICK
                # ---------------------------------------------------
                min_tp = train_points.min()
                max_tp = train_points.max()
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
                        ymin, ymax = 0.0, 0.010
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
