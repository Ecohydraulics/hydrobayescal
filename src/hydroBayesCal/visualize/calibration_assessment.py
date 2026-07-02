"""
Calibration assessment across competing models: summary metrics (RMSE, MAE,
NRMSE, NMAE, Spearman), observed-vs-modeled scatter, surrogate-vs-deterministic
scatter, and residual plots.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from hydroBayesCal.visualize.axis_utils import (
    compute_nice_limits,
    format_tick_label,
    padded_limits,
    scatter_node_groups,
    set_adaptive_decimal_formatter,
    set_nice_ticks,
    symmetric_limits,
    tight_metric_limits,
)


class CalibrationAssessment:
    def evaluate_calibration(
            self,
            cm_outputs_split,
            sm_outputs_split,
            sm_upper_ci_split,
            sm_lower_ci_split,
            obs_split,
            coordinates_df,
            model_names=None,
            quantity_names=None,
            plot_models=None
    ):
        save_folder = self.save_folder

        n_quantities = len(cm_outputs_split)
        P = next(iter(cm_outputs_split.values())).shape[0]
        N = next(iter(cm_outputs_split.values())).shape[1]

        spatial_records = []
        summary_records = []

        spearman_cm_per_quantity = {f"Q{i + 1}": [] for i in range(n_quantities)}
        spearman_sm_per_quantity = {f"Q{i + 1}": [] for i in range(n_quantities)}

        if model_names is None:
            model_names = [f"M{i + 1}" for i in range(P)]
        elif len(model_names) != P:
            raise ValueError(f"Expected {P} model names, got {len(model_names)}")

        if quantity_names is None:
            quantity_names = [f"Q{i + 1}" for i in range(n_quantities)]
        elif len(quantity_names) != n_quantities:
            raise ValueError(f"Expected {n_quantities} quantity names, got {len(quantity_names)}")

        for p in range(P):
            model_summary = {
                "model_id": p + 1,
                "model_name": str(model_names[p])
            }

            total_rmse_cm = []
            total_rmse_sm = []

            total_nrmse_cm = []
            total_nrmse_sm = []

            total_nmae_cm = []
            total_nmae_sm = []

            cm_quantities_matrix = []
            sm_quantities_matrix = []
            obs_quantities_matrix = []

            for i in range(n_quantities):
                cm_vals = cm_outputs_split[f'cm_outputs_{i + 1}'][p]
                sm_vals = sm_outputs_split[f'sm_outputs_{i + 1}'][p]
                upper_ci_vals = sm_upper_ci_split[f'sm_upper_ci_{i + 1}'][p]
                lower_ci_vals = sm_lower_ci_split[f'sm_lower_ci_{i + 1}'][p]

                obs_vals_raw = obs_split[f'obs_{i + 1}']
                obs_vals = obs_vals_raw[0] if obs_vals_raw.ndim > 1 else obs_vals_raw

                cm_quantities_matrix.append(cm_vals)
                sm_quantities_matrix.append(sm_vals)
                obs_quantities_matrix.append(obs_vals)

                residuals_cm = cm_vals - obs_vals
                residuals_sm = sm_vals - obs_vals

                # ---------------------------------------------------------
                # Error metrics per calibration target
                # ---------------------------------------------------------
                rmse_cm_total = np.sqrt(np.mean(residuals_cm ** 2))
                rmse_sm_total = np.sqrt(np.mean(residuals_sm ** 2))

                mae_cm_total = np.mean(np.abs(residuals_cm))
                mae_sm_total = np.mean(np.abs(residuals_sm))

                rmse_mae_ratio_cm = (
                    np.nan if np.isclose(mae_cm_total, 0.0)
                    else rmse_cm_total / mae_cm_total
                )

                rmse_mae_ratio_sm = (
                    np.nan if np.isclose(mae_sm_total, 0.0)
                    else rmse_sm_total / mae_sm_total
                )

                # Normalized RMSE and normalized MAE.
                # Both are normalized by the observed standard deviation.
                obs_std = np.std(obs_vals)

                if np.isclose(obs_std, 0.0):
                    nrmse_cm_total = np.nan
                    nrmse_sm_total = np.nan
                    nmae_cm_total = np.nan
                    nmae_sm_total = np.nan
                else:
                    nrmse_cm_total = rmse_cm_total / obs_std
                    nrmse_sm_total = rmse_sm_total / obs_std
                    nmae_cm_total = mae_cm_total / obs_std
                    nmae_sm_total = mae_sm_total / obs_std

                spearman_cm = spearmanr(cm_vals, obs_vals).correlation
                spearman_sm = spearmanr(sm_vals, obs_vals).correlation

                # ---------------------------------------------------------
                # Summary CSV columns per calibration target
                # ---------------------------------------------------------
                model_summary[f"RMSE_CM_Q{i + 1}"] = rmse_cm_total
                model_summary[f"RMSE_SM_Q{i + 1}"] = rmse_sm_total

                model_summary[f"MAE_CM_Q{i + 1}"] = mae_cm_total
                model_summary[f"MAE_SM_Q{i + 1}"] = mae_sm_total

                model_summary[f"NRMSE_CM_Q{i + 1}"] = nrmse_cm_total
                model_summary[f"NRMSE_SM_Q{i + 1}"] = nrmse_sm_total

                model_summary[f"NMAE_CM_Q{i + 1}"] = nmae_cm_total
                model_summary[f"NMAE_SM_Q{i + 1}"] = nmae_sm_total

                model_summary[f"RMSE_MAE_CM_Q{i + 1}"] = rmse_mae_ratio_cm
                model_summary[f"RMSE_MAE_SM_Q{i + 1}"] = rmse_mae_ratio_sm

                model_summary[f"Spearman_CM_Q{i + 1}"] = spearman_cm
                model_summary[f"Spearman_SM_Q{i + 1}"] = spearman_sm

                total_rmse_cm.append(rmse_cm_total)
                total_rmse_sm.append(rmse_sm_total)

                total_nrmse_cm.append(nrmse_cm_total)
                total_nrmse_sm.append(nrmse_sm_total)

                total_nmae_cm.append(nmae_cm_total)
                total_nmae_sm.append(nmae_sm_total)

                spearman_cm_per_quantity[f"Q{i + 1}"].append(spearman_cm)
                spearman_sm_per_quantity[f"Q{i + 1}"].append(spearman_sm)

                cm_ranks = pd.Series(cm_vals).rank().values
                sm_ranks = pd.Series(sm_vals).rank().values
                obs_ranks = pd.Series(obs_vals).rank().values

                for j in range(N):
                    ci_width = upper_ci_vals[j] - lower_ci_vals[j]

                    spatial_records.append({
                        "model_id": p + 1,
                        "model_name": str(model_names[p]),
                        "quantity": f"Q{i + 1}",
                        "x": coordinates_df.iloc[j]['x'],
                        "y": coordinates_df.iloc[j]['y'],
                        "residuals_cm": residuals_cm[j],
                        "residuals_sm": residuals_sm[j],
                        "ci_width": ci_width,
                        "cm_rank": cm_ranks[j],
                        "sm_rank": sm_ranks[j],
                        "obs_rank": obs_ranks[j],
                        "cm_output": cm_vals[j],
                        "sm_output": sm_vals[j],
                        "obs": obs_vals[j]
                    })

            cm_matrix = np.array(cm_quantities_matrix).T
            sm_matrix = np.array(sm_quantities_matrix).T
            obs_matrix = np.array(obs_quantities_matrix).T

            scaler_cm = StandardScaler()
            scaler_sm = StandardScaler()
            scaler_obs = StandardScaler()

            cm_standardized = scaler_cm.fit_transform(cm_matrix)
            sm_standardized = scaler_sm.fit_transform(sm_matrix)
            obs_standardized = scaler_obs.fit_transform(obs_matrix)

            cm_composite = np.mean(cm_standardized, axis=1)
            sm_composite = np.mean(sm_standardized, axis=1)
            obs_composite = np.mean(obs_standardized, axis=1)

            overall_spearman_cm = spearmanr(cm_composite, obs_composite).correlation
            overall_spearman_sm = spearmanr(sm_composite, obs_composite).correlation

            model_summary["Overall_NRMSE_CM"] = np.nanmean(total_nrmse_cm)
            model_summary["Overall_NRMSE_SM"] = np.nanmean(total_nrmse_sm)

            model_summary["Overall_NMAE_CM"] = np.nanmean(total_nmae_cm)
            model_summary["Overall_NMAE_SM"] = np.nanmean(total_nmae_sm)

            model_summary["Overall_Spearman_CM"] = overall_spearman_cm
            model_summary["Overall_Spearman_SM"] = overall_spearman_sm

            summary_records.append(model_summary)

        df_summary = pd.DataFrame(summary_records)

        df_summary["Rank_NRMSE_CM"] = df_summary["Overall_NRMSE_CM"].rank(method="min")
        df_summary["Rank_NRMSE_SM"] = df_summary["Overall_NRMSE_SM"].rank(method="min")

        df_summary["Rank_NMAE_CM"] = df_summary["Overall_NMAE_CM"].rank(method="min")
        df_summary["Rank_NMAE_SM"] = df_summary["Overall_NMAE_SM"].rank(method="min")

        df_summary["Rank_Spearman_CM"] = df_summary["Overall_Spearman_CM"].rank(
            ascending=False,
            method="min"
        )
        df_summary["Rank_Spearman_SM"] = df_summary["Overall_Spearman_SM"].rank(
            ascending=False,
            method="min"
        )

        for i in range(n_quantities):
            q = f"Q{i + 1}"

            cm_ranks = pd.Series(spearman_cm_per_quantity[q]).rank(
                ascending=False,
                method="min"
            )
            sm_ranks = pd.Series(spearman_sm_per_quantity[q]).rank(
                ascending=False,
                method="min"
            )

            df_summary[f"Rank_Spearman_CM_{q}"] = cm_ranks.values
            df_summary[f"Rank_Spearman_SM_{q}"] = sm_ranks.values

        if plot_models is not None:
            df_plot = df_summary.iloc[plot_models]
        else:
            df_plot = df_summary

        all_spearman_overall = pd.concat(
            [df_plot["Overall_Spearman_CM"], df_plot["Overall_Spearman_SM"]],
            ignore_index=True
        )

        all_spearman_per_quantity = []

        for i in range(n_quantities):
            q = f"Q{i + 1}"

            spearman_all = pd.concat(
                [df_plot[f"Spearman_CM_{q}"], df_plot[f"Spearman_SM_{q}"]],
                ignore_index=True
            )

            all_spearman_per_quantity.extend(spearman_all.values)

        all_spearman_combined = list(all_spearman_overall.values) + all_spearman_per_quantity
        shared_ylim_global = symmetric_limits(all_spearman_combined, pad=0.10)

        all_nrmse_cm = df_plot["Overall_NRMSE_CM"]
        all_nrmse_sm = df_plot["Overall_NRMSE_SM"]

        nrmse_cm_margin = (
            0.15 * (all_nrmse_cm.max() - all_nrmse_cm.min())
            if not np.isclose(all_nrmse_cm.max(), all_nrmse_cm.min())
            else 0.1 * abs(all_nrmse_cm.mean())
        )

        nrmse_sm_margin = (
            0.15 * (all_nrmse_sm.max() - all_nrmse_sm.min())
            if not np.isclose(all_nrmse_sm.max(), all_nrmse_sm.min())
            else 0.1 * abs(all_nrmse_sm.mean())
        )

        xlim_cm_overall = (
            all_nrmse_cm.min() - nrmse_cm_margin,
            all_nrmse_cm.max() + nrmse_cm_margin
        )

        xlim_sm_overall = (
            all_nrmse_sm.min() - nrmse_sm_margin,
            all_nrmse_sm.max() + nrmse_sm_margin
        )

        xlims_cm_quantity = []
        xlims_sm_quantity = []

        for i in range(n_quantities):
            q = f"Q{i + 1}"

            rmse_cm = df_plot[f"RMSE_CM_{q}"]
            rmse_sm = df_plot[f"RMSE_SM_{q}"]

            cm_margin = (
                0.15 * (rmse_cm.max() - rmse_cm.min())
                if not np.isclose(rmse_cm.max(), rmse_cm.min())
                else 0.1 * abs(rmse_cm.mean())
            )

            sm_margin = (
                0.15 * (rmse_sm.max() - rmse_sm.min())
                if not np.isclose(rmse_sm.max(), rmse_sm.min())
                else 0.1 * abs(rmse_sm.mean())
            )

            xlims_cm_quantity.append(
                (rmse_cm.min() - cm_margin, rmse_cm.max() + cm_margin)
            )
            xlims_sm_quantity.append(
                (rmse_sm.min() - sm_margin, rmse_sm.max() + sm_margin)
            )

        ylims_cm_quantity = []
        ylims_sm_quantity = []

        for i in range(n_quantities):
            q = f"Q{i + 1}"

            spearman_cm = df_plot[f"Spearman_CM_{q}"]
            spearman_sm = df_plot[f"Spearman_SM_{q}"]

            cm_min, cm_max = spearman_cm.min(), spearman_cm.max()
            sm_min, sm_max = spearman_sm.min(), spearman_sm.max()

            cm_margin = (
                0.15 * (cm_max - cm_min)
                if not np.isclose(cm_max, cm_min)
                else (0.1 * abs(cm_min) if cm_min != 0 else 0.1)
            )

            sm_margin = (
                0.15 * (sm_max - sm_min)
                if not np.isclose(sm_max, sm_min)
                else (0.1 * abs(sm_min) if sm_min != 0 else 0.1)
            )

            cm_ylim = (cm_min - cm_margin, cm_max + cm_margin)
            sm_ylim = (sm_min - sm_margin, sm_max + sm_margin)

            ylimit_cm = cm_ylim if cm_max < 0.25 else shared_ylim_global
            ylimit_sm = sm_ylim if sm_max < 0.25 else shared_ylim_global

            ylims_cm_quantity.append(ylimit_cm)
            ylims_sm_quantity.append(ylimit_sm)

        plt.rcParams.update({
            'xtick.labelsize': 20,
            'ytick.labelsize': 20
        })

        # ---------- Subplots per quantity: RMSE vs Spearman (CM and SM) ----------
        ncols = 2
        nrows = math.ceil(n_quantities / ncols)

        def plot_rmse_vs_spearman_subplots(metric_tag, xlims_quantity, ylims_quantity, filename):
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(12, 4 * nrows),
                sharey=False
            )

            axes = axes.flatten()
            colors = plt.cm.get_cmap('tab10', len(df_plot))

            for i in range(n_quantities):
                ax = axes[i]
                q = f"Q{i + 1}"

                for color_idx, (_, row) in enumerate(df_plot.iterrows()):
                    ax.scatter(
                        row[f"RMSE_{metric_tag}_{q}"],
                        row[f"Spearman_{metric_tag}_{q}"],
                        color=colors(color_idx),
                        label=row["model_name"],
                        s=100,
                        alpha=0.8
                    )

                ax.set_title(quantity_names[i], fontsize=20)
                ax.set_xlim(xlims_quantity[i])
                ax.set_ylim(ylims_quantity[i])

                ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
                ax.set_xlabel("RMSE", fontsize=16)
                ax.set_ylabel(r"Spearman $\rho$", fontsize=16)

                set_nice_ticks(ax, 'x', n_ticks=5)
                set_nice_ticks(ax, 'y', n_ticks=5, start_at_zero=True)

                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            for j in range(n_quantities, len(axes)):
                fig.delaxes(axes[j])

            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper center',
                bbox_to_anchor=(0.5, 1.02),
                ncol=max(1, len(by_label)),
                fontsize=16
            )

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(save_folder, filename), dpi=300)

        plot_rmse_vs_spearman_subplots(
            metric_tag="CM",
            xlims_quantity=xlims_cm_quantity,
            ylims_quantity=ylims_cm_quantity,
            filename="per_quantity_rmse_vs_spearman_CM.svg"
        )

        plot_rmse_vs_spearman_subplots(
            metric_tag="SM",
            xlims_quantity=xlims_sm_quantity,
            ylims_quantity=ylims_sm_quantity,
            filename="per_quantity_rmse_vs_spearman_SM.svg"
        )

        # ---------- Unified subplots for NRMSE vs Spearman ----------
        def plot_nrmse_vs_spearman_subplots(metric_tag, overall_xlim, quantity_ylims, filename):
            n_panels = n_quantities + 1
            ncols_local = 2
            nrows_local = math.ceil(n_panels / ncols_local)

            fig, axes = plt.subplots(
                nrows=nrows_local,
                ncols=ncols_local,
                figsize=(12, 4 * nrows_local),
                sharey=False
            )

            axes = axes.flatten()
            colors = plt.cm.get_cmap('tab10', len(df_plot))

            # Overall panel
            ax = axes[0]

            overall_x_col = f"Overall_NRMSE_{metric_tag}"
            overall_y_col = f"Overall_Spearman_{metric_tag}"

            for color_idx, (_, row) in enumerate(df_plot.iterrows()):
                ax.scatter(
                    row[overall_x_col],
                    row[overall_y_col],
                    color=colors(color_idx),
                    label=row["model_name"],
                    s=150,
                    alpha=0.8,
                    marker='o'
                )

            ax.set_title("Overall", fontsize=20)
            ax.set_xlabel("NRMSE", fontsize=16)
            ax.set_ylabel(r"Spearman $\rho$", fontsize=16)
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

            ax.set_xlim(overall_xlim)

            y_vals = df_plot[overall_y_col].values
            y_min, y_max = symmetric_limits(y_vals, pad=0.10)
            ax.set_ylim(y_min, y_max)

            set_nice_ticks(ax, 'x', n_ticks=5)
            set_nice_ticks(ax, 'y', n_ticks=5, start_at_zero=True)

            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # Quantity panels
            for i in range(n_quantities):
                ax = axes[i + 1]
                q = f"Q{i + 1}"

                x_col = f"NRMSE_{metric_tag}_{q}"
                y_col = f"Spearman_{metric_tag}_{q}"

                for color_idx, (_, row) in enumerate(df_plot.iterrows()):
                    ax.scatter(
                        row[x_col],
                        row[y_col],
                        color=colors(color_idx),
                        label=row["model_name"],
                        s=100,
                        alpha=0.8
                    )

                ax.set_title(quantity_names[i], fontsize=20)
                ax.set_xlabel("NRMSE", fontsize=16)
                ax.set_ylabel(r"Spearman $\rho$", fontsize=16)
                ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

                current_xlim = padded_limits(df_plot[x_col].values)
                ax.set_xlim(current_xlim)
                ax.set_ylim(quantity_ylims[i])

                set_nice_ticks(ax, 'x', n_ticks=5)
                set_nice_ticks(ax, 'y', n_ticks=5, start_at_zero=True)

                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            for j in range(n_panels, len(axes)):
                fig.delaxes(axes[j])

            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper center',
                bbox_to_anchor=(0.5, 1.02),
                ncol=max(1, len(by_label)),
                fontsize=16
            )

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(save_folder, filename), dpi=300)

        plot_nrmse_vs_spearman_subplots(
            metric_tag="CM",
            overall_xlim=xlim_cm_overall,
            quantity_ylims=ylims_cm_quantity,
            filename="combined_nrmse_vs_spearman_CM.svg"
        )

        plot_nrmse_vs_spearman_subplots(
            metric_tag="SM",
            overall_xlim=xlim_sm_overall,
            quantity_ylims=ylims_sm_quantity,
            filename="combined_nrmse_vs_spearman_SM.svg"
        )

        # ---------- Unified subplots for NMAE vs NRMSE ----------
        def plot_nmae_vs_nrmse_subplots(metric_tag, filename):
            n_panels = n_quantities + 1
            ncols_local = 2
            nrows_local = math.ceil(n_panels / ncols_local)

            fig, axes = plt.subplots(
                nrows=nrows_local,
                ncols=ncols_local,
                figsize=(12, 4 * nrows_local),
                sharey=False
            )

            axes = axes.flatten()
            colors = plt.cm.get_cmap('tab10', len(df_plot))

            # Overall panel
            ax = axes[0]

            overall_x_col = f"Overall_NRMSE_{metric_tag}"
            overall_y_col = f"Overall_NMAE_{metric_tag}"

            for color_idx, (_, row) in enumerate(df_plot.iterrows()):
                ax.scatter(
                    row[overall_x_col],
                    row[overall_y_col],
                    color=colors(color_idx),
                    label=row["model_name"],
                    s=150,
                    alpha=0.8,
                    marker='o'
                )

            ax.set_title("Overall", fontsize=20)
            ax.set_xlabel("NRMSE", fontsize=16)
            ax.set_ylabel("NMAE", fontsize=16)
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

            overall_x_vals = df_plot[overall_x_col].values
            overall_y_vals = df_plot[overall_y_col].values

            ax.set_xlim(tight_metric_limits(overall_x_vals))
            ax.set_ylim(tight_metric_limits(overall_y_vals))

            # Important: do not force zero here.
            # For tiny differences, start_at_zero=True destroys visual separation.
            set_nice_ticks(ax, 'x', n_ticks=5)
            set_nice_ticks(ax, 'y', n_ticks=5)

            set_adaptive_decimal_formatter(ax, axis='x', values=overall_x_vals)
            set_adaptive_decimal_formatter(ax, axis='y', values=overall_y_vals)

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # Quantity panels
            for i in range(n_quantities):
                ax = axes[i + 1]
                q = f"Q{i + 1}"

                x_col = f"NRMSE_{metric_tag}_{q}"
                y_col = f"NMAE_{metric_tag}_{q}"

                for color_idx, (_, row) in enumerate(df_plot.iterrows()):
                    ax.scatter(
                        row[x_col],
                        row[y_col],
                        color=colors(color_idx),
                        label=row["model_name"],
                        s=100,
                        alpha=0.8,
                        marker='o'
                    )

                ax.set_title(quantity_names[i], fontsize=20)
                ax.set_xlabel("NRMSE", fontsize=16)
                ax.set_ylabel("NMAE", fontsize=16)
                ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

                x_vals = df_plot[x_col].values
                y_vals = df_plot[y_col].values

                ax.set_xlim(tight_metric_limits(x_vals))
                ax.set_ylim(tight_metric_limits(y_vals))

                # Important: do not force zero here.
                set_nice_ticks(ax, 'x', n_ticks=5)
                set_nice_ticks(ax, 'y', n_ticks=5)

                set_adaptive_decimal_formatter(ax, axis='x', values=x_vals)
                set_adaptive_decimal_formatter(ax, axis='y', values=y_vals)

                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            for j in range(n_panels, len(axes)):
                fig.delaxes(axes[j])

            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper center',
                bbox_to_anchor=(0.5, 1.02),
                ncol=max(1, len(by_label)),
                fontsize=16
            )

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(save_folder, filename), dpi=300)

        plot_nmae_vs_nrmse_subplots(
            metric_tag="CM",
            filename="combined_nMAE_vs_nRMSE_CM.svg"
        )

        plot_nmae_vs_nrmse_subplots(
            metric_tag="SM",
            filename="combined_nMAE_vs_nRMSE_SM.svg"
        )

        df_spatial = pd.DataFrame(spatial_records)

        df_spatial.to_csv(
            os.path.join(save_folder, "location_metrics_models.csv"),
            index=False
        )

        df_summary.to_csv(
            os.path.join(save_folder, "summary_metrics_models.csv"),
            index=False
        )

        return df_spatial, df_summary
    def observed_vs_modeled_compare(self, df_spatial, df_summary, model_ids, quantity_names,
                                    points_group_1=None, points_group_2=None):
        r"""
        Plots Modeled vs Observed with:
            Rows    = models
            Columns = calibration targets / quantities

        Parameters
        ----------
        df_spatial : DataFrame
            Spatial data with observations and model outputs
        df_summary : DataFrame
            Summary data with model names
        model_ids : list
            List of model IDs to plot
        quantity_names : list
            List of quantity names to plot, ordered as Q1, Q2, Q3, ...
            Example: ["h", r"\bar{U}", r"\delta_z"]
        points_group_1 : list or range, optional
            First point group (e.g. downstream nodes)
        points_group_2 : list or range, optional
            Second point group (e.g. upstream nodes)
        """

        save_folder = self.save_folder
        n_models = len(model_ids)
        n_quantities = len(quantity_names)

        downstream_set = set(points_group_1) if points_group_1 is not None else None
        upstream_set = set(points_group_2) if points_group_2 is not None else None

        # ------------------------------------------------------------
        # Create subplot grid
        # ------------------------------------------------------------
        fig, axes = plt.subplots(
            nrows=n_models,
            ncols=n_quantities,
            figsize=(9.5 * n_quantities, 5.0 * n_models),
            sharex=False,
            sharey=False
        )

        # Force axes into 2D array
        if n_models == 1 and n_quantities == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes[np.newaxis, :]
        elif n_quantities == 1:
            axes = axes[:, np.newaxis]

        # ------------------------------------------------------------
        # Compute shared nice axis limits per quantity column
        # ------------------------------------------------------------
        axis_limits_by_quantity = {}

        for q_idx, qname in enumerate(quantity_names):
            all_obs_q = []
            all_cm_q = []

            for model_id in model_ids:
                df_model = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == f"Q{q_idx + 1}")
                    ]

                if not df_model.empty:
                    all_obs_q.append(df_model["obs"].values)
                    all_cm_q.append(df_model["cm_output"].values)

            if len(all_obs_q) == 0 or len(all_cm_q) == 0:
                axis_limits_by_quantity[q_idx] = (0.0, 1.0)
                continue

            all_obs_q = np.concatenate(all_obs_q)
            all_cm_q = np.concatenate(all_cm_q)

            combined = np.concatenate([all_obs_q, all_cm_q])
            min_val = np.nanmin(combined)
            max_val = np.nanmax(combined)

            axis_limits_by_quantity[q_idx] = compute_nice_limits(min_val, max_val, n_ticks=5)

        # ------------------------------------------------------------
        # Legend bookkeeping
        # ------------------------------------------------------------
        legend_handles = []
        legend_labels = []

        units_map = {
            r"$h$": "m",
            r"$\bar{U}$": "m/s",
            r"$\delta_z$": "m",
        }

        # ------------------------------------------------------------
        # Plot loop
        # ------------------------------------------------------------
        for row_idx, model_id in enumerate(model_ids):
            model_name_series = df_summary.loc[df_summary["model_id"] == model_id, "model_name"]
            model_name = model_name_series.iloc[0] if not model_name_series.empty else f"M{model_id}"

            for col_idx, qname in enumerate(quantity_names):
                ax = axes[row_idx, col_idx]

                df_model = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == f"Q{col_idx + 1}")
                    ]

                if df_model.empty:
                    ax.text(
                        0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=18
                    )
                    ax.set_axis_off()
                    continue

                obs = df_model["obs"].values
                cm = df_model["cm_output"].values

                axis_limits = axis_limits_by_quantity[col_idx]

                # ----------------------------------------------------
                # Scatter points
                # ----------------------------------------------------
                group_handles, group_labels = scatter_node_groups(
                    ax, obs, cm,
                    downstream_set, upstream_set,
                    collect_legend=(row_idx == 0 and col_idx == 0),
                    s=160, alpha=0.85, marker="*"
                )
                legend_handles.extend(group_handles)
                legend_labels.extend(group_labels)

                # 1:1 line
                ax.plot(axis_limits, axis_limits, color="red", linestyle="--", lw=1.2)

                # Same x/y limits per quantity column
                ax.set_xlim(axis_limits)
                ax.set_ylim(axis_limits)

                # Keep subplot symmetric
                ax.set_aspect("equal", adjustable="box")

                # ----------------------------------------------------
                # Titles and labels
                # ----------------------------------------------------
                if row_idx == 0:
                    ax.set_title(f"{qname}", fontsize=32, pad=16)

                if row_idx == 0 or row_idx == n_models - 1:
                    ax.set_xlabel(f"Observed {qname}", fontsize=36)

                ax.set_ylabel(f"Modeled {qname}", fontsize=36)

                if col_idx == 0:
                    ax.annotate(
                        model_name,
                        xy=(-0.48, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=28,
                        fontweight="bold"
                    )

                # ----------------------------------------------------
                # Tick formatting
                # 6 ticks exactly, show label every second tick
                # ----------------------------------------------------
                ticks = np.linspace(axis_limits[0], axis_limits[1], 6)

                # Clean tiny floating-point noise
                ticks = np.array([0.0 if np.isclose(t, 0.0, atol=1e-12) else t for t in ticks])

                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

                xlabels = [format_tick_label(tick) if k % 2 == 0 else "" for k, tick in enumerate(ticks)]
                ylabels = [format_tick_label(tick) if k % 2 == 0 else "" for k, tick in enumerate(ticks)]

                ax.set_xticklabels(xlabels)
                ax.set_yticklabels(ylabels)

                ax.tick_params(axis="both", which="both", direction="in", labelsize=28)

                # Grid and spines
                ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)

                # ----------------------------------------------------
                # Metrics box: RMSE only
                # ----------------------------------------------------
                residuals = cm - obs
                rmse = np.sqrt(np.mean(residuals ** 2))

                unit = units_map.get(qname, "")
                rmse_text = f"RMSE={rmse:.3f}" + (f" {unit}" if unit else "")

                ax.text(
                    0.97, 0.97,
                    rmse_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=24,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
                )

        # ------------------------------------------------------------
        # Figure legend
        # ------------------------------------------------------------
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels,
                loc="upper center",
                ncol=len(legend_handles),
                fontsize=18,
                framealpha=0.9,
                bbox_to_anchor=(0.5, 0.995)
            )
            fig.tight_layout(rect=[0.08, 0.03, 1, 0.95])
        else:
            fig.tight_layout(rect=[0.08, 0.03, 1, 0.98])

        save_path = os.path.join(save_folder, "scatter_observed_vs_modeled_rows_models_cols_targets.svg")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Observed vs modeled figure saved to {save_path}")

    def surrogate_vs_deterministic_compare(
            self,
            df_spatial,
            df_summary,
            model_ids,
            quantity_names,
            points_group_1=None,
            points_group_2=None,
    ):
        """
        Surrogate Model (X-axis) vs Deterministic/Complex Model (Y-axis)

        • Rows = quantities
        • Columns = surrogate models
        • Wider subplots
        • Red dashed 1:1 line
        • Star markers

        Parameters
        ----------
        points_group_1 : list or range, optional
            Node positions to color as downstream (gray).
        points_group_2 : list or range, optional
            Node positions to color as upstream (black).
        """

        save_folder = self.save_folder
        n_models = len(model_ids)
        n_quantities = len(quantity_names)

        # Convert to sets for faster lookup (only if provided)
        downstream_set = set(points_group_1) if points_group_1 is not None else None
        upstream_set = set(points_group_2) if points_group_2 is not None else None

        # ---- Wider figure ----
        fig, axes = plt.subplots(
            nrows=n_quantities,
            ncols=n_models,
            figsize=(7.5 * n_models, 6 * n_quantities),  # widened
            sharey=False
        )

        if n_quantities == 1:
            axes = axes[np.newaxis, :]
        if n_models == 1:
            axes = axes[:, np.newaxis]

        colors = plt.cm.get_cmap('tab10', n_models)

        # Track if we need to create legend (only when grouping is specified)
        legend_handles = []
        legend_labels = []

        for i, qname in enumerate(quantity_names):

            # ===================================
            # Collect all outputs for this quantity
            # ===================================
            all_surrogate = []
            all_complex = []

            for model_id in model_ids:
                df_q = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == f"Q{i + 1}")
                    ]

                all_surrogate.append(df_q["sm_output"].values)
                all_complex.append(df_q["cm_output"].values)

            all_surrogate = np.concatenate(all_surrogate)
            all_complex = np.concatenate(all_complex)

            vmin = min(all_surrogate.min(), all_complex.min())
            vmax = max(all_surrogate.max(), all_complex.max())
            margin = 0.05 * (vmax - vmin)
            lims = (vmin - margin, vmax + margin)

            # ===================================
            # Plot per surrogate model
            # ===================================
            for j, model_id in enumerate(model_ids):

                ax = axes[i, j]

                df_q = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == f"Q{i + 1}")
                    ]

                model_name = df_summary[
                    df_summary["model_id"] == model_id
                    ]["model_name"].values[0]

                surrogate = df_q["sm_output"].values
                complex_m = df_q["cm_output"].values

                group_handles, group_labels = scatter_node_groups(
                    ax, surrogate, complex_m,
                    downstream_set, upstream_set,
                    collect_legend=(i == 0 and j == 0),
                    s=110, marker='*', alpha=0.9, edgecolor='k', linewidth=0.6
                )
                legend_handles.extend(group_handles)
                legend_labels.extend(group_labels)

                # ---- Red dashed 1:1 line (no label) ----
                ax.plot(
                    lims,
                    lims,
                    linestyle='--',
                    color='red',
                    linewidth=1.5
                )

                # ===================================
                # Statistics
                # ===================================
                rmse = np.sqrt(np.mean((complex_m - surrogate) ** 2))
                rho = spearmanr(complex_m, surrogate).correlation

                # Axis limits
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_aspect("equal", adjustable="box")

                ax.set_title(f"{model_name} — {qname}", fontsize=20)
                ax.set_xlabel("Surrogate Model", fontsize=18)
                ax.set_ylabel("Deterministic Model", fontsize=18)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.tick_params(axis='both', direction='in', labelsize=20)

                ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

                # ---- Metrics box ----
                textstr = (
                    f"RMSE = {rmse:.3f}\n"
                    f"$\\rho$ = {rho:.2f}"
                )

                ax.text(
                    0.05,
                    0.95,
                    textstr,
                    transform=ax.transAxes,
                    va='top',
                    ha='left',
                    fontsize=14,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
                )

        # Add general horizontal legend at the top if grouping was specified
        if legend_handles:
            fig.legend(legend_handles, legend_labels,
                       loc='upper center',
                       ncol=len(legend_handles),
                       fontsize=14,
                       framealpha=0.9,
                       bbox_to_anchor=(0.5, 0.98))
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(
            save_folder,
            "scatter_surrogate_vs_complex.svg"
        )

        fig.savefig(save_path, dpi=300)

        print(f"Surrogate vs deterministic plots saved to {save_path}")

    def plot_residuals(
            self,
            df_spatial,
            df_summary,
            model_ids,
            quantity_names,
            points_group_1=None,
            points_group_2=None,
            mm_col="sm_output",
            cm_col="cm_output",
            figsize_per_panel=(6, 4)
    ):
        """
        Residuals (Complex Model - Metamodel) vs Location Index for multiple models and quantities.
        Rows    -> quantities
        Columns -> models

        Parameters
        ----------
        df_spatial : pd.DataFrame
            Must contain: model_id, quantity, mm_output, cm_output
        df_summary : pd.DataFrame
            Must contain: model_id, model_name
        model_ids : list[int]
            Model IDs to visualize
        quantity_names : list[str]
            Names for subplot titles (ordered as Q1, Q2, ...)
        points_group_1 : list or range, optional
            Node positions to color as downstream (gray).
        points_group_2 : list or range, optional
            Node positions to color as upstream (black).
        mm_col : str
            Column name of metamodel output
        cm_col : str
            Column name of complex model output
        figsize_per_panel : tuple
            Size per subplot (width, height)
        """
        save_folder = self.save_folder
        n_models = len(model_ids)
        n_quantities = len(quantity_names)

        # Convert to sets for faster lookup (only if provided)
        downstream_set = set(points_group_1) if points_group_1 is not None else None
        upstream_set = set(points_group_2) if points_group_2 is not None else None

        # Arrange subplots: rows = quantities, cols = models
        fig, axes = plt.subplots(
            nrows=n_quantities,
            ncols=n_models,
            figsize=(figsize_per_panel[0] * n_models,
                     figsize_per_panel[1] * n_quantities),
            sharex=False,
            sharey=False
        )

        # Ensure axes is 2D array even if n_models or n_quantities = 1
        if n_quantities == 1:
            axes = axes[np.newaxis, :]
        if n_models == 1:
            axes = axes[:, np.newaxis]

        # Track if we need to create legend (only when grouping is specified)
        legend_handles = []
        legend_labels = []

        for i, qname in enumerate(quantity_names):
            # ----- Compute per-variable global limits across all models -----
            max_points = 0
            all_residuals_q = []

            for model_id in model_ids:
                df_model = df_spatial[(df_spatial["model_id"] == model_id) &
                                      (df_spatial["quantity"] == f"Q{i + 1}")]
                cm = df_model[cm_col].values
                mm = df_model[mm_col].values
                residuals = cm - mm
                all_residuals_q.append(residuals)
                max_points = max(max_points, len(df_model))

            # Concatenate all residuals for this quantity
            all_residuals_q = np.concatenate(all_residuals_q)

            # X-axis limits based on location indices (1 to n_points)
            x_limits = (0.5, max_points + 0.5)

            # Y-axis limits based on all residuals for this quantity
            min_residual = all_residuals_q.min()
            max_residual = all_residuals_q.max()
            margin = 0.05 * (max_residual - min_residual)
            y_limits = (min_residual - margin, max_residual + margin)

            # ----- Plot each model in this row -----
            for j, model_id in enumerate(model_ids):
                ax = axes[i, j]

                df_model = df_spatial[(df_spatial["model_id"] == model_id) &
                                      (df_spatial["quantity"] == f"Q{i + 1}")]
                model_name = df_summary[df_summary["model_id"] == model_id]["model_name"].values[0]

                cm = df_model[cm_col].values
                mm = df_model[mm_col].values
                residuals = cm - mm
                n_points = len(cm)

                # Create location indices (1, 2, 3, ..., n_points)
                location_indices = np.arange(1, n_points + 1)

                group_handles, group_labels = scatter_node_groups(
                    ax, location_indices, residuals,
                    downstream_set, upstream_set,
                    collect_legend=(i == 0 and j == 0),
                    s=60, alpha=0.8, marker='*'
                )
                legend_handles.extend(group_handles)
                legend_labels.extend(group_labels)

                # Zero reference line
                ax.axhline(0.0, color='red', linestyle='--', linewidth=1)

                # Apply identical X and Y axis limits for all models in this row
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)

                # Titles and labels
                ax.set_title(f"{model_name} — {qname}", fontsize=18)
                ax.set_xlabel(f"Location Index", fontsize=16)
                if j == 0:  # first column
                    ax.set_ylabel(f"Residuals", fontsize=16)

                # Tick formatting - synchronized across row
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.tick_params(axis='both', which='both', direction='in', labelsize=20)

                # Grid and spines
                ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

                # Metrics box: RMSE and Mean residual
                rmse = np.sqrt(np.mean(residuals ** 2))
                mean_res = np.mean(residuals)

                # Position text box at top right
                ax.text(0.98, 0.98,
                        f"RMSE={rmse:.3f} $\\mathrm{{m/s}}$\nMean={mean_res:.3f} $\\mathrm{{m/s}}$",
                        transform=ax.transAxes,
                        va='top', ha='right',
                        fontsize=12,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        # Add general horizontal legend at the top if grouping was specified
        if legend_handles:
            fig.legend(legend_handles, legend_labels,
                       loc='upper center',
                       ncol=len(legend_handles),
                       fontsize=14,
                       framealpha=0.9,
                       bbox_to_anchor=(0.5, 0.98))
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_folder is not None:
            save_path = os.path.join(
                save_folder,
                "residuals_CM_vs_MM_individual_models.svg"
            )
            fig.savefig(save_path, dpi=300)
            print(f"Residuals plot saved to {save_path}")


