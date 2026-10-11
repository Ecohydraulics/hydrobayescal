"""
Calibration assessment across competing models: summary metrics (RMSE, MAE,
NRMSE, NMAE, Spearman), observed-vs-modeled scatter, surrogate-vs-deterministic
scatter, and residual plots.
"""

import math
import os
import re
import unicodedata


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
            err_split,
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

            # Store all uncertainty-normalized residuals from all quantities.
            # These arrays are pooled after the quantity loop to calculate
            # the true overall NRMSE and NMAE for the current model.
            all_normalized_residuals_cm = []
            all_normalized_residuals_sm = []

            # Store residuals normalized by the observed spatial standard
            # deviation of each quantity. Each quantity therefore has its
            # own normalization scale, shared by all locations for that target.
            all_stdobs_normalized_residuals_cm = []
            all_stdobs_normalized_residuals_sm = []

            for i in range(n_quantities):
                cm_vals = cm_outputs_split[f'cm_outputs_{i + 1}'][p]
                sm_vals = sm_outputs_split[f'sm_outputs_{i + 1}'][p]
                upper_ci_vals = sm_upper_ci_split[f'sm_upper_ci_{i + 1}'][p]
                lower_ci_vals = sm_lower_ci_split[f'sm_lower_ci_{i + 1}'][p]

                obs_vals_raw = obs_split[f'obs_{i + 1}']
                obs_vals = obs_vals_raw[0] if obs_vals_raw.ndim > 1 else obs_vals_raw

                # Measurement uncertainty (standard deviation) at each location.
                err_vals_raw = err_split[f'err_{i + 1}']
                err_vals = err_vals_raw[0] if err_vals_raw.ndim > 1 else err_vals_raw

                obs_vals = np.asarray(obs_vals, dtype=float).squeeze()
                err_vals = np.asarray(err_vals, dtype=float).squeeze()

                if obs_vals.shape != err_vals.shape:
                    raise ValueError(
                        f"Observation and uncertainty shapes differ for Q{i + 1}: "
                        f"{obs_vals.shape} versus {err_vals.shape}"
                    )

                if np.any(~np.isfinite(err_vals)):
                    raise ValueError(
                        f"Non-finite measurement uncertainties found for Q{i + 1}"
                    )

                if np.any(err_vals <= 0.0):
                    raise ValueError(
                        f"Measurement uncertainties must be greater than zero for Q{i + 1}"
                    )

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

                # ---------------------------------------------------------
                # Uncertainty-normalized RMSE and MAE
                # ---------------------------------------------------------
                # Normalize each residual with the measurement uncertainty
                # associated with the same location before aggregating.
                normalized_residuals_cm = residuals_cm / err_vals
                normalized_residuals_sm = residuals_sm / err_vals

                nrmse_cm_total = np.sqrt(
                    np.mean(normalized_residuals_cm ** 2)
                )
                nrmse_sm_total = np.sqrt(
                    np.mean(normalized_residuals_sm ** 2)
                )

                nmae_cm_total = np.mean(
                    np.abs(normalized_residuals_cm)
                )
                nmae_sm_total = np.mean(
                    np.abs(normalized_residuals_sm)
                )

                # Ratio of uncertainty-normalized RMSE to uncertainty-normalized MAE.
                # This is different from RMSE / MAE when the measurement
                # uncertainty varies among observation locations.
                nrmse_nmae_ratio_cm = (
                    np.nan
                    if (
                        not np.isfinite(nmae_cm_total)
                        or np.isclose(nmae_cm_total, 0.0)
                    )
                    else nrmse_cm_total / nmae_cm_total
                )

                nrmse_nmae_ratio_sm = (
                    np.nan
                    if (
                        not np.isfinite(nmae_sm_total)
                        or np.isclose(nmae_sm_total, 0.0)
                    )
                    else nrmse_sm_total / nmae_sm_total
                )

                # ---------------------------------------------------------
                # Observation-standard-deviation-normalized RMSE and MAE
                # ---------------------------------------------------------
                # The normalization is target-specific: for quantity Q_i,
                # compute one standard deviation from its observations across
                # locations, then normalize every location residual of Q_i
                # using that same target-specific scale.
                #
                # ddof=1 gives the sample standard deviation. If you want the
                # population standard deviation of the finite set of observed
                # locations instead, change ddof=1 to ddof=0.
                finite_obs = obs_vals[np.isfinite(obs_vals)]

                if finite_obs.size < 2:
                    raise ValueError(
                        f"At least two finite observations are required to compute "
                        f"the observation standard deviation for Q{i + 1}"
                    )

                obs_std = np.std(finite_obs, ddof=1)

                if not np.isfinite(obs_std) or obs_std <= 0.0:
                    raise ValueError(
                        f"Observation standard deviation must be finite and greater "
                        f"than zero for Q{i + 1}; got {obs_std}"
                    )

                stdobs_normalized_residuals_cm = residuals_cm / obs_std
                stdobs_normalized_residuals_sm = residuals_sm / obs_std

                nrmse_stdobs_cm_total = np.sqrt(
                    np.nanmean(stdobs_normalized_residuals_cm ** 2)
                )
                nrmse_stdobs_sm_total = np.sqrt(
                    np.nanmean(stdobs_normalized_residuals_sm ** 2)
                )

                nmae_stdobs_cm_total = np.nanmean(
                    np.abs(stdobs_normalized_residuals_cm)
                )
                nmae_stdobs_sm_total = np.nanmean(
                    np.abs(stdobs_normalized_residuals_sm)
                )

                # Keep both normalization schemes for pooled overall metrics.
                all_normalized_residuals_cm.append(normalized_residuals_cm)
                all_normalized_residuals_sm.append(normalized_residuals_sm)

                all_stdobs_normalized_residuals_cm.append(
                    stdobs_normalized_residuals_cm
                )
                all_stdobs_normalized_residuals_sm.append(
                    stdobs_normalized_residuals_sm
                )

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

                # Observation-standard-deviation normalization.
                model_summary[f"Obs_STD_Q{i + 1}"] = obs_std
                model_summary[f"NRMSE_STDOBS_CM_Q{i + 1}"] = nrmse_stdobs_cm_total
                model_summary[f"NRMSE_STDOBS_SM_Q{i + 1}"] = nrmse_stdobs_sm_total
                model_summary[f"NMAE_STDOBS_CM_Q{i + 1}"] = nmae_stdobs_cm_total
                model_summary[f"NMAE_STDOBS_SM_Q{i + 1}"] = nmae_stdobs_sm_total

                # Raw RMSE / MAE ratio.
                model_summary[f"RMSE_MAE_CM_Q{i + 1}"] = rmse_mae_ratio_cm
                model_summary[f"RMSE_MAE_SM_Q{i + 1}"] = rmse_mae_ratio_sm

                # Uncertainty-normalized NRMSE / NMAE ratio.
                model_summary[f"NRMSE_NMAE_CM_Q{i + 1}"] = nrmse_nmae_ratio_cm
                model_summary[f"NRMSE_NMAE_SM_Q{i + 1}"] = nrmse_nmae_ratio_sm

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
                        "measurement_error": err_vals[j],
                        "normalized_residual_cm": normalized_residuals_cm[j],
                        "normalized_residual_sm": normalized_residuals_sm[j],
                        "obs_std": obs_std,
                        "stdobs_normalized_residual_cm": stdobs_normalized_residuals_cm[j],
                        "stdobs_normalized_residual_sm": stdobs_normalized_residuals_sm[j],
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

            # ---------------------------------------------------------
            # Pooled overall uncertainty-normalized metrics
            # ---------------------------------------------------------
            # Concatenate the normalized residuals from every quantity and
            # every measurement location. This does not average the separate
            # per-quantity NRMSE/NMAE values.
            pooled_normalized_residuals_cm = np.concatenate(
                all_normalized_residuals_cm
            )
            pooled_normalized_residuals_sm = np.concatenate(
                all_normalized_residuals_sm
            )

            valid_overall_cm = np.isfinite(pooled_normalized_residuals_cm)
            valid_overall_sm = np.isfinite(pooled_normalized_residuals_sm)

            if np.any(valid_overall_cm):
                valid_cm_values = pooled_normalized_residuals_cm[valid_overall_cm]

                model_summary["Overall_NRMSE_CM"] = np.sqrt(
                    np.mean(valid_cm_values ** 2)
                )
                model_summary["Overall_NMAE_CM"] = np.mean(
                    np.abs(valid_cm_values)
                )
            else:
                model_summary["Overall_NRMSE_CM"] = np.nan
                model_summary["Overall_NMAE_CM"] = np.nan

            if np.any(valid_overall_sm):
                valid_sm_values = pooled_normalized_residuals_sm[valid_overall_sm]

                model_summary["Overall_NRMSE_SM"] = np.sqrt(
                    np.mean(valid_sm_values ** 2)
                )
                model_summary["Overall_NMAE_SM"] = np.mean(
                    np.abs(valid_sm_values)
                )
            else:
                model_summary["Overall_NRMSE_SM"] = np.nan
                model_summary["Overall_NMAE_SM"] = np.nan

            # Ratio calculated from the pooled overall uncertainty-normalized
            # metrics. It is not an average of the per-quantity ratios.
            overall_nmae_cm = model_summary["Overall_NMAE_CM"]
            overall_nmae_sm = model_summary["Overall_NMAE_SM"]

            model_summary["Overall_NRMSE_NMAE_CM"] = (
                np.nan
                if (
                    not np.isfinite(overall_nmae_cm)
                    or np.isclose(overall_nmae_cm, 0.0)
                )
                else model_summary["Overall_NRMSE_CM"] / overall_nmae_cm
            )

            model_summary["Overall_NRMSE_NMAE_SM"] = (
                np.nan
                if (
                    not np.isfinite(overall_nmae_sm)
                    or np.isclose(overall_nmae_sm, 0.0)
                )
                else model_summary["Overall_NRMSE_SM"] / overall_nmae_sm
            )

            # ---------------------------------------------------------
            # Pooled overall observation-STD-normalized metrics
            # ---------------------------------------------------------
            # Each residual was first normalized by the observation standard
            # deviation of its own quantity. Pooling afterward makes the
            # quantities dimensionless and comparable in the global metric.
            pooled_stdobs_residuals_cm = np.concatenate(
                all_stdobs_normalized_residuals_cm
            )
            pooled_stdobs_residuals_sm = np.concatenate(
                all_stdobs_normalized_residuals_sm
            )

            valid_stdobs_cm = np.isfinite(pooled_stdobs_residuals_cm)
            valid_stdobs_sm = np.isfinite(pooled_stdobs_residuals_sm)

            if np.any(valid_stdobs_cm):
                valid_cm_values = pooled_stdobs_residuals_cm[valid_stdobs_cm]
                model_summary["Overall_NRMSE_STDOBS_CM"] = np.sqrt(
                    np.mean(valid_cm_values ** 2)
                )
                model_summary["Overall_NMAE_STDOBS_CM"] = np.mean(
                    np.abs(valid_cm_values)
                )
            else:
                model_summary["Overall_NRMSE_STDOBS_CM"] = np.nan
                model_summary["Overall_NMAE_STDOBS_CM"] = np.nan

            if np.any(valid_stdobs_sm):
                valid_sm_values = pooled_stdobs_residuals_sm[valid_stdobs_sm]
                model_summary["Overall_NRMSE_STDOBS_SM"] = np.sqrt(
                    np.mean(valid_sm_values ** 2)
                )
                model_summary["Overall_NMAE_STDOBS_SM"] = np.mean(
                    np.abs(valid_sm_values)
                )
            else:
                model_summary["Overall_NRMSE_STDOBS_SM"] = np.nan
                model_summary["Overall_NMAE_STDOBS_SM"] = np.nan

            model_summary["Overall_Spearman_CM"] = overall_spearman_cm
            model_summary["Overall_Spearman_SM"] = overall_spearman_sm

            summary_records.append(model_summary)

        df_summary = pd.DataFrame(summary_records)

        df_summary["Rank_NRMSE_CM"] = df_summary["Overall_NRMSE_CM"].rank(method="min")
        df_summary["Rank_NRMSE_SM"] = df_summary["Overall_NRMSE_SM"].rank(method="min")

        df_summary["Rank_NMAE_CM"] = df_summary["Overall_NMAE_CM"].rank(method="min")
        df_summary["Rank_NMAE_SM"] = df_summary["Overall_NMAE_SM"].rank(method="min")

        df_summary["Rank_NRMSE_STDOBS_CM"] = df_summary[
            "Overall_NRMSE_STDOBS_CM"
        ].rank(method="min")
        df_summary["Rank_NRMSE_STDOBS_SM"] = df_summary[
            "Overall_NRMSE_STDOBS_SM"
        ].rank(method="min")

        df_summary["Rank_NMAE_STDOBS_CM"] = df_summary[
            "Overall_NMAE_STDOBS_CM"
        ].rank(method="min")
        df_summary["Rank_NMAE_STDOBS_SM"] = df_summary[
            "Overall_NMAE_STDOBS_SM"
        ].rank(method="min")

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
        def set_precise_metric_ticks(
                ax,
                axis,
                values,
                n_ticks=6
        ):
            """
            Set tight metric limits with a fixed number of ticks and always
            display tick labels with exactly two decimal places.

            Six ticks are used for the STDOBS figures, i.e. one more horizontal
            and vertical tick than the previous five-tick layout.
            """
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                return

            vmin, vmax = tight_metric_limits(vals)

            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return

            if np.isclose(vmin, vmax):
                pad = 0.02 * max(abs(vmin), 1.0)
                vmin -= pad
                vmax += pad

            ticks = np.linspace(vmin, vmax, n_ticks)
            formatter = FormatStrFormatter('%.3f')

            if axis == 'x':
                ax.set_xlim(vmin, vmax)
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(formatter)
            elif axis == 'y':
                ax.set_ylim(vmin, vmax)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(formatter)
            else:
                raise ValueError("axis must be either 'x' or 'y'")

        def plot_nmae_vs_nrmse_subplots(
                metric_tag,
                filename,
                normalization_tag=None
        ):
            """
            Plot NMAE versus NRMSE for one model-output type.

            normalization_tag
                None     -> existing measurement-uncertainty normalization
                "STDOBS" -> observation-standard-deviation normalization

            For STDOBS plots, the axis labels use subscripts rather than
            '/ sigma_obs'. The metric has already been normalized by
            sigma_obs, so writing 'NRMSE / sigma_obs' would incorrectly imply
            a second normalization.
            """
            n_panels = n_quantities + 1
            ncols_local = 2
            nrows_local = math.ceil(n_panels / ncols_local)

            fig, axes = plt.subplots(
                nrows=nrows_local,
                ncols=ncols_local,
                figsize=(12, 4 * nrows_local),
                sharey=False
            )

            axes = np.atleast_1d(axes).flatten()
            colors = plt.cm.get_cmap('tab10', len(df_plot))

            if normalization_tag is None:
                overall_x_col = f"Overall_NRMSE_{metric_tag}"
                overall_y_col = f"Overall_NMAE_{metric_tag}"
                x_label = "NRMSE"
                y_label = "NMAE"

                def quantity_cols(q):
                    return (
                        f"NRMSE_{metric_tag}_{q}",
                        f"NMAE_{metric_tag}_{q}"
                    )

            elif normalization_tag == "STDOBS":
                overall_x_col = f"Overall_NRMSE_STDOBS_{metric_tag}"
                overall_y_col = f"Overall_NMAE_STDOBS_{metric_tag}"

                # Correct notation: these quantities are already normalized
                # by the target-specific observation standard deviation.
                x_label = r"$\mathrm{NRMSE}_{\sigma_{\mathrm{obs}}}$"
                y_label = r"$\mathrm{NMAE}_{\sigma_{\mathrm{obs}}}$"

                def quantity_cols(q):
                    return (
                        f"NRMSE_STDOBS_{metric_tag}_{q}",
                        f"NMAE_STDOBS_{metric_tag}_{q}"
                    )

            else:
                raise ValueError(
                    "normalization_tag must be None or 'STDOBS'"
                )

            # Overall panel
            ax = axes[0]

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
            ax.set_xlabel(x_label, fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

            overall_x_vals = df_plot[overall_x_col].values
            overall_y_vals = df_plot[overall_y_col].values

            if normalization_tag == "STDOBS":
                # Six ticks = one more than the previous five-tick layout.
                # Tick labels are forced to exactly two decimal places.
                set_precise_metric_ticks(
                    ax, 'x', overall_x_vals, n_ticks=6
                )
                set_precise_metric_ticks(
                    ax, 'y', overall_y_vals, n_ticks=6
                )
            else:
                ax.set_xlim(tight_metric_limits(overall_x_vals))
                ax.set_ylim(tight_metric_limits(overall_y_vals))

                # Keep the existing uncertainty-normalized plots unchanged.
                set_nice_ticks(ax, 'x', n_ticks=5)
                set_nice_ticks(ax, 'y', n_ticks=5)

                set_adaptive_decimal_formatter(
                    ax, axis='x', values=overall_x_vals
                )
                set_adaptive_decimal_formatter(
                    ax, axis='y', values=overall_y_vals
                )

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # Quantity panels
            for i in range(n_quantities):
                ax = axes[i + 1]
                q = f"Q{i + 1}"
                x_col, y_col = quantity_cols(q)

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
                ax.set_xlabel(x_label, fontsize=16)
                ax.set_ylabel(y_label, fontsize=16)
                ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

                x_vals = df_plot[x_col].values
                y_vals = df_plot[y_col].values

                if normalization_tag == "STDOBS":
                    set_precise_metric_ticks(
                        ax, 'x', x_vals, n_ticks=6
                    )
                    set_precise_metric_ticks(
                        ax, 'y', y_vals, n_ticks=6
                    )
                else:
                    ax.set_xlim(tight_metric_limits(x_vals))
                    ax.set_ylim(tight_metric_limits(y_vals))

                    set_nice_ticks(ax, 'x', n_ticks=5)
                    set_nice_ticks(ax, 'y', n_ticks=5)

                    set_adaptive_decimal_formatter(
                        ax, axis='x', values=x_vals
                    )
                    set_adaptive_decimal_formatter(
                        ax, axis='y', values=y_vals
                    )

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
            plt.close(fig)

        # Existing measurement-uncertainty normalization.
        plot_nmae_vs_nrmse_subplots(
            metric_tag="CM",
            filename="combined_nMAE_vs_nRMSE_CM.svg"
        )

        plot_nmae_vs_nrmse_subplots(
            metric_tag="SM",
            filename="combined_nMAE_vs_nRMSE_SM.svg"
        )

        # Observation-standard-deviation normalization.
        plot_nmae_vs_nrmse_subplots(
            metric_tag="CM",
            filename="combined_nMAE_vs_nRMSE_STDOBS_CM.svg",
            normalization_tag="STDOBS"
        )

        plot_nmae_vs_nrmse_subplots(
            metric_tag="SM",
            filename="combined_nMAE_vs_nRMSE_STDOBS_SM.svg",
            normalization_tag="STDOBS"
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
                    s=80, alpha=0.85, marker="*"
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
                    ax.set_title(f"{qname}", fontsize=22, pad=16)

                if row_idx == 0 or row_idx == n_models - 1:
                    ax.set_xlabel(f"Observed {qname}", fontsize=22)

                ax.set_ylabel(f"Modeled {qname}", fontsize=22)

                if col_idx == 0:
                    ax.annotate(
                        model_name,
                        xy=(-0.48, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=22,
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

                ax.tick_params(axis="both", which="both", direction="in", labelsize=22)

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
                    fontsize=22,
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
        r"""
        Plots Surrogate Model vs Deterministic/Complex Model with:

            Rows    = surrogate models
            Columns = quantities / calibration targets

        The plotting style is consistent with observed_vs_modeled_compare():
            - Same subplot dimensions
            - Same font sizes
            - Same tick formatting
            - 6 ticks per axis, every second label displayed
            - Shared axis limits per quantity
            - Equal x/y aspect ratio
            - Model names shown vertically on the left
            - Quantity names shown at the top
            - Common legend above the figure

        Parameters
        ----------
        df_spatial : DataFrame
            Spatial comparison data containing:
                - model_id
                - quantity
                - sm_output
                - cm_output

        df_summary : DataFrame
            Summary table containing model_id and model_name.

        model_ids : list
            List of surrogate model IDs to plot.

        quantity_names : list
            Quantity names corresponding to Q1, Q2, Q3, ...
            Example:
                [r"$h$", r"$\bar{U}$", r"$\delta_z$"]

        points_group_1 : list or range, optional
            First group of points, e.g. downstream nodes.

        points_group_2 : list or range, optional
            Second group of points, e.g. upstream nodes.
        """

        save_folder = self.save_folder

        n_models = len(model_ids)
        n_quantities = len(quantity_names)

        downstream_set = (
            set(points_group_1)
            if points_group_1 is not None
            else None
        )

        upstream_set = (
            set(points_group_2)
            if points_group_2 is not None
            else None
        )

        # ------------------------------------------------------------
        # Create subplot grid
        # Rows    = models
        # Columns = quantities
        # ------------------------------------------------------------
        fig, axes = plt.subplots(
            nrows=n_models,
            ncols=n_quantities,
            figsize=(9.5 * n_quantities, 5.0 * n_models),
            sharex=False,
            sharey=False
        )

        # Force axes into a 2D array
        if n_models == 1 and n_quantities == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes[np.newaxis, :]
        elif n_quantities == 1:
            axes = axes[:, np.newaxis]

        # ------------------------------------------------------------
        # Compute shared nice axis limits for each quantity column
        # ------------------------------------------------------------
        axis_limits_by_quantity = {}

        for q_idx, qname in enumerate(quantity_names):

            all_sm_q = []
            all_cm_q = []

            for model_id in model_ids:

                df_model = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == f"Q{q_idx + 1}")
                    ]

                if not df_model.empty:
                    all_sm_q.append(df_model["sm_output"].values)
                    all_cm_q.append(df_model["cm_output"].values)

            # Fallback if no data exists for this quantity
            if len(all_sm_q) == 0 or len(all_cm_q) == 0:
                axis_limits_by_quantity[q_idx] = (0.0, 1.0)
                continue

            all_sm_q = np.concatenate(all_sm_q)
            all_cm_q = np.concatenate(all_cm_q)

            combined = np.concatenate([
                all_sm_q,
                all_cm_q
            ])

            min_val = np.nanmin(combined)
            max_val = np.nanmax(combined)

            axis_limits_by_quantity[q_idx] = compute_nice_limits(
                min_val,
                max_val,
                n_ticks=5
            )

        # ------------------------------------------------------------
        # Legend bookkeeping
        # ------------------------------------------------------------
        legend_handles = []
        legend_labels = []

        # Units used for RMSE
        units_map = {
            r"$h$": "m",
            r"$\bar{U}$": "m/s",
            r"$\delta_z$": "m",
        }

        # ------------------------------------------------------------
        # Plot loop
        # ------------------------------------------------------------
        for row_idx, model_id in enumerate(model_ids):

            # Model name
            model_name_series = df_summary.loc[
                df_summary["model_id"] == model_id,
                "model_name"
            ]

            model_name = (
                model_name_series.iloc[0]
                if not model_name_series.empty
                else f"M{model_id}"
            )

            for col_idx, qname in enumerate(quantity_names):

                ax = axes[row_idx, col_idx]

                df_model = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == f"Q{col_idx + 1}")
                    ]

                # ----------------------------------------------------
                # Handle missing data
                # ----------------------------------------------------
                if df_model.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=18
                    )

                    ax.set_axis_off()
                    continue

                surrogate = df_model["sm_output"].values
                deterministic = df_model["cm_output"].values

                axis_limits = axis_limits_by_quantity[col_idx]

                # ----------------------------------------------------
                # Scatter points
                # ----------------------------------------------------
                group_handles, group_labels = scatter_node_groups(
                    ax,
                    surrogate,
                    deterministic,
                    downstream_set,
                    upstream_set,
                    collect_legend=(
                            row_idx == 0 and col_idx == 0
                    ),
                    s=80,
                    alpha=0.85,
                    marker="*"
                )

                legend_handles.extend(group_handles)
                legend_labels.extend(group_labels)

                # ----------------------------------------------------
                # 1:1 line
                # ----------------------------------------------------
                ax.plot(
                    axis_limits,
                    axis_limits,
                    color="red",
                    linestyle="--",
                    lw=1.2
                )

                # Same x/y limits within each quantity column
                ax.set_xlim(axis_limits)
                ax.set_ylim(axis_limits)

                # Keep the plotting area geometrically symmetric
                ax.set_aspect(
                    "equal",
                    adjustable="box"
                )

                # ----------------------------------------------------
                # Titles and axis labels
                # ----------------------------------------------------

                # Quantity title only on top row
                if row_idx == 0:
                    ax.set_title(
                        f"{qname}",
                        fontsize=22,
                        pad=16
                    )

                # Same behavior as observed_vs_modeled_compare:
                # x-axis label on first and last model rows
                if row_idx == 0 or row_idx == n_models - 1:
                    ax.set_xlabel(
                        f"Surrogate Model {qname}",
                        fontsize=22
                    )

                ax.set_ylabel(
                    f"Deterministic Model {qname}",
                    fontsize=22
                )

                # ----------------------------------------------------
                # Model name on left side of each row
                # ----------------------------------------------------
                if col_idx == 0:
                    ax.annotate(
                        model_name,
                        xy=(-0.48, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=22,
                        fontweight="bold"
                    )

                # ----------------------------------------------------
                # Tick formatting
                #
                # 6 ticks exactly.
                # Display label every second tick.
                # ----------------------------------------------------
                ticks = np.linspace(
                    axis_limits[0],
                    axis_limits[1],
                    6
                )

                # Remove tiny floating-point values around zero
                ticks = np.array([
                    0.0 if np.isclose(t, 0.0, atol=1e-12) else t
                    for t in ticks
                ])

                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

                xlabels = [
                    format_tick_label(tick)
                    if k % 2 == 0
                    else ""
                    for k, tick in enumerate(ticks)
                ]

                ylabels = [
                    format_tick_label(tick)
                    if k % 2 == 0
                    else ""
                    for k, tick in enumerate(ticks)
                ]

                ax.set_xticklabels(xlabels)
                ax.set_yticklabels(ylabels)

                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    labelsize=22
                )

                # ----------------------------------------------------
                # Grid and spines
                # ----------------------------------------------------
                ax.grid(
                    True,
                    linestyle="--",
                    linewidth=0.5,
                    color="gray"
                )

                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)

                # ----------------------------------------------------
                # Metrics
                # ----------------------------------------------------
                residuals = deterministic - surrogate

                rmse = np.sqrt(
                    np.mean(residuals ** 2)
                )

                rho = spearmanr(
                    deterministic,
                    surrogate
                ).correlation

                unit = units_map.get(qname, "")

                rmse_text = (
                        f"RMSE={rmse:.3f}"
                        + (f" {unit}" if unit else "")
                )

                metrics_text = (
                    f"{rmse_text}\n"
                    f"$\\rho$={rho:.2f}"
                )

                # ----------------------------------------------------
                # Metrics box
                # Same position/font size as observed-vs-modeled
                # ----------------------------------------------------
                ax.text(
                    0.97,
                    0.97,
                    metrics_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=22,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="none"
                    )
                )

        # ------------------------------------------------------------
        # Figure legend
        # ------------------------------------------------------------
        if legend_handles:

            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=len(legend_handles),
                fontsize=18,
                framealpha=0.9,
                bbox_to_anchor=(0.5, 0.995)
            )

            fig.tight_layout(
                rect=[0.08, 0.03, 1, 0.95]
            )

        else:

            fig.tight_layout(
                rect=[0.08, 0.03, 1, 0.98]
            )

        # ------------------------------------------------------------
        # Save figure
        # ------------------------------------------------------------
        save_path = os.path.join(
            save_folder,
            "scatter_surrogate_vs_deterministic_rows_models_cols_targets.svg"
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Surrogate vs deterministic figure saved to {save_path}"
        )

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
            figsize_per_panel=(6, 4),
            residual_limits=None
    ):
        """
        Residuals (Complex Model - Metamodel) vs Location Index.

        Rows    -> quantities
        Columns -> models

        Residuals are defined as:

            residual = Complex Model - Metamodel

        Parameters
        ----------
        df_spatial : pd.DataFrame
            Must contain:
                model_id, quantity, mm_col, cm_col

        df_summary : pd.DataFrame
            Must contain:
                model_id, model_name

        model_ids : list[int]
            Model IDs to visualize.

        quantity_names : list[str]
            Quantity names ordered as Q1, Q2, Q3, ...

            Example:
                [
                    r"$h$",
                    r"$\\bar{U}$",
                ]

        points_group_1 : list or range, optional
            First point group, e.g. downstream nodes.

        points_group_2 : list or range, optional
            Second point group, e.g. upstream nodes.

        mm_col : str
            Column name containing metamodel predictions.

        cm_col : str
            Column name containing complex/deterministic model outputs.

        figsize_per_panel : tuple
            Width and height of each subplot.

        residual_limits : dict, optional
            User-defined residual limits for each quantity.

            Example:

                residual_limits = {
                    "Q1": (-0.05, 0.05),
                    "Q2": (-0.10, 0.10),
                }

            The same limits are applied to all models belonging to
            a given quantity.

            If a quantity is not specified, its limits are calculated
            automatically from all models.
        """

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        save_folder = self.save_folder

        n_models = len(model_ids)
        n_quantities = len(quantity_names)

        # ============================================================
        # Plot settings
        # ============================================================

        # EXACT number of Y-axis ticks in every subplot.
        #
        # Seven is useful for symmetric residual plots because zero
        # becomes the central tick:
        #
        #  -3  -2  -1   0   1   2   3
        #
        n_y_ticks = 7

        # ============================================================
        # Node groups
        # ============================================================
        downstream_set = (
            set(points_group_1)
            if points_group_1 is not None
            else None
        )

        upstream_set = (
            set(points_group_2)
            if points_group_2 is not None
            else None
        )

        # ============================================================
        # Units
        # ============================================================
        units_map = {
            r"$h$": "m",
            r"$\bar{U}$": "m/s",
            r"$\delta_z$": "m",
        }

        # ============================================================
        # Create subplot grid
        #
        # Rows    = quantities
        # Columns = models
        # ============================================================
        fig, axes = plt.subplots(
            nrows=n_quantities,
            ncols=n_models,
            figsize=(
                figsize_per_panel[0] * n_models,
                figsize_per_panel[1] * n_quantities
            ),
            sharex=False,
            sharey=False,
            squeeze=False
        )

        # ============================================================
        # Legend bookkeeping
        # ============================================================
        legend_handles = []
        legend_labels = []

        # ============================================================
        # LOOP OVER QUANTITIES
        # ============================================================
        for i, qname in enumerate(quantity_names):

            quantity_id = f"Q{i + 1}"

            # --------------------------------------------------------
            # Collect residuals across ALL selected models for this
            # quantity.
            #
            # This is used only if residual_limits is not explicitly
            # provided for the quantity.
            # --------------------------------------------------------
            all_residuals_q = []
            max_points = 0

            for model_id in model_ids:

                df_model = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == quantity_id)
                    ]

                if df_model.empty:
                    continue

                cm = df_model[
                    cm_col
                ].to_numpy(dtype=float)

                mm = df_model[
                    mm_col
                ].to_numpy(dtype=float)

                valid = (
                        np.isfinite(cm) &
                        np.isfinite(mm)
                )

                residuals = (
                        cm[valid] -
                        mm[valid]
                )

                if residuals.size > 0:
                    all_residuals_q.append(
                        residuals
                    )

                max_points = max(
                    max_points,
                    len(df_model)
                )

            # ========================================================
            # X-axis limits
            # ========================================================
            if max_points > 0:
                x_limits = (
                    0.5,
                    max_points + 0.5
                )
            else:
                x_limits = (
                    0.5,
                    1.5
                )

            # ========================================================
            # Y-axis residual limits
            # ========================================================

            # --------------------------------------------------------
            # User-defined limits
            # --------------------------------------------------------
            if (
                    residual_limits is not None
                    and quantity_id in residual_limits
            ):

                y_limits = residual_limits[
                    quantity_id
                ]

            # --------------------------------------------------------
            # Automatic fallback
            # --------------------------------------------------------
            else:

                if len(all_residuals_q) > 0:

                    all_residuals_q = np.concatenate(
                        all_residuals_q
                    )

                    max_abs = np.max(
                        np.abs(all_residuals_q)
                    )

                    if np.isclose(
                            max_abs,
                            0.0
                    ):
                        max_abs = 1.0

                    # 5 % margin
                    max_abs *= 1.05

                    y_limits = (
                        -max_abs,
                        max_abs
                    )

                else:

                    y_limits = (
                        -1.0,
                        1.0
                    )

            # --------------------------------------------------------
            # Validate limits
            # --------------------------------------------------------
            if y_limits[0] >= y_limits[1]:
                raise ValueError(
                    f"Invalid residual limits for {quantity_id}: "
                    f"{y_limits}. Lower limit must be smaller "
                    f"than upper limit."
                )

            # ========================================================
            # EXACT SAME NUMBER OF Y TICKS FOR EVERY VARIABLE
            # ========================================================
            y_ticks = np.linspace(
                y_limits[0],
                y_limits[1],
                n_y_ticks
            )

            # Remove tiny floating point values around zero
            y_ticks = np.array([
                0.0
                if np.isclose(tick, 0.0, atol=1e-12)
                else tick
                for tick in y_ticks
            ])

            # ========================================================
            # LOOP OVER MODELS
            # ========================================================
            for j, model_id in enumerate(model_ids):

                ax = axes[i, j]

                df_model = df_spatial[
                    (df_spatial["model_id"] == model_id) &
                    (df_spatial["quantity"] == quantity_id)
                    ]

                # ====================================================
                # Model name
                # ====================================================
                model_name_series = df_summary.loc[
                    df_summary["model_id"] == model_id,
                    "model_name"
                ]

                model_name = (
                    model_name_series.iloc[0]
                    if not model_name_series.empty
                    else f"M{model_id}"
                )

                # ====================================================
                # No data
                # ====================================================
                if df_model.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=16
                    )

                    ax.set_axis_off()

                    continue

                # ====================================================
                # Data
                # ====================================================
                cm = df_model[
                    cm_col
                ].to_numpy(dtype=float)

                mm = df_model[
                    mm_col
                ].to_numpy(dtype=float)

                valid = (
                        np.isfinite(cm) &
                        np.isfinite(mm)
                )

                # ----------------------------------------------------
                # Preserve original spatial indices
                # ----------------------------------------------------
                location_indices = np.arange(
                    1,
                    len(df_model) + 1
                )

                location_indices = (
                    location_indices[valid]
                )

                residuals = (
                        cm[valid] -
                        mm[valid]
                )

                # ====================================================
                # No valid values
                # ====================================================
                if residuals.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=16
                    )

                    ax.set_axis_off()

                    continue

                # ====================================================
                # Residual scatter
                # ====================================================
                group_handles, group_labels = scatter_node_groups(
                    ax,
                    location_indices,
                    residuals,
                    downstream_set,
                    upstream_set,
                    collect_legend=(
                            i == 0 and j == 0
                    ),
                    s=60,
                    alpha=0.8,
                    marker="*"
                )

                legend_handles.extend(
                    group_handles
                )

                legend_labels.extend(
                    group_labels
                )

                # ====================================================
                # Zero residual reference line
                # ====================================================
                ax.axhline(
                    0.0,
                    color="red",
                    linestyle="--",
                    linewidth=1.2
                )

                # ====================================================
                # Axis limits
                #
                # Identical Y limits across all models for the same
                # quantity.
                # ====================================================
                ax.set_xlim(
                    x_limits
                )

                ax.set_ylim(
                    y_limits
                )

                # ====================================================
                # Titles and labels
                #
                # Use normal hyphen instead of Unicode em dash to
                # avoid problems with text.usetex=True.
                # ====================================================
                ax.set_title(
                    f"{model_name} - {qname}",
                    fontsize=18
                )

                ax.set_xlabel(
                    "Location Index",
                    fontsize=16
                )

                # Y label only on first column
                if j == 0:

                    unit = units_map.get(
                        qname,
                        ""
                    )

                    if unit:

                        ax.set_ylabel(
                            f"Residual [{unit}]",
                            fontsize=16
                        )

                    else:

                        ax.set_ylabel(
                            "Residual",
                            fontsize=16
                        )

                # ====================================================
                # X-axis ticks
                # ====================================================
                ax.xaxis.set_major_locator(
                    MaxNLocator(
                        integer=True,
                        nbins=10
                    )
                )

                # ====================================================
                # Y-axis ticks
                #
                # IMPORTANT:
                # Do not use MaxNLocator here.
                #
                # np.linspace guarantees exactly the same number of
                # major ticks for Q1, Q2, Q3, etc.
                # ====================================================
                ax.set_yticks(
                    y_ticks
                )

                # ====================================================
                # Tick appearance
                # ====================================================
                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    labelsize=20
                )

                # ====================================================
                # Grid
                # ====================================================
                ax.grid(
                    True,
                    which="major",
                    linestyle="--",
                    linewidth=0.5,
                    color="gray"
                )

                # Minor ticks
                ax.minorticks_on()

                ax.grid(
                    True,
                    which="minor",
                    linestyle=":",
                    linewidth=0.5,
                    alpha=0.4
                )

                # ====================================================
                # Spines
                # ====================================================
                for spine in ax.spines.values():
                    spine.set_linewidth(
                        1.5
                    )

                # ====================================================
                # Statistics
                # ====================================================
                rmse = np.sqrt(
                    np.mean(
                        residuals ** 2
                    )
                )

                mean_res = np.mean(
                    residuals
                )

                unit = units_map.get(
                    qname,
                    ""
                )

                if unit:

                    metrics_text = (
                        f"RMSE={rmse:.3f} {unit}\n"
                        f"Mean={mean_res:.3f} {unit}"
                    )

                else:

                    metrics_text = (
                        f"RMSE={rmse:.3f}\n"
                        f"Mean={mean_res:.3f}"
                    )

                # ====================================================
                # Metrics box
                # ====================================================
                ax.text(
                    0.98,
                    0.98,
                    metrics_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=12,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="none"
                    )
                )

        # ============================================================
        # General figure legend
        # ============================================================
        if legend_handles:

            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=len(legend_handles),
                fontsize=14,
                framealpha=0.9,
                bbox_to_anchor=(0.5, 0.98)
            )

            fig.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.96
                ]
            )

        else:

            fig.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.95
                ]
            )

        # ============================================================
        # Save figure
        # ============================================================
        if save_folder is not None:
            save_path = os.path.join(
                save_folder,
                "residuals_CM_vs_MM_individual_models.svg"
            )

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            print(
                f"Residuals plot saved to {save_path}"
            )


