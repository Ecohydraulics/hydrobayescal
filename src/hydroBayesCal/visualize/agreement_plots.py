"""
Modeled versus measured calibration targets: the 1:1 scatter that shows whether a
calibration removed a systematic deviation.

One panel per calibration target and calibration state (before / after calibration).
Measured values on the x-axis, modeled values on the y-axis, one open black circle per
calibration point with the measurement uncertainty (instrument error and measured
fluctuations) as horizontal error bars, against the 45-degree line of perfect agreement.
Points consistently above the line mean the model overestimates the target, points
consistently below mean it underestimates it, and a cloud straddling the line means the
residuals are scatter that no calibration parameter can remove.

The series and the verdicts come from
:mod:`~hydroBayesCal.surrogate.target_agreement`; this module only draws them.
"""

import matplotlib.pyplot as plt
import numpy as np

from hydroBayesCal.visualize.base import PlotterBase

#: Short panel headlines per verdict of
#: :func:`~hydroBayesCal.surrogate.target_agreement.diagnose_target_agreement`.
VERDICT_LABELS = {
    "agreement": "agreement within uncertainty",
    "overestimation": "systematic overestimation",
    "underestimation": "systematic underestimation",
    "scatter": "scatter, no systematic offset",
    "unavailable": "no verdict",
}

#: Short roughness readings, added under the column title so that the figure states
#: whether roughness was the identifiable calibration parameter in each state.
ROUGHNESS_LABELS = {
    "roughness_too_high": "roughness identifiable and too high",
    "roughness_too_low": "roughness identifiable and too low",
    "not_identifiable": "roughness not identifiable, second parameter needed",
    "inconclusive": "roughness reading inconclusive",
    "unavailable": "",
}

#: Figure-local font sizes. The global style in
#: :class:`~hydroBayesCal.visualize.base.PlotterBase` is sized for single-panel figures
#: and its 26 pt labels would collide in a grid of square panels.
_FONT_SIZES = {
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
}


def _escape(text, usetex):
    """Make ``text`` safe for the active text renderer."""
    if not usetex:
        return text
    for character in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(character, f"\\{character}")
    return text


class AgreementPlots:
    def plot_target_agreement(
            self,
            data,
            diagnosis=None,
            quantity_labels=None,
            units=None,
            show_model_spread=False,
            file_name="calibration-target-agreement",
            file_formats=("png", "svg"),
            dpi=300,
    ):
        """Scatter the modeled against the measured calibration targets.

        Parameters
        ----------
        data : dict
            Output of
            :func:`~hydroBayesCal.surrogate.target_agreement.target_agreement_data`:
            the measured values, the measurement errors and one modeled series per
            calibration state.
        diagnosis : dict, optional
            Output of
            :func:`~hydroBayesCal.surrogate.target_agreement.diagnose_target_agreement`.
            Supplies the per-panel verdict, bias and RMSE annotation and the roughness
            reading in the column titles. Without it the panels are drawn bare.
        quantity_labels : list of str, optional
            Axis labels per calibration target, e.g. ``[r"$h$", r"$\\bar{U}$"]``.
            Default: the calibration target names from ``data``.
        units : list of str, optional
            Unit strings per calibration target, appended in parentheses to the axis
            labels, e.g. ``["m", "m/s"]``.
        show_model_spread : bool
            ``True`` adds vertical error bars for the central ensemble range of the
            modeled values (the prior ensemble before calibration, the posterior
            predictive after it). Default ``False``: the measurement uncertainty is what
            the 1:1 comparison is judged against, and the vertical bars of a wide prior
            ensemble hide the circles.
        file_name : str
            Base name of the figure file, written to the plotter's save folder.
        file_formats : tuple of str
            File formats to write. Default ``("png", "svg")``.
        dpi : int
            Raster resolution. Default 300.

        Returns
        -------
        list of pathlib.Path
            The files written.
        """
        save_folder = self.save_folder
        save_folder.mkdir(parents=True, exist_ok=True)

        # A LaTeX installation is a system dependency that a compute node may not have,
        # and it only fails when the figure is rendered. The final quality check of a
        # calibration is not the place to lose a run to a missing font package, so a
        # failed render is retried with the built-in mathtext renderer.
        attempts = [True, False] if plt.rcParams["text.usetex"] else [False]
        last_error = None
        for usetex in attempts:
            try:
                with plt.rc_context({"text.usetex": usetex, **_FONT_SIZES}):
                    figure = self._build_target_agreement_figure(
                        data, diagnosis, quantity_labels, units, show_model_spread,
                        usetex)
                    written = []
                    for file_format in file_formats:
                        path = save_folder / f"{file_name}.{file_format}"
                        figure.savefig(path, dpi=dpi, bbox_inches="tight")
                        written.append(path)
                    plt.close(figure)
                return written
            except Exception as error:                            # pragma: no cover
                last_error = error
                plt.close("all")
        raise last_error

    def _build_target_agreement_figure(self, data, diagnosis, quantity_labels, units,
                                       show_model_spread, usetex):
        """Assemble the panel grid: one row per calibration target, one column per state."""
        quantities = data["quantities"]
        n_quantities = data["n_quantities"]
        states = [key for key in ("pre", "post") if key in data["states"]]

        if quantity_labels is None:
            quantity_labels = [_escape(str(quantity), usetex) for quantity in quantities]
        if units is None:
            units = [""] * n_quantities

        figure, axes = plt.subplots(
            nrows=n_quantities, ncols=len(states),
            figsize=(4.6 * len(states), 4.4 * n_quantities),
            squeeze=False)

        measured_all = np.asarray(data["measured"], dtype=float)
        errors_all = np.asarray(data["errors"], dtype=float)

        for row, quantity in enumerate(quantities):
            measured = measured_all[row::n_quantities]
            errors = errors_all[row::n_quantities]

            # One pair of limits per calibration target, so that the panels of a target
            # are comparable before and after calibration and the 45-degree line is a
            # true diagonal in every one of them.
            limits = self._agreement_limits(
                measured, errors,
                [data["states"][key]["modeled"][row::n_quantities] for key in states])

            for column, key in enumerate(states):
                state = data["states"][key]
                axis = axes[row][column]
                modeled = np.asarray(state["modeled"], dtype=float)[row::n_quantities]

                y_error = None
                if show_model_spread:
                    lower = np.asarray(state["lower"], dtype=float)[row::n_quantities]
                    upper = np.asarray(state["upper"], dtype=float)[row::n_quantities]
                    y_error = np.vstack([np.clip(modeled - lower, 0.0, None),
                                         np.clip(upper - modeled, 0.0, None)])

                axis.plot(limits, limits, linestyle="--", color="0.35", linewidth=1.1,
                          zorder=2,
                          label=_escape("1:1 line (perfect agreement)", usetex))
                axis.errorbar(
                    measured, modeled, xerr=errors, yerr=y_error,
                    fmt="o", markersize=5, markerfacecolor="none",
                    markeredgecolor="black", markeredgewidth=0.9,
                    ecolor="0.45", elinewidth=0.8, capsize=2, linestyle="none",
                    zorder=3, label=_escape("calibration points", usetex))

                axis.set_xlim(limits)
                axis.set_ylim(limits)
                axis.set_aspect("equal", adjustable="box")
                axis.grid(True, color="0.9", linewidth=0.6, zorder=0)
                axis.set_axisbelow(True)
                axis.tick_params(axis="both", which="both", direction="in")

                unit = f" ({units[row]})" if units[row] else ""
                axis.set_xlabel(_escape("measured ", usetex) + quantity_labels[row]
                                + _escape(unit, usetex))
                if column == 0:
                    axis.set_ylabel(_escape("modeled ", usetex) + quantity_labels[row]
                                    + _escape(unit, usetex))

                if row == 0:
                    axis.set_title(self._state_title(key, state, diagnosis, usetex),
                                   pad=10)

                annotation = self._panel_annotation(diagnosis, key, quantity, usetex)
                if annotation:
                    # Points above the 1:1 line fill the upper left half of the panel and
                    # points below it the lower right one, so the annotation goes into
                    # whichever corner the residuals leave empty.
                    above = np.nanmean(modeled - measured) > 0
                    position = (0.96, 0.04) if above else (0.04, 0.96)
                    axis.text(*position, annotation, transform=axis.transAxes,
                              va="bottom" if above else "top",
                              ha="right" if above else "left", fontsize=9,
                              bbox={"boxstyle": "round,pad=0.35",
                                    "facecolor": "white",
                                    "edgecolor": "0.75", "alpha": 0.9})

        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(labels),
                      frameon=False, fontsize=10,
                      bbox_to_anchor=(0.5, -0.02))
        figure.tight_layout()
        return figure

    @staticmethod
    def _agreement_limits(measured, errors, modeled_series, pad=0.06):
        """Shared square limits covering the measurements, their error bars and all
        modeled series of one calibration target."""
        values = [np.asarray(measured, dtype=float) - np.asarray(errors, dtype=float),
                  np.asarray(measured, dtype=float) + np.asarray(errors, dtype=float)]
        values.extend(np.asarray(series, dtype=float) for series in modeled_series)
        stacked = np.concatenate([series.ravel() for series in values])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size == 0:
            return (0.0, 1.0)

        low, high = float(np.min(stacked)), float(np.max(stacked))
        span = high - low
        if np.isclose(span, 0.0):
            span = max(abs(high), 1.0)
        margin = pad * span
        return (low - margin, high + margin)

    @staticmethod
    def _state_title(key, state, diagnosis, usetex):
        """Column title: the calibration state, its ensemble size and the roughness
        reading of that state."""
        title = f"{state['label']} ({state['n_members']} runs)"
        reading = ((diagnosis or {}).get("states", {}).get(key, {}) or {}).get("roughness")
        if reading:
            label = ROUGHNESS_LABELS.get(reading.get("verdict"), "")
            if label:
                title += f"\n{label}"
        return _escape(title, usetex)

    @staticmethod
    def _panel_annotation(diagnosis, key, quantity, usetex):
        """Verdict, bias, RMSE and error-bar coverage of one panel."""
        statistics = (((diagnosis or {}).get("states", {}).get(key, {}) or {})
                      .get("targets", {}).get(quantity))
        if not statistics:
            return ""

        lines = [VERDICT_LABELS.get(statistics["verdict"], statistics["verdict"])]
        if np.isfinite(statistics["bias"]):
            lines.append(f"bias = {statistics['bias']:+.3g} "
                         f"({statistics['relative_bias'] * 100:+.1f}%)")
        if np.isfinite(statistics["rmse"]):
            lines.append(f"RMSE = {statistics['rmse']:.3g}")
        if np.isfinite(statistics["coverage"]):
            lines.append(f"within error bars: {statistics['coverage'] * 100:.0f}%")
        return _escape("\n".join(lines), usetex)


class AgreementPlotter(AgreementPlots, PlotterBase):
    """The agreement figure on its own, without the rest of
    :class:`~hydroBayesCal.visualize.BayesianPlotter`.

    The full plotter aggregates every plotting mixin of the package and therefore every
    plotting dependency. This figure is written at the end of each calibration, so it
    composes the minimum it needs: a missing dependency of an unrelated plot must not
    cost a finished calibration its final quality check.
    """

