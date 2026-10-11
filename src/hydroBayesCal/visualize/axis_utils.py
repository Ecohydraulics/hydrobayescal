"""
Shared axis, tick, and limit helpers for the visualize package.

These functions were previously duplicated as closures inside several
BayesianPlotter methods.
"""

import numpy as np
from matplotlib.ticker import FormatStrFormatter


def set_grid_style(ax):
    """Apply the dashed light-grey grid and thin spines used across plots."""
    ax.grid(True, linestyle='--', color='lightgrey', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)  # Set axis border thickness


def adjust_margins(fig):
    """Apply the default subplot margins used across plots."""
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)


def set_nice_ticks(ax, axis='x', n_ticks=5, start_at_zero=False):
    """Set n_ticks 'nice' ticks (1-2-2.5-5-10 sequence) on the given axis."""
    if axis == 'x':
        vmin, vmax = ax.get_xlim()
    else:
        vmin, vmax = ax.get_ylim()

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    if start_at_zero:
        vmin = 0.0

    if np.isclose(vmin, vmax):
        delta = 0.1 * abs(vmin) if vmin != 0 else 0.1
        vmin -= delta
        vmax += delta

    raw_step = (vmax - vmin) / (n_ticks - 1)

    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    step = nice_fraction * (10 ** exponent)

    if start_at_zero:
        ticks = step * np.arange(n_ticks)
    else:
        center = 0.5 * (vmin + vmax)
        total_span = step * (n_ticks - 1)
        tick_min = center - total_span / 2
        ticks = tick_min + step * np.arange(n_ticks)

    ticks = np.round(ticks, 10)

    if axis == 'x':
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_xticks(ticks)
    else:
        ax.set_ylim(ticks[0], ticks[-1])
        ax.set_yticks(ticks)


def padded_limits(values, pad=0.08):
    """Min/max of values padded by a fraction of the span."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (-1.0, 1.0)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        delta = 0.1 * abs(vmin) if vmin != 0 else 0.1
        return vmin - delta, vmax + delta

    dv = vmax - vmin
    return vmin - pad * dv, vmax + pad * dv


def tight_metric_limits(values, pad_fraction=0.12, min_span_fraction=1e-4, min_abs_span=1e-12):
    """
    Axis limits for metrics with very small inter-model differences.

    Uses the real min/max range when available.
    If values are almost identical, it creates only a small artificial span,
    instead of expanding by 10% of the metric magnitude.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (-1.0, 1.0)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    center = 0.5 * (vmin + vmax)
    span = vmax - vmin

    scale = max(abs(vmin), abs(vmax), abs(center), 1.0)
    min_span = max(min_abs_span, min_span_fraction * scale)

    if span < min_span:
        half_span = 0.5 * min_span
        return center - half_span, center + half_span

    pad = pad_fraction * span
    return vmin - pad, vmax + pad


def set_adaptive_decimal_formatter(ax, axis='both', values=None, max_decimals=2):
    """
    Uses adaptive decimals, but caps the number of decimals.

    max_decimals=2 means the axis labels will never show more than two decimals.
    """
    if values is None:
        decimals = max_decimals
    else:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            decimals = max_decimals
        else:
            span = np.nanmax(values) - np.nanmin(values)

            if span < 1e-4:
                decimals = 6
            elif span < 1e-3:
                decimals = 5
            elif span < 1e-2:
                decimals = 4
            elif span < 1e-1:
                decimals = 3
            else:
                decimals = 2

            # Cap the number of decimals.
            decimals = min(decimals, max_decimals)

    formatter = FormatStrFormatter(f'%.{decimals}f')

    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)

    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)


def symmetric_limits(values, pad=0.10):
    """Limits centered on the mean of values, symmetric around it."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (-1.0, 1.0)

    y_center = np.mean(values)
    y_range = np.max(np.abs(values - y_center))

    if np.isclose(y_range, 0.0):
        y_range = 0.1 * abs(y_center) if y_center != 0 else 0.1

    margin = pad * y_range
    return y_center - y_range - margin, y_center + y_range + margin


def nice_step(raw_step):
    """
    Round step up to a 'nice' value based on the 1-2-2.5-5-10 sequence.
    """
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0

    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    for nice_fraction in [1.0, 2.0, 2.5, 5.0, 10.0]:
        if fraction <= nice_fraction:
            return nice_fraction * (10 ** exponent)

    return 10.0 ** (exponent + 1)


def format_tick_label(value):
    """
    Format tick labels like 0, 0.05, 0.1, 0.15, 0.5
    without useless trailing zeros.
    """
    if np.isclose(value, 0.0, atol=1e-12):
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def compute_nice_limits(min_val, max_val, n_ticks=6):
    """
    Compute nice axis limits with exactly n_ticks.
    Rules:
    - If all data are >= 0, axis starts at 0.
    - If all data are <= 0, axis ends at 0.
    - If data cross 0, adapt to nice rounded limits.
    """
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return 0.0, 1.0

    if np.isclose(min_val, max_val):
        # Expand degenerate case
        if min_val >= 0:
            min_val = 0.0
            max_val = max(max_val * 1.2, 1.0)
        elif max_val <= 0:
            max_val = 0.0
            min_val = min(min_val * 1.2, -1.0)
        else:
            delta = max(abs(min_val), abs(max_val), 1.0) * 0.5
            min_val -= delta
            max_val += delta

    tol = 1e-12

    # Case 1: all non-negative -> force start at 0
    if min_val >= -tol:
        start = 0.0
        raw_step = (max_val - start) / (n_ticks - 1)
        step = nice_step(raw_step if raw_step > 0 else max(abs(max_val), 1.0) / (n_ticks - 1))
        end = start + (n_ticks - 1) * step

        while end < max_val - tol:
            step = nice_step(step * 1.001)
            end = start + (n_ticks - 1) * step

        return start, end

    # Case 2: all non-positive -> force end at 0
    if max_val <= tol:
        end = 0.0
        raw_step = (end - min_val) / (n_ticks - 1)
        step = nice_step(raw_step if raw_step > 0 else max(abs(min_val), 1.0) / (n_ticks - 1))
        start = end - (n_ticks - 1) * step

        while start > min_val + tol:
            step = nice_step(step * 1.001)
            start = end - (n_ticks - 1) * step

        return start, end

    # Case 3: mixed negative / positive values
    raw_step = (max_val - min_val) / (n_ticks - 1)
    step = nice_step(raw_step)

    start = np.floor(min_val / step) * step
    end = start + (n_ticks - 1) * step

    while end < max_val - tol:
        step = nice_step(step * 1.001)
        start = np.floor(min_val / step) * step
        end = start + (n_ticks - 1) * step

    return start, end


def scatter_node_groups(ax, x_values, y_values, downstream_set, upstream_set,
                        collect_legend=False, **scatter_kwargs):
    """
    Scatter x/y values split into downstream (gray), upstream (black), and
    other (blue) node groups. Node positions are 1-based indices along the
    arrays. Without any group sets, all points are plotted in black.

    Returns (handles, labels) of the plotted groups when collect_legend is
    True, otherwise ([], []).
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    n_points = len(x_values)

    handles = []
    labels = []

    if downstream_set is None and upstream_set is None:
        ax.scatter(x_values, y_values, color='black', **scatter_kwargs)
        return handles, labels

    downstream_mask = np.array(
        [node_position in downstream_set for node_position in range(1, n_points + 1)],
        dtype=bool
    ) if downstream_set is not None else np.zeros(n_points, dtype=bool)

    upstream_mask = np.array(
        [node_position in upstream_set for node_position in range(1, n_points + 1)],
        dtype=bool
    ) if upstream_set is not None else np.zeros(n_points, dtype=bool)

    other_mask = ~(downstream_mask | upstream_mask)

    for mask, color, label in (
            (downstream_mask, 'gray', 'Downstream nodes'),
            (upstream_mask, 'black', 'Upstream nodes'),
            (other_mask, 'blue', 'Other nodes'),
    ):
        if np.any(mask):
            handle = ax.scatter(x_values[mask], y_values[mask], color=color, **scatter_kwargs)
            if collect_legend:
                handles.append(handle)
                labels.append(label)

    return handles, labels
