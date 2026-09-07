"""
Parameter identifiability analysis from HydroBayesCal BAL dictionaries.

The analysis intentionally uses only two posterior-contraction metrics:

1. Variance reduction
       VR = 1 - Var(theta | y) / Var(theta)

2. 95% HDI contraction
       HC = 1 - Width(HDI_post) / Width(HDI_prior)

The number of measurement points per roughness zone is reported as
observational support, but it is NOT combined mathematically with VR or HC.

Example
-------
python parameter_identifiability_analysis.py \
    --config config_Telemac.py \
    --bal-dictionaries \
        BAL_dictionary_SOGPE_U_hydrodynamics.pkl \
        BAL_dictionary_SOGPE_W_hydrodynamics.pkl \
        BAL_dictionary_MOGPE_hydrodynamics.pkl \
    --labels SO-GPE-U SO-GPE-W MO-GPE \
    --output-dir identifiability_results
"""

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Observational support for the Ering hydrodynamic calibration.
# These values are used only for reporting/interpretation.
# ---------------------------------------------------------------------------
DEFAULT_MEASUREMENT_COUNTS = {
    "zone2": 12,  # Pool
    "zone3": 2,   # Slackwater
    "zone4": 8,   # Glide
    "zone5": 3,   # Riffle
    "zone6": 12,  # Run
}


def load_config(config_path):
    """Load a HydroBayesCal Python configuration file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load configuration from {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_bal_dictionary(path):
    """Load one BAL_dictionary*.pkl file."""
    with open(path, "rb") as pickle_file:
        data = pickle.load(pickle_file)

    required = {
        "prior",
        "posterior",
        "calibration_parameters",
        "param_values",
    }
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(
            f"{path} is missing required BAL keys: {sorted(missing)}"
        )

    return data


def shortest_hdi(samples, probability=0.95):
    """
    Return the shortest sample interval containing `probability` of samples.

    This reproduces the HDI construction used by the supplied BAL dictionaries:
    sort the samples and find the narrowest interval spanning floor(p * N)
    sample spacings.

    Parameters
    ----------
    samples : array-like
        One-dimensional posterior/prior samples.
    probability : float
        Probability mass of the HDI. Default is 0.95.

    Returns
    -------
    np.ndarray
        [lower, upper]
    """
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]

    if values.ndim != 1:
        raise ValueError("shortest_hdi expects a one-dimensional sample.")
    if values.size < 2:
        raise ValueError("At least two finite samples are needed for an HDI.")
    if not 0.0 < probability < 1.0:
        raise ValueError("HDI probability must lie between 0 and 1.")

    values = np.sort(values)
    n = values.size

    interval_index = int(np.floor(probability * n))
    interval_index = min(max(interval_index, 1), n - 1)

    widths = values[interval_index:] - values[: n - interval_index]
    start = int(np.argmin(widths))

    return np.array(
        [values[start], values[start + interval_index]],
        dtype=float,
    )


def parse_measurement_counts(text, parameters):
    """
    Parse 'zone2=12,zone3=2,...' into a dictionary.

    If no text is supplied, use DEFAULT_MEASUREMENT_COUNTS.
    """
    counts = dict(DEFAULT_MEASUREMENT_COUNTS)

    if text:
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    f"Invalid measurement-count entry '{item}'. "
                    "Use parameter=count, e.g. zone2=12."
                )
            name, value = item.split("=", 1)
            counts[name.strip()] = int(value)

    missing = [name for name in parameters if name not in counts]
    if missing:
        raise KeyError(
            "No measurement counts are defined for: "
            + ", ".join(missing)
            + ". Supply them with --measurement-counts."
        )

    return counts


def select_iteration(sequence, iteration, name):
    """Select and validate a BAL iteration from a list-like history."""
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(f"BAL key '{name}' must be a list/tuple over iterations.")

    try:
        selected = sequence[iteration]
    except IndexError as exc:
        raise IndexError(
            f"Iteration {iteration} does not exist for BAL key '{name}' "
            f"(available entries: {len(sequence)})."
        ) from exc

    return np.asarray(selected)


def analyze_dictionary(
    bal_data,
    method_label,
    config_parameters,
    config_bounds,
    measurement_counts,
    iteration=-1,
    hdi_probability=0.95,
):
    """
    Compute variance reduction and HDI contraction for one BAL dictionary.
    """
    bal_parameters = list(bal_data["calibration_parameters"])
    if list(config_parameters) != bal_parameters:
        raise ValueError(
            f"Parameter mismatch for {method_label}.\n"
            f"Config: {list(config_parameters)}\n"
            f"BAL:    {bal_parameters}"
        )

    bal_bounds = np.asarray(bal_data["param_values"], dtype=float)
    config_bounds = np.asarray(config_bounds, dtype=float)

    if bal_bounds.shape != config_bounds.shape or not np.allclose(
        bal_bounds, config_bounds
    ):
        raise ValueError(
            f"Parameter bounds in {method_label} do not match the config."
        )

    prior = np.asarray(bal_data["prior"], dtype=float)
    posterior = select_iteration(
        bal_data["posterior"], iteration, "posterior"
    )

    n_parameters = len(bal_parameters)
    expected_shape = (prior.shape[0], n_parameters)

    if prior.ndim != 2 or prior.shape[1] != n_parameters:
        raise ValueError(
            f"Unexpected prior shape {prior.shape}; expected N x {n_parameters}."
        )
    if posterior.ndim != 2 or posterior.shape[1] != n_parameters:
        raise ValueError(
            f"Unexpected posterior shape {posterior.shape}; "
            f"expected N x {n_parameters}."
        )

    # ---------------------------------------------------------------------
    # Metric 1: Variance reduction
    #
    # The supplied HydroBayesCal BAL dictionaries use population variance
    # (numpy ddof=0). Recompute it directly from the stored samples.
    # ---------------------------------------------------------------------
    prior_variance = np.var(prior, axis=0, ddof=0)
    posterior_variance = np.var(posterior, axis=0, ddof=0)

    variance_reduction = 1.0 - posterior_variance / prior_variance

    # If the dictionary stores variance_reduction, verify that our calculation
    # reproduces it at the selected BAL iteration.
    stored_variance_reduction = None
    if "variance_reduction" in bal_data:
        stored_variance_reduction = select_iteration(
            bal_data["variance_reduction"],
            iteration,
            "variance_reduction",
        ).astype(float)

        if not np.allclose(
            variance_reduction,
            stored_variance_reduction,
            rtol=1e-8,
            atol=1e-10,
        ):
            max_difference = np.max(
                np.abs(variance_reduction - stored_variance_reduction)
            )
            print(
                f"WARNING [{method_label}]: recomputed variance reduction "
                f"differs from stored values; max difference={max_difference:.3e}"
            )

    # ---------------------------------------------------------------------
    # Metric 2: HDI contraction
    #
    # Prior HDI is computed from the prior samples using the same shortest-
    # sample-interval method. If a stored 95% marginal_hdi exists, use it
    # for the posterior; otherwise recompute from posterior samples.
    # ---------------------------------------------------------------------
    prior_hdi = np.vstack(
        [
            shortest_hdi(prior[:, j], hdi_probability)
            for j in range(n_parameters)
        ]
    )

    use_stored_hdi = (
        np.isclose(hdi_probability, 0.95)
        and "marginal_hdi" in bal_data
    )

    if use_stored_hdi:
        posterior_hdi = select_iteration(
            bal_data["marginal_hdi"],
            iteration,
            "marginal_hdi",
        ).astype(float)

        # Verify the stored 95% HDI against the posterior samples.
        recomputed_hdi = np.vstack(
            [
                shortest_hdi(posterior[:, j], hdi_probability)
                for j in range(n_parameters)
            ]
        )
        if not np.allclose(
            posterior_hdi,
            recomputed_hdi,
            rtol=1e-8,
            atol=1e-10,
        ):
            print(
                f"WARNING [{method_label}]: stored marginal_hdi does not "
                "exactly match the recomputed 95% shortest-sample HDI."
            )
    else:
        posterior_hdi = np.vstack(
            [
                shortest_hdi(posterior[:, j], hdi_probability)
                for j in range(n_parameters)
            ]
        )

    prior_hdi_width = prior_hdi[:, 1] - prior_hdi[:, 0]
    posterior_hdi_width = posterior_hdi[:, 1] - posterior_hdi[:, 0]

    hdi_contraction = 1.0 - posterior_hdi_width / prior_hdi_width

    records = []
    for j, parameter in enumerate(bal_parameters):
        records.append(
            {
                "parameter": parameter,
                "method": method_label,
                "n_measurement_points": int(measurement_counts[parameter]),
                "n_prior_samples": int(prior.shape[0]),
                "n_posterior_samples": int(posterior.shape[0]),
                "prior_variance": float(prior_variance[j]),
                "posterior_variance": float(posterior_variance[j]),
                "variance_reduction": float(variance_reduction[j]),
                "variance_reduction_percent": float(
                    100.0 * variance_reduction[j]
                ),
                "prior_hdi_lower": float(prior_hdi[j, 0]),
                "prior_hdi_upper": float(prior_hdi[j, 1]),
                "prior_hdi_width": float(prior_hdi_width[j]),
                "posterior_hdi_lower": float(posterior_hdi[j, 0]),
                "posterior_hdi_upper": float(posterior_hdi[j, 1]),
                "posterior_hdi_width": float(posterior_hdi_width[j]),
                "hdi_contraction": float(hdi_contraction[j]),
                "hdi_contraction_percent": float(
                    100.0 * hdi_contraction[j]
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def make_wide_table(results):
    """Create a compact comparison table for reporting."""
    vr = results.pivot(
        index=["parameter", "n_measurement_points"],
        columns="method",
        values="variance_reduction_percent",
    )
    hc = results.pivot(
        index=["parameter", "n_measurement_points"],
        columns="method",
        values="hdi_contraction_percent",
    )

    vr.columns = [f"{col}_VR_percent" for col in vr.columns]
    hc.columns = [f"{col}_HDI_contraction_percent" for col in hc.columns]

    wide = pd.concat([vr, hc], axis=1).reset_index()
    return wide


def parameter_display_names(config, parameters):
    """Use plotting.parameter_names when available; otherwise raw names."""
    plotting = getattr(config, "plotting", None)
    if isinstance(plotting, dict):
        names = plotting.get("parameter_names")
        if names is not None and len(names) == len(parameters):
            return dict(zip(parameters, names))
    return {name: name for name in parameters}


def plot_metric(
    results,
    metric_column,
    ylabel,
    output_path,
    display_names,
):
    """Grouped bar plot of one identifiability metric."""
    pivot = results.pivot(
        index="parameter",
        columns="method",
        values=metric_column,
    )

    # Preserve config/BAL parameter order rather than alphabetical ordering.
    parameter_order = list(dict.fromkeys(results["parameter"].tolist()))
    pivot = pivot.reindex(parameter_order)

    ax = pivot.plot(kind="bar", figsize=(10, 5.5), width=0.8)
    ax.set_xlabel("Calibration parameter")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(
        [display_names.get(p, p) for p in pivot.index],
        rotation=0,
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.legend(title="Calibration approach")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def print_summary(results, wide):
    """Print concise tables to the terminal."""
    print("\nPARAMETER IDENTIFIABILITY ANALYSIS")
    print("=" * 80)
    print(
        "Metrics: variance reduction and 95% HDI contraction.\n"
        "Measurement-point counts are observational support only; "
        "they are not included in either metric.\n"
    )

    console = results[
        [
            "parameter",
            "method",
            "n_measurement_points",
            "n_posterior_samples",
            "variance_reduction_percent",
            "hdi_contraction_percent",
        ]
    ].copy()

    console["variance_reduction_percent"] = (
        console["variance_reduction_percent"].round(2)
    )
    console["hdi_contraction_percent"] = (
        console["hdi_contraction_percent"].round(2)
    )

    print(console.to_string(index=False))

    print("\nCOMPACT COMPARISON TABLE")
    print("-" * 80)
    print(wide.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assess parameter identifiability from one or more HydroBayesCal "
            "BAL dictionaries using variance reduction and HDI contraction."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_Telemac.py",
        help="Path to config_Telemac.py (default: config_Telemac.py).",
    )
    parser.add_argument(
        "--bal-dictionaries",
        nargs="+",
        required=True,
        help=(
            "Paths to BAL_dictionary*.pkl files. Supply one or several "
            "files to compare calibration approaches."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Labels corresponding to --bal-dictionaries, e.g. "
            "SO-GPE-U SO-GPE-W MO-GPE. If omitted, file stems are used."
        ),
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="BAL iteration to analyze; default -1 = final iteration.",
    )
    parser.add_argument(
        "--hdi-probability",
        type=float,
        default=0.95,
        help="HDI probability mass; default 0.95.",
    )
    parser.add_argument(
        "--measurement-counts",
        type=str,
        default=None,
        help=(
            "Optional comma-separated override, e.g. "
            "'zone2=12,zone3=2,zone4=8,zone5=3,zone6=12'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parameter_identifiability",
        help="Directory for CSV tables and PNG figures.",
    )
    args = parser.parse_args()

    if args.labels is not None and (
        len(args.labels) != len(args.bal_dictionaries)
    ):
        raise ValueError(
            "--labels must contain exactly one label per BAL dictionary."
        )

    config = load_config(args.config)

    config_parameters = list(config.calibration["parameters"])
    config_bounds = list(config.calibration["param_values"])

    measurement_counts = parse_measurement_counts(
        args.measurement_counts,
        config_parameters,
    )

    labels = (
        args.labels
        if args.labels is not None
        else [Path(path).stem for path in args.bal_dictionaries]
    )

    frames = []
    for path, label in zip(args.bal_dictionaries, labels):
        bal_data = load_bal_dictionary(path)
        frame = analyze_dictionary(
            bal_data=bal_data,
            method_label=label,
            config_parameters=config_parameters,
            config_bounds=config_bounds,
            measurement_counts=measurement_counts,
            iteration=args.iteration,
            hdi_probability=args.hdi_probability,
        )
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    wide = make_wide_table(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_csv = output_dir / "parameter_identifiability_long.csv"
    wide_csv = output_dir / "parameter_identifiability_comparison.csv"

    results.to_csv(long_csv, index=False)
    wide.to_csv(wide_csv, index=False)

    display_names = parameter_display_names(config, config_parameters)

    vr_plot = output_dir / "variance_reduction.png"
    hdi_plot = output_dir / "hdi_contraction.png"

    plot_metric(
        results=results,
        metric_column="variance_reduction_percent",
        ylabel="Variance reduction (%)",
        output_path=vr_plot,
        display_names=display_names,
    )
    plot_metric(
        results=results,
        metric_column="hdi_contraction_percent",
        ylabel=f"{100 * args.hdi_probability:.0f}% HDI contraction (%)",
        output_path=hdi_plot,
        display_names=display_names,
    )

    print_summary(results, wide)

    print("\nFiles written:")
    print(f"  {long_csv}")
    print(f"  {wide_csv}")
    print(f"  {vr_plot}")
    print(f"  {hdi_plot}")


if __name__ == "__main__":
    main()
