"""
Modeled versus measured calibration targets, before and after calibration.

The closing quality check of a calibration, in observation space: for every calibration
target, the modeled values are scattered against the measured ones as open black circles
with the measurement uncertainty (instrument error plus measured fluctuations) as error
bars, against the 45-degree line of perfect agreement. Two columns, one per calibration
state:

* **before calibration** - the initial-design ensemble of full-complexity runs, i.e. the
  model across the prior range of the calibration parameters;
* **after calibration** - the posterior predictive, i.e. the surrogate at the parameter
  sets that rejection sampling accepted.

Points consistently above the 1:1 line mean the model overestimates that calibration
target, points consistently below mean it underestimates it, and a cloud straddling the
line means the residuals are scatter that no calibration-parameter value can remove. The
column titles carry the roughness reading of
:func:`~hydroBayesCal.function_pool.diagnose_roughness_identifiability` for that state,
so the figure states whether roughness was the identifiable calibration parameter and
whether the calibration moved it in the right direction.

The ``bal_*`` drivers already write this figure when a calibration finishes. This script
reproduces it from archived results, with different thresholds or labels, and prints the
verdicts again:

    python src/hydroBayesCal/drivers/plot_target_agreement.py --config config_Telemac.py
    python src/hydroBayesCal/drivers/plot_target_agreement.py \\
        --bal-dictionary <res_dir>/auto-saved-results-HydroBayesCal/calibration-data/<targets>/BAL_dictionary.pkl

The ``--bal-dictionary`` form needs no model instance and no configuration, so it works
on TELEMAC, OpenFOAM and Delft3D results alike. It is report-only and never launches a
simulation.
"""
import argparse
import importlib.util
import pathlib
import pickle

from hydroBayesCal.surrogate.target_agreement import (
    AGREEMENT_KEY,
    diagnose_target_agreement,
    log_target_agreement,
    plot_target_agreement,
)
from hydroBayesCal.utils.config_logging import logger


def load_config(config_path):
    """
    Load configuration from Python file.

    Parameters
    ----------
    config_path : str
        Path to the Python configuration file

    Returns
    -------
    module
        Configuration module with all variables
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def locate_bal_dictionary(path):
    """Resolve a file, or the single ``BAL_dictionary.pkl`` under a results folder."""
    path = pathlib.Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.glob("**/BAL_dictionary.pkl"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"No BAL_dictionary.pkl under {path}.")
        raise ValueError(
            f"{len(matches)} BAL_dictionary.pkl files under {path}; name one of them "
            f"explicitly:\n  " + "\n  ".join(str(match) for match in matches))
    raise FileNotFoundError(f"{path} is neither a file nor a folder.")


def bal_dictionary_from_config(config_path):
    """Result-folder layout of a configuration, via a TELEMAC model instance.

    Imported lazily so that the ``--bal-dictionary`` form of this script stays free of
    the TELEMAC stack and usable for OpenFOAM and Delft3D results.
    """
    from hydroBayesCal.telemac.control_telemac import TelemacModel

    config = load_config(config_path)
    model = TelemacModel(
        res_dir=config.paths["res_dir"],
        calibration_pts_file_path=config.paths["calibration_pts_file_path"],
        init_runs=config.sampling["init_runs"],
        calibration_parameters=config.calibration["parameters"],
        param_values=config.calibration["param_values"],
        calibration_quantities=config.calibration["calibration_quantities"],
    )
    return pathlib.Path(model.calibration_folder) / "BAL_dictionary.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Plot modeled vs. measured calibration targets before and after "
                    "calibration, from a stored BAL_dictionary.pkl (report-only).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=str,
        help="Path to the Python configuration file of the calibration, from which the "
             "results folder is derived (TELEMAC layout).")
    source.add_argument(
        "--bal-dictionary",
        type=str,
        help="Path to BAL_dictionary.pkl, or to a results folder containing exactly "
             "one. Solver-agnostic; needs no configuration.")
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.02,
        help="A mean residual below this fraction of the mean measured value counts as "
             "agreement (default: 0.02).")
    parser.add_argument(
        "--sign-threshold",
        type=float,
        default=0.6,
        help="Fraction of calibration points that must share the sign of the mean "
             "residual for the deviation to count as systematic rather than scatter "
             "(default: 0.6).")
    parser.add_argument(
        "--show-model-spread",
        action="store_true",
        help="Add vertical error bars for the central range of the modeled ensemble "
             "(prior ensemble before calibration, posterior predictive after it).")
    parser.add_argument(
        "--units",
        type=str,
        default=None,
        help="Comma-separated unit strings, one per calibration target, appended to the "
             "axis labels (e.g. 'm,m/s,m').")
    parser.add_argument(
        "--file-name",
        type=str,
        default="calibration-target-agreement",
        help="Base name of the figure file (default: calibration-target-agreement).")
    args = parser.parse_args()

    if args.config:
        dictionary_path = bal_dictionary_from_config(args.config)
    else:
        dictionary_path = locate_bal_dictionary(args.bal_dictionary)

    with open(dictionary_path, "rb") as archive:
        bayesian_dict = pickle.load(archive)

    stored = bayesian_dict.get(AGREEMENT_KEY)
    if not stored:
        raise KeyError(
            f"{dictionary_path} carries no '{AGREEMENT_KEY}' block. It was written by a "
            f"calibration that ran before this diagnostic existed; the series it needs "
            f"(the initial-design outputs and the posterior predictive) are not stored "
            f"anywhere else in the file. Re-run the calibration in only_bal_mode to "
            f"rebuild the surrogate from the stored runs and write the block.")

    data = stored["data"]
    # Re-diagnosed rather than read back, so that --deadband and --sign-threshold have
    # an effect; the stored diagnosis is the one from the calibration's own thresholds.
    diagnosis = log_target_agreement(
        diagnose_target_agreement(data, deadband=args.deadband,
                                  sign_threshold=args.sign_threshold))

    # <res_dir>/auto-saved-results-HydroBayesCal/calibration-data/<targets>/BAL_dictionary.pkl
    # -> the plot goes to <res_dir>/auto-saved-results-HydroBayesCal/plots/<targets>/
    results_folder = dictionary_path.parents[2]
    variable_name = dictionary_path.parent.name

    units = ([unit.strip() for unit in args.units.split(",")] if args.units else None)
    if units is not None and len(units) != data["n_quantities"]:
        raise ValueError(f"{len(units)} units for {data['n_quantities']} calibration "
                         f"targets {data['quantities']}.")

    written = plot_target_agreement(
        data,
        diagnosis=diagnosis,
        results_folder_path=results_folder,
        variable_name=variable_name,
        units=units,
        show_model_spread=args.show_model_spread,
        file_name=args.file_name,
    )
    for path in written:
        logger.info(f"Calibration-target agreement plot written to {path}")


if __name__ == "__main__":
    main()
