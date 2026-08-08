"""
Derive calibrated parameter sets from a finished Bayesian Active Learning run.

Bayesian active learning selects its training points by information gain: the
parameter set that most reduces uncertainty about the posterior, not the one that
best reproduces the measurements. The last training point of a calibration is
therefore not a calibration result, and neither is any single parameter combination
that the exploitation/exploration state happened to propose.

This script reads the joint posterior stored in ``BAL_dictionary.pkl`` and turns it
into the parameter sets that a modeller actually needs:

* the peak of every calibration parameter's own posterior marginal, with credible
  intervals and identifiability flags;
* the joint posterior optimum;
* representatives of the distinct posterior modes, where the calibration is
  equifinal;

together with an explicit verdict on whether the vector assembled from the
independent marginal peaks is a jointly plausible parameter set at all, or whether
parameter correlation makes that combination a point of near-zero posterior density.

The script is report-only and never launches a simulation. With ``--write-csv`` it
writes the candidates to ``restart_data/user-collocation-points.csv``, so the final
full-complexity runs use the existing ``user_param_values`` path:

    python src/hydroBayesCal/drivers/derive_calibrated_parameters.py --config config_Telemac.py --write-csv
    # then, in the config: user_param_values=True, complete_bal_mode=False,
    # only_bal_mode=False, init_runs=<number of candidates>
    python src/hydroBayesCal/drivers/bal_telemac.py --config config_Telemac.py
    python src/hydroBayesCal/drivers/assess_calibration.py

Only ``paths``, ``calibration`` and ``sampling`` are read from the configuration, so
the same script works for the OpenFOAM and Delft3D bindings by swapping the imported
model class.
"""
import argparse
import importlib.util
import os
import pickle

from hydroBayesCal.telemac.control_telemac import TelemacModel
from hydroBayesCal.surrogate.posterior_analysis import (
    analyze_posterior,
    log_posterior_analysis,
    write_candidate_report,
    write_user_collocation_points,
)
from hydroBayesCal.utils.config_logging import logger, logger_warn


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


def load_surrogate(surrogate_path):
    """Unpickle a trained GPE for the ``likelihood`` joint-optimum method."""
    with open(surrogate_path, "rb") as pickle_file:
        return pickle.load(pickle_file)


def main():
    parser = argparse.ArgumentParser(
        description="Derive calibrated parameter sets from the stored BAL posterior "
                    "(report-only; writes CSV files, never launches simulations).")
    parser.add_argument(
        "--config",
        type=str,
        default="config_Telemac.py",
        help="Path to Python configuration file (default: config_Telemac.py)")
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="BAL iteration to analyse. Default -1, i.e. the last iteration with "
             "accepted posterior samples.")
    parser.add_argument(
        "--joint-method",
        type=str,
        default="auto",
        choices=["auto", "kde", "knn", "likelihood"],
        help="How to locate the joint posterior optimum. 'auto' (default) estimates "
             "the density of the accepted sample and needs no surrogate; "
             "'likelihood' re-evaluates a surrogate and needs --surrogate.")
    parser.add_argument(
        "--surrogate",
        type=str,
        default=None,
        help="Path to a pickled GPE, required for --joint-method likelihood.")
    parser.add_argument(
        "--candidates",
        type=str,
        default="marginal_peak,joint_map,posterior_mean,modes",
        help="Comma-separated candidate kinds to assemble.")
    parser.add_argument(
        "--max-modes",
        type=int,
        default=5,
        help="Maximum number of posterior modes to look for (default: 5).")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write restart_data/user-collocation-points.csv, backing up any existing "
             "file. Without this flag the run is report-only.")
    args = parser.parse_args()
    config = load_config(args.config)

    # ---------------------------------------------------------------------
    # Model instance for the result-folder layout and the calibration metadata.
    # user_param_values is deliberately not set here: it would make the model try
    # to read the very CSV this script is about to write.
    # ---------------------------------------------------------------------
    full_complexity_model = TelemacModel(
        res_dir=config.paths["res_dir"],
        calibration_pts_file_path=config.paths["calibration_pts_file_path"],
        init_runs=config.sampling["init_runs"],
        calibration_parameters=config.calibration["parameters"],
        param_values=config.calibration["param_values"],
        calibration_quantities=config.calibration["calibration_quantities"],
    )

    bayesian_data = full_complexity_model.read_data(
        full_complexity_model.calibration_folder, "BAL_dictionary.pkl")
    if bayesian_data is None:
        raise FileNotFoundError(
            f"No BAL_dictionary.pkl in {full_complexity_model.calibration_folder}. "
            f"Run the calibration first.")

    surrogate = load_surrogate(args.surrogate) if args.surrogate else None
    if args.joint_method == "likelihood" and surrogate is None:
        raise ValueError("--joint-method likelihood requires --surrogate <path.pkl>")

    analysis = analyze_posterior(
        bayesian_dict=bayesian_data,
        parameter_names=config.calibration["parameters"],
        prior_bounds=config.calibration["param_values"],
        iteration=args.iteration,
        joint_method=args.joint_method,
        surrogate=surrogate,
        observations=full_complexity_model.observations,
        error=full_complexity_model.variances,
        include=tuple(kind.strip() for kind in args.candidates.split(",") if kind.strip()),
        max_modes=args.max_modes,
    )
    log_posterior_analysis(analysis)

    restart_folder = full_complexity_model.restart_data_folder
    write_candidate_report(analysis, restart_folder)

    n_candidates = len(analysis["candidates"]["labels"])
    if args.write_csv:
        write_user_collocation_points(
            analysis["candidates"], config.calibration["parameters"], restart_folder)

        logger.info("")
        logger.info("To run the full complexity model at these candidate parameter "
                    "sets, set in %s:", os.path.basename(args.config))
        logger.info("    execution['user_param_values']  = True")
        logger.info("    execution['complete_bal_mode']  = False")
        logger.info("    execution['only_bal_mode']      = False")
        logger.info("    sampling['init_runs']           = %d", n_candidates)
        logger.info("and then run:")
        logger.info("    python src/hydroBayesCal/drivers/bal_telemac.py --config %s", args.config)
        logger.info("    python src/hydroBayesCal/drivers/assess_calibration.py --config %s", args.config)
        if config.sampling["init_runs"] != n_candidates:
            logger_warn.warning(
                f"init_runs is currently {config.sampling['init_runs']} but "
                f"{n_candidates} candidate parameter sets were written. The run loop "
                f"is bounded by init_runs, so a smaller value silently ignores "
                f"candidates and a larger one fails with an IndexError.")
    else:
        logger.info(
            f"Report-only run: {n_candidates} candidate parameter sets were analysed "
            f"but not written. Re-run with --write-csv to stage them for the final "
            f"full-complexity runs.")


if __name__ == "__main__":
    main()
