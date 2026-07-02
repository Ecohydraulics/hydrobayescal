"""
Code that plots BAL posterior results.

Plots:
- BME and RE evolution over BAL iterations
- Posterior histograms of uncertain parameters

Author: Andres Heredia Hidalgo MSc
"""

import argparse
import importlib.util
import numpy as np

from src.hydroBayesCal.telemac.control_telemac import TelemacModel
from src.hydroBayesCal.plots.plots import BayesianPlotter


def load_config(config_path):
    """
    Load configuration from Python file.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():

    parser = argparse.ArgumentParser(
        description="Plot BAL posterior results from BAL_dictionary.pkl"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.py",
        help="Path to Python configuration file. Default: config.py"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # ---------------------------------------------------------------------
    # Instance of Telemac model for reading paths and calibration metadata
    # ---------------------------------------------------------------------
    full_complexity_model = TelemacModel(
        res_dir=config.paths["res_dir"],
        calibration_pts_file_path=config.paths["calibration_pts_file_path"],
        init_runs=config.sampling["init_runs"],
        calibration_parameters=config.calibration["parameters"],
        param_values=config.calibration["param_values"],
        calibration_quantities=config.calibration["calibration_quantities"],
    )

    results_folder_path = full_complexity_model.asr_dir
    quantities_str = "_".join(full_complexity_model.calibration_quantities)

    plotter = BayesianPlotter(
        results_folder_path=results_folder_path,
        variable_name=quantities_str
    )

    # ---------------------------------------------------------------------
    # User settings
    # ---------------------------------------------------------------------
    iterations_to_plot = config.plotting["iterations_to_plot"]  # BAL iteration to plot (0-based index)



    # ---------------------------------------------------------------------
    # Read BAL dictionary
    # ---------------------------------------------------------------------
    bayesian_data = full_complexity_model.read_data(
        full_complexity_model.calibration_folder,
        "BAL_dictionary.pkl"
    )

    posterior_arrays = bayesian_data["posterior"]
    prior = bayesian_data["prior"]

    # ---------------------------------------------------------------------
    # Find last valid posterior if requested iteration is empty
    # ---------------------------------------------------------------------
    valid_posterior_iterations = [
        i for i, posterior in enumerate(posterior_arrays)
        if posterior is not None and np.asarray(posterior).size > 0
    ]

    if len(valid_posterior_iterations) == 0:
        raise ValueError("No valid posterior found in BAL_dictionary.pkl")

    last_valid_iteration = valid_posterior_iterations[-1]

    if iterations_to_plot not in valid_posterior_iterations:
        print(
            f"Iteration {iterations_to_plot} has no valid posterior. "
            f"Using last valid posterior iteration: {last_valid_iteration}"
        )
        iterations_to_plot = last_valid_iteration

    print(f"Plotting posterior iteration: {iterations_to_plot}")

    # ---------------------------------------------------------------------
    # Plot BME and RE evolution
    # ---------------------------------------------------------------------
    # plotter.plot_bme_re(
    #     bayesian_dict=bayesian_data,
    #     num_bal_iterations=iterations_to_plot,
    #     plot_type="both"
    # )

    # ---------------------------------------------------------------------
    # Plot posterior updates
    # ---------------------------------------------------------------------
    plotter.plot_posterior_updates(
        posterior_arrays=posterior_arrays,
        parameter_names=config.plotting["parameter_names"],
        prior=prior,
        param_values=full_complexity_model.param_values,
        iterations_to_plot=[iterations_to_plot],
        bins=10,
        plot_prior=True,
        parameter_units=config.plotting["parameter_units"],
        parameter_indices = config.plotting["parameter_indices"],
    )


if __name__ == "__main__":
    main()