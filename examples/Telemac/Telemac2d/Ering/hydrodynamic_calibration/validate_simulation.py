import os
import argparse
import importlib.util

from hydroBayesCal.telemac.control_telemac import TelemacModel
from hydroBayesCal.visualize import BayesianPlotter


def load_config(config_path):
    """
    Load configuration from Python file.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def _collect_train_points(folder, suffix):
    """
    Find all training-point numbers for files matching a given suffix.

    Example
    -------
    gpr_gpy_TP50_bal_quantities_['WATER DEPTH', 'SCALAR VELOCITY']_variables.pkl
    """
    train_points = set()

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Surrogate folder does not exist: {folder}")

    prefix = "gpr_gpy_TP"

    for filename in os.listdir(folder):
        if not filename.startswith(prefix):
            continue

        if not filename.endswith(suffix):
            continue

        try:
            tp_string = filename[len(prefix):].split("_bal_quantities_", 1)[0]
            train_points.add(int(tp_string))
        except (ValueError, IndexError):
            continue

    return train_points


def discover_common_train_points(
    surrogate_folder,
    calibration_quantities,
    multitask_selection,
):
    """
    Return training-point values for which both MO and SO surrogates exist.

    MO:
        ..._[quantities]_variables.pkl

    SO sequential:
        ..._[quantities]_SO_sequential.pkl

    Standard independent SO:
        ..._['WATER DEPTH'].pkl
        ..._['SCALAR VELOCITY'].pkl
        ...
    """
    mo_suffix = (
        f"_bal_quantities_{calibration_quantities}_variables.pkl"
    )

    mo_tps = _collect_train_points(
        surrogate_folder,
        mo_suffix,
    )

    if multitask_selection == "SO_sequential":

        so_suffix = (
            f"_bal_quantities_{calibration_quantities}_SO_sequential.pkl"
        )

        so_tps = _collect_train_points(
            surrogate_folder,
            so_suffix,
        )

    else:
        # Legacy / normal independent SO-GPE files.
        so_tp_sets = []

        for quantity in calibration_quantities:
            so_suffix = (
                f"_bal_quantities_{[quantity]}.pkl"
            )

            so_tp_sets.append(
                _collect_train_points(
                    surrogate_folder,
                    so_suffix,
                )
            )

        # TP must exist for every calibration quantity.
        if so_tp_sets:
            so_tps = set.intersection(*so_tp_sets)
        else:
            so_tps = set()

    common_tps = sorted(mo_tps & so_tps)

    if not common_tps:
        raise FileNotFoundError(
            "No common training-point surrogate files were found.\n"
            f"MO TPs: {sorted(mo_tps)}\n"
            f"SO TPs: {sorted(so_tps)}\n"
            f"Multitask selection: {multitask_selection}\n"
        )

    return common_tps


def save_metrics(
    surrogate_metrics,
    plotter,
    sm_output,
    cm_output,
    sm_upper_ci,
    sm_lower_ci,
    train_points,
    quantity_name,
    surrogate_type,
    n_loc,
    save_location_metrics=True,
):
    """
    Compute surrogate validation metrics and append them to the results dict.
    """

    (
        overall_mse,
        overall_rmse,
        overall_mae,
        overall_corr,
        ci_range_evolution,
        locations_metrics,
        ci_range_evolution_location,
    ) = plotter.compute_evolution_metrics(
        sm_output,
        cm_output,
        sm_upper_ci,
        sm_lower_ci,
        selected_locations=list(range(n_loc)),
    )

    surrogate_metrics["TrainPoints"].append(train_points)
    surrogate_metrics["Quantity"].append(quantity_name)
    surrogate_metrics["SurrogateType"].append(surrogate_type)
    surrogate_metrics["MSE"].append(overall_mse)
    surrogate_metrics["RMSE"].append(overall_rmse)
    surrogate_metrics["MAE"].append(overall_mae)
    surrogate_metrics["Correlation"].append(overall_corr)
    surrogate_metrics["CI"].append(ci_range_evolution)

    if save_location_metrics:

        n_locations_metrics = locations_metrics.shape[0]

        per_loc_data = {
            "TrainPoints": train_points,
            "Quantity": quantity_name,
            "SurrogateType": [surrogate_type] * n_locations_metrics,
            "LocationIdx": list(range(n_locations_metrics)),
            "MSE": locations_metrics[:, 0].tolist(),
            "RMSE": locations_metrics[:, 1].tolist(),
            "MAE": locations_metrics[:, 2].tolist(),
            "Correlation": locations_metrics[:, 3].tolist(),
            "CI": ci_range_evolution_location,
        }

        surrogate_metrics["metrics_per_location"].append(per_loc_data)


def main():

    parser = argparse.ArgumentParser(
        description="Validate MO-GPE and SO-GPE surrogate models."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config_Ering.py",
        help="Path to Python configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # ---------------------------------------------------------
    # Multitask configuration
    # ---------------------------------------------------------

    multitask_selection = config.sampling.get(
        "multitask_selection",
        "variables",
    )

    print(f"Configured multitask selection: {multitask_selection}")

    # ---------------------------------------------------------
    # Initialize TELEMAC model
    # ---------------------------------------------------------

    full_complexity_model = TelemacModel(
        res_dir=config.paths["res_dir"],
        calibration_pts_file_path=config.paths["calibration_pts_file_path"],
        init_runs=40,  # number of validation simulations
        calibration_parameters=config.calibration["parameters"],
        calibration_quantities=config.calibration["calibration_quantities"],
        extraction_quantities=config.calibration["extraction_quantities"],
        validation=True,

        # IMPORTANT
        multitask_selection=multitask_selection,
    )

    # Keep this unchanged throughout the validation.
    calibration_quantities = list(
        full_complexity_model.calibration_quantities
    )

    n_quantities = len(calibration_quantities)
    n_loc = full_complexity_model.nloc

    print(f"Calibration quantities: {calibration_quantities}")
    print(f"Number of quantities: {n_quantities}")
    print(f"Number of locations: {n_loc}")

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    results_folder_path = full_complexity_model.asr_dir
    restart_data_folder = full_complexity_model.restart_data_folder

    surrogate_folder = os.path.join(
        results_folder_path,
        "surrogate-gpe",
        "bal_dkl",
    )

    coordinates = full_complexity_model.calibration_pts_df[
        ["x", "y"]
    ]

    plotter = BayesianPlotter(
        results_folder_path=results_folder_path
    )

    # ---------------------------------------------------------
    # Discover available TP values
    # ---------------------------------------------------------

    surrogate_to_analyze = discover_common_train_points(
        surrogate_folder=surrogate_folder,
        calibration_quantities=calibration_quantities,
        multitask_selection=multitask_selection,
    )

    print(
        "Training-point values available for both surrogate types:",
        surrogate_to_analyze,
    )

    surrogates_to_evaluate = set(surrogate_to_analyze)

    # ---------------------------------------------------------
    # Load validation input set ONCE
    # ---------------------------------------------------------

    validation_sets = full_complexity_model.read_data(
        restart_data_folder,
        "collocation-points-validation.csv",
    )

    # ---------------------------------------------------------
    # Load full-complexity validation results ONCE
    # ---------------------------------------------------------

    cm_outputs = full_complexity_model.output_processing(
        output_data_path=os.path.join(
            restart_data_folder,
            "model-results-validation.json",
        ),
        validation=full_complexity_model.validation,
        filter_outputs=True,
        run_range_filtering=(
            1,
            full_complexity_model.init_runs,
        ),
    )

    expected_columns = n_loc * n_quantities

    if cm_outputs.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected number of columns in validation model outputs.\n"
            f"Expected: {expected_columns} "
            f"({n_loc} locations x {n_quantities} quantities)\n"
            f"Obtained: {cm_outputs.shape[1]}"
        )

    # The MO output arrangement is:
    #
    # P1_q1, P1_q2, P2_q1, P2_q2, ...
    #
    # Therefore idx::n_quantities selects one quantity across
    # all locations.
    cm_outputs_by_quantity = {
        quantity: cm_outputs[:, idx::n_quantities]
        for idx, quantity in enumerate(calibration_quantities)
    }

    # ---------------------------------------------------------
    # Results container
    # ---------------------------------------------------------

    surrogate_metrics = {
        "TrainPoints": [],
        "Quantity": [],
        "SurrogateType": [],
        "MSE": [],
        "RMSE": [],
        "MAE": [],
        "Correlation": [],
        "CI": [],
        "metrics_per_location": [],
    }

    # =========================================================
    # VALIDATION LOOP
    # =========================================================

    for train_points in surrogate_to_analyze:

        print("\n" + "=" * 70)
        print(f"VALIDATING TP = {train_points}")
        print("=" * 70)

        # =====================================================
        # 1. MULTIOUTPUT GPE -- variables
        # =====================================================

        mo_filename = (
            f"gpr_gpy_TP{train_points}"
            f"_bal_quantities_{calibration_quantities}"
            f"_variables.pkl"
        )

        mo_relative_path = os.path.join(
            "surrogate-gpe",
            "bal_dkl",
            mo_filename,
        )

        print(f"Loading MO surrogate: {mo_filename}")

        mo_surrogate = full_complexity_model.read_data(
            results_folder_path,
            mo_relative_path,
        )

        mo_predictions = mo_surrogate.predict_(
            input_sets=validation_sets,
            get_conf_int=True,
        )

        mo_output = mo_predictions["output"]
        mo_upper_ci = mo_predictions["upper_ci"]
        mo_lower_ci = mo_predictions["lower_ci"]

        if mo_output.shape[1] != expected_columns:
            raise ValueError(
                f"MO surrogate TP{train_points} returned "
                f"{mo_output.shape[1]} columns; "
                f"expected {expected_columns}."
            )

        # Evaluate each calibration target separately.
        for idx, quantity_name in enumerate(
            calibration_quantities
        ):

            sm_output_q = mo_output[:, idx::n_quantities]
            sm_upper_q = mo_upper_ci[:, idx::n_quantities]
            sm_lower_q = mo_lower_ci[:, idx::n_quantities]

            cm_output_q = cm_outputs_by_quantity[quantity_name]

            save_metrics(
                surrogate_metrics=surrogate_metrics,
                plotter=plotter,
                sm_output=sm_output_q,
                cm_output=cm_output_q,
                sm_upper_ci=sm_upper_q,
                sm_lower_ci=sm_lower_q,
                train_points=train_points,
                quantity_name=quantity_name,
                surrogate_type="MO",
                n_loc=n_loc,
                save_location_metrics=(
                    train_points in surrogates_to_evaluate
                ),
            )

        # =====================================================
        # 2. SINGLE-OUTPUT GPEs
        # =====================================================

        if multitask_selection == "SO_sequential":

            # -------------------------------------------------
            # NEW structure:
            #
            # SOSequentialGPyTraining
            #     .models["WATER DEPTH"] -> GPyTraining
            #     .models["SCALAR VELOCITY"] -> GPyTraining
            # -------------------------------------------------

            so_filename = (
                f"gpr_gpy_TP{train_points}"
                f"_bal_quantities_{calibration_quantities}"
                f"_SO_sequential.pkl"
            )

            so_relative_path = os.path.join(
                "surrogate-gpe",
                "bal_dkl",
                so_filename,
            )

            print(
                f"Loading sequential SO surrogate container: "
                f"{so_filename}"
            )

            so_container = full_complexity_model.read_data(
                results_folder_path,
                so_relative_path,
            )

            if not hasattr(so_container, "models"):
                raise AttributeError(
                    f"{so_filename} does not contain a "
                    "'models' attribute."
                )

            for quantity_name in calibration_quantities:

                if quantity_name not in so_container.models:
                    raise KeyError(
                        f"Quantity '{quantity_name}' is not present "
                        f"in {so_filename}.\n"
                        f"Available models: "
                        f"{list(so_container.models.keys())}"
                    )

                # Extract the actual GPyTraining object.
                so_surrogate = so_container.models[
                    quantity_name
                ]

                print(
                    f"  Extracting SO model for "
                    f"{quantity_name}"
                )

                so_predictions = so_surrogate.predict_(
                    input_sets=validation_sets,
                    get_conf_int=True,
                )

                sm_output_q = so_predictions["output"]
                sm_upper_q = so_predictions["upper_ci"]
                sm_lower_q = so_predictions["lower_ci"]

                cm_output_q = cm_outputs_by_quantity[
                    quantity_name
                ]

                if sm_output_q.shape[1] != n_loc:
                    raise ValueError(
                        f"Sequential SO model for "
                        f"{quantity_name}, TP{train_points}, "
                        f"returned {sm_output_q.shape[1]} columns; "
                        f"expected {n_loc}."
                    )

                save_metrics(
                    surrogate_metrics=surrogate_metrics,
                    plotter=plotter,
                    sm_output=sm_output_q,
                    cm_output=cm_output_q,
                    sm_upper_ci=sm_upper_q,
                    sm_lower_ci=sm_lower_q,
                    train_points=train_points,
                    quantity_name=quantity_name,
                    surrogate_type="SO",
                    n_loc=n_loc,
                    save_location_metrics=(
                        train_points in surrogates_to_evaluate
                    ),
                )

        else:

            # -------------------------------------------------
            # OLD structure:
            # one completely independent pickle per quantity.
            # -------------------------------------------------

            for quantity_name in calibration_quantities:

                quantity_group = [quantity_name]

                so_filename = (
                    f"gpr_gpy_TP{train_points}"
                    f"_bal_quantities_{quantity_group}.pkl"
                )

                so_relative_path = os.path.join(
                    "surrogate-gpe",
                    "bal_dkl",
                    so_filename,
                )

                print(
                    f"Loading independent SO surrogate: "
                    f"{so_filename}"
                )

                so_surrogate = full_complexity_model.read_data(
                    results_folder_path,
                    so_relative_path,
                )

                so_predictions = so_surrogate.predict_(
                    input_sets=validation_sets,
                    get_conf_int=True,
                )

                sm_output_q = so_predictions["output"]
                sm_upper_q = so_predictions["upper_ci"]
                sm_lower_q = so_predictions["lower_ci"]

                cm_output_q = cm_outputs_by_quantity[
                    quantity_name
                ]

                save_metrics(
                    surrogate_metrics=surrogate_metrics,
                    plotter=plotter,
                    sm_output=sm_output_q,
                    cm_output=cm_output_q,
                    sm_upper_ci=sm_upper_q,
                    sm_lower_ci=sm_lower_q,
                    train_points=train_points,
                    quantity_name=quantity_name,
                    surrogate_type="SO",
                    n_loc=n_loc,
                    save_location_metrics=(
                        train_points in surrogates_to_evaluate
                    ),
                )

    # =========================================================
    # PLOTTING
    # =========================================================

    plotter.plot_metric_comparison(
        surrogate_metrics,
        calibration_quantities,
        metrics=["MAE"],
    )

    plotter.location_metrics(
        surrogate_metrics,
        coordinates,
    )


if __name__ == "__main__":
    main()