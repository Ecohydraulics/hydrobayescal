import os
import time
import numpy as np
import argparse
import importlib.util



from hydroBayesCal.telemac.control_telemac import TelemacModel
from hydroBayesCal.visualize import BayesianPlotter

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

def main():
    # Initialize full complexity model
    parser = argparse.ArgumentParser(description="Assess calibraed deterministic models  against measurements.")
    parser.add_argument(
        '--config',
        type=str,
        default='config_Ering.py',
        help='Path to Python configuration file (default: config_Ering.py)'
    )
    args = parser.parse_args()
    config = load_config(args.config)

    full_complexity_model = TelemacModel(
                # Telemac parameters
                friction_file=config.hydrodynamic_simulation['friction_file'],
                tm_xd=config.hydrodynamic_simulation['solver_name'],
                gaia_steering_file=config.morphodynamic_simulation['gaia_cas'],
                # General hydrosimulation parameters
                results_filename_base=config.hydrodynamic_simulation['results_filename_base'],
                control_file=config.hydrodynamic_simulation['control_file'],
                model_dir=config.paths['model_dir'],
                res_dir=config.paths['res_dir'],
                calibration_pts_file_path=config.paths['calibration_pts_file_path'],
                n_cpus=config.hydrodynamic_simulation['n_processors'],
                init_runs=config.sampling['init_runs'],
                calibration_parameters=config.calibration['parameters'],
                param_values=config.calibration['param_values'],
                extraction_quantities=config.calibration['extraction_quantities'],
                calibration_quantities=config.calibration['calibration_quantities'],
                dict_output_name=config.calibration['dict_output_name'],
                user_param_values=True,
                # max_runs=8,
                # complete_bal_mode=False,
                # only_bal_mode=False,
                # delete_complex_outputs=True,
                # validation=False
                )
    surrogate_to_analyze = 100
    results_folder_path = full_complexity_model.asr_dir
    restart_data_folder = full_complexity_model.restart_data_folder
    plotter = BayesianPlotter(results_folder_path=results_folder_path)
    obs = full_complexity_model.observations
    # err = np.sqrt(full_complexity_model.variances)
    err = full_complexity_model.measurement_errors
    n_loc = full_complexity_model.nloc
    calibration_names = full_complexity_model.calibration_quantities
    n_quantities = full_complexity_model.num_calibration_quantities
    num_simulations = full_complexity_model.init_runs
    coordinates = full_complexity_model.calibration_pts_df[["x", "y"]]
    collocation_points = full_complexity_model.user_collocation_points # To be used for the surrogate model predictions.
    # coordinates = full_complexity_model.calibration_pts_df[["x", "y"]]
    # -------------------------------------------------------------------------
    # Call the surrogate model
    # -------------------------------------------------------------------------
    #
    # surrogate_type = "MO"
    #     Uses the multi-output GPE.
    #
    # surrogate_type = "SO"
    #     Loads one separate single-output GPE pickle per calibration quantity.
    #
    # surrogate_type = "SOSequential"
    #     Loads one SO-sequential pickle containing all quantities.
    #     The pickle has:
    #
    #         sm.models[quantity]
    #
    #     where each entry is a GPyTraining object containing one independent
    #     GP per calibration location/target in its gp_list.
    #
    # All approaches finally produce the same interleaved output layout:
    #
    #     [Q1_loc1, Q2_loc1, ..., Qn_loc1,
    #      Q1_loc2, Q2_loc2, ..., Qn_loc2, ...]
    #
    # -------------------------------------------------------------------------

    surrogate_type = "SO_sequential"  # options: "MO", "SO", "SOSequential"

    start_time = time.time()

    # =========================================================================
    # MULTI-OUTPUT GPE
    # =========================================================================
    if surrogate_type == "MO":

        if n_quantities == 1:
            sm = full_complexity_model.read_data(
                results_folder_path,
                f"surrogate-gpe/bal_dkl/"
                f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
                f"{full_complexity_model.calibration_quantities}.pkl"
            )
        else:
            sm = full_complexity_model.read_data(
                results_folder_path,
                f"surrogate-gpe/bal_dkl/"
                f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
                f"{full_complexity_model.calibration_quantities}_"
                f"{full_complexity_model.multitask_selection}.pkl"
            )

        sm_predictions = sm.predict_(
            input_sets=collocation_points,
            get_conf_int=True
        )

        sm_outputs = np.asarray(sm_predictions["output"])
        sm_upper_ci = np.asarray(sm_predictions["upper_ci"])
        sm_lower_ci = np.asarray(sm_predictions["lower_ci"])


    # =========================================================================
    # SINGLE-OUTPUT GPEs STORED IN SEPARATE PICKLE FILES
    # =========================================================================
    elif surrogate_type == "SO":

        # Number of parameter sets/models to predict.
        n_models = collocation_points.shape[0]

        # Number of calibration/reproduction locations.
        n_points = n_loc

        surrogate_quantities = full_complexity_model.calibration_quantities

        # Empty matrices with same interleaved layout as MO-GPE.
        sm_outputs = np.full(
            (n_models, n_points * n_quantities),
            np.nan
        )
        sm_upper_ci = np.full_like(sm_outputs, np.nan)
        sm_lower_ci = np.full_like(sm_outputs, np.nan)

        for q_idx, quantity in enumerate(surrogate_quantities):

            # -------------------------------------------------------------
            # Load one pickle for this calibration quantity.
            # -------------------------------------------------------------
            sm = full_complexity_model.read_data(
                results_folder_path,
                f"surrogate-gpe/bal_dkl/"
                f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
                f"{[quantity]}.pkl"
            )

            sm_predictions_q = sm.predict_(
                input_sets=collocation_points,
                get_conf_int=True
            )

            output_q = np.asarray(sm_predictions_q["output"])
            upper_q = np.asarray(sm_predictions_q["upper_ci"])
            lower_q = np.asarray(sm_predictions_q["lower_ci"])

            # -------------------------------------------------------------
            # Standardize shapes to:
            #
            #     (n_models, n_points)
            # -------------------------------------------------------------
            if output_q.ndim == 1:
                output_q = output_q.reshape(n_models, n_points)

            if upper_q.ndim == 1:
                upper_q = upper_q.reshape(n_models, n_points)

            if lower_q.ndim == 1:
                lower_q = lower_q.reshape(n_models, n_points)

            if output_q.shape == (n_points, n_models):
                output_q = output_q.T

            if upper_q.shape == (n_points, n_models):
                upper_q = upper_q.T

            if lower_q.shape == (n_points, n_models):
                lower_q = lower_q.T

            # -------------------------------------------------------------
            # Safety checks
            # -------------------------------------------------------------
            if output_q.shape != (n_models, n_points):
                raise ValueError(
                    f"Wrong output shape for SO-GPE quantity '{quantity}'. "
                    f"Expected {(n_models, n_points)}, "
                    f"got {output_q.shape}."
                )

            if upper_q.shape != (n_models, n_points):
                raise ValueError(
                    f"Wrong upper_ci shape for SO-GPE quantity '{quantity}'. "
                    f"Expected {(n_models, n_points)}, "
                    f"got {upper_q.shape}."
                )

            if lower_q.shape != (n_models, n_points):
                raise ValueError(
                    f"Wrong lower_ci shape for SO-GPE quantity '{quantity}'. "
                    f"Expected {(n_models, n_points)}, "
                    f"got {lower_q.shape}."
                )

            # -------------------------------------------------------------
            # Interleave quantities:
            #
            # q_idx = 0 -> columns 0, n_quantities, 2*n_quantities, ...
            # q_idx = 1 -> columns 1, n_quantities+1, ...
            # etc.
            # -------------------------------------------------------------
            sm_outputs[:, q_idx::n_quantities] = output_q
            sm_upper_ci[:, q_idx::n_quantities] = upper_q
            sm_lower_ci[:, q_idx::n_quantities] = lower_q

        sm_predictions = {
            "output": sm_outputs,
            "upper_ci": sm_upper_ci,
            "lower_ci": sm_lower_ci,
        }


    # =========================================================================
    # SO-SEQUENTIAL
    # =========================================================================
    elif surrogate_type == "SO_sequential":

        # Number of parameter sets/models to predict.
        n_models = collocation_points.shape[0]

        # Number of calibration/reproduction locations.
        n_points = n_loc

        surrogate_quantities = full_complexity_model.calibration_quantities
        sm = full_complexity_model.read_data(
            results_folder_path,
            f"surrogate-gpe/bal_dkl/"
            f"gpr_gpy_TP{surrogate_to_analyze}_bal_quantities_"
            f"{full_complexity_model.calibration_quantities}_"
            f"SO_sequential.pkl"
        )

        # -------------------------------------------------------------
        # Check that this really is an SOSequential surrogate.
        # -------------------------------------------------------------
        if not hasattr(sm, "models"):
            raise AttributeError(
                "The loaded SOSequential surrogate does not contain a "
                "'models' attribute."
            )

        if not isinstance(sm.models, dict):
            raise TypeError(
                f"Expected sm.models to be a dictionary, "
                f"got {type(sm.models)}."
            )

        # Check number of locations stored in the sequential model.
        if hasattr(sm, "nloc"):
            if sm.nloc != n_points:
                raise ValueError(
                    f"SOSequential surrogate was trained with {sm.nloc} "
                    f"locations, but n_loc = {n_points}."
                )

        # -------------------------------------------------------------
        # Allocate final interleaved prediction matrices.
        # -------------------------------------------------------------
        sm_outputs = np.full(
            (n_models, n_points * n_quantities),
            np.nan
        )

        sm_upper_ci = np.full_like(sm_outputs, np.nan)
        sm_lower_ci = np.full_like(sm_outputs, np.nan)
        sm_std = np.full_like(sm_outputs, np.nan)

        # -------------------------------------------------------------
        # Loop through quantities stored INSIDE the SOSequential pickle.
        # -------------------------------------------------------------
        for q_idx, quantity in enumerate(surrogate_quantities):

            if quantity not in sm.models:
                raise KeyError(
                    f"Quantity '{quantity}' was not found inside the "
                    f"SOSequential surrogate.\n"
                    f"Available quantities: {list(sm.models.keys())}"
                )

            # ---------------------------------------------------------
            # Extract the GPyTraining object corresponding to this
            # calibration quantity.
            #
            # This object contains n_points independent GPs in gp_list.
            # ---------------------------------------------------------
            sm_quantity = sm.models[quantity]

            if not hasattr(sm_quantity, "predict_"):
                raise AttributeError(
                    f"The SOSequential model for '{quantity}' does not "
                    f"contain a predict_() method."
                )

            if hasattr(sm_quantity, "n_obs"):
                if sm_quantity.n_obs != n_points:
                    raise ValueError(
                        f"SOSequential quantity '{quantity}' contains "
                        f"{sm_quantity.n_obs} independent targets, "
                        f"but n_loc = {n_points}."
                    )

            # ---------------------------------------------------------
            # Predict ALL independent targets for this quantity.
            #
            # GPyTraining.predict_() internally loops through gp_list,
            # so this reproduces all calibration locations using their
            # own independent surrogate.
            #
            # Returned shape:
            #
            #     (n_models, n_points)
            #
            # ---------------------------------------------------------
            predictions_q = sm_quantity.predict_(
                input_sets=collocation_points,
                get_conf_int=True
            )

            output_q = np.asarray(predictions_q["output"])
            upper_q = np.asarray(predictions_q["upper_ci"])
            lower_q = np.asarray(predictions_q["lower_ci"])

            if "std" in predictions_q:
                std_q = np.asarray(predictions_q["std"])
            else:
                # Recover standard deviation from the ±2 sigma CI
                # used by GPyTraining.predict_().
                std_q = (upper_q - lower_q) / 4.0

            # ---------------------------------------------------------
            # Standardize dimensions.
            # ---------------------------------------------------------
            arrays = {
                "output": output_q,
                "upper_ci": upper_q,
                "lower_ci": lower_q,
                "std": std_q,
            }

            for name, array in arrays.items():

                # 1D -> expected 2D
                if array.ndim == 1:
                    if array.size != n_models * n_points:
                        raise ValueError(
                            f"Cannot reshape '{name}' for quantity "
                            f"'{quantity}'. Array contains {array.size} "
                            f"values, while {n_models * n_points} "
                            f"were expected."
                        )

                    array = array.reshape(n_models, n_points)

                # Transposed case.
                if (
                        array.shape == (n_points, n_models)
                        and array.shape != (n_models, n_points)
                ):
                    array = array.T

                if array.shape != (n_models, n_points):
                    raise ValueError(
                        f"Wrong {name} shape for SOSequential quantity "
                        f"'{quantity}'. Expected "
                        f"{(n_models, n_points)}, got {array.shape}."
                    )

                arrays[name] = array

            output_q = arrays["output"]
            upper_q = arrays["upper_ci"]
            lower_q = arrays["lower_ci"]
            std_q = arrays["std"]

            # ---------------------------------------------------------
            # Put this quantity into the common interleaved matrix.
            #
            # For 2 quantities:
            #
            # WATER DEPTH:
            #     columns 0, 2, 4, 6, ...
            #
            # SCALAR VELOCITY:
            #     columns 1, 3, 5, 7, ...
            #
            # resulting in:
            #
            # [h_1, U_1, h_2, U_2, ..., h_37, U_37]
            #
            # ---------------------------------------------------------
            sm_outputs[:, q_idx::n_quantities] = output_q
            sm_upper_ci[:, q_idx::n_quantities] = upper_q
            sm_lower_ci[:, q_idx::n_quantities] = lower_q
            sm_std[:, q_idx::n_quantities] = std_q

        # -------------------------------------------------------------
        # Reconstruct same prediction dictionary used by the rest of
        # the script.
        # -------------------------------------------------------------
        sm_predictions = {
            "output": sm_outputs,
            "std": sm_std,
            "upper_ci": sm_upper_ci,
            "lower_ci": sm_lower_ci,
        }


    # =========================================================================
    # UNKNOWN OPTION
    # =========================================================================
    else:
        raise ValueError(
            "surrogate_type must be one of "
            "'MO', 'SO', or 'SOSequential'."
        )

    end_time = time.time()

    print(
        f"{surrogate_type}-GPE surrogate model predictions took "
        f"{end_time - start_time:.2f} seconds."
    )

    print("sm_outputs shape:", sm_outputs.shape)
    print("sm_upper_ci shape:", sm_predictions["upper_ci"].shape)
    print("sm_lower_ci shape:", sm_predictions["lower_ci"].shape)

    # -------------------------------------------------------------------------
    # This line filters the outputs according to the calibration_quantities.
    cm_outputs = full_complexity_model.output_processing(output_data_path=os.path.join(full_complexity_model.restart_data_folder,
                                                                                              f'user-extraction-data-detailed.json'),
                                                                validation=True, # Putting validation=True to avoid rewriting the full complex model outputs .json (extraction-data-detailed.json) in the original calibration-data according to calibration_quantities= ""..." folder.
                                                                filter_outputs=True, # Filters from the .json file output_data_path the outputs according to the calibration_quantities.
                                                                run_range_filtering=(1, full_complexity_model.init_runs)) # This is to filter the outputs according to the range of runs that we want to analyze accoring to init_runs.
    # --------------------------------------------------------------------------

    # Split columns dynamically for each quantity
    cm_outputs_split = {}
    sm_outputs_split = {}
    sm_upper_ci_split = {}
    sm_lower_ci_split = {}
    obs_split = {}
    err_split = {}

    for i in range(n_quantities):
        # Extract interleaved columns for the current quantity
        cm_outputs_split[f'cm_outputs_{i+1}'] = cm_outputs[:, i::n_quantities]
        sm_outputs_split[f'sm_outputs_{i+1}'] = sm_outputs[:, i::n_quantities]
        sm_upper_ci_split[f'sm_upper_ci_{i+1}'] = sm_predictions["upper_ci"][:, i::n_quantities]
        sm_lower_ci_split[f'sm_lower_ci_{i+1}'] = sm_predictions["lower_ci"][:, i::n_quantities]
        obs_split[f'obs_{i+1}'] = obs[:, i::n_quantities]
        err_split[f'err_{i+1}'] = err[i::n_quantities]
    df_spatial,df_summary= plotter.evaluate_calibration(cm_outputs_split,
                sm_outputs_split,
                sm_upper_ci_split,
                sm_lower_ci_split,
                obs_split,
                err_split,
                coordinates,
                model_names=[
                                     r"MO-GPE (postBAL-JointOpt): $h, \bar{U}$",
                                     r"SO-GPE (postBAL-JointOpt): $h, \bar{U}$",
                                     #r"MO-GPE (preBAL-JointOpt): $h, \bar{U}$",
                                     #r"SO-GPE (preBAL-JointOpt): $h, \bar{U}$",
                                     # r"SO-GPE: $\delta_{z}$",
                                     # r"Benchmark: $k_{s} = \mathrm{mean}$",
                                     #r"Benchmark: $k_{s} = 3 \times d_{50}$"
                                 ],
                quantity_names=calibration_names,
                plot_models=list(range(2)))
    plotter.observed_vs_modeled_compare(df_spatial=df_spatial, df_summary=df_summary, model_ids=[1,2],
                                        quantity_names=[
                                            r"$h$",
                                            r"$\bar{U}$",
                                            #r"$\delta_z$"
                                        ],
                                        points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                        points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                        35, 36, 37]
                                        )
    plotter.surrogate_vs_deterministic_compare(df_spatial=df_spatial, df_summary=df_summary, model_ids=[1,2],
                                        quantity_names=[
                                            r"$h$",
                                            r"$\bar{U}$",
                                            #r"$\delta_z$"
                                        ],
                                        points_group_1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                        points_group_2=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                        35, 36, 37]
                                        )

    plotter.plot_residuals(
        df_spatial,
        df_summary,
        model_ids=[1, 2],
        quantity_names=[
            r"$h$",
            r"$\bar{U}$",
            # r"$\delta_z$"
        ],
        points_group_1=[
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17
        ],
        points_group_2=[
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37
        ],
        residual_limits={
            "Q1": (-0.012, 0.012),  # h
            "Q2": (-0.045, 0.045),  # Ubar
        }
    )


if __name__ == "__main__":
    main()
