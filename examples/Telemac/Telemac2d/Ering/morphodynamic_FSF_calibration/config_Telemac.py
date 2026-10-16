"""
TELEMAC configuration for HydroBayesCal (TELEMAC-2D + GAIA morphodynamics).

Example: Ering case, surrogate-assisted Bayesian calibration of bed-friction
zones and GAIA critical Shields parameters against measured water depth.

Consumed by ``bal_telemac.py`` via ``--config`` (default: this file); see the
dictionaries below (``paths``, ``hydrodynamic_simulation``,
``morphodynamic_simulation``, ``calibration``, ``sampling``, ``execution``) for
the configurable fields. The OpenFOAM analogue is ``config_OpenFOAM.py``.

Calibration / extraction quantity names refer to TELEMAC SELAFIN variables,
e.g. "WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U/V",
"CUMUL BED EVOL".
"""

import os

# Base directory
BASE_DIR = "/home/modelling/projects-Andres/hbc/hydrobayescal/examples/Telemac/Telemac2d/Ering/morphodynamic_FSF_calibration/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    'case_template_dir': os.path.join(BASE_DIR, ""),
    'model_dir':         os.path.join(BASE_DIR, "simulationFiles"),
    'res_dir':           os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR, "measuredData","measurements-calibration-EringCalib-FSF-CUMULBEDEVOL-post.csv"),
}

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
hydrodynamic_simulation = {
    'solver_name':           "Telemac2d",
    'n_processors':          16,
    'results_filename_base': "results_hydromorphodynamics",
    'control_file':          "Ering_hydromorphodynamics_telemac.cas",
    'friction_file':         "Ering_friction_zones.tbl",
    'fortran_file':          None
}

morphodynamic_simulation = {
    'gaia_cas':                   "Ering_hydromorphodynamics_gaia.cas",
    'gaia_results_filename_base': "results_hydromorphodynamics_gaia",

    'gaia_layer_average': {"LAY SAND RAT": {"sand_classes": [1], "layers": [1, 2], "thicknesses": [0.08, 0.60]}}
    }

# ============================================================================
# CALIBRATION PARAMETERS - TELEMAC FRICTION ZONES + GAIA SHIELDS PARAMETERS
# ============================================================================
calibration = {
    # GAIA critical Shields parameters (per sediment class) and TELEMAC
    # bed-friction zones; names must match update_model_controls / the .cas.
    'parameters': ["gaiaCLASSES SHIELDS PARAMETERS 1",
                   "gaiaCLASSES SHIELDS PARAMETERS 2",
                   #"gaiaCLASSES SHIELDS PARAMETERS 3",
                   "gaiaCLASSES SHIELDS PARAMETERS 3",
                   ], 

    'param_values': [[0.02, 0.080],  # critical shields parameter class 1
                     [0.02, 0.080],
                     #[0.03, 0.070],
                     [0.02, 0.080]],
    # Quantities to extract from simulation - USE STANDARD NAMES
    'extraction_quantities': ["WATER DEPTH","SCALAR VELOCITY","CUMUL BED EVOL","LAY SAND RAT"],

    # Quantities used for BAL calibration - must match columns in measurements.csv
     #'calibration_quantities': ["SAND RAT1","CUMUL BED EVOL"],
     #'calibration_quantities': ["CUMUL BED EVOL"],
     'calibration_quantities': ["LAY SAND RAT"],

    # Three relative error terms, each a fraction of every measured value, added to
    # the observation variance alongside the absolute <target>_ERROR column:
    #   measurement_error       the instrument/campaign is imprecise.
    #   gpe_error               flat stand-in for the emulator's own uncertainty.
    #                           Leave at 0.0 while include_surrogate_error is True:
    #                           the inference then uses the real per-prediction GPE
    #                           standard deviation, and a value here would count the
    #                           same uncertainty twice.
    #   model_structural_error  the solver itself is an imperfect description of the
    #                           site (unresolved processes, geometry, boundary
    #                           conditions). Independent of the emulator and NOT
    #                           supplied by include_surrogate_error. Set it only if
    #                           you can defend a value.
    'measurement_error':      0.05,
    'gpe_error':              0.02,
    'model_structural_error': 0.0,

    'dict_output_name': "extraction-data",
}

# ============================================================================
# SAMPLING AND BAL SETTINGS
# ============================================================================
sampling = {
    'init_runs': 25,   # Number of initial parameter samples
    'max_runs': 100,   # Total runs (initial + BAL iterations)

    # Experimental design
    'parameter_distribution':   "uniform",
    'parameter_sampling_method': "sobol",
    'tp_selection_criteria':    "dkl",

    # BAL specific0
    'eval_steps':    5,      # Save surrogate and evaluate every iteration
    'prior_samples': 25000,
    'mc_samples_al': 2000,
    'mc_exploration': 1000,
    'gp_library':    "gpy",
    'multitask_selection': 'variables',  # 'locations' or 'variables' or 'all' or 'SO_sequential'
    # Feed the GPE predictive standard deviation into the Bayesian inference rather
    # than treating the surrogate predictions as exact. On by default: the emulator's
    # uncertainty is genuine uncertainty, and the BAL utility already accounts for
    # it. Keep calibration['gpe_error'] at 0.0 while this is True.
    'include_surrogate_error': True,
}

# ============================================================================
# OUTPUT EXTRACTION
# ============================================================================
extraction = {
    # When to read the model outputs: "mean_last" averages the last n_last time
    # steps (steady state), "last" takes the final one, "index" a fixed index.
    'output_extraction_time': "mean_last",
    'n_last':                 20,
    'calibration_quantities': ["CUMUL BED EVOL","SAND RAT1"],
    'extraction_quantities': ["CUMUL BED EVOL","SAND RAT1"],
    'time_index': 20,
    'input_slf_file' : "results_hydromorphodynamics_gaia.slf",
}

# ============================================================================
# EXECUTION MODES
# ============================================================================
execution = {
    'complete_bal_mode':      True,
    'only_bal_mode':          False,
    'delete_complex_outputs': True,
    'validation':             False,
}
# ============================================================================
# PLOTTING AND REPORTING SETTINGS
# ============================================================================
plotting = {

    # Used for plotting and reporting - must be in same order as 'parameters'
    'parameter_names': [
        r"$\tau_{*,\mathrm{cr},d_{10}}$",
        r"$\tau_{*,\mathrm{cr},d_{16}}$",
        r"$\tau_{*,\mathrm{cr},d_{50}}$",
        #r"$\tau_{*,\mathrm{cr},d_{70}}$",
    ],
    # Units for reporting and plotting - must be in same order as 'parameters'
    'parameter_units': ["-", "-", "-"],
    # Order of parameters in the BAL posterior arrays - must be in same order as 'parameters', used for plotting selected parameters.
    # When all parameters are plotted all indices must be included.
    'parameter_indices': [0,1,2],
    'iterations_to_plot': [0],
    # -------------------------
    # posterior plotting options
    # -------------------------
    # "posterior_mean",
    # "posterior_marginal_peak",
    # "joint_posterior_MAP"
    'posterior_plotting_option': 'posterior_marginal_peak'
}
