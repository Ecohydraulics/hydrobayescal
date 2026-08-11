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
BASE_DIR = "/home/user/hydrobayescal/examples/Telemac/Hydromorphodynamic/Ering/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    'case_template_dir': os.path.join(BASE_DIR, ""),
    'model_dir':         os.path.join(BASE_DIR, "simulationFiles"),
    'res_dir':           os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR, "measuredData", "measurements-calibration-EringCalib.csv"),
}

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
hydrodynamic_simulation = {
    'solver_name':           "Telemac2d",
    'n_processors':          16,
    'results_filename_base': "results2m3",
    'control_file':          "tel_ering_initial_NIKU.cas",
    'friction_file':         "friction_ering_MU_initial_NIKU.tbl", #Telemac friction file (if needed)
    'fortran_file':          None
}
morphodynamic_simulation= {
    'gaia_cas':                     "gaia_ering_initial_NIKU.cas",
    'gaia_results_filename_base':   "results2m3",
}

# ============================================================================
# CALIBRATION PARAMETERS - TELEMAC FRICTION ZONES + GAIA SHIELDS PARAMETERS
# ============================================================================
calibration = {
    # GAIA critical Shields parameters (per sediment class) and TELEMAC
    # bed-friction zones; names must match update_model_controls / the .cas.
    'parameters': ["gaiaCLASSES SHIELDS PARAMETERS 1",
                                    "gaiaCLASSES SHIELDS PARAMETERS 2",
                                    "zone2", # Pool
                                    "zone3", # Slackwater
                                    "zone4", # Glide
                                    "zone5", # Riffle
                                    "zone6"], # Run,

    # Parameter ranges [min, max] in the same order as 'parameters' above.
    'param_values': [[0.047, 0.070],  # critical shields parameter class 1
                          [0.047, 0.070],  # critical shields parameter class 2
                          [0.002, 0.6],  # zone2
                          [0.002, 0.6],  # zone3
                          [0.002, 0.6],  # zone4
                          [0.002, 0.6],  # zone5
                          [0.002, 0.6]],

    # Quantities to extract from simulation - USE STANDARD NAMES
    'extraction_quantities': ["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V","CUMUL BED EVOL"],

    # Quantities used for BAL calibration - must match columns in measurements.csv
    'calibration_quantities': ["WATER DEPTH"],

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
    'measurement_error':      0.10,
    'gpe_error':              0.0,
    'model_structural_error': 0.0,

    'dict_output_name': "extraction-data",
}

# ============================================================================
# SAMPLING AND BAL SETTINGS
# ============================================================================
sampling = {
    'init_runs': 100,   # Number of initial parameter samples
    'max_runs':  101,   # Total runs (initial + BAL iterations)

    # Experimental design. Sobol is the default: a low-discrepancy sequence covers the
    # parameter space far more evenly than random or Latin hypercube sampling at the
    # same cost, and it is extensible, so the staged design below can grow along it.
    # Valid: random, latin_hypercube, sobol, halton, hammersley, chebyshev, grid, user.
    'parameter_distribution':   "uniform",
    'parameter_sampling_method': "sobol",

    # Grow the initial design in Sobol blocks and stop as soon as it is measurably
    # sufficient, instead of always running all init_runs. init_runs stays the ceiling,
    # so this can only ever save simulations; the ones it saves go to BAL iterations.
    # Set to False for the fixed-size initial design of earlier versions.
    'adaptive_init_runs': True,
    'init_runs_min':      None,    # first block; None -> 2**ceil(log2(4 * n_parameters))
    'tp_selection_criteria':    "dkl",

    # BAL specific
    'eval_steps':    5,      # Save surrogate and evaluate every iteration
    'prior_samples': 25000,
    'mc_samples_al': 2000,
    'mc_exploration': 1000,
    'gp_library':    "gpy",

    # Feed the GPE predictive standard deviation into the Bayesian inference rather
    # than treating the surrogate predictions as exact. On by default: the emulator's
    # uncertainty is genuine uncertainty, and the BAL utility already accounts for
    # it. Keep calibration['gpe_error'] at 0.0 while this is True.
    'include_surrogate_error': True,

    # Exploration/exploitation of the BAL point selection. 'auto' (default) exploits the
    # posterior until the per-iteration diagnostic finds more than one well-separated
    # mode, then adds exploration for the remaining iterations: pure exploitation keeps
    # refining the mode it started in, which is how a local maximum survives to the end
    # of a calibration. True or False force the behaviour.
    'bal_exploration_tradeoff': 'auto',
}

# ============================================================================
# OUTPUT EXTRACTION
# ============================================================================
extraction = {
    # When to read the model outputs: "mean_last" averages the last n_last time
    # steps (steady state), "last" takes the final one, "index" a fixed index.
    'output_extraction_time': "mean_last",
    'n_last':                 80,
}

# ============================================================================
# EXECUTION MODES
# ============================================================================
execution = {
    'complete_bal_mode':      True,
    'only_bal_mode':          False,
    'delete_complex_outputs': True,
    'validation':             False,
    'user_param_values':      False,
}
