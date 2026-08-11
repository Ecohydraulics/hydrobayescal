"""
Delft3D-FLOW configuration for HydroBayesCal.

Surrogate-assisted Bayesian calibration of a Delft3D-FLOW (Delft3D 4 suite)
case, e.g. the uniform bed roughness (Manning/Chezy/White-Colebrook) and the
horizontal eddy viscosity ``Vicouv`` against measured water depths.

Consumed by ``bal_delft3d.py`` via ``--config`` (default: this file); see the
dictionaries below (``paths``, ``simulation``, ``delft3d``, ``calibration``,
``sampling``, ``execution``) for the configurable fields. The TELEMAC and
OpenFOAM analogues are ``config_Telemac.py`` and ``config_OpenFOAM.py``.

Note the Delft3D-specific schema: the case template must contain the
``<case>.mdf`` master definition file, ``config_d_hydro.xml`` and all
attribute files (grid, bathymetry, boundary conditions). Calibration
parameter names are either ``"roughness"`` (uniform bed roughness written to
``Roumet``/``Ccofu``/``Ccofv``) or literal MDF keywords (e.g. ``"Vicouv"``,
``"Dicouv"``). Quantity names use the standard HydroBayesCal field names:
"WATER_LEVEL", "WATER_DEPTH", "U_x", "U_y", "U_MAG".
"""

import os

# Base directory holding the Delft3D-FLOW case template and results.
BASE_DIR = "/home/user/hydrobayescal/examples/delft3d-case/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    # Delft3D-FLOW case template that is copied for each run. Must contain the
    # <case>.mdf, config_d_hydro.xml, grid (.grd/.enc), bathymetry (.dep) and
    # boundary files (.bnd/.bct/...).
    'case_template_dir':         os.path.join(BASE_DIR, "delft3d_case_template"),
    'model_dir':                 os.path.join(BASE_DIR, "simulations"),
    'res_dir':                   os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR, "measurements-calibration.csv"),
}

# ============================================================================
# SIMULATION SETTINGS (Delft3D-FLOW)
# ============================================================================
simulation = {
    # env.sh of the Delft3D-FLOW installation prefix (native build convention,
    # see hydro-informatics.com/get-started/delft3d.html). Sourced before each run.
    'env_script':      "~/opt/delft3d-flow/env.sh",
    # Runtime configuration read by run_dflow2d3d.sh; its <mdfFile> entry
    # names the master definition file.
    'd_hydro_config':  "config_d_hydro.xml",
    'n_processors':    1,    # 1 = run_dflow2d3d.sh, >1 = run_dflow2d3d_parallel.sh
    'n_avg_timesteps': 1,    # final map time steps averaged on extraction
}

# ============================================================================
# DELFT3D-FLOW SPECIFIC SETTINGS
# ============================================================================
delft3d = {
    # Bed-roughness law written to the Roumet keyword when the "roughness"
    # calibration parameter is updated: "Chezy", "Manning" or "WhiteColebrook".
    'roughness_formulation': "Manning",
}

# ============================================================================
# CALIBRATION PARAMETERS - BED ROUGHNESS + HORIZONTAL EDDY VISCOSITY
# ============================================================================
calibration = {
    # "roughness" -> Roumet/Ccofu/Ccofv in the .mdf; any other name is written
    # as a literal MDF keyword (see Delft3DModel.update_model_controls).
    'parameters': ["roughness",  # uniform bed roughness (units follow Roumet)
                   "Vicouv"],    # horizontal eddy viscosity [m2/s]

    # Parameter ranges [min, max] in the same order as 'parameters' above.
    'param_values': [[0.02, 0.04],   # Manning n [s/m^(1/3)]
                     [0.1, 10.0]],   # Vicouv [m2/s]

    # Quantities to extract from the NetCDF map output - USE STANDARD NAMES.
    'extraction_quantities': ["WATER_LEVEL", "WATER_DEPTH", "U_x", "U_y", "U_MAG"],

    # Quantities used for BAL calibration - must match the <quantity>_DATA /
    # <quantity>_ERROR columns in the calibration points CSV.
    'calibration_quantities': ["WATER_DEPTH"],

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
    'n_cpus':    1,     # CPUs available to the BAL/surrogate layer
    'init_runs': 30,    # Number of initial parameter samples
    'max_runs':  50,    # Total runs (initial + BAL iterations)

    # Experimental design. Sobol is the default: a low-discrepancy sequence covers the
    # parameter space far more evenly than random or Latin hypercube sampling at the
    # same cost, and it is extensible, so the staged design below can grow along it.
    # Valid: random, latin_hypercube, sobol, halton, hammersley, chebyshev, grid, user.
    'parameter_distribution':    "uniform",
    'parameter_sampling_method': "sobol",

    # Grow the initial design in Sobol blocks and stop as soon as it is measurably
    # sufficient, instead of always running all init_runs. init_runs stays the ceiling,
    # so this can only ever save simulations; the ones it saves go to BAL iterations.
    # Set to False for the fixed-size initial design of earlier versions.
    'adaptive_init_runs': True,
    'init_runs_min':      None,    # first block; None -> 2**ceil(log2(4 * n_parameters))
    'tp_selection_criteria':     "dkl",

    # BAL specific
    'eval_steps':     1,      # Save surrogate and evaluate every iteration
    'prior_samples':  25000,
    'mc_samples_al':  2000,
    'mc_exploration': 1000,
    'gp_library':     "gpy",

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
# EXECUTION MODES
# ============================================================================
execution = {
    'complete_bal_mode':      True,
    'only_bal_mode':          False,
    'delete_complex_outputs': False,
    'validation':             False,
    'user_param_values':      False,
}
