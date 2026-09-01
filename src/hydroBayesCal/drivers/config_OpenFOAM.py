"""
OpenFOAM configuration for HydroBayesCal (interFoam free-surface solver).

Surrogate-assisted Bayesian calibration of an OpenFOAM ``interFoam`` case, e.g.
the turbulence coefficient ``Cmu`` and/or the wall roughness height ``ks``
against measured velocities.

Consumed by ``bal_openfoam.py`` via ``--config`` (default: this file); see the
dictionaries below (``paths``, ``simulation``, ``interfoam``, ``calibration``,
``sampling``, ``execution``) for the configurable fields. The TELEMAC analogue
is ``config_Telemac.py``.

Note the OpenFOAM-specific schema differences from the TELEMAC config:
``simulation`` (not ``hydrodynamic_simulation``), an ``interfoam`` block with
real values, and ``n_cpus`` living in ``sampling``. Quantity names use the
standard HydroBayesCal field names. The OpenFOAM binding extracts "U_x", "U_y",
"U_z", "U_MAG", "TKE", "ALPHA_WATER" and the fluctuation components "u_fluct",
"v_fluct", "w_fluct"; anything else in ``calibration_quantities`` is rejected at
model construction (see ``OpenFOAMModel.EXTRACTABLE_QUANTITIES``). Free-surface
quantities such as "WATER_DEPTH" and "FREE_SURFACE" are available in the Delft3D
and TELEMAC bindings but are not yet extracted here.

Choosing the calibration quantities after the runs: the binding writes *every*
extractable quantity to ``results-detailed-*.csv`` for every run x control point,
regardless of ``calibration_quantities``, and mirrors that table to
``restart_data/detailed-model-outputs.csv``. To re-calibrate on a different
subset without re-running OpenFOAM, change ``calibration_quantities`` (or pass
``--calibration_quantities "U_x TKE"``) and run with ``--only_bal_mode True``:
``output_processing`` rebuilds the surrogate training data for the new subset
from that mirrored table.
"""

import os

# Base directory holding the interFoam case template and results.
BASE_DIR = "/home/user/hydrobayescal/examples/openfoam-case/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    # OpenFOAM case template that is copied for each run.
    'case_template_dir':         os.path.join(BASE_DIR, "interfoam_case_template"),
    'model_dir':                 os.path.join(BASE_DIR, "simulations"),
    'res_dir':                   os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR, "measurements-calibration.csv"),
}

# ============================================================================
# SIMULATION SETTINGS (interFoam)
# ============================================================================
simulation = {
    'solver_name':           "interFoam",          # OpenFOAM solver executable
    'n_processors':          8,                     # subdomains for decomposePar / mpirun
    'results_filename_base': "results_interfoam",
    'control_file':          "system/controlDict",  # OpenFOAM control dictionary
    'n_avg_timesteps':       1,                     # final time steps averaged on extraction
}

# ============================================================================
# INTERFOAM FREE-SURFACE SETTINGS
# ============================================================================
interfoam = {
    'alpha_water_name':    "alpha.water",  # water volume-fraction field
    'water_surface_alpha': 0.5,            # iso-value locating the free surface
    'reference_z':         0.0,            # datum for water-depth / free-surface extraction
}

# ============================================================================
# CALIBRATION PARAMETERS - Cmu TURBULENCE COEFFICIENT + ks WALL ROUGHNESS
# ============================================================================
calibration = {
    # Names must match the dispatch in control_openfoam.run_multiple_simulations
    # (currently "Cmu" -> constant/turbulenceProperties, "ks" -> 0/nut wall BC).
    'parameters': ["Cmu",   # k-epsilon turbulence coefficient
                   "ks"],   # Nikuradse wall roughness height [m]

    # Parameter ranges [min, max] in the same order as 'parameters' above.
    'param_values': [[0.06, 0.12],    # Cmu
                     [0.001, 0.05]],   # ks [m]

    # Quantities to extract from the VTK output - USE STANDARD NAMES.
    # Informational for the OpenFOAM binding: it always writes every quantity in
    # OpenFOAMModel.EXTRACTABLE_QUANTITIES ("U_x","U_y","U_z","U_MAG","TKE",
    # "ALPHA_WATER","u_fluct","v_fluct","w_fluct") to results-detailed-*.csv for
    # every run, so the calibration subset below can be changed afterwards without
    # re-running OpenFOAM (see module docstring, --only_bal_mode).
    'extraction_quantities': ["U_x", "U_y", "U_z", "TKE", "ALPHA_WATER"],

    # Quantities used for BAL calibration - must match columns in measurements.csv.
    # Can be overridden per run with --calibration_quantities; combine with
    # --only_bal_mode True to re-calibrate on a different subset from stored runs.
    'calibration_quantities': ["U_x", "U_y", "U_z"],

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
    'n_cpus':    8,     # CPUs available to the BAL/surrogate layer
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
