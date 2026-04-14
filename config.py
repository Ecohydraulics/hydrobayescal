"""
Configuration File for HydroBayesCal - OpenFOAM InterFoam
Calibration of Cmu turbulence parameter using velocity measurements

Cylinder in Channel Case:
  - ADV measurements 115cm behind cylinder center (x = 4.15m)
  - Two measurement depths: z = 3cm and z = 9cm

Standard Names Reference:
  - U_x, U_y, U_z      -> Velocity components (OpenFOAM: U[0,1,2])
  - U_MAG              -> Velocity magnitude
  - WATER_DEPTH        -> Water depth
  - FREE_SURFACE       -> Free surface elevation
  - ALPHA_WATER        -> Volume fraction (OpenFOAM only)
  - TKE                -> Turbulent kinetic energy (OpenFOAM: k)
  - CMU                -> k-epsilon Cmu parameter
"""

import os

# Base directory
BASE_DIR = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/MU2026-AllRange/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    'case_template_dir': os.path.join(BASE_DIR, ""),
    'model_dir':         os.path.join(BASE_DIR, "simulation2026MU"),
    'res_dir':           os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR,"measurements-calibration.csv"),
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
# INTERFOAM SPECIFIC SETTINGS
# ============================================================================
interfoam = {
    'alpha_water_name':   None,
    'water_surface_alpha': None,
    'reference_z':         None,
}

# ============================================================================
# CALIBRATION PARAMETERS - CMU TURBULENCE COEFFICIENT
# ============================================================================
calibration = {
    # Use "Cmu" to match the key expected by update_model_controls
    'parameters': ["gaiaCLASSES SHIELDS PARAMETERS 1",
                                    "gaiaCLASSES SHIELDS PARAMETERS 2",
                                    "zone2", # Pool
                                    "zone3", # Slackwater
                                    "zone4", # Glide
                                    "zone5", # Riffle
                                    "zone6"], # Run,

    # Cmu range: typical values 0.06-0.12 (default is 0.09)
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

    'dict_output_name': "extraction-data",
}

# ============================================================================
# SAMPLING AND BAL SETTINGS
# ============================================================================
sampling = {
    'init_runs': 100,   # Number of initial parameter samples
    'max_runs':  101,   # Total runs (initial + BAL iterations)

    # Experimental design
    'parameter_distribution':   "uniform",
    'parameter_sampling_method': "sobol",
    'tp_selection_criteria':    "dkl",

    # BAL specific
    'eval_steps':    5,      # Save surrogate and evaluate every iteration
    'prior_samples': 25000,
    'mc_samples_al': 2000,
    'mc_exploration': 1000,
    'gp_library':    "gpy",
}

# ============================================================================
# EXECUTION MODES
# ============================================================================
execution = {
    'complete_bal_mode':      True,
    'only_bal_mode':          True,
    'delete_complex_outputs': True,
    'validation':             False,
    'user_param_values':      False,
}
