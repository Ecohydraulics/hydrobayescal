"""
Configuration File for HydroBayesCal - Telemac
"""

import os

# Base directory
BASE_DIR = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/Telemac/Telemac3d/cylinderFlume/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    'case_template_dir': os.path.join(BASE_DIR, ""),
    'model_dir':         os.path.join(BASE_DIR,"simulationFiles"),
    'res_dir':           os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR, "measuredData", "measuredData_Flume3d.csv"),
}

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
hydrodynamic_simulation = {
    'solver_name':           "Telemac3d",
    'n_processors':          16,
    'results_filename_base': "3d-ref-2cm-0.5-3d-BAL",
    'control_file':          "3d_cylinder_2cm_BAL.cas",
    'friction_file':         None, #Telemac friction file (if needed)
    'fortran_file':          "cstkep.f",
}
morphodynamic_simulation= {
    'gaia_cas':                     None,
    'gaia_results_filename_base':   None,
}

# ============================================================================
# INTERFOAM SPECIFIC SETTINGS
# ============================================================================
interfoam = {
    'alpha_water_name':   "alpha.water",
    'water_surface_alpha': 0.5,
    'reference_z':         0.0,
}

# ============================================================================
# CALIBRATION PARAMETERS - CMU TURBULENCE COEFFICIENT
# ============================================================================
calibration = {
    # Use "Cmu" to match the key expected by update_model_controls
    'parameters': ["FRICTION COEFFICIENT FOR THE BOTTOM"], # Run,

    # Cmu range: typical values 0.06-0.12 (default is 0.09)
    'param_values': [[0.01,0.07]],

    # Quantities to extract from simulation - USE STANDARD NAMES
    'extraction_quantities': ['VELOCITY U','VELOCITY V','VELOCITY W','TURBULENT ENERG','DISSIPATION','3D VELOCITY MAGNITUDE'],

    # Quantities used for BAL calibration - must match columns in measurements.csv
    #'calibration_quantities': ["TURBULENT ENERG", "VELOCITY U", "VELOCITY V"],
    'calibration_quantities': ['VELOCITY U','VELOCITY V','3D VELOCITY MAGNITUDE'],
    # 'calibration_quantities': ["3D VELOCITY MAGNITUDE"],


    'dict_output_name': "extraction-data",
}

# ============================================================================
# SAMPLING AND BAL SETTINGS
# ============================================================================
sampling = {
    'init_runs': 2,   # Number of initial parameter samples
    'max_runs':  40,   # Total runs (initial + BAL iterations)

    # Experimental design
    'parameter_distribution':   "uniform",
    'parameter_sampling_method': "sobol",
    'tp_selection_criteria':    "dkl",

    # BAL specific
    'eval_steps':    1,      # Save surrogate and evaluate every iteration
    'prior_samples': 15000,
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
# ============================================================================
# EXTRACTION OPTIONS
# ============================================================================
extraction = {
    'output_extraction_time': "mean_last",  # Options: "mean_last", "last", "index"
    'time_index':             100,          # Time index for extraction (if needed)
    'n':                      80,           # Number of time steps to average (if needed)
    # -----------------------------------------------------
    # UNCOMMENT THIS for 3d .slf file extraction (example)
    # ------------------------------------------------------
    # 'extraction_quantities': ['VELOCITY U','VELOCITY V','VELOCITY W','TURBULENT ENERG','DISSIPATION','3D VELOCITY MAGNITUDE'],
    # 'calibration_quantities': ['VELOCITY U', 'VELOCITY V', '3D VELOCITY MAGNITUDE'],
    # -----------------------------------------------------
    # UNCOMMENT THIS for 2d .slf file extraction (example)
    # ------------------------------------------------------
    'extraction_quantities': ['VELOCITY U','VELOCITY V','FROUDE NUMBER','FRICTION VELOCI','WATER DEPTH'],
    'calibration_quantities': ['VELOCITY U','VELOCITY V'],
    'input_slf_file': '3d-ref-2cm-0.5-2d.slf'  # Use this when extracting data from a .slf file independent from BAL

}