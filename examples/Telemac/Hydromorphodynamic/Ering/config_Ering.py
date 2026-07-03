"""
Configuration File for HydroBayesCal - Telemac2d/3d
"""

import os

# Base directory
# Base directory of this example (resolved relative to this file, so the
# example runs from any clone location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    'case_template_dir': os.path.join(BASE_DIR, ""),
    'model_dir':         os.path.join(BASE_DIR, "simulationFiles"),
    'res_dir':           os.path.join(BASE_DIR),
    'calibration_pts_file_path': os.path.join(BASE_DIR,"measuredData","measurements-calibration-EringCalib.csv"),
}

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
hydrodynamic_simulation = {
    'solver_name':           "Telemac2d",
    'n_processors':          16,
    'results_filename_base': "results2m3_preBAL",
    'control_file':          "tel_ering_initial_NIKU.cas",
    'friction_file':         "friction_ering_MU_initial_NIKU.tbl", #Telemac friction file (if needed)
    'fortran_file':          None
}
morphodynamic_simulation= {
    'gaia_cas':                     "gaia_ering_initial_NIKU.cas",
    'gaia_results_filename_base':   "resultsGAIA2m3_preBAL",
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
    'extraction_quantities': ["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG", "VELOCITY U", "VELOCITY V", "CUMUL BED EVOL"],

    # Quantities used for BAL calibration - must match columns in measurements.csv
     'calibration_quantities': ["WATER DEPTH", "SCALAR VELOCITY", "CUMUL BED EVOL"],
    # 'calibration_quantities': ["WATER DEPTH", "SCALAR VELOCITY"],
    # 'calibration_quantities': ["WATER DEPTH"],
    #  'calibration_quantities': ["SCALAR VELOCITY"],
    # 'calibration_quantities': ["CUMUL BED EVOL"],

    'dict_output_name': "extraction-data",
}

# ============================================================================
# SAMPLING AND BAL SETTINGS
# ============================================================================
sampling = {
    'init_runs': 7,   # Number of initial parameter samples
    'max_runs':  7,   # Total runs (initial + BAL iterations)

    # Experimental design
    'parameter_distribution':   "uniform",
    'parameter_sampling_method': "user",
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
    'complete_bal_mode':      False,
    'only_bal_mode':          False,
    'delete_complex_outputs': True,
    'validation':             False,
    'user_param_values':      True,
}
# ============================================================================
# PLOTTING AND REPORTING SETTINGS
# ============================================================================
plotting = {

    # Used for plotting and reporting - must be in same order as 'parameters'
    'parameter_names': [
        r"$\tau_{*,\mathrm{cr},d_{10}}$",
        r"$\tau_{*,\mathrm{cr},d_{16}}$",
        r"$k_{\mathrm{s,pool}}$",
        r"$k_{\mathrm{s,slack}}$",
        r"$k_{\mathrm{s,glide}}$",
        r"$k_{\mathrm{s,riff}}$",
        r"$k_{\mathrm{s,run}}$"
    ],
    # Units for reporting and plotting - must be in same order as 'parameters'
    'parameter_units': ["-", "-", "m", "m", "m", "m", "m"],
    # Order of parameters in the BAL posterior arrays - must be in same order as 'parameters', used for plotting selected parameters.
    # When all parameters are plotted all indices must be included.
    'parameter_indices': [0, 1, 2, 3, 4, 5, 6],
    'iterations_to_plot': 70,
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