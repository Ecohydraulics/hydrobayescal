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
BASE_DIR = "/home/modelling/projects-Andres/hbc/hydrobayescal/examples/Telemac/Telemac3d/cylinderFlume/"

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
paths = {
    'case_template_dir': os.path.join(BASE_DIR, ""),
    'model_dir':         os.path.join(BASE_DIR, "simulationFiles"),
    'res_dir':           os.path.join(BASE_DIR,""),
    'calibration_pts_file_path': os.path.join(BASE_DIR, "measuredData", "measuredData_Flume3d_correction.csv"),
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
    'fortran_file':          "cstkep.f"
}
morphodynamic_simulation = {
    'gaia_cas':                   None,
    'gaia_results_filename_base': None,

    'gaia_layer_average': {
        "LAY1 SAND RAT": {
            "layers": None,
            "thicknesses": None
        }
    }
}

# ============================================================================
# CALIBRATION PARAMETERS - TELEMAC FRICTION ZONES + GAIA SHIELDS PARAMETERS
# ============================================================================
calibration = {
    # Use "Cmu" to match the key expected by update_model_controls
    'parameters': ["FRICTION COEFFICIENT FOR THE BOTTOM"], # Run,

    # Cmu range: typical values 0.06-0.12 (default is 0.09)
    'param_values': [
        [0.005, 0.05],  # roughness
         # vertical diffusion coefficient
    ],

    # Quantities to extract from simulation - USE STANDARD NAMES
    'extraction_quantities': ["TURBULENT ENERG", "VELOCITY U", "VELOCITY V", "VELOCITY W","3D VELOCITY MAGNITUDE"],

    # Quantities used for BAL calibration - must match columns in measurements.csv
    'calibration_quantities': ["TURBULENT ENERG", "3D VELOCITY MAGNITUDE"],
    # 'calibration_quantities': ["3D VELOCITY MAGNITUDE"],
    #'calibration_quantities': ["3D VELOCITY MAGNITUDE","VELOCITY U"],
    #'calibration_quantities': ["TURBULENT ENERG"],
 
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
    'measurement_error':      0.0,
    'gpe_error':              0.0,
    'model_structural_error': 0.15,

    'dict_output_name': "extraction-data",
}

# ============================================================================
# SAMPLING AND BAL SETTINGS
# ============================================================================
sampling = {
    'init_runs': 15,   # Number of initial parameter samples
    'max_runs':  60,   # Total runs (initial + BAL iterations)

    # Experimental design
    'parameter_distribution':   "uniform",
    'parameter_sampling_method': "sobol",
    'tp_selection_criteria':    "dkl",

    # BAL specific
    'eval_steps':    2,      # Save surrogate and evaluate every iteration
    'prior_samples': 15000,
    'mc_samples_al': 1000,
    'mc_exploration': 1000,
    'gp_library':    "gpy",
    'multitask_selection': 'variables', # 'locations' or 'variables' or 'all'
    # Feed the GPE predictive standard deviation into the Bayesian inference rather
    # than treating the surrogate predictions as exact. On by default: the emulator's
    # uncertainty is genuine uncertainty, and the BAL utility already accounts for
    # it. Keep calibration['gpe_error'] at 0.0 while this is True.
    'include_surrogate_error': True,
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
# PLOTTING AND REPORTING SETTINGS
# ============================================================================
plotting = {

    # Used for plotting and reporting - must be in same order as 'parameters'
    'parameter_names': [
        r"$k_{\mathrm{s,bed}}$",

    ],
    # Units for reporting and plotting - must be in same order as 'parameters'
    'parameter_units': ["m"],
    # Order of parameters in the BAL posterior arrays - must be in same order as 'parameters', used for plotting selected parameters.
    # When all parameters are plotted all indices must be included.
    'parameter_indices': [0],
    'iterations_to_plot': [2],

    # -------------------------
    # posterior plotting options
    # -------------------------
    # "posterior_mean",
    # "posterior_marginal_peak",
    # "joint_posterior_MAP"
    'posterior_plotting_option': 'posterior_marginal_peak'
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
