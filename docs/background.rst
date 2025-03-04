.. Full complexity model


Complex model - Metamodel Calibration Workflow
================================================

The workflow describes the use of Bayesian Model Evidence (BME) and Relative Entropy (RE) in conjunction with a Gaussian Process Emulator,
as proposed by `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_, to iteratively improve the accuracy of a surrogate model applied
for the calibration of full-complexity hydrodynamic models.

The following steps outline the process for performing a GPE surrogate-assisted calibration of any hydrodynamic model using open-source
hydrodynamic software. Currently, model calibration is supported only with Telemac.

Step 0: Wet your TELEMAC Model
------------------------------

Before the surrogate-assisted calibration can run, it needs an initial model run. The initial model needs to be fully functional with all the required simulation files.
The first model run should start with `dry conditions (read more at hydro-informatics.com) <https://hydro-informatics.com/numerics/telemac2d-steady.html>`_ and
be adapted to `wet (steady or unsteady hotstart) initial conditions <https://hydro-informatics.com/numerics/telemac2d-unsteady.html#hotstart-initial-conditions>`_ for the surrogate-assisted calibration.

.. note:: **Why hotstart the model for the surrogate-assisted calibration?**

    A hotstart simulation involves re-using the output file (.slf) of a previous simulation that began under dry conditions as a file containing the new initial conditions.
    In a typical numerical model of a fluvial ecosystem, it is common to start with dry conditions to prevent filling disconnected terrain depressions with water. However, applying wet initial
    conditions that approximately correspond to the target conditions can significantly speed up convergence.
    To expedite surrogate-assisted calibration, it is recommended to perform one dry model initialization. Afterwards, switch to fast-converging hotstart (wet initial) conditions.


Step 1: Assign user input parameters
-------------------------------

As it was mentioned before the calibration process involves two well defined parts in the code. Both processes depend on the user defined input parameters, which are essential
to run the code properly.
Firstly, the initialization of all input parameters must be done in ``bal_telemac.py`` Python script. ``bal_telemac.py`` is the main script that runs the calibration process and
calls the necessary instances of the classes that run the hydrodynamic model, creation of surrogate models and BAL.

.. _HydroSimulations_class:
------------------------
HydroSimulations Class (Global Full Complexity Model Parameters)
------------------------
The **HydroSimulations** class manages and runs hydrodynamic simulations within the context of Bayesian Calibration using a Gaussian Process Emulator (GPE). The class is designed to handle simulation setup,
execution, and result storage while managing calibration parameters and Bayesian Active Learning (BAL) iterations.

This class contains the general attributes that a hydrodynamic simulation requires to run. The attributes are:

* **control_file**: Name of the file that controls the full complexity model simulation (default is "control.cas" as an example for Telemac).

* **model_dir**: Full complexity model directory where all simulation files (mesh, control file, boundary conditions) are located.

* **res_dir**: Directory where a subfolder called "auto-saved-results-HydroBayesCal" will be created to store all the result files.
  Additionally, subfolders for plots, surrogate models, and restart data will be created.

* **calibration_pts_file_path**: File path to the calibration points data file. Please check documentation for further details of the file format.

.. table:: Measurement Data

   ======================= ================== ================== ====================== =============== ====================== ===============
   Point                   X                  Y                  MEASUREMENT 1           ERROR 1        MEASUREMENT 2           ERROR 2
   ======================= ================== ================== ====================== =============== ====================== ===============
   [Point data row 1]      [X value]          [Y value]          [Measurement 1 value]  [Error 1 value]  [Measurement 2 value]  [Error 2 value]
   [Point data row 2]      [X value]          [Y value]          [Measurement 1 value]  [Error 1 value]  [Measurement 2 value]  [Error 2 value]
   [Point data row 3]      [X value]          [Y value]          [Measurement 1 value]  [Error 1 value]  [Measurement 2 value]  [Error 2 value]
   ======================= ================== ================== ====================== =============== ====================== ===============

* **n_cpus**: Number of CPUs to be used for parallel processing (if available).

* **init_runs**: Initial runs of the full complexity model (before Bayesian Active Learning).

* **calibration_parameters**: Names of the considered calibration parameters (e.g., roughness coefficients, empirical constants, turbulent viscosity, etc.),
  any uncertain parameter that can be introduced in the numerical model for calibration purposes.

  * **Notes**:

    * No limit in the number of calibration parameters.
    * For Telemac users, the calibration parameters **MUST** coincide with the **KEYWORD** in Telemac found in the `.cas` file.
      The notation should be BOTTOM FRICTION = 0.025 in the `.cas` file. **IMPORTANT: (with ' = ' not with ' : ')**
      You can find more details in the `Telemac User Manuals <https://wiki.opentelemac.org/doku.php#principal_documentation>`_.

    .. code-block:: python

       calibration_parameters = ["LAW OF FRICTION ON LATERAL BOUNDARIES", "INITIAL ELEVATION", "BOTTOM FRICTION"]  # Correspond to KEYWORDS in TELEMAC .cas file

    * If you want to calibrate different values of roughness coefficients in roughness zones, the roughness zones description MUST be indicated in the .tbl file.
    * The friction zone name **MUST** be indicated in the friction file .tbl. More information on friction zones in Telemac in `Friction (Roughness) Zones <https://hydro-informatics.com/numerics/telemac/roughness.html>`_
    * The calibration zone **MUST** contain the word zone,ZONE or Zone as a prefix in the calib_parameter field.

    .. code-block:: python

       calibration_parameters = ['zone1', 'zone2', 'Zone3','ZONE99999100']  # 3 friction zones numbered as 1, 2, and 3

* **param_values**: Value ranges considered for parameter sampling.

    .. code-block:: python

       param_values = [[min1, max1], [min2, max2], ...]

* **calibration_quantities**: Names of the calibration targets (model outputs) used for calibration.

    .. code-block:: python

       calibration_quantities = ['WATER DEPTH']  # Single quantity
       calibration_quantities = ['WATER DEPTH', 'SCALAR VELOCITY']  # Multiple quantities


* **extraction_quantities**: Quantities to be extracted from the model output files. Generally, these are the same as or more than the **calibration_quantities**. These quantities will be extracted from the model and used for calibration purposes (using any quantity) when restarting it with the option ``only_bal_mode = True``.

    .. code-block:: python

      calibration_quantities = ['WATER DEPTH'] # WATER DEPTH as a calibration parameter.
      extraction_quantities = ['WATER DEPTH', 'SCALAR VELOCITY', 'TURBULENT ENERG', 'VELOCITY U', 'VELOCITY V'] # Calibration and additional quantities to be extracted.

    Any of these additional extracted quantities can be used for calibration purposes.
* **dict_output_name**: Base name for output dictionary files where the outputs are saved as `.json` files.

* **parameter_sampling_method**: Method used for sampling parameter values during the calibration process.

    Available options:

  * **"random"** - Random sampling.
  * **"latin_hypercube"** - Latin Hypercube Sampling (LHS).
  * **"sobol"** - Sobol sequence sampling.
  * **"halton"** - Halton sequence sampling.
  * **"hammersley"** - Hammersley sequence sampling.
  * **"chebyshev(FT)"** - Chebyshev nodes (Fourier Transform-based).
  * **"grid(FT)"** - Grid-based sampling (Fourier Transform-based).
  * **"user"** - User-defined sampling.

    If "user" is selected, a ``.csv`` file containing user-defined collocation points must be provided
    in the restart data folder. The file should follow this format:

.. table:: User-Defined Collocation Points

       ================== ================== ================== ================== ==================
       param1            param2              param3             param4             param5
       ================== ================== ================== ================== ==================
       0.148             0.770               0.014              0.014              0.700
       0.066             0.066               0.066              0.066              0.066
       ================== ================== ================== ================== ==================

* **max_runs**: Maximum (total) number of model simulations, including initial runs and Bayesian Active Learning iterations.

* **complete_bal_mode**: (Default: ``True``)

  - If ``True``: Bayesian Active Learning (BAL) is performed after the initial runs, enabling a complete surrogate-assisted calibration process.
    **This option MUST be selected if you choose to perform only BAL** (i.e., when ``only_bal_mode = True``).
  - If ``False``: Only the initial runs of the full complexity model are executed, and the model outputs are stored as ``.json`` files.

* **only_bal_mode**: (Default: ``False``)

  - If ``False``: The process will either execute a complete surrogate-assisted calibration or only the initial runs, depending on the value of ``complete_bal_mode``.
  - If ``True``: Only the surrogate model construction and Bayesian Active Learning of preexisting model outputs at predefined collocation points are performed.
    **This mode can be executed only if either a complete process has already been performed** (``complete_bal_mode = True`` and ``only_bal_mode = True``)
    **or if only the initial runs have been executed** (``complete_bal_mode = False`` and ``only_bal_mode = False``).

* **validation**: (Default: ``False``)
  If ``True``, creates output files (inputs and outputs) for validation of the surrogate model. If it is True,
the validation data is saved in the restart data folder.

* **Shortcut Combinations and Their Corresponding Tasks**:

.. table:: Task Descriptions

   ===================== =============================== ============================================================================
   **complete_bal_mode**  **only_bal_mode**               **Task Description**
   ===================== =============================== ============================================================================
   True                  False                            Complete surrogate-assisted calibration
   False                 False                            Only initial runs (no surrogate model)
   True                  True, with ``init_runs = max_runs``  Only surrogate construction with a set of predefined runs (no BAL)
   True                  True, with ``init_runs > max_runs``  Surrogate model construction and Bayesian Active Learning (BAL) applied
   ===================== =============================== ============================================================================


------------------------
TelemacModel Class (Telemac specific parameters)
------------------------

For telemac simulations, the following parameters should be defined in the **TelemacModel** class if necesarry:

* **friction_file** :
  Name of the friction file .tbl to be used in Telemac simulations (should end with ``.tbl``); do not include the directory path.

* **tm_xd** :
  Specifies the Telemac hydrodynamic solver, either ``Telemac2d`` or ``Telemac3d``.

.. code-block:: text

   tm_xd = "1"  # Telemac 2D
   tm_xd = "2"  # Telemac 3D

* **gaia_steering_file**:
  Name of the Gaia steering file; should be provided if required. Not implemented in this HydroBayesCal version.

* **results_filename_base** :
  Base name for the results file, which will be iteratively updated in the ``.cas`` file.
  This indicates the base name of the results file. In each run, the results file changes so
  it is used for data extraction.

.. code-block:: text

    results_filename_base="results"



Step 2: Data storage and extraction
-----------------------------------------
In each run, HydroBayesCal creates a results folder called "auto-saved-results-HydroBayesCal" in the directory specified by the user.
This folder contains the following subfolders:

* **calibration-data**: Contains the calibration data for the considered calibration quantity/ies. The files are:

  * **``BAL_dictionary.pkl``**: Contains the Bayesian Active Learning data after iterations, including: prior distributions, posterior distributions, observations, errors, BME, RE.
  * **``collocation-points-<CALIBRATION_QUANTITY>.csv``**: Stores the collocation points used in the calibration process, including initial collocation points and those added during the BAL iterations for the specified <CALIBRATION_QUANTITY>.
  * **``extraction-data-detailed.json``**: Contains the output data as a dictionary JSON from the complex model simulations, for all collocation points and for the variables in ``extraction_quantities``.
  * **``model-results-calibration-<CALIBRATION_QUANTITY>.csv``**: Stores the model results used for all collocation points and for the specified <CALIBRATION_QUANTITY> as .csv file.
  * **``model-results-extraction.csv``**: Contains extracted model results from the simulations and for the variables in ``extraction_quantities``.
  * **``<CALIBRATION_QUANTITY>-detailed.json``**: Provides a detailed JSON file of the extracted <CALIBRATION_QUANTITY> for each collocation points and location.

* **plots**: Stores the plots after the calibration process. The Python script called plots.py is used to generate the plots.
* **surrogate_models**: Contains the surrogate models created during the calibration process. The surrogate models are saved as pickle files.
* **restart_data**: Contains the restart data for the calibration process. Typically, the files saved in this folder are:
  * **``initial-collocation-points.csv``**: Contains the initial collocation points (parameter combinations) for the calibration process. The number of
        collocation points corresponds to the value assigned in init_runs.
  * **``initial-outputs.json``**: Contains the initial outputs (expressed in extraction_quantities) of the full complexity model at the collocation points as dictionary .json.


Step 3: Bayesian Model Optimization
-----------------------------------

With the initial model setup and the measurement points, the Bayesian model optimization process has everything it needs for its iterative score calculation. The number of iterations corresponds to the user-defined limit in **``max_runs``** and the following tasks are performed in every iteration:

1. Initial surrogate model with the initial collocation points and the corresponding model outputs:
    * **Training a initial metamodel** using single or multitask Gaussian Process Regression. To train a GP metamodel, a coviariance function (kernel) must be defined.
        - `Single GP Regression  <https://docs.gpytorch.ai/en/v1.13/examples/01_Exact_GPs/Simple_GP_Regression.html>`_
        - `Multi-task GP Regression <https://docs.gpytorch.ai/en/v1.13/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html>`_
        - `Gaussian Process Kernels <https://docs.gpytorch.ai/en/v1.13/kernels.html>`_
    *  **Surrogate model predictions**  using the trained metamodel to predict the model outputs at  Monte Carlo collocation points according to the user-defined prior samples (taken from a uniform distribution).
2. Bayesian Inference in light of measured data
    *  **Bayesian Inference** through the calculation of likelihood functions based on surrogate model predictions , measurements and the errors. Note that the errors are taken from the calibration points file (.csv) in **calibration_pts_file_path**. Those errors must include measurement and surrogate errors :math:`{\varepsilon}^2=({\varepsilon}^2_{measured} + {\varepsilon}^2_{surrogate})`
    *  **Uncertainty quantification** of calibration parameters by estimating their posterior distributions using rejection sampling.
3. Bayesian Active Learning (BAL) iterations (**heavy computation load**)
    In each BAL iteration, the following steps are performed:
    *  From the original prior sample pool (``prior_samples``), the code selects the MC samples using their indices (i.e. collocation points) that have not been used in the previous steps and taken according to the number expressed in (``mc_samples_al``).
    *  Instantiate an active learning output space as a function of a user-defined size (``mc_samples_al``), and the calculated surrogate prediction and standard deviation arrays.
    *  Calculate Bayesian model evidence (BME) and relative entropy (RE) according to the user-defined (``mc_exploration``).
           - **Bayesian model evidence** rates the model quality compared with available measured data `Bayesian Model Evidence <https://en.wikipedia.org/wiki/Marginal_likelihood>`_.
           - **Relative Entropy** also known as `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_ and measures the so-called **information geometry** in moving from the prior :math:`p(\omega)` to the posterior :math:`p(\omega | D)`.
     `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_.
    *  Find the best performing calibration parameter values (maximum BME/RE scores) and set it as the new best parameter set for use with the deterministic (TELEMAC) model
    *  Run the complex model (i.e., TELEMAC) with the best best performing calibration parameter values.
4.  Repeat the process until the maximum number of iterations or a convergence in BME/RE is reached. The last iteration step corresponds to the supposedly best solution. Consider trying more iteration steps, other calibration parameters, or other value ranges if the calibration results in physical non-sense combinations.

