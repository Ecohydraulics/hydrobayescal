.. Full complexity model


Full-complexity - Surrogate Calibration Workflow
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
calls the necesary instances of the classes that run the hydrodynamic model, creation of surrogate models and BAL.

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
In each run, HydroBayesCal creates a results folder called "auto-saved-results-HydroBayesCal" in the results directory specified by the user.
This folder contains the following subfolders:

* **calibration-data**: Contains the calibration data (model outputs) for each calibration parameter set.
* **plots**: Contains plots of the calibration parameters and the calibration quantities.
* **surrogate_models**: Contains the surrogate models created during the calibration process.
* **restart_data**: Contains the restart data for the calibration process.



Step 3: Bayesian Model Optimization
-----------------------------------

With the initial model setup and the measurement points, the Bayesian model optimization process has everything it needs for its iterative score calculation. The number of iterations corresponds to the user-defined limit (recall, the default is ``it_limit = 15``) and the following tasks are performed in every iteration:

1. Compute a surrogate model prediction for all collocation (measurement) points
    * Instantiate a prediction and a standard deviation array, each with the size of of measurement points.
    * Loop over the model predictions at the collocation points:
        - Instantiate a `radial-basis function (RBF) kernel <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html>`_ corresponding to the possible value ranges of the selected calibration parameters.
        - Instantiate a `Gaussian process regressor <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html?highlight=gaussianprocessregressor>`_ with the RBF kernel.
        - Fit the Gaussian process regression model.
        - Create parameter predictions with the Gaussian process regression (also known as `kriging <https://en.wikipedia.org/wiki/Kriging>`_ ) model, which represent the **surrogate predictions** (i.e., fill the previously instantiated prediction arrays).
2. Calculate the error in the likelihood functions as :math:`{\varepsilon}^2=({\varepsilon}^2_{measured} + {\varepsilon}^2_{surrogate})`
3. Calculate Bayesian model evidence (BME) and relative entropy (RE)
    * Bayesian model evidence rates the model quality compared with available data and is here estimated as the expectancy value of a Monte Carlo sampling.
    * Relative entropy is also known as `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_ and measures the difference (distance) between two probability distributions.
4. Run Bayesian active learning (BAL) on the output space (**heavy computation load**):
    * Use the indices of priors (i.e. collocation points) that have not been used in the previous steps.
    * Instantiate an active learning output space as a function of a user-defined size (``mc_samples_al``), and the above-calculated surrogate prediction and standard deviation arrays (see item 1)
    * Calculate Bayesian scores as a function of the user-defined strategy (BME or RE), the observations, and the active learning output space.
5. Find the best performing calibration parameter values (maximum BME/RE scores) and set it as the new best parameter set for use with the deterministic (TELEMAC) model
6. Run TELEMAC with the best best performing calibration parameter values.

Step 4: Get Best Performing solution
------------------------------------

The last iteration step corresponds to the supposedly best solution. Consider trying more iteration steps, other calibration parameters, or other value ranges if the calibration results in physical non-sense combinations.
