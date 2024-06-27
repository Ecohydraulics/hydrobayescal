.. Stochastic surrogate workflow.


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
    To expedite surrogate-assisted calibration, it is recommended to perform one dry model initialization initially. Afterward, switch to fast-converging hotstart (wet initial) conditions.


Step 1: Assign user inpuit parameters
-------------------------------

As it was mentioned before the calibration process involves to well defined parts in the code. Both processes depend on the user defined input parameters, which are essential
to run the code properly.
Firstly, the initialization of all input parameters must be done through the ``user_settings.py`` Python script. The file is divided in two parts, the full complexity model global parameters
and the Bayesian Active Learning **(BAL)** global parameters. Each of the parameters has a purpose in the code running so be sure to follow the instructions properly.

------------------------
Global Full Complexity Model Parameters
------------------------

* **control_file_name**: Name of the TELEMAC steering file (.cas)

* **telemac_solver**: TELEMAC solver.

  Two options are possible:

  * 1 = Telemac 2D
  * 2 = Telemac 3D

* **model_simulation_path**: Folder path where all the necessary Telemac simulation files (.cas ; .cli ; .tbl ; .slf) are located

* **results_folder_path**: Folder path where all simulation outputs will be stored. Inside this folder a subfolder called ``auto-saved-results`` will be created with the following files:

  * .slf (For all simulations)
  * collocation_points.csv
  * collocation_points.npy
  * model_results.npy
  * surrogate.pickle (One for each surrogate evaluation)

* **calib_pts_file_path**: Complete folder path where the ``calibration_points.csv`` file is located. ``calibration_points.csv`` is the file which holds the information of measured data for calibration purposes. The .csv file MUST be structured as follows:

.. table:: Measurement Data

   ======================= ================== ================== ====================== ===============
   Point                   X                  Y                  MEASUREMENT 1           ERROR 1
   ======================= ================== ================== ====================== ===============
   [Point data row 1]      [X value]          [Y value]          [Measurement 1 value]  [Error 1 value]
   [Point data row 2]      [X value]          [Y value]          [Measurement 1 value]  [Error 1 value]
   [Point data row 3]      [X value]          [Y value]          [Measurement 1 value]  [Error 1 value]
   ======================= ================== ================== ====================== ===============
* **n_cpus**: Number of CPUs for Telemac Simulations.

* **init_runs**: Number of initial full-complexity model runs. The initial simulations will run with the initial training points for surrogate model construction.

* **friction_file**: Friction file .tbl that contains the information of friction zones for the Telemac simulation. This name MUST be indicated in the Telemac .cas file with the keyword **FRICTION DATA FILE**.

* **dict_output_name**: Desired name of the external file .json file containing the model outputs of the calibration quantities

* **results_filename_base**: Desired name of the results.slf ile to be iteratively created after each simulation. Add the name WITHOUT the extension .slf (i.e., "telemac_rfile"). The results file (.slf) will be stored inside the *auto-saved-results* folder, inside the *results* folder.

* **Calibration parameters**: Assign calibration parameters. They must be assigned as *strings*. Please consider these recommendations before assigning the calibration parameters.

    * Notes:
        * MAXIMUM number of calibration parameters = 4.
        * The calibration parameters MUST coincide with the Telemac KEYWORD in the .cas file. You can find more details in the Telemac User Manuals `http://wiki.opentelemac.org/doku.php#principal_documentation <https://wiki.opentelemac.org/doku.php#principal_documentation>`_
             Example: calib_parameter_1 = "LAW OF FRICTION ON LATERAL BOUNDARIES"
                      calib_parameter_2 = "INITIAL ELEVATION"
        * If you want to calibrate different values of roughness coefficients in roughness zones, the roughness zones description MUST be indicated in the .tbl file.
        * The .tbl file name MUST be indicated in the friction file input.
        * The calibration zone MUST contain the word zone,ZONE or Zone as a prefix in the calib_parameter field.
             Example: calib_parameter_1='zone99999100'   , if the zone description is: 99999100

* calib_parameter_1
    * calib_parameter_2
    * calib_parameter_3
    * calib_parameter_4

    *Calibration ranges*

    * param_range_1
    * param_range_2
    * param_range_3
    * param_range_4

    *Calibration quantities*

    * calib_quantity_1
    * calib_quantity_2
    * calib_quantity_3
    * calib_quantity_4

* **dict_output_name**

* **results_file_name_base**




Step 2: Read Collocation Points
-------------------------------

The second step consist of reading the (initial) collocation (measurement) point file. The measurement points correspond to the target values for the model optimization regarding, for instance, topographic change, water depth, or flow velocity. The measurement point's coordinates must correspond to mesh nodes of the computational mesh. Rather than forcing the numerical mesh to exactly fit the coordinates of a measurement point, we recommend to interpolate measurement data the closest measurement point(s) onto selected mesh nodes.

.. tip::

    The number of measurement points scales exponentially with the run time for the surrogate-assisted calibration process. Therefore, we recommend to use **no more than 200 measurement points** (speed criterion) and **at least 100 measurement points** (quality criterion).

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
