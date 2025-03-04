.. BAL Telemac

Complete Bayesian Active Learning (BAL) for the Gaussian Process Emulator (GPE) using Telemac
=============================================================================================
A complete surrogate-assisted calibration using Gaussian Process and Bayesian Active Learning (BAL) is performed for a hydrodynamic model using Telemac and is launched with the main script as follows:

.. image:: _static/UML-bal-reduced.png
   :alt: UML complete surrogate assisted calibration
   :width: 80%
   :align: center

Telemac simulation folder
------------------------------

To run HydroBayesCal using Telemac, you need to have Telemac and all the necesary files to run a hydrodynamic model.
Create a folder called **telemac_simulation** and copy the necessary files for a Telemac simulation into it:
For example (hydrodynamic numerical model):

- **telemac.cas:** Numerical configuration of the hydrodynamic model.
- **liquid.liq:** Liquid boundary condition (flow inflow/outflow) (in case of unsteady flow, a .liq file is needed)
- **boundary-conditions.cli:** File that defines the type and location of the boundary conditions.
- **geometry.slf:** File that defines the mesh structure for the hydrodynamic model.
- **zones.tbl:** File that defines the roughness zones of the mesh.
- **rating-curve.txt:**: File that defines the stage-discharge rating curve of the outlet boundary condition.

Until now, the code cannot run sediment transport model with GAIA. Only hydrodynamic simulations are possible.

OpenFoam simulation folder
------------------------------

Definition of HydroBayesCal parameters
---------------------------------------

A complete surrogate assisted calibration of a hydrodynamic model requires the definition of some parameters corresponding to the complex model (e.g. Telemac or OpenFoam) and parameters for the metamodel construction based on Gaussian Process.

complex_model instance:

.. code-block:: python

    complex_model = initialize_model(
        TelemacModel(
            friction_file="/path/to/friction_file.tbl",
            tm_xd="1",
            gaia_steering_file="",
            results_filename_base="results",
            control_file="control_file.cas",
            model_dir="/path/to/model_directory/telemac_simulation",
            res_dir="/path/to/results_directory/",
            calibration_pts_file_path="/path/to/calibration_points.csv",
            n_cpus=8,
            init_runs=15,
            calibration_parameters=["zone1", "zone2", "zone3", "ROUGHNESS COEFFICIENT OF BOUNDARIES", ],
            param_values=[[0.011, 0.79], [0.011, 0.79], [0.0016, 0.060], [0.018, 0.028]],
            extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERGY", "VELOCITY U", "VELOCITY V"],
            calibration_quantities= ["WATER DEPTH", "SCALAR VELOCITY"]
            dict_output_name="output-data",
            user_param_values=False,
            max_runs=30,
            complete_bal_mode=True,
            only_bal_mode=False,
            delete_complex_outputs=True,
            validation=False
        )
    )


In this example, the **Telemac** files are saved in **telemac_simulation** folder. The path to this folder is defined in ``model_dir``.
.. code::

    model_dir="/path/to/model_directory/telemac_simulation".

The prior assumptions for these uncertain calibration parameters are defined as four ranges in ``param_values`` following a uniform distribution limited by the minimum and maximum limits. The model is calibrated for three roughness zones and the roughness coefficients of the boundaries.
.. code::

    param_values=[[0.011, 0.79], [0.011, 0.79], [0.0016, 0.060], [0.018, 0.028]].

The measured data, stored in a `.csv` file, should consists of water depth and scalar velocity. Each of these quantities has a measurement error which is also assigned in the corresponding column in the .csv file. These quantities will be the calibration targets and are extracted from the model. The user-specified ``calibration_quantities``are ["WATER DEPTH", "SCALAR VELOCITY"].
.. code::

    calibration_quantities=["WATER DEPTH", "SCALAR VELOCITY"]

For more details on the assignment of complex model parameters, please refer to the section :ref:`HydroSimulations_class`.


Experiment Design Definition
----------------------------

The calibration model parameters are associated with uncertainty and are described as probability distributions.
To define the values of the input parameters, **HydroBayesCal** uses the classes `ExpDesigns` and `Input` from BayesValidRox:
`Priors, input space and experimental design <https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/input_description.html>`_

If the uncertain parameters are defined as distribution types, they must be specified as follows:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    exp_design = setup_experiment_design(
        complex_model=full_complexity_model,
        tp_selection_criteria='dkl',
        parameter_distribution='uniform',
        parameter_sampling_method='sobol'
    )

This function returns an instance of the ``ExpDesigns`` class from BayesValidRox,
which will be used in subsequent steps.

The attribute ``exp_design.X`` stores the collocation points for the initial execution of the complex model.

Parameters
^^^^^^^^^^

**complex_model** : object
    An instance representing the hydrodynamic model to be used in the experiment.

**tp_selection_criteria** : str, optional
    The criteria for selecting new training points (TP) during the Bayesian Active Learning process.
    Default: ``'dkl'`` (relative entropy).

**parameter_distribution** : str, optional
    The criteria for selecting the parameter distribution.
    Default: ``'uniform'`` (uniform distribution).

**parameter_sampling** : str, optional
    The criteria for selecting the parameter sampling method.
    Default: ``'sobol'``.

Returns
^^^^^^^^

**exp_design** : object
    An instance of the experiment design object configured with the specified model and selection criteria.

.. autofunction:: bal.setup_experiment_design

Run Complex Model with Experiment Design
-----------------------------------------

This step executes the hydrodynamic model for a given experiment design and returns
the collocation points (previously obtained in the experiment design) and the model outputs.
The collocation points serve as the input parameters for the initial model runs.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    init_collocation_points, model_evaluations = run_complex_model(
        complex_model=complex_model,
        experiment_design=exp_design,
    )

Parameters
^^^^^^^^^^

**complex_model** : obj
    Instance representing the hydrodynamic model to be evaluated.

**experiment_design** : obj
    Instance of the experiment design object that specifies the settings for the experimental runs.

Returns
^^^^^^^^

**collocation_points** : array
    Contains the collocation points (parameter combination sets) with shape
    ``[number of runs x number of calibration parameters]`` used for model evaluations.

**model_outputs** : array
    Contains the model outputs. The shape of the array depends on the number of quantities:

    - **For 1 quantity**: ``[number of runs x number of locations]``
    - **For 2 quantities**: ``[number of runs x 2 * number of locations]``
      *(Each pair of columns contains the two quantities for each location.)*

Run Bayesian Active Learning Calibration
----------------------------------------

This step performs **stochastic calibration** of the Telemac hydrodynamic model using
**Surrogate-Assisted Bayesian Inversion**. The surrogate model is constructed with
**Gaussian Process Regression (GPR)**, supporting both **Single-Output GP** and **Multi-Output GP** formulations.

This approach enables:

- **Bayesian Model Inversion**, allowing uncertainty quantification of model input parameters
  through **Bayesian Inference**.
- **Iterative surrogate training**, where the GP metamodel is refined dynamically by adding new training points using
  **Bayesian Active Learning (BAL)** to improve calibration efficiency. The criteria for adding new training points is selected from **DKL (relative entropy)**, **BME (Bayesian Model Evidence)**
- **Methods by:** `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python
    run_bal_model(
        collocation_points=init_collocation_points,
        model_outputs=model_evaluations,
        complex_model=complex_model,
        experiment_design=exp_design,
        eval_steps=20,
        prior_samples=15000,
        mc_samples_al=2000,
        mc_exploration=1000,
        gp_library="gpy"
    )

Parameters
^^^^^^^^^^

**collocation_points** : array
    An array containing the collocation points used for model evaluations, with shape
    ``[number of initial runs x number of calibration parameters]``.

**model_outputs** : array
    Contains the outputs from the hydrodynamic model, with shape dependent on the number of quantities and locations.

**complex_model** : obj
    An instance representing the hydrodynamic model instance to be evaluated.

**experiment_design** : obj
    Contains the experiment design object.

**eval_steps** : int, optional
    Specifies how often the surrogate model is evaluated and saved in the surrogate model folder.
    Default is ``1`` (evaluates the surrogate model at every BAL iteration).
    Example: if 10 the surrogate model is evaluated every 10 BAL iterations.

**prior_samples** : int, optional
    The number of samples drawn from the prior distribution (prior pool).
    Default is ``10,000``.

**mc_samples_al** : int, optional
    The number of Monte Carlo samples used for the Bayesian inference process (taken from prior pool).
    Default is ``5,000``.

**mc_exploration** : int, optional
    The number of samples used for exploring the parameter space during the Bayesian Active Learning process. (from mc_samples_al)
    Default is ``1,000``.

**gp_library** : str, optional
    The Gaussian Process library to be used for modeling. Options include ``"gpy"`` (for GPyTorch) or ``"skl"`` (for SciKitLearn).
    Default is ``gpy``.

Returns
^^^^^^^^

**None**

The following files are saved in the user-defined results directory ``res_dir`` under
the name **auto-saved-results-HydroBayesCal**:

- **BAL_dictionary**: Dictionary and ``.pkl`` file containing the data from Bayesian Active Learning.
- **updated_collocation_points**: Array and ``.csv`` file containing all the collocation points (Initial + BAL-added).
- **model_outputs**: Files ``.csv`` and ``.json`` containing all model outputs obtained from the
  collocation points and required model variables.

Surrogate-assisted BAL calibration outputs
----------------------------------------

The surrogate assisted calibration process will run until max_runs is reached. When the calibration process gets to that point, the output files have been created. Please refer to: :ref:`calibration_outputs` to see which files are created.
Each of the files contain data that can be extracted or used for further anaylsis.

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

Methods by:
Oladyshkin, S., Mohammadi, F., Kroeker, I., & Nowak, W. (2020). Bayesian3 Active Learning for the Gaussian Process Emulator Using Information Theory. Entropy, 22(8), 890.
----------------------

#external_libraries Folder:
The library pputils-master by Pat Prodanovic (https://github.com/pprodano/pputils) is used to extract the results of the simulation file (.slf) into a .txt file, which is then stored in the results Folder.

#scripts Folder:
-auxiliary_fuctions_BAL: Auxiliary functions for the stochastic calibration of model using Surrogate-Assisted Bayesian inversion
- auxiliary_functions_telemac: Contains auxiliary functions used to modify the input and output of the telemac files. These functions are specific to the parameters that wanted to be changed at the time, but they can be used as a base on how to modify Telemac's input and output files
-init.py: Reference other folders.
