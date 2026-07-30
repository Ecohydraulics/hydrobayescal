.. Full complexity model


Bayesian Calibration Workflow
=============================

The workflow describes the use of Bayesian Model Evidence (BME) and Relative Entropy (RE) in conjunction with a Gaussian Process Emulator,
as proposed by `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_, to iteratively improve the accuracy of a surrogate model applied
for the calibration of full-complexity hydrodynamic models.

The following steps outline the process for performing a GPE surrogate-assisted calibration of any hydrodynamic model using open-source
hydrodynamic software. Currently, model calibration is supported only with Telemac.

Step 0: Wet your (TELEMAC) Model
--------------------------------

Before the surrogate-assisted calibration can run, it needs an initial model run. The initial model needs to be fully functional with all the required simulation files.
The first model run should start with `dry conditions (read more at hydro-informatics.com) <https://hydro-informatics.com/numerics/telemac2d-steady.html>`_ and
be adapted to `wet (steady or unsteady hotstart) initial conditions <https://hydro-informatics.com/numerics/telemac2d-unsteady.html#hotstart-initial-conditions>`_ for the surrogate-assisted calibration.

.. note:: **Why hotstart the model for the surrogate-assisted calibration?**

    A hotstart simulation involves re-using the output file (.slf) of a previous simulation that began under dry conditions as a file containing the new initial conditions.
    In a typical numerical model of a fluvial ecosystem, it is common to start with dry conditions to prevent filling disconnected terrain depressions with water. However, applying wet initial
    conditions that approximately correspond to the target conditions can significantly speed up convergence.
    To expedite surrogate-assisted calibration, it is recommended to perform one dry model initialization. Afterwards, switch to fast-converging hotstart (wet initial) conditions.


Step 1: Assign the calibration settings
---------------------------------------

As it was mentioned before the calibration process involves two well defined parts in the code. Both processes depend on the user-defined settings, i.e. the calibration parameters to adjust
and the calibration targets to fit against (see :ref:`terminology`), which are essential
to run the code properly.
Firstly, the initialization of all settings must be done in the ``templates/bal_telemac.py`` Python script. ``bal_telemac.py`` is the main script that runs the calibration process and
calls the necessary instances of the classes that run the hydrodynamic model, creation of surrogate models and BAL.


.. _HydroSimulations_class:

Functioning of the HydroSimulations Class
+++++++++++++++++++++++++++++++++++++++++

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

* **calibration_quantities**: Names of the calibration targets, i.e. the measured variables
  the model is fitted against (see :ref:`terminology`).

    .. code-block:: python

       calibration_quantities = ['WATER DEPTH']  # Single calibration target
       calibration_quantities = ['WATER DEPTH', 'SCALAR VELOCITY']  # Multiple calibration targets


* **extraction_quantities**: Variables to be extracted from the model output files. Generally the same as, or more than, the calibration targets. Any extracted variable can be promoted to a calibration target when restarting with ``only_bal_mode = True``, without re-running the model.

    .. code-block:: python

      calibration_quantities = ['WATER DEPTH'] # WATER DEPTH as the calibration target.
      extraction_quantities = ['WATER DEPTH', 'SCALAR VELOCITY', 'TURBULENT ENERG', 'VELOCITY U', 'VELOCITY V'] # Calibration and additional variables to be extracted.

    Any of these additional extracted variables can be used for calibration purposes.
* **dict_output_name**: Base name for output dictionary files where the outputs are saved as `.json` files.

* **user_param_values**: (Default: ``False``). Boolean variable that enables the use of user-defined collocation points taken from a .csv file located in the restart folder.

  - If ``True``: Collocation points are taken from the user-defined .csv file.
  - If ``False``: Sampling methods from BayesValidRox according to the available sampling options.

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
   **param1**          **param2**        **param3**         **param4**         **param5**
   ================== ================== ================== ================== ==================
   0.148              0.770               0.014              0.014              0.700
   0.066              0.066               0.066              0.066              0.066
   ================== ================== ================== ================== ==================

* **max_runs**: Maximum (total) number of model simulations, including initial runs and Bayesian Active Learning iterations.

* **complete_bal_mode**: (Default: ``True``). Boolean variable to select a complete BAL calibration or not.

  - If ``True``: Bayesian Active Learning (BAL) is performed after the initial runs, enabling a complete surrogate-assisted calibration process.
    **This option MUST be selected if you choose to perform only BAL** (i.e., when ``only_bal_mode = True``).
  - If ``False``: Only the initial runs of the full complexity model are executed, and the model outputs are stored as ``.json`` and ``.csv`` files.

* **only_bal_mode**: (Default: ``False``). Boolean variable to select the BAL or not.

  - If ``False``: The process will either execute a complete surrogate-assisted calibration or only the initial runs, depending on the value of ``complete_bal_mode``.
  - If ``True``: Only the surrogate model construction and Bayesian Active Learning of preexisting model outputs at predefined collocation points are performed.
    **This mode can be executed only if either a complete process has already been performed** (``complete_bal_mode = True`` and ``only_bal_mode = True``)
    **or if only the initial runs have been executed** (``complete_bal_mode = False`` and ``only_bal_mode = False``). It is possible also to build the surrogate model with either **ALL** the restart data or just a **PART** of it. To use only a part of it, initialize the ``initial_runs`` accordingly.

* **validation**: (Default: ``False``). Boolean variable to select the creation of a independent set of collocation points and outputs for surrogate validation purposes.
  If ``True``, creates output files (inputs and outputs) for validation of the surrogate model. If it is True, the validation data is saved in the restart data folder.

* **Shortcut Combinations and Their Corresponding Tasks**:


.. table:: Task Descriptions

   ===================== =================================== ============================================================================
   **complete_bal_mode**  **only_bal_mode**                   **Task Description**
   ===================== =================================== ============================================================================
   True                  False                                Complete surrogate-assisted calibration
   False                 False                                Only initial runs (no surrogate model)
   False                 False, with ``validation=True``      Only initial runs (for validation data)
   True                  True, with ``init_runs = max_runs``  Only surrogate construction with a set of predefined runs (no BAL)
   True                  True, with ``init_runs > max_runs``  Surrogate model construction and Bayesian Active Learning (BAL) applied
   ===================== =================================== ============================================================================



TelemacModel Class (Telemac specific parameters)
++++++++++++++++++++++++++++++++++++++++++++++++

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


.. note:: **Pre-BAL check: is roughness the identifiable calibration knob?**

    After the initial full-complexity runs (and before BAL), HydroBayesCal reports a
    physics-based *roughness-identifiability* diagnostic whenever the calibration
    targets include both a depth-like and a velocity-like target
    (``diagnose_roughness_identifiability`` / ``log_roughness_identifiability`` in
    ``hydroBayesCal.function_pool``, invoked from
    ``templates/prebal_telemac_error_analysis.py``). At a fixed discharge
    (:math:`Q = U\,A`), raising the bottom roughness slows the flow, so the water
    deepens to keep passing :math:`Q` -- depth goes up, velocity goes down. Hence the
    sign pattern of the depth-vs-velocity residuals (simulated minus observed) at the
    calibration points is diagnostic:

    * **Anti-correlated** residuals (one target simulated too high while the other
      is too low) are the fingerprint of a *roughness* error, and roughness
      calibration will converge -- the signs even tell you which way to move it:
      too deep **and** too slow means roughness is too high; too shallow **and** too
      fast means it is too low.
    * **Correlated** residuals (both targets too high, or both too low) *cannot*
      be produced by roughness alone. Roughness calibration then fights itself and
      its optimum tends to pin at a prior bound. The diagnostic logs a **warning**
      recommending a second calibration parameter -- e.g. ``VELOCITY DIFFUSIVITY``,
      the boundary friction, or the turbulence closure.

    The check is **report-only**: it never alters the sampling or the parameter set.
    It is model-agnostic (TELEMAC, OpenFOAM, ...) and works for single- and
    multi-flow calibrations. Heed the warning before committing solver hours to BAL:
    calibrating a non-identifiable roughness wastes runs and yields a bound-pinned,
    physically meaningless optimum.


Step 2: Bayesian model optimization
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
3. Bayesian Active Learning (BAL) iterations (**heavy computation load**).
   In each BAL iteration, the following steps are performed:

   * From the original prior sample pool (``prior_samples``), the code selects the MC samples using their indices (i.e. collocation points) that have not been used in previous steps, according to the number set in ``mc_samples_al``.
   * Instantiate an active-learning output space as a function of a user-defined size (``mc_samples_al``) and the computed surrogate prediction and standard-deviation arrays.
   * Calculate Bayesian model evidence (BME) and relative entropy (RE) according to the user-defined ``mc_exploration``:

     - **Bayesian model evidence** rates the model quality compared with available measured data (`Bayesian Model Evidence <https://en.wikipedia.org/wiki/Marginal_likelihood>`_).
     - **Relative entropy**, also known as `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_, measures the **information geometry** in moving from the prior :math:`p(\omega)` to the posterior :math:`p(\omega | D)` (`Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_).

   * Find the best-performing calibration parameter values (maximum BME/RE scores) and set them as the new parameter set for the deterministic (TELEMAC) model.
   * Run the complex model (i.e., TELEMAC) with the best-performing calibration parameter values.
4.  Repeat the process until the maximum number of iterations or a convergence in BME/RE is reached. Consider trying more iteration steps, other calibration parameters, or other value ranges if the calibration results in physical non-sense combinations.

.. warning::

    The last training point of the calibration is **not** the calibrated parameter set.
    Bayesian active learning selects every training point by *information gain*: the
    parameter combination that most reduces uncertainty about the posterior, which is
    not the combination that best reproduces the measurements. The calibrated
    parameter sets are derived from the posterior itself, see
    :ref:`calibrated-parameters`.


Step 3: Post-calibration data
------------------------------

The Bayesian Active Learning (BAL) process runs iteratively until the specified ``max_runs`` limit is reached.
After completion, the post-calibration data is automatically saved in a directory named
**auto-saved-results-HydroBayesCal**.

Inside this directory, you will find four subfolders containing all the necessary information
for analyzing the calibration process, including the trained GPR metamodels.

For a detailed explanation of the saved data, please refer to :ref:`outputs-folder`.


.. _calibrated-parameters:

Step 4: Derive the calibrated parameter sets
--------------------------------------------

The BAL loop stores, for every iteration, a **joint** posterior sample in
``BAL_dictionary.pkl``: the prior samples accepted by rejection sampling against the
joint likelihood over all calibration points and calibration targets. That sample, not any
single training point, is the calibration result.

Run the derivation on a finished calibration:

.. code-block:: bash

   python templates/derive_calibrated_parameters.py --config config_Telemac.py

It reports, for every calibration parameter:

* the **peak of that parameter's own posterior marginal**, with the credible interval.
  The peak is read directly off a histogram of the accepted posterior samples (the
  most populated bin, refined to the mean of the samples inside it), with no
  smoothing, so the reported optimum is a property of the posterior rather than of a
  fitted curve. The number of bins follows from the sample size and the spread of the
  posterior instead of being fixed in advance, see
  :func:`~hydroBayesCal.surrogate.posterior_analysis.marginal_bin_count`;
* the **variance reduction** relative to the prior, i.e. how strongly the measurements
  constrain that parameter at all;
* flags for parameters whose optimum is **pinned at a prior bound** (the range is too
  narrow, or that parameter alone cannot compensate the model error), whose marginal is
  **multimodal**, or that are **not identifiable** from the calibration targets.

Why the per-parameter optima are not automatically a parameter set
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Each calibration parameter has its own posterior marginal and therefore its own
optimum. Stacking those optima into one vector, however, implicitly assumes the
parameters are independent. Under equifinality they are not: a friction zone and a
critical Shields parameter can trade off along a ridge in parameter space, so the
combination of their individual peaks falls in the empty middle of that ridge, a
parameter set the posterior considers implausible even though each component is
individually optimal.

The script therefore also reports:

* the **joint posterior optimum**, i.e. the parameter vector of highest joint posterior
  density, which is an actual posterior sample and hence jointly consistent by
  construction;
* an **equifinality verdict** based on the posterior correlation between the
  parameters, the Mahalanobis distance of the marginal-peak vector under the posterior
  covariance, and the joint posterior density at that vector relative to the accepted
  samples. ``consistent`` and ``acceptable`` mean the per-parameter optima can be used
  as a parameter set; ``coupled`` means they happen to be compatible but the parameters
  trade off so tightly that the marginals do not determine the combination;
  ``inconsistent`` means the assembled vector sits in a low-density region and is not a
  valid parameter set at all;
* representatives of the **distinct posterior modes**, where several different parameter
  combinations explain the measurements comparably well. Where that happens, no single
  optimum is a defensible answer. A continuous trade-off *ridge* is reported as one
  mode rather than several, since every combination along it is about as good; only
  solutions separated by a genuine drop in posterior density are counted separately.

Running the full complexity model at the candidates
+++++++++++++++++++++++++++++++++++++++++++++++++++

With ``--write-csv`` the candidate parameter sets (marginal peaks, joint optimum,
posterior mean and one representative per posterior mode) are written to
``restart_data/user-collocation-points.csv``, together with two labelled report files
that carry the diagnostics. Then set in the configuration:

.. code-block:: python

   execution['user_param_values']  = True
   execution['complete_bal_mode']  = False
   execution['only_bal_mode']      = False
   sampling['init_runs']           = <number of candidates>

``init_runs`` matters: the run loop is bounded by it, not by the number of rows in the
CSV file. Then run the full complexity model at each candidate and compare the outputs
against the measurements:

.. code-block:: bash

   python templates/bal_telemac.py --config config_Telemac.py
   python templates/assess_calibration.py --config config_Telemac.py

Letting the full complexity model arbitrate between a handful of labelled candidates is
the honest way to close a calibration whose posterior is equifinal.

Watching the optima converge
+++++++++++++++++++++++++++++

The same diagnostics are recorded at every BAL iteration, so
``templates/plot_posteriors.py`` produces two additional figures:

* ``parameter_optimum_convergence``: each parameter's own optimum against the number of
  training points, with its credible interval and the calibration range. A trace still
  drifting at the end means the calibration has not converged for that parameter; a trace
  sitting on a bound means it is pinned.
* ``marginal_vs_joint``: where the combination of the per-parameter optima sits in the
  joint posterior density, per iteration. A trace that stays low is a quantitative
  equifinality warning.

Both are reconstructed from the stored posteriors when a result file predates these
diagnostics, so archived calibrations can be analysed without being re-run.

.. note::

   All figures are rendered through LaTeX, which needs a few system packages that
   ``pip`` cannot install. If a plotting call fails with a ``RuntimeError`` quoting a
   LaTeX error, see :ref:`latex-for-plots`.

Accounting for the surrogate uncertainty
+++++++++++++++++++++++++++++++++++++++++

The Bayesian inference accounts for the emulator's own predictive uncertainty by
default (``sampling['include_surrogate_error'] = True``): the GPE standard deviation at
each predicted point is added to the observation variance, so a parameter set the
emulator is unsure about is not treated as if it had been simulated exactly. This is
also what the active-learning utility has always done, so the inference and the point
selection now make the same assumption.

The observation variance is built from three *relative* terms, each a fraction of every
measured value, plus the absolute ``<target>_ERROR`` column of the calibration-points
file:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Setting
     - Default
     - Represents
   * - ``calibration['measurement_error']``
     - 0.10
     - The instrument or campaign is imprecise.
   * - ``calibration['gpe_error']``
     - 0.0
     - Flat stand-in for the emulator's uncertainty. Zero because
       ``include_surrogate_error`` now supplies the real per-prediction value; a
       non-zero value here would count the same uncertainty twice.
   * - ``calibration['model_structural_error']``
     - 0.0
     - The solver itself is an imperfect description of the site: unresolved
       processes, geometry, boundary conditions. Independent of the emulator and
       **not** supplied by ``include_surrogate_error``, which only ever accounts for
       the surrogate's approximation of the solver, never for the solver's own error.
       Set it if you can defend a value.

.. warning::

   Expect the posterior to become **sharper**, not broader, relative to earlier
   versions. The old defaults added a flat 10 % emulator term on top of the 10 %
   measurement term, i.e. 14.1 % of each measured value before any site-specific
   error. A trained GPE is usually a good deal tighter than 10 %, so the total
   variance typically falls. Turning the flag on *while holding* ``gpe_error`` at 0.10
   would broaden the posterior, but that is the double-counted combination.

To reproduce the behaviour of earlier versions exactly, set **both**:

.. code-block:: python

   sampling['include_surrogate_error'] = False
   calibration['gpe_error']            = 0.10

Setting only the first leaves the emulator uncertainty represented nowhere at all,
which gives the sharpest and least defensible posterior of the four combinations. The
drivers warn about that case, and log the effective settings on every run so an
archived ``logfile.log`` records which convention produced its numbers.

.. note::

   ``RE`` (relative entropy) is the score to compare across runs: it is invariant to a
   sample-independent rescaling of the likelihood. ``BME`` and ``ELPD`` are on a
   different scale depending on whether a model error was included, so compare those
   only within a run. The stored ``log_BME`` is the exact evidence; ``BME`` is kept for
   backward compatibility and can reach ``0.0`` or ``inf`` on large problems.