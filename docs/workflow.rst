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


.. _initial-design:

Step 1: Size and sample the initial design
------------------------------------------

Bayesian active learning does not start from nothing. It starts from a Gaussian Process
Emulator trained on an **initial design**, a set of full-complexity runs spread over the
calibration ranges, and everything it does afterwards is a refinement of that emulator.
This step decides whether the calibration can find the global maximum of the posterior at
all, and it is the one step whose cost is paid entirely in solver runs.

Why the initial design decides the outcome
+++++++++++++++++++++++++++++++++++++++++++

BAL picks each new training point where the *current* emulator says the most information
is to be gained. An emulator that has never seen a region of the parameter space cannot
report that anything is missing there, so a region the initial design skipped stays
skipped: the loop keeps refining the likelihood peak it already knows about. If that peak
is an artefact of an emulator that is wrong elsewhere, the calibration converges,
confidently, onto a parameter set the full complexity model does not support. The failure
is silent, because every diagnostic BAL computes is a diagnostic of the emulator.

Too many initial runs are wasteful in the opposite direction. Every initial run is a full
TELEMAC, OpenFOAM or Delft3D simulation, and a run spent covering the prior is a run not
spent on a BAL iteration, which targets the posterior. The initial design therefore has to
be **large enough to make the emulator trustworthy everywhere, and no larger**.

Sobol sampling, and why it is the default
++++++++++++++++++++++++++++++++++++++++++

``sampling['parameter_sampling_method'] = "sobol"`` (the default) draws the collocation
points from a Sobol sequence, a low-discrepancy quasi-random sequence. Compared with the
alternatives at the same number of runs:

* **random** sampling clusters and leaves holes, purely by chance. In five dimensions and
  50 runs the largest empty region is routinely a substantial fraction of the range.
* **latin_hypercube** fixes the one-dimensional projections, so every parameter is evenly
  covered on its own, but says nothing about the joint coverage: a Latin hypercube design
  can still leave a whole corner of the joint space empty.
* **sobol**, **halton** and **hammersley** control the *joint* discrepancy, which is
  exactly the property a Gaussian process needs, since its prediction error at a point is
  governed by the distance to the nearest training points in the full space.

Sobol has one further property the workflow depends on: it is **extensible**. The first
:math:`n` points of a Sobol sequence of length :math:`2n` are exactly the sequence of
length :math:`n`, so the design can be grown without discarding a single simulation that
has already been run. Latin hypercube and random designs have no such guarantee.

The full list of accepted methods is ``random``, ``latin_hypercube``, ``sobol``,
``halton``, ``hammersley``, ``chebyshev``, ``grid`` and ``user``. Invalid values are
rejected before the first simulation starts rather than several minutes into the run.

.. note::

   Older configurations used ``"chebyshev(FT)"`` and ``"grid(FT)"``. Those spellings are
   not rules that the underlying sampler knows; they are accepted with a warning and
   mapped to ``chebyshev`` and ``grid``.

How many initial runs are needed
+++++++++++++++++++++++++++++++++

The a-priori rule is **ten runs per calibration parameter, rounded up to a power of two**,
with an absolute floor of 16:

.. math::

   n_\mathrm{init} = 2^{\left\lceil \log_2 \max(10\,d,\ 16) \right\rceil}

where :math:`d` is the number of calibration parameters. Ten runs per dimension is the
established rule of thumb for an initial design a Gaussian process is to be fitted to. The
rounding to a power of two is not cosmetic: an unscrambled Sobol sequence is balanced at
:math:`n = 2^m`, so 128 points cover the space measurably better than 100 do.

.. table:: Recommended number of initial runs

   ========================== =================== ============================
   **calibration parameters**  **10 d**            **recommended init_runs**
   ========================== =================== ============================
   2                          20                  32
   3                          30                  32
   5                          50                  64
   7                          70                  128
   10                         100                 128
   ========================== =================== ============================

:func:`~hydroBayesCal.surrogate.initial_design.recommended_init_runs` computes this and
compares it against the configured ``sampling['init_runs']`` **before the first
simulation starts**. An undersized design is logged as a warning, with the reason and the
recommended value. It is never enforced and never silently raised: spending days of
additional CPU time is the modeller's decision, not the code's.

Meeting the requirement without paying for it: the staged ladder
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The rule above is a rule of thumb, and how many runs a *particular* calibration needs
depends on how non-linear the solver's response is, how many calibration points there are
and how precise the measurements are. None of that is known in advance, but all of it can
be measured once some runs exist. With ``sampling['adaptive_init_runs'] = True`` (the
default) the initial design is therefore run in **blocks along the same Sobol sequence**,
and measured in between:

.. code-block:: text

   block 1: runs   1-32   ->  Q2 = 0.71,  posterior samples 95    ->  insufficient
   block 2: runs  33-64   ->  Q2 = 0.89,  posterior samples 410   ->  marginal
   block 3: runs  65-128  ->  Q2 = 0.96,  posterior samples 980   ->  SUFFICIENT
                                                                      start BAL

Runs 33 to 64 are Sobol points 33 to 64 of the *same* sequence, so block 2 extends block 1
instead of replacing it and no simulation is ever wasted. The ladder starts at
:math:`2^{\lceil \log_2 4d \rceil}` runs (override with ``sampling['init_runs_min']``),
doubles, and is **capped by** ``init_runs``, so a staged design can only ever run fewer
simulations than the configuration authorises, never more. Runs saved by stopping early
are not lost either: ``max_runs`` is the total budget, so they become BAL iterations,
which target the posterior instead of covering the whole prior.

What "sufficient" means
++++++++++++++++++++++++

After every block,
:func:`~hydroBayesCal.surrogate.initial_design.initial_design_sufficiency` fits an
independent Gaussian process to the runs carried out so far and measures five things.
Each of them breaks BAL in a different way when it fails:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Criterion
     - Threshold
     - What its failure means
   * - Predictivity ``Q2``
     - median 0.90, worst column 0.70
     - The emulator cannot predict a parameter set it has not been trained on, so the
       likelihood surface BAL refines is not the solver's. ``Q2`` is a leave-one-out
       score, i.e. it is measured at points the emulator did not see; the training fit
       itself is always perfect for an interpolating GP and says nothing.
   * - Error-bar calibration
     - at least 0.85
     - Fraction of leave-one-out residuals inside the 95 % predictive interval. Below
       that the emulator is overconfident, and the active-learning utility is an
       expectation over exactly those error bars.
   * - Posterior resolution
     - at least max(200, 25 d)
     - Accepted posterior samples. Everything downstream, the maximum included, is
       estimated from the accepted rejection sample; a handful of samples is noise and
       its maximum is a random draw. The check enlarges its own prior sample first, so
       this measures the design and not ``prior_samples``.
   * - Data-driven shape
     - at most 0.50
     - Median emulator standard deviation over the observation standard deviation, at the
       accepted samples. Above that, the posterior is a picture of what the emulator does
       not know rather than of what the measurements say.
   * - Stability
     - at most 0.25 std, and a log-evidence change of at most 1
     - How far the posterior moved since the previous block. Needs two blocks, so a first
       block is never "sufficient". Measured on the posterior *mean*, not on its maximum:
       with a few hundred accepted samples the maximum wanders by a third of a standard
       deviation between two rejection samplings of the same emulator, so a criterion
       built on it would test the random number generator instead of the design.

The verdict is ``sufficient`` when all five pass, ``marginal`` when the emulator and the
posterior resolution are adequate but the design has not settled, and ``insufficient``
otherwise. When the ladder reaches the ``init_runs`` ceiling while still insufficient, the
calibration proceeds and says so: the posterior maximum it eventually reports has to be
treated as provisional.

Configuration
++++++++++++++

.. code-block:: python

   sampling = {
       'init_runs':                 128,      # ceiling, from the table above
       'max_runs':                  180,      # total budget: initial runs plus BAL
       'parameter_sampling_method': "sobol",
       'adaptive_init_runs':        True,     # grow in blocks, stop when sufficient
       'init_runs_min':             None,     # first block; None -> 2**ceil(log2(4 d))
   }

Set ``adaptive_init_runs = False`` for the fixed-size initial design of earlier versions,
in which all ``init_runs`` simulations are run in one block and no gate is applied.

.. warning::

   When the ladder stops early, the number of initial runs actually carried out is
   smaller than the configured ``init_runs``. That number is written to
   ``restart_data/initial-design.json`` and logged, and it is the value to put into
   ``sampling['init_runs']`` before restarting the calibration with ``only_bal_mode``,
   which reads exactly ``init_runs`` rows from ``initial-collocation-points.csv``.


Step 2: Assign the calibration settings
---------------------------------------

As it was mentioned before the calibration process involves two well defined parts in the code. Both processes depend on the user-defined settings, i.e. the calibration parameters to adjust
and the calibration targets to fit against (see :ref:`terminology`), which are essential
to run the code properly.
Firstly, the initialization of all settings must be done in the ``src/hydroBayesCal/drivers/bal_telemac.py`` Python script. ``bal_telemac.py`` is the main script that runs the calibration process and
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

  * **"sobol"** - Sobol sequence sampling (default, see :ref:`initial-design`).
  * **"random"** - Random sampling.
  * **"latin_hypercube"** - Latin Hypercube Sampling (LHS).
  * **"halton"** - Halton sequence sampling.
  * **"hammersley"** - Hammersley sequence sampling.
  * **"chebyshev"** - Chebyshev nodes.
  * **"grid"** - Grid-based sampling.
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
    ``src/hydroBayesCal/drivers/prebal_telemac_error_analysis.py``). At a fixed discharge
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


Step 3: Bayesian model optimization
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

Staying out of a local maximum
+++++++++++++++++++++++++++++++

Every training point above is chosen by exploiting the current posterior, which is the
efficient thing to do while the posterior has a single maximum: the runs go where the
answer is. It stops being the right thing to do as soon as the posterior turns out to have
more than one mode, because pure exploitation keeps refining whichever mode the design
started in, and a mode that is never sampled again cannot overtake it. The calibration
then converges cleanly onto a local maximum.

With ``sampling['bal_exploration_tradeoff'] = 'auto'`` (the default), the per-iteration
posterior diagnostic counts the well-separated modes, and from the first iteration that
finds more than one, the point selection balances exploitation against exploration of the
parameter space for the remaining iterations. The switch is logged, and recorded per
iteration in ``BAL_dictionary.pkl`` so an archived result shows which points were chosen
under which regime. Setting the key to ``True`` or ``False`` forces one behaviour for the
whole calibration.

.. warning::

    The last training point of the calibration is **not** the calibrated parameter set.
    Bayesian active learning selects every training point by *information gain*: the
    parameter combination that most reduces uncertainty about the posterior, which is
    not the combination that best reproduces the measurements. The calibrated
    parameter sets are derived from the posterior itself, see
    :ref:`calibrated-parameters`.


Step 4: Post-calibration data
------------------------------

The Bayesian Active Learning (BAL) process runs iteratively until the specified ``max_runs`` limit is reached.
After completion, the post-calibration data is automatically saved in a directory named
**auto-saved-results-HydroBayesCal**.

Inside this directory, you will find four subfolders containing all the necessary information
for analyzing the calibration process, including the trained GPR metamodels.

For a detailed explanation of the saved data, please refer to :ref:`outputs-folder`.


.. _calibrated-parameters:

Step 5: Derive the calibrated parameter sets
--------------------------------------------

The BAL loop stores, for every iteration, a **joint** posterior sample in
``BAL_dictionary.pkl``: the prior samples accepted by rejection sampling against the
joint likelihood over all calibration points and calibration targets. That sample, not any
single training point, is the calibration result.

The calibrated parameter set is the **maximum of the joint posterior probability density
function**, i.e. the single parameter combination the measurements support most strongly
once all parameters are considered together. The per-parameter marginal optima answer a
different question, and most of this step is about telling the two apart and deciding
which one to report.

Run the derivation on a finished calibration:

.. code-block:: bash

   python src/hydroBayesCal/drivers/derive_calibrated_parameters.py --config config_Telemac.py \
       --surrogate <path/to/a/pickled/gpe.pkl> --refine

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

The marginal is a *projection* of the joint posterior, and a projection loses exactly the
information that says which combinations are allowed. That is why the joint posterior is
the object of inference here and the marginals are a summary of it, never a substitute.

Locating the maximum of the joint posterior
++++++++++++++++++++++++++++++++++++++++++++

The joint maximum is found in three stages, each of which removes a specific way of
getting the wrong answer:

1. **Density ranking over the accepted samples.** The joint posterior density is estimated
   at every accepted sample from nearest-neighbour distances, and the densest sample is
   taken. Being an actual posterior sample, it is jointly consistent by construction. No
   smoothing is applied: a kernel density estimate would return the peak of the smoothed
   curve rather than of the posterior, and would pull the peak away from a prior bound.
2. **Mode detection.** A Gaussian mixture over the accepted samples, pruned by weight,
   separation and *level-set connectivity*, reports how many genuinely distinct solutions
   the posterior has. A continuous trade-off ridge counts as one mode, since every
   combination along it is about as good; only solutions separated by a real drop in
   density between them count as separate.
3. **Multi-start local refinement** (``--refine``, needs ``--surrogate``). The emulator's
   joint posterior is maximised over the continuous calibration ranges with a bounded
   quasi-Newton method, started from the highest-density samples **and from one
   representative of every detected mode**.

Stage 3 is what makes the reported optimum both precise and global:

* Without it, the answer is the best of finitely many prior draws. With 10 000 draws in
  five dimensions the nearest draw to the true maximum is typically several percent of the
  calibration range away, and it moves from one rejection sampling to the next.
* Seeding every mode is what makes the search *global*. A local optimiser started only in
  the densest basin climbs to the top of that basin however much deeper another one is,
  and the accepted sample is densest where the emulator happens to have been sampled most,
  which is not necessarily where the best fit is. One start per basin removes that.

The agreement between the starts is reported: when all of them converge to the same
parameter set, the maximum is global over the basins the posterior exhibits; when they do
not, the posterior has several local maxima, the reported one is the highest of those
found, and the run says so.

Vetting the marginal optima against the joint optimum
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Whether the marginal peaks may be reported as a parameter set is decided, not guessed,
from three measurements on the accepted sample:

* the **posterior correlation** between the parameters, i.e. whether the independence
  assumption behind stacking the marginals holds at all;
* the **Mahalanobis distance** of the marginal-peak vector under the posterior covariance,
  i.e. how unusual that combination is compared with the accepted samples themselves;
* the **joint posterior density** at that vector, as a percentile of the density at the
  accepted samples, i.e. whether the posterior puts any mass there.

Those give the **equifinality verdict**: ``consistent`` (the parameters are effectively
independent), ``acceptable`` (mildly coupled, the vector is in the posterior bulk),
``coupled`` (in the bulk, but the parameters trade off so tightly that the marginals do
not determine the combination) and ``inconsistent`` (the vector sits in a low-density
region and is not a valid parameter set at all).

How the calibrated optima are chosen
+++++++++++++++++++++++++++++++++++++

:func:`~hydroBayesCal.surrogate.posterior_analysis.select_calibrated_parameters` turns
those measurements into one decision. The joint maximum is the default and the marginal
peaks have to earn their promotion:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Posterior situation
     - Reported calibrated parameter set
   * - More than one well-separated mode
     - **None.** Several different parameter combinations reproduce the measurements
       comparably well and the posterior cannot tell them apart. Every mode
       representative is reported for the full complexity model to arbitrate.
   * - ``inconsistent``
     - **Joint maximum.** The marginal-peak vector is explicitly rejected as a parameter
       set; the marginal peaks remain valid as a per-parameter summary.
   * - ``coupled``
     - **Joint maximum.** The marginals do not determine the combination, and that they
       happen to be compatible here is a coincidence rather than a confirmation.
   * - ``consistent`` **and** every marginal peak within 0.25 posterior standard
       deviations of the joint maximum
     - **Marginal-peak vector**, confirmed by the joint maximum: both readings of the
       posterior give the same answer, and the marginals additionally carry credible
       intervals per parameter.
   * - anything else (``acceptable``)
     - **Joint maximum**, with the marginal-peak vector to be checked against it by
       running the solver at both.

Parameters flagged **non-identifiable** are reported from the joint maximum and marked:
any value in their range fits about as well, so the number is a placeholder the data does
not support, whichever vector it came from.

The decision, its reasoning and the parameter sets it asks to be verified are logged and
written to the ``selected`` column of ``calibrated-parameter-candidates.csv``
(``calibrated`` for the reported set, ``verify`` for the ones to run).

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

   python src/hydroBayesCal/drivers/bal_telemac.py --config config_Telemac.py
   python src/hydroBayesCal/drivers/assess_calibration.py --config config_Telemac.py

Letting the full complexity model arbitrate between a handful of labelled candidates is
the honest way to close a calibration whose posterior is equifinal.

Watching the optima converge
+++++++++++++++++++++++++++++

The same diagnostics are recorded at every BAL iteration, so
``src/hydroBayesCal/drivers/plot_posteriors.py`` produces two additional figures:

* ``parameter_optimum_convergence``: each parameter's own optimum **and the joint
  maximum** against the number of training points, with the credible interval and the
  calibration range. A trace still drifting at the end means the calibration has not
  converged for that parameter; a trace sitting on a bound means it is pinned. Where the
  two traces converge onto each other, reading the posterior per parameter and reading it
  jointly give the same answer; where they stay apart, only the joint trace is a
  parameter set that can be run.
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