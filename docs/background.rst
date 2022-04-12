.. Stochastic surrogate workflow.


Surrogate Workflow
==================

The workflow describes the usage of Bayesian model evidence and relative entropy in combination with a Gaussian Process Emulator proposed by `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_.

Step 0: Wet your TELEMAC Model
------------------------------

Before the surrogate-assisted calibration can run, it needs an initial model run. The first model run should start with `dry conditions (read more at hydro-informatics.com) <https://hydro-informatics.com/numerics/telemac2d-steady.html>`_ and be adapted to `wet (steady or unsteady hotstart) initial conditions <https://hydro-informatics.com/numerics/telemac2d-unsteady.html#hotstart-initial-conditions>`_ for the surrogate-assisted calibration.

.. note:: Why hotstart the model for the surrogate-assisted calibration?

    Instead of applying an initial water depth that covers the entire model domain (or other initial condition types), a numerical model of a fluvial ecosystem typically starts dry to avoid filling disconnected terrain depressions with water. However, wet initial conditions converge considerably faster if those approximately correspond to the target conditions. Thus, to speed up the surrogate-assisted calibration, preferably do one dry model initialization and then switch to fast converging hotstart (wet initial) conditions.

Step 1: Initialize Information Range
------------------------------------

The initialization consists of building void (zero-value) matrices for Bayesian model evidence (BME) and relative entropy (RE) scores with the size of the user-defined limit of calibration iterations (default is ``it_limit = 15``).
In addition, void matrices for the active learning sampling are instantiated based on user definitions (default is ``al_samples = 1000``).

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
