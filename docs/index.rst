.. Stochastic surrogate documentation parent file.


About
=====

This **Python3 package** uses Bayesian active learning (**BAL**) to wrap around Gaussian process emulators (**GPE**) for constructing a surrogate of complex, deterministic numerical models. To enable parameter adjustments at all levels, the code requires fully open source modeling software. This is why we decided to taylor the code for running it with **`TELEMAC <http://www.opentelemac.org/>`**.

a machine learning method)
Stochastic calibration of a Telemac2d hydro-morphodynamic model using  Surrogate-Assisted Bayesian inversion. The surrogate model is created using Gaussian Process Regression.  The codes implement the methods proposed by `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_.


This documentation is also as available as `style-adapted PDF <https://stochastic-surrogate.readthedocs.io/_/downloads/en/latest/pdf/>`_.

.. admonition:: Good to Know

    For working with this Python3 package, make sure to familiarize with the TELEMAC software suite.

    We recommend to have a look at `hydro-informatics.com`_ and read the:

    - `installation instructions for TELEMAC <https://hydro-informatics.com/get-started/install-telemac.html>`_
    - `TELEMAC tutorial <https://hydro-informatics.com/numerics/telemac.html>`_

Surrogate Workflow
==================

The stochastic surrogate technique for optimizing model calibration involved the following workflow.

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
4. Run Bayesian active learning (BAL) on the output space (**heavy computation load**):
    * Use the indices of priors (i.e. collocation points) that have not been used in the previous steps.
    * Instantiate an active learning output space as a function of a user-defined size (``mc_samples_al``), and the above-calculated surrogate prediction and standard deviation arrays (see item 1)
    * Calculate Bayesian scores as a function of the user-defined strategy (BME or RE), the observations, and the active learning output space.
5. Find the best performing calibration parameter values (maximum BME/RE scores) and set it as the new best parameter set for use with the deterministic (TELEMAC) model
6. Run TELEMAC with the best best performing calibration parameter values.

Step 4: Get Best Performing solution
------------------------------------

The last iteration step corresponds to the supposedly best solution. Consider trying more iteration steps, other calibration parameters, or other value ranges if the calibration results in physical non-sense combinations.

Requirements \& Installation
============================

*Time requirement: 5-10 min.*

Install Requirements
---------------------

To get the code running, the following software is needed and their installation instructions are provided below:

- Python `>=3.6`
- NumPy `>=1.17.4`
- Openpyxl `3.0.3`
- PPutils
- Pandas `>=1.3.5`
- Matplotlib `>=3.1.2`

Start with downloading and installing the latest version of `Anaconda Python <https://www.anaconda.com/products/individual>`_.  Alternatively, downloading and installing a pure `Python <https://www.python.org/downloads/>`_ interpreter will also work. Detailed information about installing Python is available in the `Anaconda Docs <https://docs.continuum.io/anaconda/install/windows/>`_ and at `hydro-informatics.com/python-basics <https://hydro-informatics.com/python-basics/pyinstall.html>`_.


.. admonition:: New to Python?

    Have a look at `hydro-informatics.com`_, follow the detailed `Python installation guide for your platform <https://hydro-informatics.com/python-basics/pyinstall.html>`_, and dive into the `Python tutorials <https://hydro-informatics.com/python-basics/python.html>`_

To install the requirements after installing Anaconda, open Anaconda Prompt (e.g., click on the Windows icon, tap ``anaconda prompt``, and hit ``enter``), and use one of the two following options

Option 1: Recommended
~~~~~~~~~~~~~~~~~~~~~

`Download requirements.txt <https://github.com/sschwindt/stochastic-surrogate/raw/main/requirements.txt>`_ and, in Anaconda Prompt,  ``cd`` into the folder where you downloaded *requirements.txt*. Consider creating and activating a `new conda environment (at least Python 3.6) <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_. Install the requirements for ``stochastic_surrogate`` into the active environment with the following command sequence:

.. code-block::

    conda install pip
    pip install -r requirements.txt


Option 2: Step-by-Step
~~~~~~~~~~~~~~~~~~~~~~

In Anaconda Prompt, enter the following command sequence to install the libraries in the **base** environment. The installation may take a while depending on your internet speed.

.. code-block::

    conda install -c anaconda numpy
    conda install -c anaconda openpyxl
    conda install -c conda-forge pandas
    conda install -c conda-forge matplotlib
    conda install -c phaustin pyutils

If you are struggling with the dark window and blinking cursor of Anaconda Prompt, worry not. You can also use Anaconda Navigator and install the four libraries (in the above order) in Anaconda Navigator.



Install the Stochatic Surrogate Package
---------------------------------------

Still in Anaconda Prompt (or any other Python-pip-able Terminal), enter:

.. code::

    pip install stochastic_surrogate

With the ``stochastic_surrogate`` installed you are now ready to use it for running a stochastic optimization of your TELEMAC model. The `usage section <usage>` provides detailed explanations for running the optimization.


.. toctree::
    :hidden:
    :maxdepth: 2

    Stochastic surrogate <self>

.. toctree::
    :hidden:

    Usage <usage>

.. toctree::
    :hidden:

    Developer Docs <codedocs>

.. toctree::
    :hidden:

    License <license>


.. _hydro-informatics.com: https://hydro-informatics.com