.. Stochastic surrogate documentation parent file.


Documentation for the Python Package for Bayesian Calibration for Hydrodynamic models 'HydroBayesCal'
=====

This **Python3 package** uses Bayesian active learning (**BAL**) to wrap around Gaussian process emulators (**GPE**) for constructing a surrogate (metamodel) of complex, deterministic numerical models. To enable parameter adjustments at all levels, the code requires fully open source modeling software. This is why we decided to taylor the code for running it with `TELEMAC <http://www.opentelemac.org/>`_.

The surrogate model is created using Gaussian Process Regression and evaluated using Bayesian model evidence and/or relative entropy. The codes implement the methods proposed by `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_.


This documentation is also as available as `style-adapted PDF <https://hybayescal.readthedocs.io/_/downloads/en/latest/pdf/>`_.

.. admonition:: Good to Know

    For working with this Python3 package, make sure to familiarize with the TELEMAC software suite.

    To familiarize with TELEMAC, visit `hydro-informatics.com`_ and read the:

    - `Installation instructions for TELEMAC <https://hydro-informatics.com/get-started/install-telemac.html>`_
    - `TELEMAC tutorial <https://hydro-informatics.com/numerics/telemac.html>`_

.. _requirements:

Purpose and description
============================
The package aims to make significant development and contribution on creating a tool for automated Bayesian Calibration for numerical models using open source softwares for hydrodyamic simulation such as
Telemac and/or OpenFoam. Bayesian calibration techniques require a huge number of model simulations to perform statistical analysis in light of measured data.
This is ,in fact, unfeasible when running numerical models may requiere several hours just for one realization. To make this possible, surrogate models (reduced models) are constructed as
a first step with only a few number of model realizations or so called initial collocation points which are estrategically selected using advanced parameter sampling methodology.

The program employs Bayesian Active Learning, or BAL, to iteratively add new training points—parameter combinations—that yielded the highest value of relative entropy, hence increasing the model's
accuracy in the parameter space regions that are most crucial for Bayesian inference.


Requirements \& Installation
============================

HydroBayesCal is a Python package, which inherently requires Python, and the installation of an open source numerical model such us Telemac or/and OpenFoam. Currently, only Telemac bindings are enabled.
This section guides through setting up Python for working with HydroBayesCal and the Telemac system.
Additionally the following libraries are required to be installed in a Python environment called HBenv.

Overview of the package components
============================

The package is well divided in 2 different components, one that treats the full complex model runs with user-defined mdel inputs and the second that treats the
surrogate model construction and the BAL for model calibration purposes.



Python
------

*Time requirement: 10 min.*

To get the code running, we strongly recommend creating a new conda or virtual environment as described in the `Anaconda Docs <https://docs.continuum.io/anaconda/install/windows/>`_ and at `hydro-informatics.com/python-basics <https://hydro-informatics.com/python-basics/pyinstall.html>`_ with Python 3.10. ``HyBayesCal`` potentially also works with earlier Python3 versions, but it was developed and tested with v3.10.

Windows user will have the best experience with Anaconda/conda env and Linux users with virtualenv.

.. admonition:: New to Python?

    Have a look at `hydro-informatics.com`_, follow the detailed `Python installation guide for your platform <https://hydro-informatics.com/python-basics/pyinstall.html>`_, and dive into the `Python tutorials <https://hydro-informatics.com/python-basics/python.html>`_

The package is recommended to be installed in a linux Operating System.  It gives us the flexibility for configuring the environment and optimizing settings for Telemac simulations while providing a powerful command-line interface, which is well-suited for running batch simulations and automating tasks which improves the productivity of the code when managing large data sets.
The creation of a virtual machine with the following characteristics is required:

Virtualenv on Linux
+++++++++++++++++++

To create a virtual environment called ``HBCenv`` that has access to system-wide libraries on Linux, open Terminal and enter (create a new folder called *HyBayesCal* in the directory where you opened Terminal):

.. code-block::

   python3 -m venv HBCenv

Next, activate the environment, `download requirements.txt <https://github.com/sschwindt/hybayescal/raw/main/requirements.txt>`_ and, ``cd`` into the directory where you downloaded ``requirements.txt`` to install the requirements:

.. code-block::

    source HBCenv/bin/activate
    cd <TO/REQUIREMENTStxt-DOWNLOAD/FOLDER/>
    pip install -r requirements.txt


macOS
+++++

On macOS, potentially both conda env and virtualenv work, but we could not test ``HyBayesCal`` on macOS. Thus, we recommend to follow the instructions for `installing Anaconda on macOS <https://docs.anaconda.com/anaconda/install/mac-os/>`_ and create a new conda environment as above-described in the *Conda on Windows* section.


Install HyBayesCal
------------------

*Time requirement: <5 min.*

In Anaconda Prompt or any other Python-pip-able Terminal, enter:

.. code::

    pip install HyBayesCal

With the ``HyBayesCal`` installed you are now ready to use it for running a stochastic optimization of your TELEMAC model. The `usage section <usage>` provides detailed explanations for running the optimization.

TELEMAC
-------

Install TELEMAC
+++++++++++++++
*Time requirement: 60 min.*

The calibration routines are tied to the open-source numerical modeling software TELEMAC. The developers provide installation instructions at `http://opentelemac.org <http://www.opentelemac.org/index.php/installation>`_, and we also provide a detailed installation guide at `https://hydro-informatics.com/get-started/install-telemac.html <https://hydro-informatics.com/get-started/install-telemac.html>`_ that is tweaked for HyBayesCal. We recommend to install TELEMAC with ``pysource.gfortranHPC.sh``.

Load HBCenv with TELEMAC
++++++++++++++++++++++++

The simultaneous activation of the *HyBayesCal* environment and TELEMAC environment variables requires some tweaking, which can be achieved by source-ing our environment activation templates


.. tabs::

   .. tab:: Linux

      **One-time actions**: `Download activateHBCtelemac.sh <https://github.com/sschwindt/hybayescal/raw/main/env-scripts/activateHBCtelemac.sh>`_ and right-click on *activateHBCtelemac.sh* to open it in a text editor for adapting:
      * In line 3, adapt ``TELEMAC_CONFIG_DIR`` to where your TELEMAC installation's config lives.
      * In line 4, adapt ``TELEMAC_CONFIG_NAME`` to the name of your TELEMAC bash file.
      * In line 5, adapt ``HBCenv_DIR`` to where you created ``HBCenv``.
      * Save and close *activateHBCtelemac.sh*

      **Regular load action**: To load the combined ``HBCenv`` and TELEMAC environments, open Terminal, ``cd`` to where you saved *activateHBCtelemac.sh*, and enter:

      .. code-block::

         source activateHBCtelemac.sh

      If both environments are load without errors, you are good to go for running the codes.

   .. tab:: Windows

      .. warning::
         The Windows implementation has not yet been tested and is fully experimental. It does most likely not work currently and should be used with uttermost caution. Feedback welcome.

      **One-time actions**: `Download activateHBCtelemac.bat <https://github.com/sschwindt/hybayescal/raw/main/env-scripts/activateHBCtelemac.bat>`_ and right-click on *activateHBCtelemac.sh* to open it in a text editor for adapting:
      * In line 3, adapt ``TELEMAC_CONFIG_DIR`` to where your TELEMAC installation's config lives.
      * In line 4, adapt ``TELEMAC_CONFIG_NAME`` to the name of your TELEMAC bash file.
      * In line 5, adapt ``HBCenv_name`` if you did not use the name ``HBCenv`` for your conda environment.
      * Save and close *activateHBCtelemac.bat*

      **Regular load action**: Double-click on *activateHBCtelemac.bat*.

      If both environments are load without errors, you are good to go for running the codes.

.. tip::

   The bash environment can also be used with your IDE. For instance, in PyCharm:
   
   * Find the *Run* top menu > *Edit Configurations...* tool. 
   * Select *Shell script*, enter ``HBCtelemac`` in the *Name* field, and find the ``activateHBCtelemac.sh`` script in the field *Script path*.
   * On Linux, make sure the *Interpreter path* is ``/bin/bash``, and click *OK*.
   * Activate the main Python script that you want to use for running *HyBayesCal*, and find the *Run/Debug Configurations* next to the green *Run* arrow (typically in top-right corner of the PyCharm window).
   * In the *Run/Debug Configurations* window, look for the *Before launch* box (scroll down), click on **+** (*Add*) > *Run another configuration*, and select the above-created *HBCtelemac* configuration.
   * Check the *Emulate Terminal in output console* box.
   * Click *OK* and run your Python application with the *HBCtelemac* configuration.

   Read more at `jetbrains.com/help <https://www.jetbrains.com/help/pycharm/run-debug-configuration-shell-script.html>`_.

   If this option does not work, find the *Terminal* box in PyCharm, run ``source activateHBCtelemac.sh`` and execute your TELEMAC files from the PyCharm Terminal (e.g., ``python use_case_tm2d.py``. Either way, to run a Bayesian calibration, better directly use the system Terminal, not an IDE (computation load).


Other Software
--------------

Editing the input.xlsx workbook requires an office program, such as LibreOffice or ONLYOFFICE DesktopEditors (for non-commercial use only). Fore more details, read our `office applications instructions <https://hydro-informatics.com/get-started/others.html#lo>`_.

We recommend working with GMSH for generating a computational mesh for the numerical model. Get GMSH at `https://gmsh.info <https://gmsh.info/>`_ and `QGIS <https://qgis.org>`_.

.. toctree::
    :hidden:
    :maxdepth: 2

    Stochastic surrogate <self>

.. toctree::
    :hidden:

    Workflow <background>

.. toctree::
    :hidden:

    Usage with TELEMAC <usage-telemac>

.. toctree::
    :hidden:

    Developer Docs <codedocs>

.. toctree::
    :hidden:

    License <license>


.. _hydro-informatics.com: https://hydro-informatics.com
