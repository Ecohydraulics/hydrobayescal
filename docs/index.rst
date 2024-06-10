.. Introduction.


Python Package for Bayesian Calibration of Hydrodynamic models 'HydroBayesCal'
=====

What is HydroBayesCal?
========================

**HydroBayesCal**  is a  **Python3 package** that is used for optimization and calibration of hydrodynamic models using a Bayesian Active Learning approach.
The package trains Gaussian Process Emulator (**GPE**) for constructing a surrogate model (i.e.,metamodel) of any numerical model so called full complexity model and evaluates
the metamodel using Bayesian model evidence and/or relative entropy proposed by `Oladyshkin et al. (2020) <https://doi.org/10.3390/e22080890>`_.
To enable parameter updates at every level, the code requires modeling software that is completely open source. Currently, it can only be used in conjunction with Telemac.

This documentation is also as available as `style-adapted PDF <https://hybayescal.readthedocs.io/_/downloads/en/latest/pdf/>`_.

.. admonition:: Good to Know

    For working with HydroBayesCal, make sure to familiarize with the use of TELEMAC software.

    To familiarize with TELEMAC, visit `hydro-informatics.com`_ and read the:

    - `Installation instructions for TELEMAC <https://hydro-informatics.com/get-started/install-telemac.html>`_
    - `TELEMAC tutorial <https://hydro-informatics.com/numerics/telemac.html>`_

.. _requirements:

Purpose and description
========================
The package aims to make significant development and contribution on creating a tool for automated Bayesian Calibration for hydrodynamic models using open source software such as
Telemac and/or OpenFoam. Stochastic calibration techniques require a huge number of full complexity model realizations to perform statistical analysis.
However, this is unfeasible when a single realization may require several hours or even days. To make this possible, surrogate models (reduced models) are constructed as
a first step with only a few number of model realizations or so called initial collocation points which are strategically selected using advanced parameter sampling methods.

The program employs Bayesian Active Learning, or BAL, to iteratively add new training points (parameter combinations) that yielded the highest value of relative entropy, hence increasing the model's
accuracy in the parameter space regions that are most crucial for Bayesian inference.

Prerequisites Software
======================
HydroBayesCal is a Python package, which inherently requires Python, and the installation of an open source numerical
model such as Telemac or/and OpenFoam. Currently, only Telemac 2D/ 3D bindings are enabled. The package runs in Debian Linux platforms
including Ubuntu and its derivatives and Windows, however, it is recommended to be installed in a Linux Operating System because
it gives the flexibility for configuring the virtual environment and optimizing settings for Telemac simulations while providing
a powerful command-line interface, which is well-suited for running batch simulations and automating tasks.

To start, this section guides through setting up Python and the virtual environment for working with HydroBayesCal
and the Telemac system.


Python
------

*Time requirement: 10 min.*

To get the code running, we strongly recommend creating a new conda or virtual environment as described in the `Anaconda Docs <https://docs.continuum.io/anaconda/install/windows/>`_ and
at `hydro-informatics.com/python-basics <https://hydro-informatics.com/python-basics/pyinstall.html>`_ with Python 3.10. ``HydroBayesCal`` potentially also works with earlier Python3 versions,
but it was developed and tested with v3.10.

Windows user will have the best experience with Anaconda/conda env and Linux users with virtualenv.

.. admonition:: New to Python?

    Have a look at `hydro-informatics.com`_, follow the detailed `Python installation guide for your platform <https://hydro-informatics.com/python-basics/pyinstall.html>`_,
and dive into the `Python tutorials <https://hydro-informatics.com/python-basics/python.html>`_


Telemac
------
Telemac-Mascaret or typically known as only Telemac is a robust and versatile integrated modeling tool designed
for simulating free-surface flows with a wide range of applications in both river and maritime hydraulics.
It encompasses several modules, including Telemac 2D and Telemac 3D, each tailored to specific simulation needs.

Telemac 2D is the two-dimensional hydrodynamic simulation module that solves the Saint-Venant equations using either the
finite-element or finite-volume method. On the other hand, Telemac 3D is the three-dimensional hydrodynamic simulation module
that solves the Navier-Stokes equations. In the next step, you will find a step-by-step explanation on how to install Telemac in
your system.

Install TELEMAC
+++++++++++++++
*Time requirement: 60 min.*

The calibration routines are tied to the open-source numerical modeling software TELEMAC. The developers provide installation instructions
at `http://opentelemac.org <http://www.opentelemac.org/index.php/installation>`_, and we also provide a detailed installation guide
at `https://hydro-informatics.com/get-started/install-telemac.html <https://hydro-informatics.com/get-started/install-telemac.html>`_ that
is tweaked for HydroBayesCal. We recommend to install TELEMAC with ``pysource.gfortranHPC.sh``or with your preferred pysource file.

Install HydroBayesCal
------------------
*Time requirement: <5 min.*
To install HydroBayesCal via pip from PyPI


Open an Anaconda Prompt or any other Python-pip-able Terminal and enter:

.. code::

    pip install HydroBayesCal

It is also possible to install manually by cloning HydroBayesCal from GitHub repository:

.. code::

    git clone `https://github.com/.......`_
    cd HydroBayesCal
    pip install .

With the ``HydroBayesCal`` installed you are now ready to use it for running a stochastic optimization of your TELEMAC model.
The `usage section <usage>` provides detailed explanations for running the optimization.

Create Virtual environment in Linux
===================================
The package needs access to system-wide libraries in Linux. The environment is called ``HBCenv``.
You can create your own virtual environment by following these steps:

Open the folder called *HydroBayesCal* and open a terminal in this directory.

Enter this command:
.. code-block::

   python3 -m venv HBCenv

Next, activate the environment, `download requirements.txt <https://github.com/Ecohydraulics/hydrobayescal/requirements.txt>`_ and, ``cd`` into the directory where you downloaded ``requirements.txt`` to install the requirements:

.. code-block::

    source HBCenv/bin/activate
    cd <TO/REQUIREMENTStxt-DOWNLOAD/FOLDER/>
    pip install -r requirements.txt

macOS
+++++

On macOS, potentially both conda env and virtualenv work, but we could not test ``HyBayesCal`` on macOS. Thus, we recommend to follow the instructions for `installing Anaconda on macOS <https://docs.anaconda.com/anaconda/install/mac-os/>`_ and create a new conda environment as above-described in the *Conda on Windows* section.





Load HBCenv with TELEMAC
++++++++++++++++++++++++

The simultaneous activation of the *HydroBayesCal* environment and TELEMAC environment variables requires some tweaking, which can be achieved by source-ing our environment activation templates


.. tabs::

   .. tab:: Linux

      **One-time actions**: `Download activateHBCtelemac.sh <https://github.com/sschwindt/hybayescal/raw/main/env-scripts/activateHBCtelemac.sh>`_ and right-click on *activateHBCtelemac.sh* to open it in a text editor for adapting:
      * In line 3, adapt ``TELEMAC_CONFIG_DIR`` to where your TELEMAC installation's config lives.
      * In line 4, adapt ``TELEMAC_CONFIG_NAME`` to the name of your TELEMAC bash file.
      * In line 5, adapt ``HBCenv_DIR`` to where you created ``HBCenv``.
      * Save and close *activateHBCtelemac.sh*

   .. tab:: Windows

      .. warning::
         The Windows implementation has not yet been tested and is fully experimental. It does most likely not work currently and should be used with uttermost caution. Feedback welcome.

      **One-time actions**: `Download activateHBCtelemac.bat <https://github.com/sschwindt/hybayescal/raw/main/env-scripts/activateHBCtelemac.bat>`_ and right-click on *activateHBCtelemac.sh* to open it in a text editor for adapting:
      * In line 3, adapt ``TELEMAC_CONFIG_DIR`` to where your TELEMAC installation's config lives.
      * In line 4, adapt ``TELEMAC_CONFIG_NAME`` to the name of your TELEMAC bash file.
      * In line 5, adapt ``HBCenv_name`` if you did not use the name ``HBCenv`` for your conda environment.
      * Save and close *activateHBCtelemac.bat*

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

Overview of the package components
============================

The package has two well defined parts. The first part is the one that performs the hydrodynamic simulations with any open-source hydrodynamic software and the second is the one
builds up the initial surrogate model with Gaussian Process Regression and performs Bayesian Active Learning in light of measured data with the purpose of improving the initial surrogate by
adding new collocation points to the initial one.

You will find a detailed explanation of each module functionality in the following documentation.


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
