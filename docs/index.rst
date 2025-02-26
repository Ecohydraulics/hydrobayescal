.. Introduction.


About
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
However, this is unfeasible when a single realization may require several hours or even days. To make this possible, surrogate models (approximated models) are constructed as
a first step with only a few number of model realizations or so called initial collocation points which are strategically selected using advanced parameter sampling methods.
The package builds on top of `BayesValidRox <https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/>`_ (until now only for Design of Experiments) and uses Gaussian Process Regression(GPR)
to construct single outputs and multioutputs surrogate models which are useful to predict the model outputs at any parameter combination and Bayesian Inference to quantify the uncertainty of the model
parameters.

The package employs Bayesian Active Learning, or BAL, to iteratively add new training points (parameter combinations) that yielded the highest value of relative entropy,
hence increasing the model's accuracy in the parameter space regions that are most crucial for Bayesian inference.

Prerequisites (requirements)
===========================

HydroBayesCal is a Python package, bound to the installation of a numerical model. Currently, only Telemac 2D/ 3D bindings are enabled. The package runs in Debian Linux platforms
including Ubuntu and its derivatives. he package has not been tested in Windows yet. We recommended to install the package in a Linux Operating System because
it gives the flexibility for configuring the virtual environment and optimizing settings for Telemac simulations while providing
a powerful command-line interface, which is well-suited for running batch simulations and automating tasks.

To start, this section guides through setting up Python and the virtual environment for working with HydroBayesCal
and the Telemac system.


Python
------

To get the code running, we strongly recommend creating a new conda or virtual environment as described in the `Anaconda Docs <https://docs.continuum.io/anaconda/install/windows/>`_ and
at `hydro-informatics.com/python-basics <https://hydro-informatics.com/python-basics/pyinstall.html>`_ with Python 3.10. ``HydroBayesCal`` potentially also works with earlier Python3 versions,
but it was developed and tested with v3.10.

Windows user will have the best experience with Anaconda/conda env and Linux users with virtualenv.

.. admonition:: New to Python?

Have a look at `hydro-informatics.com`_, follow the detailed `Python installation guide for your platform <https://hydro-informatics.com/python-basics/pyinstall.html>`_,
and dive into the `Python tutorials <https://hydro-informatics.com/python-basics/python.html>`_


.. todo::

    View the environment requirements for the package and the installation of the package in the `requirements section <requirements>`_.

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
at `https://hydro-informatics.com <https://hydro-informatics.com/get-started/install-telemac.html>`_ that
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
.. code::

   python3 -m venv HBCenv

Next, activate the environment, `download requirements.txt <https://github.com/Ecohydraulics/hydrobayescal/requirements.txt>`_ and, ``cd`` into the
directory where you downloaded ``requirements.txt`` to install the requirements:

.. code::

    source HBCenv/bin/activate
    cd <TO/REQUIREMENTStxt-DOWNLOAD/FOLDER/>
    pip install -r requirements.txt

Load HBCenv with TELEMAC (Linux)
================================

The simultaneous activation of the *HydroBayesCal* environment and TELEMAC environment variables requires some tweaking,
which can be achieved by source-ing the environment activation templates. To activate the environment specifically for your system,
you need to modify the ``activateHBCtelemac.sh`` file.
The steps to activate the Python and Telemac environments for your system are the following:

.. note:: 

   **One-time actions**:

   Download `activateHBCtelemac.sh <https://github.com/sschwindt/hybayescal/raw/main/env-scripts/activateHBCtelemac.sh>`_ and open it in a text editor to modify the following lines:

   * In line 3, set **``TELEMAC_CONFIG_DIR``** to the location of your TELEMAC installation's config directory.
   * In line 4, set **``TELEMAC_CONFIG_NAME``** to the name of your TELEMAC configuration file.
   * In line 5, set **``HBCenv_DIR``** to the directory where you created ``HBCenv``.
   * Save and close the file after making these changes.

   **Regular load action**:

   To load the combined ``HBCenv`` and TELEMAC environments, open a terminal, navigate to the directory
   where you saved ``activateHBCtelemac.sh``, and enter:

   .. code:: bash

      source activateHBCtelemac.sh

   If the activation was successful, a message will show up:

   .. code:: bash

      > Loading HBCenv...
      **Success**
      > Loading TELEMAC config...
      **Success**

If both environments are loaded without errors, you are good to go for running the codes.


Windows usage
=============

The `source` command is commonly used in Unix-based systems to execute shell scripts that set up environment variables and paths. In Windows, you can achieve similar functionality by using either PowerShell or a compatible shell environment (e.g., WSL, Git Bash). To run a `.sh` file in Windows, use `.\file_name.sh`. Thus, take the following actions: 

1. Open `activateHBCtelemacWindows.ps1` in a text editor and make sure to define the following parameters correctly according to your system settings:

.. code:: bash
   
   $TELEMAC_CONFIG_DIR = "C:\modeling\telemac\v8p5r0\configs"
   $TELEMAC_CONFIG_NAME = "pysource.win.sh"
   $HBCenv_DIR = "C:\USER\hydrobayescal\HBCenv"

2. Save the `.ps1` file.

3. Run the `.ps1` file in PowerShell:

.. code:: bash
   
   .\activateHBCtelemacWindows.ps1

After setting up the environment, test if the Telemac API is working by running:

.. code:: bash
   
   python -c "import telapy; print(telapy.__version__)"

If both environments are loaded without errors, you are good to go for running the codes. There are a couple of issues that can be caused by the execution policy. To allow script execution, you may need to adjust your PowerShell execution policy using:

.. code:: bash
   
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned




Overview of the package components
==================================

The package consists of two well-defined parts:

1. **Hydrodynamic Simulations**:
   This part performs hydrodynamic simulations using any open-source hydrodynamic software.

2. **Surrogate Model and Bayesian Active Learning**:
   This part builds the initial surrogate model using Gaussian Process Regression and performs Bayesian Active Learning. The goal is to improve the initial surrogate by adding new collocation points.

All user input parameters are assigned in the ``user_input.py`` file.

You will find a detailed explanation of each module's functionality in the following documentation.


.. _hydro-informatics.com: https://hydro-informatics.com
