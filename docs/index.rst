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