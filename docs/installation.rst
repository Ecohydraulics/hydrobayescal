.. _installation:

Installation
============

HydroBayesCal is a Python package bound to an external numerical model
(TELEMAC is fully supported, OpenFOAM is in progress, Delft3D-FLOW is planned).
It is developed and tested
on **Debian/Ubuntu Linux**, which we recommend: Linux gives the flexibility to
configure the environment and TELEMAC together, and a command-line interface
well suited to batch simulations. Windows has not been tested.

This page covers (1) the Python environment and the package itself, and
(2) the numerical-model bindings.

.. contents::
   :local:
   :depth: 2

Python environment
------------------

Dependencies are declared once in ``pyproject.toml``; the legacy
``requirements*.txt`` files simply install the project. Use a dedicated
environment.

.. important::

   HydroBayesCal currently targets **Python 3.10 or 3.11**. The upper bound is
   imposed by the ``bayesvalidrox`` dependency, which declares ``python < 3.12``.

Option A — from PyPI (recommended)
++++++++++++++++++++++++++++++++++

Install the released package with pip (preferably into a fresh virtual
environment):

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install hydroBayesCal

The core install pulls the full surrogate/UQ stack (NumPy, SciPy, pandas,
scikit-learn, PyTorch, GPyTorch, emcee, chaospy, PyVista and BayesValidRox).

Option B — conda / mamba (development)
++++++++++++++++++++++++++++++++++++++

A ready-made environment file is provided. From the repository root:

.. code-block:: bash

   mamba env create -f environment.yml   # or: conda env create -f environment.yml
   mamba activate hbenv

This creates a Python 3.11 environment and installs HydroBayesCal in editable
mode with its documentation, mesh and model-server extras.

Option C — editable install from a clone
++++++++++++++++++++++++++++++++++++++++

HydroBayesCal uses a ``src`` layout and is installed like any modern Python
package:

.. code-block:: bash

   git clone https://github.com/sschwindt/hydrobayescal.git
   cd hydrobayescal
   pip install -e ".[dev,docs,mesh]"

Optional feature sets (*extras*) can be added in brackets (e.g.
``pip install "hydroBayesCal[mesh]"``):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Purpose
   * - ``mesh``
     - Extra mesh / geospatial IO (``vtk``, ``meshio``, ``rasterio``) for some
       post-processing utilities.
   * - ``server``
     - UM-Bridge model-server coupling (``umbridge``).
   * - ``mpi``
     - Parallel execution on HPC clusters (``mpi4py``; needs an MPI toolchain).
   * - ``docs``
     - Build this documentation (Sphinx + RTD theme + Mermaid).
   * - ``dev``
     - Development tooling (``pytest``, ``build``, ``twine``, ``ruff``).

For example, a developer install with documentation and mesh tools:

.. code-block:: bash

   pip install -e ".[dev,docs,mesh]"

.. note::

   PyTorch and VTK wheels are large (several hundred MB). On clusters without
   a working MPI toolchain, omit the ``mpi`` extra.

Numerical-model bindings
------------------------

TELEMAC
+++++++

*Time requirement: ~60 min.*

The calibration routines drive the open-source modelling system
**TELEMAC-Mascaret** (TELEMAC 2D solves the depth-averaged shallow-water
equations; TELEMAC 3D solves the Reynolds-averaged Navier–Stokes equations).
Install it following the developers' instructions at `opentelemac.org
<http://www.opentelemac.org/index.php/installation>`_, or the
HydroBayesCal-tailored guide at `hydro-informatics.com/install-telemac
<https://hydro-informatics.com/install-telemac/>`_. We recommend
the ``pysource.gfortranHPC.sh`` configuration (or your preferred ``pysource``
file).

Loading HydroBayesCal **and** TELEMAC together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a calibration needs both the HydroBayesCal environment and the TELEMAC
environment variables active at once. The repository ships an activation
template in ``env-scripts/activateHBCtelemac.sh``.

.. note::

   **One-time setup** — open ``env-scripts/activateHBCtelemac.sh`` in a text
   editor and set:

   * ``TELEMAC_CONFIG_DIR`` — your TELEMAC installation's config directory.
   * ``TELEMAC_CONFIG_NAME`` — your TELEMAC configuration file name.
   * the path/name of your HydroBayesCal environment.

   **Each session** — from the repository root, either source the script
   directly or use the convenience launcher:

   .. code-block:: bash

      source env-scripts/activateHBCtelemac.sh
      # or
      python activate_HBC_Telemac.py

   A successful activation reports that both the HydroBayesCal and TELEMAC
   environments were loaded.

Verify the TELEMAC Python API is reachable:

.. code-block:: bash

   python -c "import telapy; print(telapy.__version__)"

OpenFOAM
++++++++

.. note::

   The OpenFOAM (interFoam) binding is under active development. It requires a
   working `OpenFOAM <https://www.openfoam.com/>`_ installation on ``PATH``
   (``decomposePar``, ``reconstructPar``, the solver, and ``foamToVTK``); see
   the installation guide at `hydro-informatics.com/install-openfoam
   <https://hydro-informatics.com/install-openfoam/>`_. The ``mesh`` extra
   (PyVista/VTK) is used to read the VTK output. See :doc:`usage-openfoam` for
   running a calibration.

Delft3D-FLOW
++++++++++++

.. note::

   The Delft3D-FLOW (Deltares) binding is **planned and not yet implemented**.
   A placeholder interface
   (:class:`hydroBayesCal.delft3d.control_delft3d.Delft3DModel`) records the
   intended coupling and the Delft3D-specific file conventions (``.mdf`` master
   definition file, ``config_d_hydro.xml`` / ``d_hydro`` launcher, NEFIS
   ``trim``/``trih`` output). It raises :class:`NotImplementedError`. See
   :doc:`usage-delft3d` for the planned workflow.

Check your installation
------------------------

With the environment active, confirm the package imports:

.. code-block:: bash

   python -c "import hydroBayesCal; print('HydroBayesCal OK')"

You are now ready to set up a calibration — continue with :doc:`workflow`.

Building and publishing (maintainers)
-------------------------------------

The project is configured for PyPI via :file:`pyproject.toml` (``src`` layout,
SPDX license, build backend ``setuptools``). To build and check the
distribution artifacts:

.. code-block:: bash

   pip install -e ".[dev]"          # provides build + twine
   python -m build                  # creates dist/*.whl and dist/*.tar.gz
   twine check dist/*               # validate the metadata/long description

Upload to TestPyPI first, then to PyPI:

.. code-block:: bash

   twine upload --repository testpypi dist/*
   twine upload dist/*

Bump ``version`` in :file:`pyproject.toml` (and tag the release) before each
upload.
