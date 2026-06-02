.. _usage-openfoam:

Using HydroBayesCal with OpenFOAM
=================================

This page explains how to run a surrogate-assisted Bayesian calibration of an
**OpenFOAM** (``interFoam``) case. It mirrors :doc:`usage-telemac` and
complements the generic :doc:`workflow`.

.. note::

   The OpenFOAM binding is **under active development**. The interface and the
   calibration-parameter format may still change; treat the examples below as a
   starting point and check :mod:`hydroBayesCal.openfoam.control_openfoam` for
   the current behaviour.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

* A working **OpenFOAM** installation (the binding is developed against the
  ``com`` releases, e.g. OpenFOAM v2412) with the standard utilities on the
  ``PATH``: the solver (``interFoam``), ``decomposePar``, ``reconstructPar``
  and ``foamToVTK``. See the installation guide at
  `hydro-informatics.com/install-openfoam
  <https://hydro-informatics.com/install-openfoam/>`_.
* A **fully functional interFoam case** that runs to completion on its own
  before calibration.
* HydroBayesCal installed with the ``mesh`` extra (PyVista/VTK), which is used
  to read the VTK output:

  .. code-block:: bash

     pip install "hydroBayesCal[mesh]"

The OpenFOAM case template
--------------------------

HydroBayesCal copies a **case template** (``case_template_dir``) for each run
and modifies the relevant dictionaries. A standard interFoam case contains the
usual OpenFOAM directory structure:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Path
     - Purpose
   * - ``system/``
     - ``controlDict``, ``fvSchemes``, ``fvSolution``, ``decomposeParDict``.
   * - ``constant/``
     - Mesh (``polyMesh``), ``transportProperties``, ``turbulenceProperties``
       (e.g. ``kEpsilonCoeffs`` for the ``Cmu`` coefficient).
   * - ``0/``
     - Initial/boundary fields (``U``, ``p_rgh``, ``alpha.water``, ``k``,
       ``epsilon``, ``nut`` …).

.. important::

   The turbulent kinetic energy ``k`` must be written to the VTK output,
   otherwise TKE and velocity-fluctuation outputs are ``NaN``. HydroBayesCal
   checks ``system/controlDict`` on start-up and warns if ``k`` is not listed.
   A parallel run additionally needs ``system/decomposeParDict``.

OpenFOAM-specific parameters
----------------------------

In addition to the common :ref:`HydroSimulations_class` parameters, the
``OpenFOAMModel`` class accepts:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Parameter
     - Description
   * - ``case_template_dir``
     - Path to the OpenFOAM case template that is copied for each run.
   * - ``solver_name``
     - OpenFOAM solver, default ``"interFoam"``.
   * - ``n_processors``
     - Number of subdomains for the parallel run (``decomposePar`` /
       ``mpirun``).
   * - ``control_file``
     - Control dictionary, default ``"system/controlDict"``.
   * - ``alpha_water_name``
     - Name of the water volume-fraction field, default ``"alpha.water"``.
   * - ``water_surface_alpha``
     - Volume-fraction threshold used to locate the free surface (e.g. ``0.5``).
   * - ``reference_z``
     - Reference elevation for water-depth/free-surface extraction.
   * - ``n_avg_timesteps``
     - Number of final time steps to average when extracting outputs.

Calibration parameters
-----------------------

OpenFOAM calibration parameters map to model coefficients and boundary/
dictionary entries that ``OpenFOAMController`` writes into the case, for example:

* **Turbulence coefficient** ``Cmu`` in ``constant/turbulenceProperties``
  (``kEpsilonCoeffs``).
* **Wall roughness** ``ks`` applied as a boundary condition (e.g. on a wall
  patch in ``0/nut``).
* Other **boundary-condition values** or **dictionary entries**, updated via
  ``update_boundary_condition`` / ``update_dictionary_entry``.

``calibration_quantities`` / ``extraction_quantities`` use the standard field
names, e.g. ``"U_x"``, ``"U_y"``, ``"U_z"``, ``"U_MAG"`` (velocity components /
magnitude), ``"TKE"`` (turbulent kinetic energy ``k``), ``"WATER_DEPTH"`` and
``"FREE_SURFACE"``. As for TELEMAC, the calibration CSV provides a
``<quantity>_DATA`` and ``<quantity>_ERROR`` column per quantity, together with
the ``X``, ``Y`` (and ``Z``) coordinates of the measurement points.

Running the calibration
-----------------------

Define a configuration file and launch the OpenFOAM driver:

.. code-block:: bash

   python bal_openfoam.py --config config.py

The driver builds an ``OpenFOAMModel``, runs the initial simulations
(``decomposePar`` → ``interFoam`` → ``reconstructPar`` → ``foamToVTK``),
extracts the requested fields at the calibration points, trains the GPE and
performs Bayesian Active Learning. Equivalently, in a script:

.. code-block:: python

   from hydroBayesCal.openfoam.control_openfoam import OpenFOAMModel

   model = OpenFOAMModel(
       case_template_dir="/path/to/interfoam_case_template",
       solver_name="interFoam",
       n_processors=8,
       control_file="system/controlDict",
       alpha_water_name="alpha.water",
       water_surface_alpha=0.5,
       reference_z=0.0,
       model_dir="/path/to/model",
       res_dir="/path/to/results",
       calibration_pts_file_path="/path/to/measurements-calibration.csv",
       n_cpus=8,
       init_runs=10,
       max_runs=30,
       calibration_parameters=["Cmu"],
       param_values=[[0.06, 0.12]],
       calibration_quantities=["U_x", "U_y", "U_z"],
       extraction_quantities=["U_x", "U_y", "U_z", "TKE"],
   )

Results are written to the same ``auto-saved-results-HydroBayesCal`` layout as
for TELEMAC (see :doc:`gpe-bal-telemac`), so post-processing is identical across
solvers.

See also
--------

* :doc:`installation` — environment and OpenFOAM setup.
* :doc:`workflow` — the calibration workflow and all configuration parameters.
* :doc:`usage-telemac` — the analogous TELEMAC guide.
