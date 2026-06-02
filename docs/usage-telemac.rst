.. _usage-telemac:

Using HydroBayesCal with TELEMAC
================================

This page explains how to run a surrogate-assisted Bayesian calibration of a
**TELEMAC** (2D/3D) model. It complements the generic :doc:`workflow` and the
binding walk-through in :doc:`gpe-bal-telemac`.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

* A working **TELEMAC v9** installation with its Python API reachable. Follow
  the installation guide at `hydro-informatics.com/install-telemac
  <https://hydro-informatics.com/install-telemac/>`_ and the HydroBayesCal
  :doc:`installation` page for coupling the two environments.
* A **fully functional TELEMAC model** that runs to completion on its own
  *before* you start any calibration.
* HydroBayesCal installed in an environment that is loaded together with the
  TELEMAC environment variables (see :doc:`installation` —
  ``env-scripts/activateHBCtelemac.sh``).

.. note::

   With TELEMAC v9 the solver is launched through ``telemac2d.py`` /
   ``telemac3d.py`` from the active ``systel`` configuration (``pysource``
   file). HydroBayesCal calls these launchers for you; you only need the
   combined environment loaded so that ``telemac2d.py`` is on the ``PATH``.

The TELEMAC simulation folder
-----------------------------

Collect all files needed for one TELEMAC run into a single model directory
(``model_dir``). A typical hydrodynamic setup contains:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Purpose
   * - ``*.cas``
     - Steering (control) file with the numerical configuration. Its name is
       passed as ``control_file``.
   * - ``*.cli``
     - Boundary-condition file (type and location of the boundaries).
   * - ``*.slf``
     - SELAFIN geometry/mesh file.
   * - ``*.tbl``
     - Friction (roughness) table, required when calibrating friction *zones*
       (passed as ``friction_file``).
   * - ``*.liq``
     - Liquid boundary conditions for unsteady flow (if applicable).
   * - rating curve / stage-discharge
     - Outflow boundary definition (if applicable).

.. admonition:: Before calibrating

   * Verify the model runs properly on its own.
   * Start from `dry initial conditions
     <https://hydro-informatics.com/numerics/telemac2d-steady.html>`_ once, then
     switch to fast-converging `hotstart (wet) conditions
     <https://hydro-informatics.com/numerics/telemac2d-unsteady.html#hotstart-initial-conditions>`_
     to speed up the many surrogate-training runs.

Measurement / calibration points
---------------------------------

The calibration targets are provided in a CSV file (``calibration_pts_file_path``)
with one row per measurement location. The header uses the coordinates and, for
each calibration quantity, a ``<quantity>_DATA`` and a ``<quantity>_ERROR``
column (matched case-insensitively)::

    X, Y, WATER DEPTH_DATA, WATER DEPTH_ERROR, SCALAR VELOCITY_DATA, SCALAR VELOCITY_ERROR
    ...

The ``_ERROR`` column holds the measurement error in the physical units of the
quantity. Measurements and the computational mesh (``.slf``) must use the same
coordinate reference system so modelled and measured values can be compared.

TELEMAC-specific parameters
---------------------------

In addition to the common :ref:`HydroSimulations_class` parameters, the
``TelemacModel`` class accepts:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Parameter
     - Description
   * - ``tm_xd``
     - Hydrodynamic solver: ``"Telemac2d"`` or ``"Telemac3d"``.
   * - ``control_file``
     - Name of the ``.cas`` steering file (no path).
   * - ``friction_file``
     - Name of the ``.tbl`` friction file (no path); needed for friction zones.
   * - ``results_filename_base``
     - Base name of the results ``.slf`` file, updated for each run and used for
       output extraction.
   * - ``gaia_steering_file``
     - GAIA steering file for morphodynamics (optional).

Calibration parameters and TELEMAC keywords
--------------------------------------------

For TELEMAC, each entry of ``calibration_parameters`` must match a keyword in
the model files:

* **Steering-file keywords** must be written exactly as in the ``.cas`` file,
  using ``=`` (not ``:``). See the `TELEMAC user manuals
  <https://wiki.opentelemac.org/doku.php#principal_documentation>`_.

  .. code-block:: python

     calibration_parameters = ["LAW OF FRICTION ON LATERAL BOUNDARIES",
                               "INITIAL ELEVATION", "BOTTOM FRICTION"]

* **Friction zones** are calibrated per zone; the zone names must be defined in
  the ``.tbl`` friction file, and each name must contain ``zone``/``Zone``/
  ``ZONE`` as a prefix. See `friction (roughness) zones
  <https://hydro-informatics.com/numerics/telemac/roughness.html>`_.

  .. code-block:: python

     calibration_parameters = ["zone1", "zone2", "Zone3"]

``calibration_quantities`` / ``extraction_quantities`` use SELAFIN variable
names, e.g. ``"WATER DEPTH"``, ``"SCALAR VELOCITY"``, ``"TURBULENT ENERG"``,
``"VELOCITY U"``, ``"VELOCITY V"``, ``"CUMUL BED EVOL"``.

.. note::

   Bayesian active learning only handles numeric (scalar) parameters. A
   parameter that takes a non-numeric value (e.g. a solver-type keyword) cannot
   be used as a calibration parameter.

Running the calibration
-----------------------

Define a configuration file (see the example ``config.py``) and launch the
TELEMAC driver:

.. code-block:: bash

   python bal_telemac.py --config config.py

The driver builds a ``TelemacModel``, sets up the experimental design, runs the
initial simulations, trains the GPE and performs Bayesian Active Learning.
Equivalently, in a script:

.. code-block:: python

   from hydroBayesCal.telemac.control_telemac import TelemacModel

   model = TelemacModel(
       control_file="tel_model.cas",
       tm_xd="Telemac2d",
       friction_file="friction.tbl",
       results_filename_base="results",
       model_dir="/path/to/telemac_simulation",
       res_dir="/path/to/results",
       calibration_pts_file_path="/path/to/measurements-calibration.csv",
       n_cpus=8,
       init_runs=15,
       max_runs=30,
       calibration_parameters=["zone1", "zone2", "BOTTOM FRICTION"],
       param_values=[[0.011, 0.79], [0.011, 0.79], [0.018, 0.028]],
       calibration_quantities=["WATER DEPTH", "SCALAR VELOCITY"],
       extraction_quantities=["WATER DEPTH", "SCALAR VELOCITY", "TURBULENT ENERG"],
   )

See :doc:`gpe-bal-telemac` for the full driver, the experiment-design and BAL
functions, and the structure of the output folders.

See also
--------

* :doc:`installation` — environment and TELEMAC setup.
* :doc:`workflow` — the calibration workflow and all configuration parameters.
* :doc:`usage-openfoam` — the analogous OpenFOAM guide.
