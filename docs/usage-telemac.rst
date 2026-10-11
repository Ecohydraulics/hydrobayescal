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
each calibration target, a ``<target>_DATA`` and a ``<target>_ERROR``
column (matched case-insensitively)::

    X, Y, WATER DEPTH_DATA, WATER DEPTH_ERROR, SCALAR VELOCITY_DATA, SCALAR VELOCITY_ERROR
    ...

The ``_ERROR`` column holds the measurement error in the physical units of the
calibration target. Measurements and the computational mesh (``.slf``) must use the same
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
   * - ``gaia_layer_average``
     - Sand ratios averaged over GAIA bed layers as derived extraction quantities (optional, keyword-only), see :ref:`gaia-layer-average`.

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

A quantity that is not a variable of the TELEMAC or the GAIA result file, and is not configured as a layer average (next section), is refused at the first extraction with a message that names it.

.. note::

   Bayesian active learning only handles numeric (scalar) parameters. A
   parameter that takes a non-numeric value (e.g. a solver-type keyword) cannot
   be used as a calibration parameter.

.. _gaia-layer-average:

Sand ratio averaged over GAIA bed layers
----------------------------------------

A sediment sample taken over a known depth below the bed surface, e.g. the fine sediment fraction of a freeze core, corresponds to no single variable of a GAIA result file. GAIA writes the ratio of every sand class in every bed layer as its own variable, named ``LAY<layer> SAND RAT<class>``: the number after ``LAY`` is the bed layer, counted from the bed surface, and the trailing number is the sand class, in the order of ``CLASSES SEDIMENT DIAMETERS`` in the GAIA steering file. ``LAY1 SAND RAT2`` is thus the ratio of sand class 2 in the top layer, and ``LAY2 SAND RAT1`` the ratio of sand class 1 in the second layer.

HydroBayesCal derives the quantity that matches such a sample: the ratio of one or several sand classes, averaged over several bed layers with the layer thicknesses as weights. It is configured in the ``morphodynamic_simulation`` block:

.. code-block:: python

   morphodynamic_simulation = {
       'gaia_cas': "gaia_model.cas",
       'gaia_results_filename_base': "results_gaia",
       'gaia_layer_average': {
           "FINE SEDIMENT FRACTION": {       # name of the derived quantity
               "sand_classes": [1],          # ratios of these classes are summed in every layer
               "layers": [1, 2],             # bed layers that are averaged
               "thicknesses": [0.08, 0.60],  # weight of every layer, in meters
           },
       },
   }

   calibration = {
       'extraction_quantities': ["WATER DEPTH", "CUMUL BED EVOL", "FINE SEDIMENT FRACTION"],
       'calibration_quantities': ["FINE SEDIMENT FRACTION"],
       # ...
   }

With this configuration the extraction reads ``LAY1 SAND RAT1`` and ``LAY2 SAND RAT1`` and stores :math:`(0.08\,r_{1,1} + 0.60\,r_{2,1}) / 0.68` under the name ``FINE SEDIMENT FRACTION``, where :math:`r_{l,c}` is the ratio of sand class :math:`c` in layer :math:`l`. The name of the derived quantity is free; the calibration points file needs its ``<name>_DATA`` and ``<name>_ERROR`` columns. The GAIA steering file has to list the layer ratios among its printout variables, and the model needs at least as many bed layers and sand classes as the configuration names; a missing variable is reported with its name. All three entries are required. An entry without ``sand_classes`` is rejected when the model is constructed, because reading the class number for the layer number yields a value between 0 and 1 that looks plausible and is wrong.

Running the calibration
-----------------------

Define a configuration file (see the example ``src/hydroBayesCal/drivers/config_Telemac.py``) and launch
the TELEMAC driver:

.. code-block:: bash

   python src/hydroBayesCal/drivers/bal_telemac.py --config src/hydroBayesCal/drivers/config_Telemac.py

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

Extract & compare 2d/3d simulation data
---------------------------------------

During calibration, HydroBayesCal extracts model results at the calibration
points through ``TelemacModel.extract_data_point``. For TELEMAC-3D results,
this extraction is *depth-explicit*: the routine reads the ``ELEVATION Z``
planes of the 3D SELAFIN file, reconstructs the vertical column above every
horizontal point, and selects the vertical plane whose elevation is closest to
the requested measurement height above the local bed. Horizontal sampling uses
inverse-distance-weighted (IDW) interpolation over the ``k`` nearest mesh
nodes (or the nearest node only), and the time series can be reduced to the
last time step, a specific time step, or the mean of the last ``n`` steps.

The same functionality is available **outside the calibration workflow**
through the top-level function :func:`hydroBayesCal.extract_results`. It works
on any result file - a 2D SELAFIN file, a 3D SELAFIN file, the 2D file
generated by a TELEMAC-3D run, or OpenFOAM VTK output - without a
``TelemacModel`` instance or configuration file:

.. code-block:: python

   import hydroBayesCal as hbc

   # 3D TELEMAC results: extract the velocity magnitude 5 cm above the bed
   # at two points; the closest vertical plane is selected per point
   df = hbc.extract_results(
       "r3d_steady.slf",
       variable="velocity",           # or exact names, e.g. "VELOCITY U"
       x=[371522.5, 371540.0],
       y=[5345152.0, 5345170.0],
       z=0.05,                        # height above the local bed [m]
   )
   print(df)  # columns: point, x, y, z, plane, z_plane, z_above_bed, velocity

   # Full vertical profile (one row per plane) by omitting z
   profile = hbc.extract_results(
       "r3d_steady.slf",
       variable=["velocity", "VELOCITY W"],
       x=371522.5, y=5345152.0,
   )

   # 2D results work with the same call (z is ignored); time-averaging
   # over the last 10 time steps with nearest-node sampling
   df_2d = hbc.extract_results(
       "r2d_steady.slf",
       variable=["WATER DEPTH", "velocity"],
       x=[371522.5, 371540.0], y=[5345152.0, 5345170.0],
       time="mean_last", n_last=10, method="nearest",
   )

The returned :class:`pandas.DataFrame` has one row per point (or per point and
vertical plane for a 3D profile), which makes model-to-model or
model-to-measurement comparisons a simple DataFrame merge.

The ``variable`` argument accepts exact SELAFIN variable names
(case-insensitive, e.g. ``"WATER DEPTH"``, ``"TURBULENT ENERG"``) and the
aliases ``"velocity"`` (flow velocity magnitude: ``SCALAR VELOCITY`` or
``sqrt(U^2+V^2)`` in 2D, ``sqrt(U^2+V^2+W^2)`` in 3D) and ``"depth"``. An
unknown name raises an error that lists the variables available in the file.

The function works analogously for OpenFOAM: pass a ``.vtu``/``.vtk`` file or
a case directory containing the ``VTK/`` folder written by ``foamToVTK``.
OpenFOAM results are inherently three-dimensional, so ``z`` is the absolute
vertical coordinate and is required:

.. code-block:: python

   df_of = hbc.extract_results(
       "/path/to/openfoam_case",       # or .../VTK/case_1234/internal.vtu
       variable=["velocity", "k"],     # fields as written by foamToVTK
       x=[0.5, 0.7], y=[0.2, 0.2], z=[0.05, 0.05],
       time="mean_last", n_last=3,     # averages the last 3 VTK time steps
   )

For the wall-law diagnostics of the calibration workflow (friction velocity,
y-plus, and the complete modeled vertical velocity profile at each calibration
point), see ``TelemacModel.extract_data_point`` with
``compute_wall_law_diagnostics=True`` in the :doc:`codedocs`.

See also
--------

* :doc:`installation` — environment and TELEMAC setup.
* :doc:`workflow` — the calibration workflow and all configuration parameters.
* :doc:`usage-openfoam` — the analogous OpenFOAM guide.
