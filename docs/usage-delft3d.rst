.. _usage-delft3d:

Using HydroBayesCal with Delft3D-FLOW (Experimental)
====================================================

This page explains how to run a surrogate-assisted Bayesian calibration of a
**Delft3D-FLOW** (Delft3D 4 suite, Deltares) model, which is functionally integrated in HydroBayesCal yet **not tested**. It mirrors
:doc:`usage-telemac` and :doc:`usage-openfoam` and complements the generic
:doc:`workflow`.


.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

.. warning::

   This workflow is experimental and has not yet been fully tested. It may not work as expected and should be used with caution. Please review the results carefully before relying on them in production or decision-critical contexts. Feedback, bug reports, and suggestions for improvement are welcome.


* A working **Delft3D-FLOW** installation compiled from source, following the
  guide at `hydro-informatics.com/get-started/delft3d
  <https://hydro-informatics.com/get-started/delft3d.html>`_. The default
  installation prefix is ``~/opt/delft3d-flow``, which provides:

  * ``env.sh``: the environment script that exports ``DELFT3D_HOME``, the
    Intel oneAPI runtime, ``PATH`` and ``LD_LIBRARY_PATH``. HydroBayesCal
    sources this script automatically before every run (parameter
    ``env_script``).
  * ``run_dflow2d3d.sh`` and ``run_dflow2d3d_parallel.sh``: the launchers that
    HydroBayesCal calls for serial and parallel runs, respectively.

* A **fully functional Delft3D-FLOW case** that runs to completion on its own
  *before* you start any calibration, that is::

     source ~/opt/delft3d-flow/env.sh
     cd /path/to/case && run_dflow2d3d.sh

  finishes with exit code 0, writes ``trim-<case>.*`` output and reports no
  ``*** ERROR`` lines in ``tri-diag.<case>``.

* HydroBayesCal installed with the ``delft3d`` extra (netCDF4), which is used
  to read the NetCDF map output:

  .. code-block:: bash

     pip install "hydroBayesCal[delft3d]"

.. note::

   HydroBayesCal reads the **NetCDF** map file ``trim-<case>.nc`` and enforces
   ``FlNcdf = #maphis#`` in the ``.mdf`` before every run, so the binary NEFIS
   files (``trim-<case>.dat``/``.def``) never need to be parsed in Python.

The Delft3D-FLOW case template
------------------------------

HydroBayesCal copies a **case template** (``case_template_dir``) for each run
and modifies the calibration keywords in the copy. A standard Delft3D-FLOW
setup contains:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Purpose
   * - ``<case>.mdf``
     - Master definition FLOW file with the numerical configuration; the
       calibration parameters are written into this file.
   * - ``config_d_hydro.xml``
     - Runtime configuration read by ``run_dflow2d3d.sh``; its ``<mdfFile>``
       entry names the ``.mdf`` file (passed as ``d_hydro_config``).
   * - ``*.grd`` / ``*.enc``
     - Curvilinear grid and grid enclosure.
   * - ``*.dep``
     - Bathymetry (depth) file.
   * - ``*.bnd`` / ``*.bct`` / ``*.bcc``
     - Boundary definitions and time series.
   * - further attribute files
     - Initial conditions, dry points, thin dams, and so on, as referenced by
       the ``.mdf``.

.. admonition:: Before calibrating

   * Verify the case runs properly on its own (see prerequisites above).
   * Keep the simulated period as short as the physics allow: the surrogate
     training runs the model tens of times.

Measurement / calibration points
---------------------------------

The calibration targets are provided in a CSV file
(``calibration_pts_file_path``) with one row per measurement location. The
header uses the ``X``, ``Y`` coordinates and, for each calibration quantity, a
``<quantity>_DATA`` and a ``<quantity>_ERROR`` column (matched
case-insensitively)::

    X, Y, WATER_DEPTH_DATA, WATER_DEPTH_ERROR, U_MAG_DATA, U_MAG_ERROR
    ...

The ``_ERROR`` column holds the measurement error in the physical units of the
quantity. Measurements and the computational grid (``.grd``) must use the same
coordinate reference system. Model values are sampled at the nearest active
grid cell (zeta point) of each measurement location.

Delft3D-specific parameters
---------------------------

In addition to the common :ref:`HydroSimulations_class` parameters, the
``Delft3DModel`` class accepts:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Parameter
     - Description
   * - ``case_template_dir``
     - Path to the Delft3D-FLOW case template that is copied for each run.
   * - ``env_script``
     - ``env.sh`` of the Delft3D-FLOW installation, default
       ``"~/opt/delft3d-flow/env.sh"``; sourced before every run.
   * - ``d_hydro_config``
     - Runtime configuration file, default ``"config_d_hydro.xml"``.
   * - ``n_processors``
     - ``1`` runs ``run_dflow2d3d.sh``; larger values run
       ``run_dflow2d3d_parallel.sh <n>``.
   * - ``roughness_formulation``
     - Bed-roughness law written to the ``Roumet`` keyword when the
       ``"roughness"`` parameter is calibrated: ``"Chezy"``, ``"Manning"`` or
       ``"WhiteColebrook"``.
   * - ``n_avg_timesteps``
     - Number of final map time steps to average when extracting outputs.

Calibration parameters and MDF keywords
---------------------------------------

Delft3D calibration parameters map to keywords in the ``.mdf`` master
definition file:

* ``"roughness"`` (or ``"Ccof"``) is treated specially: it writes the
  formulation to ``Roumet`` (``#C#``, ``#M#`` or ``#Z#``) and the value to the
  uniform roughness coefficients ``Ccofu`` and ``Ccofv``.

  .. code-block:: python

     calibration_parameters = ["roughness"]
     param_values = [[0.02, 0.04]]   # e.g. Manning n

* Any **other name** is written as a literal MDF keyword, for example the
  horizontal eddy viscosity ``Vicouv`` or diffusivity ``Dicouv``:

  .. code-block:: python

     calibration_parameters = ["roughness", "Vicouv"]
     param_values = [[0.02, 0.04], [0.1, 10.0]]

``calibration_quantities`` / ``extraction_quantities`` use the standard
HydroBayesCal field names: ``"WATER_LEVEL"``, ``"WATER_DEPTH"``, ``"U_x"``,
``"U_y"`` and ``"U_MAG"``. Velocities are rotated from the curvilinear grid
directions to east/north components using the local grid orientation
(``ALFAS``); for 3D runs the layers are averaged.

.. note::

   Bayesian active learning only handles numeric (scalar) parameters. A
   parameter that takes a non-numeric value (e.g. a file-name keyword) cannot
   be used as a calibration parameter.

Running the calibration
-----------------------

Define a configuration file (see the example ``templates/config_Delft3D.py``)
and launch the Delft3D driver:

.. code-block:: bash

   python templates/bal_delft3d.py --config templates/config_Delft3D.py

The driver builds a ``Delft3DModel``, sets up the experimental design, runs
the initial simulations (source ``env.sh``, then ``run_dflow2d3d.sh`` in a
fresh copy ``run_0000``, ``run_0001``, ... of the case template), extracts the
requested quantities from ``trim-<case>.nc`` at the calibration points, trains
the GPE and performs Bayesian Active Learning. Each run is checked for
``*** ERROR`` lines in ``tri-diag.<case>`` and for the presence of the map
output before extraction. Equivalently, in a script:

.. code-block:: python

   from hydroBayesCal.delft3d.control_delft3d import Delft3DModel

   model = Delft3DModel(
       case_template_dir="/path/to/delft3d_case_template",
       env_script="~/opt/delft3d-flow/env.sh",
       d_hydro_config="config_d_hydro.xml",
       n_processors=1,
       roughness_formulation="Manning",
       model_dir="/path/to/simulations",
       res_dir="/path/to/results",
       calibration_pts_file_path="/path/to/measurements-calibration.csv",
       n_cpus=1,
       init_runs=15,
       max_runs=30,
       calibration_parameters=["roughness", "Vicouv"],
       param_values=[[0.02, 0.04], [0.1, 10.0]],
       calibration_quantities=["WATER_DEPTH"],
       extraction_quantities=["WATER_LEVEL", "WATER_DEPTH",
                              "U_x", "U_y", "U_MAG"],
   )

Results are written to the same ``auto-saved-results-HydroBayesCal`` layout as
for TELEMAC (see :doc:`gpe-bal-telemac`), so post-processing is identical
across solvers.

See also
--------

* :doc:`installation` - environment and Delft3D-FLOW setup.
* :doc:`workflow` - the calibration workflow and all configuration parameters.
* :doc:`usage-telemac`, :doc:`usage-openfoam` - the analogous guides.
