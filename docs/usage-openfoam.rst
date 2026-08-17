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
       (the ``kEpsilonCoeffs`` subdictionary holds the calibratable turbulence
       coefficients).
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
dictionary entries that ``OpenFOAMController`` writes into the case:

* **k-epsilon turbulence coefficients** in the ``kEpsilonCoeffs``
  subdictionary of ``constant/turbulenceProperties``. The supported names are
  ``Cmu``, ``C1``, ``C2``, ``C3``, ``sigmak`` and ``sigmaEps``, listed in
  ``hydroBayesCal.openfoam.control_openfoam.KEPSILON_COEFFS``.
* **Wall roughness** ``ks`` applied as a boundary condition on the
  ``nutkRoughWallFunction`` patch in ``0/nut``. The patch is auto-detected from
  the case template.
* Other **boundary-condition values** or **dictionary entries**, updated via
  ``update_boundary_condition`` / ``update_dictionary_entry`` by code driving
  ``OpenFOAMController`` directly.

Parameter names are matched case-insensitively, so ``"sigmaeps"`` and
``"sigmaEps"`` both work; the key written into the case file always uses the
spelling OpenFOAM expects. Any other name raises a ``ValueError`` when the model
is constructed, before a simulation starts.

.. important::

   A coefficient must already be present in the case template's
   ``kEpsilonCoeffs`` subdictionary, otherwise the run stops with
   ``ValueError: Key '<name>' not found``. This is deliberate: OpenFOAM falls
   back to a built-in default for any coefficient it does not find in the
   dictionary, so a write that quietly did nothing would leave every run of the
   calibration using the same value while the surrogate was told the parameter
   had changed.

``calibration_quantities`` uses the standard field names. The OpenFOAM binding
extracts ``"U_x"``, ``"U_y"``, ``"U_z"``, ``"U_MAG"`` (velocity components and
magnitude), ``"TKE"`` (turbulent kinetic energy ``k``) and the isotropic
fluctuation components ``"u_fluct"``, ``"v_fluct"``, ``"w_fluct"``. These are
listed in ``OpenFOAMModel.EXTRACTABLE_QUANTITIES``, and any other name raises a
``ValueError`` when the model is constructed, before a simulation starts.
Free-surface quantities such as ``"WATER_DEPTH"`` and ``"FREE_SURFACE"`` are
available in the Delft3D and TELEMAC bindings but are not yet extracted here.
As for TELEMAC, the calibration CSV provides a
``<target>_DATA`` and ``<target>_ERROR`` column per calibration target, together with
the ``X``, ``Y`` (and ``Z``) coordinates of the measurement points.

Running the calibration
-----------------------

Define a configuration file and launch the OpenFOAM driver:

.. code-block:: bash

   python src/hydroBayesCal/drivers/bal_openfoam.py --config src/hydroBayesCal/drivers/config_OpenFOAM.py

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

To sample fields from OpenFOAM VTK output at arbitrary (x, y, z) points
outside the calibration workflow, use :func:`hydroBayesCal.extract_results`
with the case directory or a ``.vtu`` file - see the section
*Extract & compare 2d/3d simulation data* in :doc:`usage-telemac`.

See also
--------

* :doc:`installation` — environment and OpenFOAM setup.
* :doc:`workflow` — the calibration workflow and all configuration parameters.
* :doc:`usage-telemac` — the analogous TELEMAC guide.
