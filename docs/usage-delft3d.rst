.. _usage-delft3d:

Using HydroBayesCal with Delft3D-FLOW
=====================================

This page is a **placeholder** for the planned **Delft3D-FLOW** binding. It
mirrors :doc:`usage-telemac` and :doc:`usage-openfoam` and will describe how to
run a surrogate-assisted Bayesian calibration of a Delft3D-FLOW model once the
binding is implemented.

.. note::

   The Delft3D-FLOW binding is **planned and not yet implemented**. The
   :class:`hydroBayesCal.delft3d.control_delft3d.Delft3DModel` class is a stub
   that raises :class:`NotImplementedError`; it defines the intended interface
   so the coupling can be filled in incrementally. Track progress in
   :mod:`hydroBayesCal.delft3d.control_delft3d`.

.. contents::
   :local:
   :depth: 2

Intended scope
--------------

`Delft3D-FLOW <https://www.deltares.nl/en/software-and-data/products/delft3d-flexible-mesh-suite>`_
is the hydrodynamic/morphodynamic solver of the Delft3D suite (Deltares). The
binding targets the structured-grid Delft3D-FLOW engine driven through the
``d_hydro`` / ``config_d_hydro.xml`` mechanism, so that HydroBayesCal can:

* copy a Delft3D-FLOW case template for each experimental-design run,
* edit the calibration parameters in the **master definition file**
  (``<case>.mdf``) and the associated attribute files,
* launch the solver and read its output, and
* extract the calibration quantities at the measurement points.

Software-specific keywords (planned)
------------------------------------

As for TELEMAC and OpenFOAM, the Python attribute names are shared across
solvers, but the *string and file conventions* are Delft3D-specific and must be
preserved. The binding is expected to use:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Delft3D-FLOW convention
   * - Control / master file
     - ``<case>.mdf`` (master definition FLOW file); runtime is launched through
       ``config_d_hydro.xml`` and the ``d_hydro`` executable.
   * - Roughness parameters
     - Bed roughness via Chézy / Manning / White-Colebrook (``.rgh`` roughness
       file or ``Roughness`` keywords in the ``.mdf``).
   * - Other calibration inputs
     - Horizontal/vertical eddy viscosity and diffusivity (``Vicouv``,
       ``Dicouv``), wind drag, and boundary-condition values.
   * - Map (field) output
     - ``trim-<case>.dat`` / ``trim-<case>.def`` (NEFIS map files).
   * - History (point) output
     - ``trih-<case>.dat`` / ``trih-<case>.def`` (NEFIS history files at
       monitoring points).
   * - Output quantities
     - e.g. water level / depth, depth-averaged and 3D velocity components,
       mapped onto the common HydroBayesCal quantity names.

These keywords are recorded here so the eventual implementation keeps them
distinct from the TELEMAC (``.cas`` / SELAFIN) and OpenFOAM
(``system/controlDict`` / VTK) conventions.

See also
--------

* :doc:`installation` — environment and numerical-model bindings.
* :doc:`workflow` — the calibration workflow and configuration parameters.
* :doc:`usage-telemac`, :doc:`usage-openfoam` — the implemented bindings.
