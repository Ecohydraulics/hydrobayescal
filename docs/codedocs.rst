.. API reference

API Reference
=============

This section documents the public classes and functions, generated from the
in-code docstrings. See :doc:`uml` for how these modules fit together.

Binding layer
-------------

Abstract base class
+++++++++++++++++++

.. automodule:: hydroBayesCal.hysim
   :members:
   :show-inheritance:

TELEMAC binding
+++++++++++++++

.. automodule:: hydroBayesCal.telemac.control_telemac
   :members:
   :show-inheritance:

OpenFOAM binding
++++++++++++++++

.. automodule:: hydroBayesCal.openfoam.control_openfoam
   :members:
   :show-inheritance:

Delft3D-FLOW binding (planned)
++++++++++++++++++++++++++++++

.. automodule:: hydroBayesCal.delft3d.control_delft3d
   :members:
   :show-inheritance:

Surrogate model and Bayesian Active Learning
--------------------------------------------

Gaussian Process Emulators
++++++++++++++++++++++++++

.. automodule:: hydroBayesCal.surrogate.gpe_gpytorch
   :members:
   :show-inheritance:

.. automodule:: hydroBayesCal.surrogate.gpe_skl
   :members:
   :show-inheritance:

Bayesian inference and sequential design
++++++++++++++++++++++++++++++++++++++++

.. automodule:: hydroBayesCal.surrogate.bal_functions
   :members:
   :show-inheritance:

.. automodule:: hydroBayesCal.surrogate.exploration
   :members:
   :show-inheritance:

Shared utilities
-----------------

.. automodule:: hydroBayesCal.function_pool
   :members:
