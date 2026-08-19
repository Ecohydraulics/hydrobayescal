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

Delft3D-FLOW binding
++++++++++++++++++++

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

Initial design
++++++++++++++

Sizing the initial design from the number of calibration parameters, extensible Sobol
blocks and the sufficiency gate that decides when the design is good enough to start
Bayesian active learning on; see :ref:`initial-design` for the workflow.

.. automodule:: hydroBayesCal.surrogate.initial_design
   :members:
   :show-inheritance:

Calibrated parameters from the posterior
++++++++++++++++++++++++++++++++++++++++

Per-parameter marginal optima, the maximum of the joint posterior, identifiability
flags, the equifinality diagnostic and the rule that decides which of the two is the
calibrated parameter set; see :ref:`calibrated-parameters` for the workflow.

.. automodule:: hydroBayesCal.surrogate.posterior_analysis
   :members:
   :show-inheritance:

Agreement with the calibration targets
++++++++++++++++++++++++++++++++++++++

Modeled against measured calibration targets before and after calibration, the verdict
that separates a systematic over- or underestimation from scatter, and the closing
post-processing step of every driver; see :ref:`target-agreement` for the workflow.

.. automodule:: hydroBayesCal.surrogate.target_agreement
   :members:
   :show-inheritance:

Result extraction
-----------------

Standalone 2D/3D point extraction from TELEMAC SELAFIN and OpenFOAM VTK
result files; see :doc:`usage-telemac` for a usage guide.

.. automodule:: hydroBayesCal.extract
   :members: extract_results

Visualization
-------------

.. automodule:: hydroBayesCal.visualize.plotter
   :members:
   :show-inheritance:

.. automodule:: hydroBayesCal.visualize.agreement_plots
   :members:
   :show-inheritance:

Shared utilities
-----------------

.. automodule:: hydroBayesCal.function_pool
   :members:
