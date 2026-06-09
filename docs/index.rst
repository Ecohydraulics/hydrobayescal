HydroBayesCal: Surrogate-Assisted Bayesian Calibration
======================================================

**HydroBayesCal** is a Python package for the surrogate-assisted Bayesian
calibration of computationally expensive hydro- and morphodynamic models.
Calibrating a full-complexity numerical model directly is often infeasible,
because a single simulation can take hours to days and stochastic calibration
needs many runs. HydroBayesCal sidesteps this cost by training a **Gaussian
Process Emulator (GPE)** as a fast surrogate of the numerical model from a small
set of strategically sampled simulations, and then refining it with **Bayesian
Active Learning (BAL)** — iteratively adding the training points that maximise
the information gain (relative entropy) and Bayesian model evidence for the
calibration. The package supports both single-output and **multi-output GPEs**,
and couples to open-source modelling software through a common binding layer:
**TELEMAC** (2D/3D) is fully supported, an **OpenFOAM** binding is under
active development, and a **Delft3D-FLOW** binding is planned. Adding bindings
for further solvers — or swapping the
experimental-design backend (`BayesValidRox
<https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/>`_) — is
designed to be straightforward.

Scientific background and references
------------------------------------

The methods implemented here build on the Bayesian active-learning framework of
Oladyshkin et al. (2020) and on Gaussian-process regression as described by
Rasmussen & Williams (2006). The calibration strategy and its application to
reservoir sedimentation and three-dimensional reservoir hydrodynamics are
documented in Mouris et al. (2023) and Schwindt et al. (2023).

The full bibliography, including DOIs, is collected on the :doc:`references`
page. (The development repository keeps local copies of these works in a
git-ignored ``ExportedItems/`` folder for convenience; they are not
redistributed.)

.. note::

   HydroBayesCal is research software developed at the University of Stuttgart
   and collaborators. It is provided under a BSD 3-Clause license (see
   :doc:`license`).

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   introduction
   installation

.. toctree::
   :maxdepth: 2
   :caption: User guide

   workflow
   gpe-bal-telemac
   usage-telemac
   usage-openfoam
   usage-delft3d
   use-cases

.. toctree::
   :maxdepth: 2
   :caption: Reference

   uml
   codedocs
   references
   license

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
