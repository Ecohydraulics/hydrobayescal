.. Introduction

About HydroBayesCal
===================

What is HydroBayesCal?
----------------------

**HydroBayesCal** is a **Python 3** package for the optimisation and calibration
of hydrodynamic and morphodynamic models using a **surrogate-assisted Bayesian
Active Learning** approach. It trains a **Gaussian Process Emulator (GPE)** as a
fast surrogate (metamodel) of a *full-complexity* numerical model, and evaluates
that surrogate using **Bayesian model evidence (BME)** and **relative entropy
(RE)**, following `Oladyshkin et al. (2020)
<https://doi.org/10.3390/e22080890>`_.

Because every parameter update must be propagated back into the numerical model,
HydroBayesCal couples to fully open-source modelling software through a common
binding layer:

* **TELEMAC** (2D and 3D) — fully supported.
* **OpenFOAM** (interFoam) — binding under active development.

The architecture is deliberately solver-agnostic: a binding only needs to
implement how the model is run and how outputs are extracted (see
:doc:`uml`), so support for additional solvers can be added with limited effort.

.. admonition:: Good to know

    To work with HydroBayesCal and TELEMAC, familiarise yourself with the
    TELEMAC software. Useful starting points:

    - `Installation instructions for TELEMAC
      <https://hydro-informatics.com/install-telemac/>`_
    - `TELEMAC tutorial <https://hydro-informatics.com/numerics/telemac.html>`_

Purpose and motivation
----------------------

Stochastic calibration techniques require a large number of full-complexity
model realisations to perform statistical analysis. This is unfeasible when a
single realisation takes hours or days. HydroBayesCal makes it tractable by
first constructing a surrogate model from only a few realisations at carefully
chosen *initial collocation points*, sampled with advanced design-of-experiments
methods.

The package builds on `BayesValidRox
<https://pages.iws.uni-stuttgart.de/inversemodeling/bayesvalidrox/>`_ for the
design of experiments, and uses **Gaussian Process Regression (GPR)** to build
**single-output and multi-output** surrogate models. These predict the model
outputs for any parameter combination, and **Bayesian inference** then quantifies
the uncertainty of the calibration parameters.

Bayesian Active Learning (BAL) is used to *iteratively* add new training points
(parameter combinations) where the expected information gain — measured by
relative entropy and Bayesian model evidence — is highest, increasing the
surrogate's accuracy precisely in the regions of parameter space that matter
most for the calibration.

Where to go next
----------------

* :doc:`installation` — set up a suitable computing environment and the solver
  bindings.
* :doc:`workflow` — the end-to-end calibration workflow and all configuration
  parameters.
* :doc:`gpe-bal-telemac` — running a TELEMAC (and OpenFOAM) calibration.
* :doc:`uml` — code structure and architecture.
* :doc:`references` — scientific background and citations.
