.. UML

Code Structure
==============

.. components

Package components
-------------------

The package consists of two well-defined parts:

1. **Hydrodynamic Simulations**:
   This part performs hydrodynamic simulations using any open-source hydrodynamic software (currently, only Telemac is supported).

2. **Surrogate Model and Bayesian Active Learning (BAL)**:
   This part builds the initial surrogate model using Gaussian Process Regression and performs Bayesian Active Learning. The goal is to iteratively add new training points using the methodology in `Oladyshkin, S., et al (2020) <https://doi.org/10.3390/e22080890>`_, hence increasing the model's accuracy in the parameter space regions that are most crucial for Bayesian inference.

All user input parameters are assigned in the ``bal_telemac.py`` file.

You will find a detailed explanation of each module's functionality in the following documentation.



UML diagram
-----------

.. figure:: _static/UML.png
   :alt: Complete UML of HydroBayesCal
   :width: 100%
   :align: center
   :scale: 15%


