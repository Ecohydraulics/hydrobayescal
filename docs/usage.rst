
Usage
=====

Prepare Input
-------------

Direct Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, all Telemac2d and Gaia parameters that are available in the `user-input.xlsx`_ workbook are understood by the code (and can therefore also be theoretically used - we did not test every parameter combination, so some combinations might just crash). Select up to four direct calibration parameters and define physically meaningful parameter ranges. The ``lower_limit`` and ``upper_limit`` define the parameter ranges (e.g. use ``0.01, 0.1`` to define the selected parameter's ``lower_limit=0.01`` and ``upper_limit=0.1``, respectively). Between the given parameter ranges, the code will draw ``mc_samples`` (defined by the Active Learning parameters in `user-input.xlsx`_) to build the surrogate.

.. note:: Why four parameters only?

   In theory, more parameters can easily be added. However, if four parameters are selected, running the surrogate-based optimization will take days to weeks. Adding a fifth parameter or more will quickly lead to calculation times of weeks to months, which seems not affordable, even in open-outcome research projects.

For instance, to calibrate a suspended sediment load model (Gaia parameters) as a function of observed topographic change, consider defining the following direct calibration parameters and associated value ranges in `user-input.xlsx`_:

.. csv-table:: Exemplary Direct Calibration Parameters for a Suspended Load Model with Value Ranges
   :header: "Direct Calibration Parameter", "Value Range"
   :widths: 50, 30

   "CLASSES CRITICAL SHEAR STRESS FOR MUD DEPOSITION", "0.01, 0.1"
   "LAYERS CRITICAL EROSION SHEAR STRESS OF THE MUD", "0.05, 0.4"
   "LAYERS MUD CONCENTRATION", "200, 500"


Indirect Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indirect calibration parameters will be varied within a multiplier range. For instance if the multiplier range is defined with ``0.8, 1.7`` the indirect parameter will be varied in a range between 0.8 to 1.7 times the initial values.
Thus, the selected Indirect Calibration Parameter requires the definition of initial values (rather than a variation interval, as opposed to Direct Calibration Parameters).


.. csv-table:: Exemplary Indirect Calibration Parameters for a Suspended Load Model with Value Ranges
   :header: "Linked Calibration Parameter", "Value Range"
   :widths: 50, 30

   "CLASSES SEDIMENT DIAMETERS", "0.001, 0.000024, 0.0000085, 0.0000023"

   "CLASSES SEDIMENT DIAMETERS", "0.001, 0.000024, 0.0000085, 0.0000023"
   "CLASSES SETTLING VELOCITIES", "0.8, 1.7"

Recalculation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

Updating of some parameters in the control (steering CAS) file will require updating also other parameters. Such recalculation parameters are automatically detected in `user-input.xlsx`_ as a function of available routines for **one Indirect Calibration Parameter only**. To avoid that the code uses automatically detected recalculation parameters, deactivate the detected parameter by switching the **Apply?** field to ``False`` (or ``0`` in LibreOffice).

For instance, if ``CLASSES SEDIMENT DIAMETERS`` is an indirect calibration parameter, those will affect the  ``CLASSES SETTLING VELOCITIES`` in suspended load calculations. However, for running a bedload calculation, the ``CLASSES SEDIMENT VELOCITIES`` keyword does not make sense and should not be applied.


Regular Usage
-------------

Coming soon

.. figure:: https://github.com/sschwindt/stochastic-surrogate/raw/main/docs/img/browser-icon-large.jpg
   :alt: calibrate surrogate bayesian gaussian bal gpe

   *Intro figure.*

Implement the following code in a Python script and run that Python script:

.. code-block::

    import stochastic_surrogate as sur
    model_dir = r"C:\telemac\\v8p3\\models\\training-example"
    sur.optimize(model_dir)


.. important::

    The model directory may not end on any ``\`` or  ``/`` .

- After a successful run, the code will have produced the following files in ``...\your-data\``:
    + ``files`` das

Usage Example
-------------

For example, consider your model lives in a folder called ``C:\telemac\models\reservoir2d``.

Multiparametric Optimization
----------------------------
Run multiple hydro-morphodynamic simulations of Telemac-2d, using a list of parameter combinations. The code is specific to the parameters that wanted to be changed at the time, but it can be used as the base to run other specific numerical configurations.

Run run_multiple_telemac.py using the main folder as a current directory from a console/terminal in which Telemac and GAIA have already been compiled.

Do not use with an IDE, better use command line Python!

Provide main simulation folder (directory) with:
-run_multiple_telemac.py: runs a hydro-morphodynamic simulation using the Telemac 2D software coupled with the GAIA module for all the parameter combinations located in the file parameter_comb.txt. The parameters modified in each run are named in the variable parameters_name in the USER INPUT section of the code. These parameters should be one of the KeyWords listed in Telemac or GAIA.
-parameter_comb.txt: Have the numerical value of the parameter combinations for which the telemac software is going to be run. For this example, 4 parameters (4 columns) are going to be modified 3 times (3 lines). Therefore when the code is run, there are going to be 3 different simulations.
-calibration_points.txt: This file contains the number of the nodes that will be used in case the values of a specific variable want to be extracted from particular nodes of the mesh.
-init.py: Reference other folders.
- Files necessary to run the hydro-morhodynamic model using Telemac2D and GAIA:
    - bc_liquid.liq: Liquid boundary condition (flow, sediment or tracers inflow/outflow)
    - bc_steady_tel.cli: File that defines the type and location of the boundary conditions.
    - geo_banda.slf: File that defines the mesh structure for the hydro-morphodynamic model.
    -run_liquid_gaia.cas: Numerical configuration of the sediment transport model.
    - run_liquid_tel.cas: Numerical configuretion of the hydrodynamic model.

**simulations Folder:**
After each simulation is completed, the simulation files will be stored in this folder.

**results Folder:**
After each simulation is completed, a .txt file with the values of a specified variable (water elevation, bottom elevation, ...) in the nodes listed in calibration_points.txt will be generated and stored in this folder.

**external_libraries Folder:**
The library pputils-master by Pat Prodanovic (https://github.com/pprodano/pputils) is used to extract the results of the simulation file (.slf) into a .txt file, which is then stored in the results Folder.

**scripts Folder:**
- auxiliary_functions_telemac: Contains auxiliary functions used to modify the input and output of the telemac files. These functions are specific to the parameters that wanted to be changed at the time, but they can be used as a base on how to modify Telemac's input and output files
-init.py: Reference other folders.

.. _user-input.xlsx: https://github.com/sschwindt/stochastic-surrogate/raw/main/user-input.xlsx


