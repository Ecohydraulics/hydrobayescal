
Usage
=====

TELEMAC Model
-------------

Follow the `TELEMAC tutorial <https://hydro-informatics.com/numerics/telemac.html>`_ to learn more.

Measurement Data
----------------

Calibration Points
^^^^^^^^^^^^^^^^^^

The surrogate-assisted calibration procedure requires calibration points associated with mesh node IDs to layers of a Selafin (``.slf``) file. The calibration points should involve 120-180 measurements and provide measurements regarding either:

* Topographic change (``.slf`` layer: BOTTOM)
* Water depth (``.slf`` layer: DEPTH)
* Flow velocity (``.slf`` layer: VELOCITY)

.. important::

    The measurements and computational mesh (``.slf``) must use the same spatial coordinate reference system (CRS) to ensure that the calibration routines will be able to compare modeled and measured quantities.


The points should be stored in a text or csv file-like format where the file ending is less important than the requirement of using a **comma** as **column separator**, and **no header**:

* The first column of the calibration points file should indicate the mesh node IDs
* The second column should indicate the absolute measurement (in *m* or *m/s*)
* The third column should indicate the measurement error.

For instance, create a csv file that contains the following exemplary information in the first three rows (the ``...`` indicates 120-180 more rows needed):

.. csv-table:: Exemplary Representation of a **calibration-points.csv** file.
   :header: "Mesh node ID", "Absolute Value", "Error
   :widths: 40, 40, 40

   "3895", "155.4", "0.60"
   "4884", "165.2", "0.25"
   "4887", "161.8", "1.03"
   "...", "...", "..."

Download our `calibration-points.csv file <https://github.com/sschwindt/stochastic-surrogate/raw/main/calibration-points.csv>`_ to see a working example.

Prepare Input Workbook
----------------------

Scalar Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important:: Only use numeric scalar values

   Bayesian active learning can only deal with numbers. Thus, any calibration parameter that takes non-numeric (scalar) values, such as a solver type, can currently not be understood by the code.

All Telemac2d and Gaia parameters that are available in the `user-input.xlsx`_ workbook are understood by the code as scalars (and can therefore also be theoretically used - we did not test every parameter combination, so some combinations might just crash). Select up to four scalar calibration parameters and define physically meaningful parameter ranges. The ``lower_limit`` and ``upper_limit`` define the parameter ranges (e.g. use ``0.01, 0.1`` to define the selected parameter's ``lower_limit=0.01`` and ``upper_limit=0.1``, respectively). Between the given parameter ranges, the code will draw ``mc_samples`` (defined by the Active Learning parameters in `user-input.xlsx`_) to build the surrogate.

.. note:: Why four parameters only?

   In theory, more parameters can easily be added. However, if four parameters are selected, running the surrogate-based optimization will take days to weeks. Adding a fifth parameter or more will quickly lead to calculation times of weeks to months, which seems not affordable, even in open-outcome research projects.

For instance, to calibrate a suspended sediment load model (Gaia parameters) as a function of observed topographic change, consider defining the following direct calibration parameters and associated value ranges in `user-input.xlsx`_:

.. csv-table:: Exemplary Direct Calibration Parameters for a Suspended Load Model with Value Ranges
   :header: "Direct Calibration Parameter", "Value Range"
   :widths: 50, 30

   "CLASSES CRITICAL SHEAR STRESS FOR MUD DEPOSITION", "0.01, 0.1"
   "LAYERS CRITICAL EROSION SHEAR STRESS OF THE MUD", "0.05, 0.4"
   "LAYERS MUD CONCENTRATION", "200, 500"

.. important::

    Make sure your Selafin file (``.slf``)  has the attribute ``BOTTOM`` for morphodynamic calibrations with respect to ``TOPOGRAPHIC CHANGE``.
    For ``DEPTH`` and ``VELOCITY`` based-calibration, the Selafin also needs to get these layer names (i.e, a ``DEPTH`` and/or ``VELOCITY`` layer, respectively) with values assigned from the calculation results and measurements!

List-like Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important:: Only use numeric values in lists

   Bayesian active learning can only deal with numbers. Thus, any calibration parameter that takes non-numeric values as a list, such as ``CLASSES TYPE OF SEDIMENT``, can currently not be understood by the code.


List-like (vector) calibration parameters are non-scalar model controls, such as multiple diameters for sediment classes. For instance, a morphodynamic model with three sediment classes may have three ``CLASSES SEDIMENT DIAMETERS: 0.001, 0.02, 0.05``. To enable the variation of such vector-like calibration parameters, we use a multiplier that can be defined in the **Multiplier range** field and is then applied to a list of parameter values. For example, if the multiplier range is defined with ``0.8, 1.7`` the multi-class parameter will be varied in a range between 0.8 to 1.7 times the initial values.
Thus, a list-like calibration parameter requires the definition of initial values (rather than a variation interval that can be defined for Scalar Calibration Parameters). It is recommended the list of values corresponds to means.


.. csv-table:: Exemplary Indirect Calibration Parameters for a Suspended Load Model with Value Ranges
   :header: "Linked Calibration Parameter", "Value Range"
   :widths: 50, 30

   "CLASSES SEDIMENT DIAMETERS", "0.001, 0.000024, 0.0000085, 0.0000023"


Recalculation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

Updating of some parameters in the control (steering CAS) file will require updating also other parameters. Such recalculation parameters are automatically detected in `user-input.xlsx`_ as a function of available routines for **max. two Multi-Class Calibration Parameter only** (computing time!). To avoid that the code uses automatically detected recalculation parameters, deactivate the detected parameter by switching the **Apply?** field to ``False`` (or ``0`` in LibreOffice/ONLYOFFICE).
In addition, if an automatically detected recalculation parameter is in the list of scalar or multi-class calibration parameters, it will be automatically updated and you may completely ignore the recalculation parameter section. In this case, the direct calibration parameter will automatically be recalculated as a function of the indirect calibration parameter.

.. important::

   Recalculation parameters are detected in the order of Multi-Class parameters. Thus, if a list-like calibration parameter selected in cell B33 is detected to be associated with a recalculation parameter, the corresponding recalculation parameter will show up in cell B37. Analogously, if a multi-class parameter is selected in cell B34, any potential recalculation parameter will show up in cell B38.

For instance, if ``CLASSES SEDIMENT DIAMETERS`` is a multi-class calibration parameter, which affects the ``CLASSES SETTLING VELOCITIES`` in suspended load calculations. However, for running a bedload calculation, the ``CLASSES SEDIMENT VELOCITIES`` keyword does not make sense and should not be applied. If ``CLASSES SETTLING VELOCITIES`` was already selected in the list of direct calibration parameters, the settling velocities will be automatically recalculated.


Regular Usage
-------------

Launch Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

The extraction of geospatial information uses QGIS' mesh operations, which is why the Python environment needs to understand QGIS (read more in the :ref:`requirements` section). The standard sudo-installation of QGIS on Linux will make that the system's Python interpreter knows the ``qgis`` library. On Windows and in Anaconda environments, the qgis capacity can be installed



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

Output
------

The optimized simulation is stored in a sub-folder called **opt-reults** in the provided simulation directory. The produced files involve:

* Updated steering ( ``.cas``) files for TELEMAC and, if used, Gaia:
    * TELEMAC: res-tel-PC
    * Gaia: res-gaia-PC


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
    - run_liquid_gaia.cas: Numerical configuration of the sediment transport model.
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


