
TELEMAC Calibration Examples
============================


The lower Yuba River
--------------------

This case study treats the lower Yuba River, CA, USA, between its intersection with Highway 20 and Daguerre Point Dam (DPD).


The mesh characteristics are:

* Multi-block mesh with size XX
* EPSG: 31610 (WGS 84 / UTM zone 10N)


Preparation
-----------

* Test-run the TELEMAC model to substantiate hotstart conditions for steady discharges corresponding to the lower Yuba river flows during the measurements:
  + Flow velocity from the Hammond reach (HR, Dry Creek to DPD) from *measurement-data/LYR-VELRUN-analysis-assorted.xlsx* in tabs ``HR 817`` (23.13 CMS) and ``HR 1093`` (30.95 CMS) from 2009-12-15 and 2010-01-27, respectively.
  + Combined water depth and flow velocity data is available for the HR reach is fewer quantity than velocity only in the file *measurement-data/LYR-XS-vs-Model.xlsx* in the ``HR_n04_800cfs`` tab. The 800 CFS are precisely 795 CFS (22.51 CMS) from 2009-12-08.
  + Analysis of the roughness zones covered by the two measurement files showed that [LYR-VELRUN-analysis-assorted.xlsx OR LYR-XS-vs-Model.xlsx] covers all zones. BUT/AND only LYR-XS-vs-Model.xlsx provides combined data on U and H, which is why a 22.51 CMS steady-state model with data from LYR-XS-vs-Model.xlsx was used for calibration.

* Create ``calibration-pts.csv`` containing SCALAR VELOCITY and WATER DEPTH measurements (see above bullet item) at points identical to mesh nodes with the following shape (file does not contain the header):

.. code-block::
  :linenos:
  :emphasize-lines: 1

  X          ,Y          ,U    ,H
  637847.6583,4342828.999,0.679,0.828600629
  637689.6481,4342733.228,0.661,0.694408771
  ...

* Create ``measurement-error.csv`` (adapt from Handique/results/Error.txt)

* Define limits for calibration parameters that the DoE code will use to create ``parameter-file.csv`` (ref. to Handique/results/parameter_file.txt), which will take the following shape where ``PC`` stands for *parameter combination* (without first line header). The headers ``a1j`` and ``a2j`` refer to the calibration parameters in `Ferguson (2007) <https://onlinelibrary.wiley.com/doi/abs/10.1029/2006WR005422>`_ for the roughness zones ``j=1,...4``.

.. code-block::
  :linenos:
  :emphasize-lines: 1

  PCi; a11; a21; a12; a22; a13; a23; a14; a24
  PC1; 7.9; 3.3; 6.7; 1.8; 7.8; 1.5; 7.1; 1.9
  PC2; 6.2; 2.0; 7.8; 2.9; 6.4; 1.0; 7.7; 3.6
  ...


Procedure
---------


* Ran Telemac Mesh Quality Analysis in QGIS