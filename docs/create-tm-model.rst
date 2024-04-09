
TELEMAC Model
=============

.. admonition:: Requirements

  The TELEMAC model requires the following data:
  
  * A digital elevation model (DEM) in GeoTIFF raster format (`.tif` file)
  * Knowledge about the maximum wetted extend that the model needs to cover
  * Inflow or outflow (discharge) rate(s) for steady or unsteady simulations
  * A stage-discharge relation at the model outlet (i.e., downstream end) for at least one discharge


Create QGIS baseline files
--------------------------

The TELEMAC model builds on a numerical mesh that can be build based on a DEM and knowledge about the landscape. In particular, the following files need to be created with QGIS (or other GIS software) and according to the detailed instructions in the following:

- An XYZ (comma-serparated value - `.csv` formatted) DEM raster
- A boundary polyline shapefile (**boundary.shp**) that delineates the maximum wetted area
- A breaklines (internal lines) polyline shapefile to delineate regions with different properties (e.g., roughness properties or mesh density)
- An embedded points file



.. tip::

  No idea about QGIS? Consider to complete our 60-minutes tutorial at `hydro-informatics.com/geopy/use-qgis <https://hydro-informatics.com/geopy/use-qgis.html>`_ 

To start, open QGIS and load the DEM (GeoTIFF) layer. In addition, a satellite imagery basemap might be usefull if no own UAV imagery is available (read more at `hydro-informatics.com/.../basemaps <https://hydro-informatics.com/geopy/use-qgis.html#basemap>`_). To facilitate drawing lines in the following, **enable snapping**.


XYZ DEM Raster
~~~~~~~~~~~~~~

Raster data are typically provided in common geospatial formats, such as GeoTIFF. However, the application with `pputils` requires the DEM raster to be stored in a comma-separated value (`.csv`) ASCII file.

.. note::

   * The ASCII / csv text format is a poor way of storing raster files. It is only needed here for working with `pputils`, but should be avoided in general.
   * The projection of all geospatial formats used with `pputils` should be the same and work with metric SI units (not degrees) with a precision of at least 0.1 m (thus, do not use `EPSG:4326` or `EPSG:3857`, which both have maximum precisions of 2 m)
   * In this example, we use `EPSG:5678` for all files. If required, reproject the DEM raster into the target projection format from **Processing Toolbox** > **GDAL** > **Raster projections** > **Warp (reproject)**.

To export the DEM to XYZ-csv format complete the following workflow:

* In QGIS, go to the **Raster** top menu **Conversion** > **Translate (Convert Format)**
* In the popup window:
  * Select the source DEM in the **Input layer** drop-down menu
  * Keep all other defaults
  * Click on the **...** button > **Save to File...** to save the file in `.xyz` format (e.g., `dem-4326.csv` in a directory for storing the geodata in this tutorial.
  * Click **Run** to finalize the format conversion.
* Leave QGIS and open the newly created `.xyz` file with the spreadsheet editor of an office application (e.g., MS Excel, LibreOffice, or ONLYOFFICE):
  * When importing the `.xyz` file, make sure to use the correct separator (*space* should be enabled).
  * The three columns are the X, Y, and Z coordinates of the DEM, but those still contain all not-a-number (nan) points, which makes the file unnecessarily heavy.
  * To remove the nan points, create a table, and sort by the Z-column (if the nan-value is `-9999` use a largest to smallest sort).
  * As a result, only relevant points (with valid Z-values that are not `-9999`) should be shown on the top of the table.
  * Scroll down the table until the first useless nan-value (e.g. `-9999`) appears and highlight all following rows (hold down `CTRL` + `SHIFT` and click first on the right, then on the down arrow keys).
  * Press `DEL` to delete the nan-value points and make sure that no other values than required, valid XYZ points are in the spreadsheet.
  * Save the spreadsheet as `.csv` file (e.g., `dem-xyz.csv`).

.. tip:: Use a Python script to remove NAN values

   The following script does all steps in the office program:
   
   .. codeblock::
      
      import os
      import pandas as pd  # requires pandas

      # set user variables
      wd = os.path.abspath("") + "/"
      xyz_fn = "dem.xyz"
      nan_value = -9999

      # open xyz file, remove all rows where the Z-column contains the nan_value, and save as csv file
      df = pd.read_csv(xyz_fn, sep="\s+", names=["X", "Y", "Z"])
      df = df[df.Z != nan_value]
      df.to_csv(wd + xyz_fn.split(".")[0] + ".csv", header=False, index=False)


The successfull run of the tool has created the first required geospatial data item in the form of the **dem-xyz.csv** (ASCII) raster.

Create boundary.shp
~~~~~~~~~~~~~~~~~~~

1. In QGIS go to **Layer** > **Create Layer** > **New Shapefile Layer...** and fill the following fields:
   * **File name**: click on the `...` button to navigate to a target directory and type the file name `boundary-raw` (thus, the field should show something like `/home/user/tm-calib-ex/boundary-raw.shp`)
   * **Geometry type**: `LineString`
   * For **EPSG**, make sure to select the same projection as the DEM raster uses. For instance, the example `dem.tif` provided with this tutorial uses `EPSG:31494`.
   * Keep all other defaults and click **OK**.
1. Enable **editing** of the new **boundary-raw** layer, and delineate the maximum wetted boundary of the model by drawing a polyline around it. **Important** is that the polyline is closed in the end (possible when **snapping is enabled**). When drawing the line, make sure that it is entirely in the region covered by the DEM. After drawing the boundary, **save the edits and disable (*toggle*) editing**.
1. The raw boundary has an irregular spacing of corner points the stem from individual choices in the previous work step. Yet, the density of points will drive the mesh resolution in the following, which is why the raw boundary should be adapted with a re-sampling trick to yield regular point spacing at the model boundary. For this purpose, resample points at the target mesh resolution by opening QGIS' processing toolbox: expand the **Vector geometry** entry and click on **Split lines by maximum length**. In the popup window make the following settings:
   * **Input layer**: select **boundary-raw**
   * **Maximum line length**: use the targert mesh resolution (10 meters in the example)
   * **Split** file name: click on the `...` button and define a file name (e.g., `boundary.shp`)
   * Click on **Run**

The last step can also be achieved by calling python3 in the terminal (adapt file path):

.. codeblock::

   qgis_process run native:splitlinesbylength --distance_units=meters --area_units=m2 --ellipsoid=EPSG:7004 --INPUT=/home/public/example/boundary.shp --LENGTH=10 --OUTPUT=TEMPORARY_OUTPUT

The successfull run of the tool has created the second required geospatial data item (after the DEM) in the form of the **boundary.shp** shapefile.

Export to csv.

Create breaklines.shp
~~~~~~~~~~~~~~~~~~~~~


1. In QGIS go to **Layer** > **Create Layer** > **New Shapefile Layer...** and fill the following fields:
   * **File name**: click on the `...` button to navigate to a target directory and type the file name `breaklines-raw` (thus, the field should show something like `/home/user/tm-calib-ex/boundary-raw.shp`)
   * **Geometry type**: `LineString`
   * For **EPSG**, make sure to select the same projection as the DEM GeoTIFF uses (recall, the tutorial DEM uses `EPSG:31494`).
   * Keep all other defaults and click **OK**.
1. Enable **editing** of the new **breaklines-raw** layer, and draw lines delineating model regions within the boundary (given by `boundary.shp`). For instance, breaklines are recommended to be drawn along the active river channel and the floodplains for two reasons. The first reason is that floodplains should get assigned a higher friction than the active channel later. The second reason is that the computational mesh in the channel should be denser than over the flood plains. The example in this tutorials uses the following breaklines:
   * Internal limits of rough block ramps in the riverbed
   * Gravel banks and bars
   * Shore line of wetted boundaries at average discharge
   * Additional notes: 
     - The breaklines should **not cross the boundary** (i.e., enable snapping when drawing). After drawing the breaklines, **save the edits and disable (*toggle*) editing**.
     - To enforce elongated triangles along the main streamline, consider drawing buffers between the left and rght bank shore lines. In QGIS, the **Offset lines** tool (**Processing Toolbox** > **Vector geometry** > **Offset lines**) aids in drawing a user-defined number of parallel lines between the banks at a user-defined distance. For instance, if the regular mesh size is 10 m, 9 parallel lines (i.e., segments) with distances of 5 m can be drawn to enforce a streamline-oriented mesh along an approximately 50-m wide active channel.
1. Also the raw breaklines have an irregular spacing of corner points the stem from individual drawing choices. Again, to control the mesh resolution, the raw breaklines should be adapted with the re-sampling trick. Therefore, resample points at the target mesh resolution between lines by opening QGIS' processing toolbox: expand the **Vector geometry** entry and click on **Split lines by maximum length**. In the popup window make the following settings:
   * **Input layer**: select **breaklines-raw**
   * **Maximum line length**: use the targert mesh resolution (7 meters in the example)
   * **Split** file name: click on the `...` button and define a file name (e.g., `breaklines.shp`)
   * Click on **Run**

The last step can also be achieved by calling python3 in the terminal (adapt file path):

.. codeblock::

   qgis_process run native:splitlinesbylength --distance_units=meters --area_units=m2 --ellipsoid=EPSG:7004 --INPUT=/home/public/example/breaklines.shp --LENGTH=10 --OUTPUT=TEMPORARY_OUTPUT

The successfull run of the tool has created the third required geospatial data item in the form of the **breaklines.shp** shapefile.

Export to csv.

Create embedded-nodes.shp
~~~~~~~~~~~~~~~~~~~~~~~~~

Ideally, measurements were made before the numerical is created and the measurement points are known. For the model calibration, mesh nodes ideally coincide with measurements. Thus, the measurements points should be embedded in the baseline files for the mesh generation. In the example of this tutorial, a point shapefile called `measurements.shp` (in `EPSG:31468`) is provided with water depth and flow velocity measurements.

Export to csv.














Learn more about `TELEMAC at hydro-informatics.com/numerics/telemac <https://hydro-informatics.com/numerics/telemac.html>`_ to learn more.


