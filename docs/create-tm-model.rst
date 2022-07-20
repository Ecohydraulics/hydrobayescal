
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

- A boundary polyline shapefile (**boundary.shp**) that delineates the maximum wetted area
- A breaklines polyline shapefile to delineate regions with different properties (e.g., roughness properties or mesh density)

.. tip::

  No idea about QGIS? Consider to complete our 60-minutes tutorial at `hydro-informatics.com/geopy/use-qgis <https://hydro-informatics.com/geopy/use-qgis.html>`_ 

To start, open QGIS and load the DEM (GeoTIFF) layer. In addition, a satellite imagery basemap might be usefull if no own UAV imagery is available (read more at `hydro-informatics.com/.../basemaps <https://hydro-informatics.com/geopy/use-qgis.html#basemap>`_). To facilitate drawing lines in the following, **enable snapping**.


Create boundary.shp
~~~~~~~~~~~~~~~~~~~

1. In QGIS go to **Layer** > **Create Layer** > **New Shapefile Layer...** and fill the following fields:
   * **File name**: click on the `...` button to navigate to a target directory and type the file name `boundary-raw` (thus, the field should show something like `/home/user/tm-calib-ex/boundary-raw.shp`)
   * **Geometry type**: `LineString`
   * For **EPSG**, make sure to select the same projection as the DEM GeoTIFF uses. For instance, the example `dem.tif` provided with this tutorial uses `EPSG:31494`.
   * Keep all other defaults and click **OK**.
1. Enable editing of the new **boundary-raw** layer, and delineate the maximum wetted boundary of the model by drawing a polyline around it. **Important** is that the polyline is closed in the end (possible when **snapping is enabled**). When drawing the line, make sure that it is entirely in the region covered by the DEM. After drawing the boundary, **save the edits and disable (*toggle*) editing**.
1. The raw boundary has an irregular spacing of corner points the stem from individual choices in the previous work step. Yet, the density of points will drive the mesh resolution in the following, which is why the raw boundary should be adapted with a re-sampling trick to yield regular point spacing at the model boundary. For this purpose, resample points at the target mesh resolution by opening QGIS' processing toolbox: expand the **Vector geometry** entry and click on **Points along geometry**. In the popup window make the following settings:
   * **Input layer**: select **boundary-raw**
   * **Distance**: use the targert mesh resolution (10 meters in the example)
   * **Start** and **End** offsets: keep defaults of `0`
   * **Interpolated points** file name: click on the `...` button and define a file name (e.g., `resampled-points.shp`)
   * Click on **Run**
 1. In the processing toolbox, expand the **Vector creation** entry and click on the **Points to path** tool to re-build a polyline from the previously re-sampled points:
   * **Input layer**: select **resampled-points**
   * **Activate** the optional **Create closed path** checkbox
   * **Paths** file name: click on the `...` button and define a file name (e.g., `boundary.shp`)
   * Click on **Run**

The successfull run of the tool has created the second required geospatial data item (after the DEM) in the form of the **boundary.shp** shapefile.


Create breaklines.shp
~~~~~~~~~~~~~~~~~~~~~

measurements.shp is in `EPSG:31468`














Learn more about `TELEMAC at hydro-informatics.com/numerics/telemac <https://hydro-informatics.com/numerics/telemac.html>`_ to learn more.


