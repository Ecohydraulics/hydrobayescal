"""This is a use case example for the automated stochastic calibration
with sample data from the https://hydro-informatics.com database."""

# get QGIS core
from qgis.core import *
import qgis.utils
from qgis.analysis import QgsNativeAlgorithms


# load calibration package
import stochastic_calibration as sc

# USER INPUT --------------------------
slf_mesh = "/home/public/test-data/steady2d-restart/r2dsteady.slf"
qgis_install_path = "/usr"  # find in QGIS GUI > Plugins > Python Console > enter QgsApplication.prefixPath()

# END USER INPUT ----------------------

# path to the qgis install location
QgsApplication.setPrefixPath(qgis_install_path, True)
# create a reference to the QgsApplication.
qgs = QgsApplication([], False)  # set second arg to True for GUI apps
# load providers
qgs.initQgis()


qgis_exportmeshvertices(
        INPUT='SELAFIN:"/home/public/test-data/steady2d-restart/r2dsteady.slf"',
        DATASET_GROUPS=[0,1],
        STARTING_TIME={'type': 'dataset-time-step',
                         'value': [0,30]},
        FINISHING_TIME={'type': 'dataset-time-step',
                          'value': [0,30]},
        TIME_STEP=0,
        INPUT_POINTS='/home/public/test-data/observations/measurements.shp',
        COORDINATES_DIGITS=2,
        DATASET_DIGITS=2,
        OUTPUT='/home/public/test-data/export_time_series_vals_mesh2.csv'
    )


# processing.run(
#     "native:meshexporttimeseries", {
#         'INPUT':'SELAFIN:"/home/public/test-data/steady2d-restart/r2dsteady.slf"',
#         'DATASET_GROUPS':[0,1],
#         'STARTING_TIME':{'type': 'dataset-time-step',
#                          'value': [0,30]},
#         'FINISHING_TIME':{'type': 'dataset-time-step',
#                           'value': [0,30]},
#         'TIME_STEP':0,
#         'INPUT_POINTS':'/home/public/test-data/observations/measurements.shp',
#         'COORDINATES_DIGITS':2,
#         'DATASET_DIGITS':2,
#         'OUTPUT':'/home/public/test-data/export_time_series_vals_mesh2.csv'
#     })

# exit QGIS
qgs.exitQgis()
