import os,sys
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

from src.hydroBayesCal.function_pool import *

classification = {
    0: {"name": "Banks", "velocity_range": (0, 1.30), "depth_range": (0, 1)},
    1: {"name": "Pool", "velocity_range": (0.05, 0.50), "depth_range": (0.30, 1)},
    2: {"name": "Slackwater", "velocity_range": (0.05, 0.30), "depth_range": (0.05, 0.30)},
    3: {"name": "Glide", "velocity_range": (0.30, 1), "depth_range": (0.10, 0.60)},
    4: {"name": "Riffle", "velocity_range": (0.30, 1.4), "depth_range": (0.05, 0.20)},
    5: {"name": "Run", "velocity_range": (0.50, 1.4), "depth_range": (0.10, 1)}
}
saving_folder = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation-folder-telemac-gaia/"
raster_data = rasterize(
    saving_folder=saving_folder,
    slf_file_name="results-initial.slf",
    desired_variables=["SCALAR VELOCITY", "WATER DEPTH"],
    spacing=0.05
)
velocity_data = raster_data["SCALAR VELOCITY"]["data"]
depth_data = raster_data["WATER DEPTH"]["data"]
x_regs= raster_data["SCALAR VELOCITY"]["x_regs"]
y_regs = raster_data["SCALAR VELOCITY"]["y_regs"]
spacing = raster_data["SCALAR VELOCITY"]["spacing"]
classify_mu(raster_data, classification, saving_folder, "mu_nf")
# Now, `mu_raster` contains the classified morphological units as integer values
# You can save it as an ASCII file, or process it further as needed

