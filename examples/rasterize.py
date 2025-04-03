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
    0: {"name": "Banks", "velocity_range": (0, 1.2), "depth_range": (0, 0.05), "color": "grey", "alpha": 0.5},
    1: {"name": "Banks", "velocity_range": (0, 0.05), "depth_range": (0.05, 1), "color": "grey", "alpha": 0.5},
    2: {"name": "Pool", "velocity_range": (0.05, 0.50), "depth_range": (0.35, 1.0), "color": "blue", "alpha": 0.5},
    3: {"name": "Slackwater", "velocity_range": (0.05, 0.15), "depth_range": (0.05, 0.35), "color": "green", "alpha": 0.5},
    4: {"name": "Glide", "velocity_range": (0.15, 0.50), "depth_range": (0.05, 0.35), "color": "orange", "alpha": 1.0},
    5: {"name": "Riffle", "velocity_range": (0.50, 1.2), "depth_range": (0.05, 0.35), "color": "red", "alpha": 0.5},
    6: {"name": "Run", "velocity_range": (0.50, 1.2), "depth_range": (0.35, 1.0), "color": "purple", "alpha": 0.5}
}
saving_folder = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation-folder-telemac-gaia/"
raster_data = rasterize(
    saving_folder=saving_folder,
    slf_file_name="results-initial.slf",
    desired_variables=["SCALAR VELOCITY", "WATER DEPTH"],
    spacing=0.05
)
classify_mu(raster_data, classification, saving_folder, "mu_beforeflush")


