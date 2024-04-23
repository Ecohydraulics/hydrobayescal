import os
from datetime import datetime
from functools import wraps
import random as rnd
import numpy as np
import pandas as pd
import subprocess
import shutil

# import own scripts

from src.hyBayesCal.telemac.global_config import init_runs
base_path_control_telemac = os.path.dirname(os.path.abspath(__file__))
script_path_control_telemac = os.path.join(base_path_control_telemac,"src", "hyBayesCal", "telemac", "control_telemac.py")


# =====================================================
# ===========GLOBAL SIMULATION PARAMETERS==============
# =====================================================


class TelemacSimulations():

    def run_multiple_simulations(self,script_path,simulation_num):
        try:
            subprocess.run(["python", script_path,simulation_num], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            exit()
        else:
            print("Telemac run completed.")

if __name__ == "__main__":

    for i in range(init_runs):
        simulation_num=str(i+1)
        instance = TelemacSimulations()
        instance.run_multiple_simulations(script_path_control_telemac,simulation_num)
