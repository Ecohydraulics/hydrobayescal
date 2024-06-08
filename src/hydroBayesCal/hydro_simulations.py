"""
Core code for coupling any hydrodynamic simulation software with the main script for GPE surrogate model construction
and Bayesian Active Learning .

Author: Andres Heredia M.Sc.

"""
#Import libraries
from pathlib import Path
import subprocess
import os
import pdb

# Import own scripts
from telemac.control_telemac import TelemacModel
from model_structure.control_full_complexity import FullComplexityModel


# Base directory
base_dir = Path(__file__).resolve().parent.parent.parent
print("Base directory:", base_dir)
# Relative paths to the base directory
env_script_path = base_dir / 'env-scripts'

from config_logging import *
setup_logging()
logger_info = logging.getLogger("HydroBayesCal")

class HydroSimulations(FullComplexityModel):
    def __init__(
            self,
            model_dir="",
            res_dir="",
            control_file="",
            calibration_parameters=None,
            bal_mode=True,
            n_max_tp=int(),
            init_runs=int(),
            user_inputs=None,
    ):
        """
        Constructor for the HydroSimulations Class. Wraps functions for running Telemac and OpenFoam
        in the context of Bayesian Calibration using Gaussian Process Emulation (GPE).

        Parameters
        ----------
        model_dir : str
            Full complexity model directory.
        res_dir : str
            Directory of the folder where a subfolder called "auto-saved-results" will be created to store all the results files.
        control_file : str
            Name of the file that controls the full complexity model simulation (to be called from a terminal).
        calibration_parameters : list, optional
            Names of the considered calibration parameters (maximum 4 parameters).
        bal_mode : bool, optional
            Default is True. If True, runs Bayesian Active Learning; if False, only runs the initial collocation points.
        n_max_tp : int, optional
            Total number of model simulations, including Bayesian Active Learning iterations.
        init_runs : int, optional
            Initial runs of the full complexity model (initial surrogate model before Bayesian Active Learning).
        user_inputs : dict, optional
            User input parameters.

        Attributes
        ----------
        user_inputs : dict
            User input parameters.
        bal_mode : bool
            Mode for Bayesian Active Learning. Default: True
        n_max_tp : int
            Total number of model simulations, including Bayesian Active Learning iterations.
        init_runs : int
            Initial runs of the full complexity model.
        """

        FullComplexityModel.__init__(self, model_dir=model_dir,
                                     res_dir=res_dir,
                                     calibration_parameters=calibration_parameters,
                                     control_file=control_file,
                                     init_runs=init_runs)
        self.user_inputs = user_inputs
        self.bal_mode = bal_mode
        self.n_max_tp=n_max_tp
        self.init_runs=init_runs


    def tm_simulations(
            self,
            collocation_points=None,
            bal_iteration=int(),
            bal_new_set_parameters=None,
            ):
        control_tm = TelemacModel(
            model_dir=self.model_dir,
            res_dir=self.res_dir,
            control_file=self.control_file,
            friction_file=self.user_inputs['friction_file'],
            calibration_parameters=self.calibration_parameters,
            calibration_pts_file_path=self.user_inputs['calib_pts_file_path'],
            calibration_quantities=self.user_inputs['calib_quantity_list'],
            tm_xd=self.user_inputs['Telemac_solver'],
            n_processors=self.user_inputs['n_cpus'],
            dict_output_name=self.user_inputs['dict_output_name'],
            results_file_name_base=self.user_inputs['results_file_name_base'],
            init_runs=self.init_runs,

        )
        control_tm.run_multiple_simulations(collocation_points, bal_new_set_parameters, bal_iteration,
                                            bal_mode=self.bal_mode)
        model_results = control_tm.output_processing(dict_output_name=self.user_inputs['dict_output_name'])

        return model_results

    def of_simulations(self):
        pass

    def run(self, collocation_points, bal_iteration, bal_new_set_parameters):
        if collocation_points is not None:
            logger_info.info("Running full complexity models with initial collocation points:")
            return self.tm_simulations(collocation_points=collocation_points, bal_iteration=bal_iteration)
        elif bal_new_set_parameters is not None:
            return self.tm_simulations(bal_iteration=bal_iteration, bal_new_set_parameters=bal_new_set_parameters)
        else:
            raise ValueError("Error: At least one of 'collocation_points' or 'bal_new_set_parameters' must be provided.")

    # @staticmethod
    # def activate_env (environment):
    #     script_directory = env_script_path
    #     os.chdir(script_directory)
    #     commands = {
    #         'telemac': "source activateHBCtelemac.sh",
    #         'openfoam': "source activateHBCopenfoam.sh"
    #     }
    #     if environment not in commands:
    #         print(f"Error: Invalid environment '{environment}'. Valid options are 'telemac' and 'openfoam'.")
    #         return
    #
    #     command = commands[environment]
    #
    #     # Run the command for Telemac environment activation
    #     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     stdout, stderr = process.communicate()
    #
    #     # Print the output or error
    #     if process.returncode == 0:
    #         if environment == 'telemac':
    #             print("TELEMAC Environment activated:")
    #         elif environment == 'openfoam':
    #             print("OpenFOAM Environment activated:")
    #         print(stdout.decode())
    #     else:
    #         print("Error executing command:")
    #         print(stderr.decode())

if __name__ == "__main__":
    full_complexity_simulation = HydroSimulations()
    full_complexity_simulation.run()
