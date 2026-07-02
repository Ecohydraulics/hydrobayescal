"""
Script to extract model outputs from a TELEMAC result file (.slf) based on
specified EXTRACTION QUANTITIES. The extraction is performed at locations
defined in a calibration points file, which must be provided.
Author: Andrés Heredia Hidalgo
"""

import argparse
import importlib.util
import os

# Import own scripts
from src.hydroBayesCal.telemac.control_telemac import TelemacModel

def load_config(config_path):
    """
    Load configuration from Python file.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():

    parser = argparse.ArgumentParser(
        description="Extracts Telemac2d/3d model outputs from slf files"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.py",
        help="Path to Python configuration file. Default: config.py"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # Initialize TelemacModel object
    control_tm = TelemacModel(
        model_dir=config.paths['model_dir'], # the .slf file must be in this directory.
        res_dir=config.paths['res_dir'], # an Autosaved results folder will be saved here with the results
        calibration_pts_file_path=config.paths["calibration_pts_file_path"],
        calibration_quantities=config.extraction['calibration_quantities'],
        extraction_quantities=config.extraction['extraction_quantities'],
        n_processors=1,
        dict_output_name=config.calibration['dict_output_name'],
        init_runs=1,
    )

    # Correctly define output_name and extraction_quantity
    # Call extract_data_point method
    control_tm.extract_data_point(input_file=config.extraction['input_slf_file'], calibration_pts_df=control_tm.calibration_pts_df,
                                  output_name=control_tm.dict_output_name, extraction_quantity=control_tm.extraction_quantities, simulation_number=1,
                                  model_directory=config.paths['model_dir'], results_folder_directory=control_tm.calibration_folder,
                                  output_extraction_time=config.extraction['output_extraction_time'],
                                  time_index=config.extraction['time_index'],
                                  n=config.extraction['n'],)
    control_tm.output_processing(output_data_path=os.path.join(control_tm.calibration_folder,f'{control_tm.dict_output_name}-detailed.json'),
                                 calibration_quantities=control_tm.calibration_quantities,
                                 save_extraction_outputs = True,
                                 extraction_mode = True,
                                 calibration_mode = True,
                                 delete_slf_files=False)
if __name__ == "__main__":
    main()
