"""
Creates .cas files according to the number in full complexity models
"""
from config import *
import os
import pandas as pd

def cas_creator(source_file, tm_model_dir, init_runs,results_filename_base,df_calib_param_values,calib_param_names,cas_lines):
    """
    Author - Andres
    Creates .cas files for full complexity models.

    Parameters:
    - source_file: Path to source (*.cas base) file.
    - tm_model_dir: Directory for .cas files.
    - init_runs: Number of initial runs for surrogate model construction.
    - results_filename_base: Base filename for results.
    - calib_param_range: Range for calibration parameter.

    Returns:
    - results_filename_list: List of generated results filenames.
    - random_param: List of random calibration parameters.

    """    
    results_filename_list=[]
    file_name = os.path.basename(source_file)
    prefix, extension = os.path.splitext(file_name)

    with open(source_file, 'r') as f:

        source_content = f.read()
    print(calib_param_names)

    for i in range(1, init_runs + 1):
        # Get the random parameters for the current run (indexed by i)
        random_params = df_calib_param_values.loc[f'PC{i}']

        # Create the new file name for the current run
        new_file_name = f"{prefix}-{i}{extension}"
        destination_file = os.path.join(tm_model_dir, new_file_name)

        # Read the source content from the original file
        with open(source_file, 'r') as f:
            source_content = f.readlines()

        # Modify the source content for each parameter
        new_content = source_content
        x=0
        for param_name, param_value in random_params.items():
            e=cas_lines[x]
            for line_num, line_content in enumerate(new_content):
                if cas_lines[-1] in line_content:
                    new_content[line_num] = line_content.replace(cas_lines[-1], f"RESULTS FILE:{results_filename_base}-{i}.slf")
                if e in line_content:
                    try:
                        new_content[line_num] = line_content.replace(e, f"{param_name}:{param_value}")
                        x=x+1
                        break
                    except:
                        print(f"Error in replacing the parameter value in the {i} th .cas file.")
                        break
            continue

        # Write the modified content to the new file
        results_filename_list.append(os.path.join(tm_model_dir,f"{results_filename_base}-{i}.slf"))

        with open(destination_file, 'w') as f:
            f.writelines(new_content)

    return results_filename_list

def sim_output_df(init_runs,results_filename_base,output_excel_file_name,auto_saved_results_path,calib_quantity):
    """
    Creates DataFrame from simulation outputs and saves to Excel.

    Parameters:
    - tm_model_dir: Directory for simulation results.
    - init_runs: Number of runs.
    - results_filename_base: Base filename for results. This is taken from each .cas file.
    - output_excel_file_name: Filename for Excel file containing simulation outputs for the number of model runs.
    - random_param: List of random calibration parameters.

    Returns:
    - df_outputs: DataFrame of simulation outputs for all the selected calibration quantities and parameter combinations.

    """ 
    # Initializes an empty DataFrame to store the data
    results_filename_list_txt = []
    for simulation in range(1, init_runs + 1):
        for quantity in calib_quantity:
            result_path_txt = auto_saved_results_path + f"/{results_filename_base}-{simulation}_{quantity}.txt"
            results_filename_list_txt.append(result_path_txt)
    # column_names = [txt_path.split('-')[-1].split('.')[0] for txt_path in results_filename_list_txt]
    df_outputs = pd.DataFrame()
    # Loops through each file path
    for file_path in results_filename_list_txt:
        # Reads the output text files into a DataFrame
        data = pd.read_csv(file_path, header=None, delimiter='\s+')
        df_outputs = pd.concat([df_outputs, data.iloc[:, 1]], axis=1)
    # Sets column names
    column_names = ['Outputs for .CAS file # '+txt_path.split('-')[-1].split('.')[0] for txt_path in results_filename_list_txt]
    df_outputs.columns = column_names
    df_outputs['TM Nodes'] = range(1, len(df_outputs) + 1)
    df_outputs.set_index('TM Nodes', inplace=True)
    #df_outputs_filtered_calib_nodes=
    df_outputs.to_excel(auto_saved_results_path + "/" + output_excel_file_name)
    print("DataFrame saved to Excel file:", output_excel_file_name)
    return df_outputs

