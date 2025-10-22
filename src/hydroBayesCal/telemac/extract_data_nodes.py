"""
Script that extracts the variable of interest of all or specific nodes of a Telemac mesh. If specific nodes are required, the number of the node (Node ID) should be
given.
"""
import numpy as np
from pputils.ppmodules.selafin_io_pp import ppSELAFIN  # Ensure this module is available and correctly imported

def main():
    # Define the input parameters
    file_name = '/home/IWS/hidalgo/Documents/QGIS_Ering_final_2025/constant-NKU/simulation/geometry-constantNKU.slf'  # Replace with the path to your SELEFIN file
    calibration_variable = 'FRIC_ID'  # Replace with the variable name you are interested in
    specific_nodes = None # Replace with the specific nodes you want to analyze (or set to None to analyze all nodes)
    save_name = '/home/IWS/hidalgo/Documents/QGIS_Ering_final_2025/constant-NKU/simulation/friction_ID.bfr.txt'  # Replace with the desired output file name (or leave empty if you don't want to save)

    # Call the function
    results = get_variable_value(file_name, calibration_variable, specific_nodes, save_name)

    # Print the results
    print("Formatted modelled results:")
    print(results)
def get_variable_value(file_name, calibration_variable, specific_nodes=None, save_name=""):
    # Read the SELEFIN file
    slf = ppSELAFIN(file_name)
    slf.readHeader()
    slf.readTimes()

    # Get the printout times
    times = slf.getTimes()

    # Read the variables names
    variables_names = slf.getVarNames()
    # Removed unnecessary spaces from variables_names
    variables_names = [v.strip() for v in variables_names]
    # Get the position of the value of interest
    index_variable_interest = variables_names.index(calibration_variable)

    # Read the variables values in the last time step
    slf.readVariables(len(times) - 1)

    # Get the values (for each node) for the variable of interest in the last time step
    modelled_results = slf.getVarValues()[index_variable_interest, :]
    format_modelled_results = np.zeros((len(modelled_results), 2))
    format_modelled_results[:, 0] = np.arange(1, len(modelled_results) + 1, 1)
    format_modelled_results[:, 1] = modelled_results

    # Get specific values of the model results associated in certain nodes number, in case the user want to use just
    # some nodes for the comparison. This part only runs if the user specify the parameter specific_nodes. Otherwise
    # this part is ommited and all the nodes of the model mesh are returned
    if specific_nodes is not None:
        format_modelled_results = format_modelled_results[specific_nodes.astype(int) - 1, :]

    if len(save_name) != 0:
        np.savetxt(save_name, format_modelled_results, delimiter="	", fmt=['%1.0f', '%1.3f'])

    # Return the value of the variable of interest
    return format_modelled_results

if __name__ == "__main__":
    main()
