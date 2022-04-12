"""
Instantiate global variables of user definitions made in user_input.xlsx
"""

import os as _os
import pandas as _pd
from openpyxl import load_workbook
from config import *


def assign_calib_ranges(direct_par_df, indirect_par_df, recalc_par_df):
    """Parse user calibration ranges for parameters

    :param pd.DataFrame direct_par_df: direct calibration parameters from user-input.xlsx
    :param pd.DataFrame indirect_par_df: indirect calibration parameters from user-input.xlsx
    :return:
    """
    global CALIB_PAR_SET  # dict for calibration optimization parameters and ranges
    global CALIB_ID_PAR_SET  # dict for indirect calibration parameters

    CALIB_PAR_SET
    CALIB_ID_PAR_SET


def load_input_defs():
    """loads provided input file name as dictionary

    Returns:
        (dict): user input of input.xlsx (or custom file, if provided)
    """
    return {
        "tm pars": read_wb_range(INPUT_XLSX_NAME, TM_RANGE),
        "al pars": read_wb_range(INPUT_XLSX_NAME, AL_RANGE),
        "direct priors": read_wb_range(INPUT_XLSX_NAME, PRIOR_DIR_RANGE),
        "indirect priors": read_wb_range(INPUT_XLSX_NAME, PRIOR_INDIR_RANGE),
        "recalculation priors": read_wb_range(INPUT_XLSX_NAME, PRIOR_REC_RANGE),
    }


def rewrite_globals(file_name="user-input.xlsx"):
    """rewrite globals from config

    Args:
        file_name (str): name of input file (default is user-input.xlsx)

    Returns:
        (dict): user input of input.xlsx (or custom file, if provided)
    """
    # instantiate global variables to modify
    global INPUT_XLSX_NAME  # str of path to user-input.xlsx including filename
    global CALIB_PTS  # numpy array to be loaded from calibration_points file
    global CALIB_TARGET # str of calibration target nature (e.g. topographic change)
    global AL_STRATEGY  # str for active learning strategy
    global IT_LIMIT  # int limit for Bayesian iterations
    global MC_SAMPLES  # int for Monte Carlo samples
    global MC_SAMPLES_AL  # int for Monte Carlo samples
    global N_CPUS  # int number of CPUs to use for Telemac models
    global AL_SAMPLES  # int for no. of active learning sampling size

    # update input xlsx file name globally and load user definitions
    INPUT_XLSX_NAME = file_name
    user_defs = load_input_defs()  # dict

    # update global variables with user definitions
    N_CPUS = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("CPU"), 1].values[0]
    # CALIB_PARAMETERS = user_defs["direct priors"][0].to_list()
    CALIB_PTS
    CALIB_TARGET = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("calib\_target"), 1].values[0]

    AL_STRATEGY = TM_TRANSLATOR[user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("strategy"), 1].values[0]]
    IT_LIMIT = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("it\_limit"), 1].values[0]
    AL_SAMPLES = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("al\_samples"), 1].values[0]
    MC_SAMPLES = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("mc\_samples\)"), 1].values[0]
    MC_SAMPLES_AL = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("mc\_samples\_al"), 1].values[0]


def read_wb_range(file_name, read_range, sheet_name="MAIN"):
    """Read a certain range of a workbook only with openpyxl

    :param str file_name: full path and name of input file
    :param str read_range: letter-number read range in workbook (e.g. "A2:B4")
    :param str sheet_name: name of the sheet to read (default is MAIN from user-inpux.xlsx)
    :return pd.DataFrame: xlsx contents in the defined range
    """
    ws = load_workbook(filename=file_name, read_only=True)[sheet_name]
    # Read the cell values into a list of lists
    data_rows = []
    for row in ws[read_range]:
        data_rows.append([cell.value for cell in row])
    return _pd.DataFrame(data_rows)

