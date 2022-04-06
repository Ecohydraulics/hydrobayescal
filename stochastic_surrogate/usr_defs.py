"""
Instantiate global variables of user definitions made in user_input.xlsx
"""

import os as _os
import pandas as _pd
from openpyxl import load_workbook
from config import *


def load_input_defs(file_name="user-input.xlsx"):
    """loads provided input file name as pandas dataframe

    Args:
        file_name (str): name of input file (default is user-input.xlsx)

    Returns:
        (dict): user input of input.xlsx (or costum file, if provided)
    """
    return {
        "tm pars": read_wb_range(file_name, TM_RANGE),
        "al pars": read_wb_range(file_name, AL_RANGE),
        "direct priors": read_wb_range(file_name, PRIOR_DIR_RANGE),
        "indirect piors": read_wb_range(file_name, PRIOR_INDIR_RANGE),
        "recalculation piors": read_wb_range(file_name, PRIOR_REC_RANGE),
    }


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

