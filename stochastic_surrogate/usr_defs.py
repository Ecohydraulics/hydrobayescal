"""
Instantiate global variables of user definitions made in user_input.xlsx
"""

import os as _os
import pandas as _pd


def load_input_defs(file_name="user_input.xlsx"):
    """loads provided input file name as pandas dataframe

    Args:
        file_name (str): name of input file (default is input.xlsx)

    Returns:
        (dict): user input of input.xlsx (or costum file, if provided)
    """
    input_xlsx_df = _pd.read_excel(file_name, header=0, index_col=0)



    return {
        "folder name": input_xlsx_df["VALUE"]["Input folder directory"],
        "file type": input_xlsx_df["VALUE"]["Data file ending"],
        "profile": PROFILE_KEYS[input_xlsx_df["VALUE"]["ADV direction"]],
        "bulk velocity": input_xlsx_df["VALUE"]["Flow velocity"],
        "bulk depth": input_xlsx_df["VALUE"]["Water depth"],
        "freq": input_xlsx_df["VALUE"]["ADV sampling frequency"],
        "characteristic wood length": input_xlsx_df["VALUE"]["Turbulence object length dimension"],
        "despiking method": input_xlsx_df["VALUE"]["Spike detection method"],
        "lambda a": input_xlsx_df["VALUE"]["Despike lambda a"],
        "despike k": input_xlsx_df["VALUE"]["Despike k"],
    }



