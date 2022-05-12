"""
Instantiate global variables of user definitions made in user_input.xlsx
"""

import os as _os
import pandas as _pd
import numpy as _np
from openpyxl import load_workbook
from config import *
from basic_functions import *


class UserDefs:
    def __init__(self, input_worbook_name="user-input.xlsx", *args, **kwargs):
        self.input_xlsx_name = input_worbook_name
        print("Using %s to read user settings. To change the user settings, run OBJECT(bal_gpe4telemac).write_global_settings('path/to/workbook.xlsx')" % self.input_xlsx_name)

        self.CALIB_PAR_SET = {}  # dict for direct calibration optimization parameters
        self.CALIB_ID_PAR_SET = {}  # dict for indirect calibration parameters
        self.CALIB_PTS = None  # numpy array to be loaded from calibration_points file
        self.CALIB_TARGET = str() # str of calibration target nature (e.g. topographic change)
        self.IT_LIMIT = int()  # int limit for Bayesian iterations
        self.MC_SAMPLES = int()  # int for Monte Carlo samples
        self.MC_SAMPLES_AL = int()  # int for Monte Carlo samples for active learning
        self.N_CPUS = int()  # int number of CPUs to use for Telemac models
        self.AL_SAMPLES = int()  # int for no. of active learning sampling size
        self.AL_STRATEGY = str()  # str for active learning strategy
        self.TM_CAS = str()
        self.GAIA_CAS = str()
        self.RESULTS_DIR = "../results"  # relative path for results
        self.SIM_DIR = str()  # relative path for simulations


    def assign_calib_ranges(self, direct_par_df, indirect_par_df, recalc_par_df):
        """Parse user calibration ranges for parameters

        :param pd.DataFrame direct_par_df: direct calibration parameters from user-input.xlsx
        :param pd.DataFrame indirect_par_df: indirect calibration parameters from user-input.xlsx
        :param pd.DataFrame recalc_par_df: recalculation parameters from user-input.xlsx
        :return:
        """


        dir_par_dict = dict(zip(direct_par_df[0].to_list(), direct_par_df[1].to_list()))
        for par, bounds in dir_par_dict.items():
            if not (("TELEMAC" or "GAIA") in par):
                self.CALIB_PAR_SET.update({par: {"bounds": str2seq(bounds),
                                            "distribution": None}})

        indir_par_dict = dict(zip(indirect_par_df[0].to_list(), indirect_par_df[1].to_list()))
        for par, bounds in indir_par_dict.items():
            self.CALIB_ID_PAR_SET.update({par: {"classes": str2seq(bounds),
                                           "distribution": None}})
            if not (("TELEMAC" or "GAIA") in par):
                # erase CALIB_ID_PAR_SET in the last step, if user did not enable (OK because inexpensive)
                self.CALIB_ID_PAR_SET = None

        recalc_par_dict = dict(zip(recalc_par_df[0].to_list(), recalc_par_df[1].to_list()))
        for par, bounds in recalc_par_dict.items():
            # loop not really needed but implemented for potential developments
            if not(par in dir_par_dict) and bounds:
                # overwrite or add recalculation parameter in CALIB_PAR_SET dict
                # here: bounds is a user-defined boolean
                self.CALIB_PAR_SET.update({par: {"bounds": (self.CALIB_ID_PAR_SET["Multiplier range"]["classes"][0],
                                                       self.CALIB_ID_PAR_SET["Multiplier range"]["classes"][1]),
                                            "distribution": None}})

        print(" * received direct calibration parameters: %s" % ", ".join(list(self.CALIB_PAR_SET.keys())))
        if self.CALIB_ID_PAR_SET:
            print(" * received indirect calibration parameter: %s" % ", ".join(list(self.CALIB_ID_PAR_SET.keys())))

    def check_user_input(self):
        """Check if global variables are correctly assigned"""
        print(" * verifying directories...")
        if not (_os.path.isdir(self.SIM_DIR)):
            print("ERROR: Cannot find %s - please double-check input XLSX (cell B8).")
            raise NotADirectoryError
        if not (_os.path.isfile(self.SIM_DIR + "/%s" % self.TM_CAS)):
            print("ERROR: The TELEMAC steering file %s does not exist." % str(self.SIM_DIR + "/%s" % self.TM_CAS))
            raise FileNotFoundError
        if self.GAIA_CAS:
            if not (_os.path.isfile(self.SIM_DIR + "/%s" % self.GAIA_CAS)):
                print("ERROR: The GAIA steering file %s does not exist." % str(self.SIM_DIR + "/%s" % self.GAIA_CAS))
                raise FileNotFoundError
        if not (_os.path.isfile(self.CALIB_PTS)):
            print("ERROR: The Calibration CSV file %s does not exist." % str(self.CALIB_PTS))
            raise FileNotFoundError
        if not (_os.path.isdir(self.RESULTS_DIR)):
            try:
                _os.mkdir(self.RESULTS_DIR)
            except PermissionError:
                print("ERROR: Cannot write to %s (check user rights/path consistency)" % self.RESULTS_DIR)
                raise PermissionError
            except NotADirectoryError:
                print("ERROR: %s is not a directory - adapt simulation directory (B8)" % self.RESULTS_DIR)
                raise NotADirectoryError
        if self.MC_SAMPLES < (self.AL_SAMPLES + self.IT_LIMIT):
            print("ERROR: MC_SAMPLES < (AL_SAMPLES + IT_LIMIT)!")
            raise ValueError

    def load_input_defs(self):
        """loads provided input file name as dictionary

        Returns:
            (dict): user input of input.xlsx (or custom file, if provided)
        """
        print(" * loading %s" % self.input_xlsx_name)
        return {
            "tm pars": self.read_wb_range(TM_RANGE),
            "al pars": self.read_wb_range(AL_RANGE),
            "direct priors": self.read_wb_range(PRIOR_DIR_RANGE),
            "indirect priors": self.read_wb_range(PRIOR_INDIR_RANGE),
            "recalculation priors": self.read_wb_range(PRIOR_REC_RANGE),
        }

    def write_global_settings(self, file_name=None):
        """rewrite globals from config

        Args:
            file_name (str): name of input file (default is user-input.xlsx)

        Returns:
            (dict): user input of input.xlsx (or custom file, if provided)
        """

        # update input xlsx file name globally and load user definitions
        if file_name:
            self.input_xlsx_name = file_name
        user_defs = self.load_input_defs()  # dict

        print(" * assigning user-defined variables...")
        # assign direct, indirect, and recalculation parameters
        self.assign_calib_ranges(
            direct_par_df=user_defs["direct priors"],
            indirect_par_df=user_defs["indirect priors"],
            recalc_par_df=user_defs["recalculation priors"]
        )

        # update global variables with user definitions
        self.CALIB_PTS = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("calib\_pts"), 1].values[0]
        self.CALIB_TARGET = TM_TRANSLATOR[user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("calib\_target"), 1].values[0]]

        self.AL_STRATEGY = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("strategy"), 1].values[0]
        self.IT_LIMIT = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("it\_limit"), 1].values[0]
        self.AL_SAMPLES = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("al\_samples"), 1].values[0]
        self.MC_SAMPLES = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("mc\_samples\)"), 1].values[0]
        self.MC_SAMPLES_AL = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("mc\_samples\_al"), 1].values[0]

        self.TM_CAS = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("TELEMAC"), 1].values[0]
        self.GAIA_CAS = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("Gaia"), 1].values[0]
        self.SIM_DIR = r"" + user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("Simulation"), 1].values[0]
        self.N_CPUS = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("CPU"), 1].values[0]
        self.RESULTS_DIR = self.SIM_DIR + "opt-results/"

        self.check_user_input()

    def read_wb_range(self, read_range, sheet_name="MAIN"):
        """Read a certain range of a workbook only with openpyxl

        :param str read_range: letter-number read range in workbook (e.g. "A2:B4")
        :param str sheet_name: name of the sheet to read (default is MAIN from user-inpux.xlsx)
        :return pd.DataFrame: xlsx contents in the defined range
        """
        ws = load_workbook(filename=self.input_xlsx_name, read_only=True, data_only=True)[sheet_name]
        # Read the cell values into a list of lists
        data_rows = []
        for row in ws[read_range]:
            data_rows.append([cell.value for cell in row])
        return _pd.DataFrame(data_rows)
