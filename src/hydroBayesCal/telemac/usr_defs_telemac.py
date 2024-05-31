"""
Instantiate global variables of user definitions made in user_input.xlsx
"""

import pandas as _pd
import numpy as _np
from openpyxl import load_workbook

# package scripts
from .config_telemac import *  # contains os and sys
from function_pool import *
from model_structure.usr_defs_standard import UserDefs


class UserDefsTelemac(UserDefs):
    def __init__(self, input_worbook_name="user-input.xlsx", *args, **kwargs):
        UserDefs.__init__(self, input_worbook_name=input_worbook_name)

        # initialize capital letter class variables that are defined through the user XLSX file
        self.N_CPUS = int()  # int number of CPUs to use for Telemac models
        self.TM_CAS = str()  # str of telemac steering file without its directory (CAS ONLY)
        self.tm_xD = str()  # defines to either use telemac2d or 3d
        self.GAIA_CAS = None  # default should be None for compliance with TelemacModel class

    def assign_calib_ranges(self, direct_par_df, vector_par_df, recalc_par_df):
        """Parse user calibration ranges for parameters

        :param pd.DataFrame direct_par_df: direct calibration parameters from user-input.xlsx
        :param pd.DataFrame vector_par_df: vector calibration parameters from user-input.xlsx
        :param pd.DataFrame recalc_par_df: recalculation parameters from user-input.xlsx
        :return: None
        """
        # add scalar calibration parameters to CALIB_PAR_SET
        dir_par_dict = dict(zip(direct_par_df[0].to_list(), direct_par_df[1].to_list()))
        for par, bounds in dir_par_dict.items():
            if not (("TELEMAC" or "GAIA") in par):
                self.CALIB_PAR_SET.update({par: {"bounds": str2seq(bounds),
                                                 "initial val": _np.mean(str2seq(bounds)),
                                                 "recalc par": None}})

        # add list-like vector calibration parameters to CALIB_PAR_SET and check for recalculation parameters
        vec_par_dict = dict(zip(vector_par_df[0].to_list(), vector_par_df[1].to_list()))
        recalc_par_dict = dict(zip(recalc_par_df[0].to_list(), recalc_par_df[1].to_list()))
        for par, init_list in vec_par_dict.items():
            if "multiplier" in str(par).lower():
                self.CALIB_PAR_SET.update({par: {"bounds": (str2seq(vec_par_dict["Multiplier range"])),
                                                 "initial val": 1.0,
                                                 "recalc par": None}})
            if not (("TELEMAC" or "GAIA" or "Multiplier") in str(par)):
                try:
                    self.CALIB_PAR_SET.update({par: {"bounds": (str2seq(vec_par_dict["Multiplier range"])),
                                                     "initial val": _np.array(str2seq(init_list)),
                                                     "recalc par": "Multiplier"}})
                except Exception as e:
                    print("ERROR: the list-like parameter {0} got assigned the value {1}, which I cannot convert to a numpy.array. Details:\n{2}".format(str(par), str(init_list), str(e)))
                    raise ValueError

                if par in RECALC_PARS.keys():
                    # check if parameter is a recalculation parameter (if yes -> check for user input FALSE or TRUE)
                    try:
                        if bool(recalc_par_dict[RECALC_PARS[par]]):
                            self.CALIB_PAR_SET[par]["recalc par"] = RECALC_PARS[par]
                    except KeyError:
                        print("WARNING: found recalcution parameter %s that is not defined in config_TELEMAC.py (skipping...")

        print(" * received the following calibration parameters: %s" % ", ".join(list(self.CALIB_PAR_SET.keys())))
        print(self.CALIB_PAR_SET)
    def check_user_input(self):
        """Check if global variables are correctly assigned"""
        print(" * verifying directories...")
        if not (os.path.isdir(self.SIM_DIR)):
            print(f"ERROR: Cannot find {self.SIM_DIR} - please double-check input XLSX (cell B8).")
            raise NotADirectoryError
        if not (os.path.isfile(self.SIM_DIR + "/%s" % self.TM_CAS)):
            print("ERROR: The TELEMAC steering file %s does not exist." % str(self.SIM_DIR + "/%s" % self.TM_CAS))
            raise FileNotFoundError
        if self.GAIA_CAS:
            if not (os.path.isfile(self.SIM_DIR + "/%s" % self.GAIA_CAS)):
                print("ERROR: The GAIA steering file %s does not exist." % str(self.SIM_DIR + "/%s" % self.GAIA_CAS))
                raise FileNotFoundError
        if not (os.path.isfile(self.CALIB_PTS)):
            print("ERROR: The Calibration CSV file %s does not exist." % str(self.CALIB_PTS))
            raise FileNotFoundError
        if self.MC_SAMPLES < (self.AL_SAMPLES + self.IT_LIMIT):
            print("ERROR: MC_SAMPLES < (AL_SAMPLES + IT_LIMIT)!")
            raise ValueError
        if not isinstance(self.N_CPUS, int):
            # this check is already implied in load_input_defs and kept here for consistency with later versions
            print("ERROR: %s is not a valid number of processors to use (min. 1, integer)" % str(self.N_CPUS))
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
            "meas data": self.read_wb_range(MEASUREMENT_DATA_RANGE),
            "direct priors": self.read_wb_range(PRIOR_SCA_RANGE),
            "vector priors": self.read_wb_range(PRIOR_VEC_RANGE),
            "recalculation priors": self.read_wb_range(PRIOR_REC_RANGE),
        }

    def assign_global_settings(self, file_name=None):
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
        # assign direct, vector, and recalculation parameters
        self.assign_calib_ranges(
            direct_par_df=user_defs["direct priors"],
            vector_par_df=user_defs["vector priors"],
            recalc_par_df=user_defs["recalculation priors"]
        )

        # update global variables with user definitions
        # measurement data
        for set_no in range(1, 5):
            cell_val = str(user_defs["meas data"].loc[
                               user_defs["meas data"][0].str.contains("calib\_target" + str(set_no)), 1].values[0])
            if not ("none" in cell_val.lower()):
                self.CALIB_TARGETS.append(TM_TRANSLATOR[cell_val])
        # active learning parameters
        self.CALIB_PTS = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("calib\_pts"), 1].values[0]
        self.AL_STRATEGY = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("strategy"), 1].values[0]
        self.score_method = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("score"), 1].values[0]
        self.init_runs = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("init\_runs"), 1].values[0]
        self.init_run_sampling = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("init\_run\_sampling"), 1].values[0]
        self.IT_LIMIT = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("it\_limit"), 1].values[0]
        self.AL_SAMPLES = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("al\_samples"), 1].values[0]
        self.MC_SAMPLES = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("mc\_samples\)"), 1].values[0]
        self.MC_SAMPLES_AL = user_defs["al pars"].loc[user_defs["al pars"][0].str.contains("mc\_samples\_al"), 1].values[0]
        # telemac parameters
        self.TM_CAS = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("TELEMAC steering"), 1].values[0]
        self.tm_xD = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("tm_xd"), 1].values[0]
        self.GAIA_CAS = user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("Gaia"), 1].values[0]
        self.SIM_DIR = r"" + user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("Simulation"), 1].values[0]
        try:
            self.N_CPUS = int(user_defs["tm pars"].loc[user_defs["tm pars"][0].str.contains("CPU"), 1].values[0])
        except ValueError:
            print("ERROR: %s is not a valid number of processors to use (min. 1, integer)" % str(self.N_CPUS))
            raise ValueError

        # global surrogate and active learning variables
        self.BME = _np.zeros((self.IT_LIMIT, 1))
        self.RE = _np.zeros((self.IT_LIMIT, 1))
        self.al_BME = _np.zeros((self.AL_SAMPLES, 1))
        self.al_RE = _np.zeros((self.AL_SAMPLES, 1))

        self.check_user_input()
