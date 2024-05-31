"""
Setup logging
"""
import os
import logging

# directories
SCRIPT_DIR = r"" + os.path.dirname(__file__) + os.sep

# setup logging
info_formatter = logging.Formatter("%(asctime)s - %(message)s")
warn_formatter = logging.Formatter("WARNING [%(asctime)s]: %(message)s")
error_formatter = logging.Formatter("ERROR [%(asctime)s]: %(message)s")
logger = logging.getLogger("HyBayesCal")
logger.setLevel(logging.INFO)
logger_warn = logging.getLogger("warnings")
logger_warn.setLevel(logging.WARNING)
logger_error = logging.getLogger("errors")
logger_error.setLevel(logging.ERROR)
# create console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(info_formatter)
logger.addHandler(console_handler)
console_whandler = logging.StreamHandler()
console_whandler.setLevel(logging.WARNING)
console_whandler.setFormatter(warn_formatter)
logger_warn.addHandler(console_whandler)
console_ehandler = logging.StreamHandler()
console_ehandler.setLevel(logging.ERROR)
console_ehandler.setFormatter(error_formatter)
logger_error.addHandler(console_ehandler)
# create info file handler and set level to debug
info_handler = logging.FileHandler(SCRIPT_DIR + "logfile.log", "w")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(info_formatter)
logger.addHandler(info_handler)
# create warning file handler and set level to error
warn_handler = logging.FileHandler(SCRIPT_DIR + "warnings.log", "w")
warn_handler.setLevel(logging.WARNING)
warn_handler.setFormatter(warn_formatter)
logger_warn.addHandler(warn_handler)
# create error file handler and set level to error
err_handler = logging.FileHandler(SCRIPT_DIR + "errors.log", "w")
err_handler.setLevel(logging.ERROR)
err_handler.setFormatter(error_formatter)
logger_error.addHandler(err_handler)
