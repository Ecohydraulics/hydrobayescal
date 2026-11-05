"""
Setup logging

Messages go to the console from the import on. The log files ``logfile.log``, ``warnings.log`` and ``errors.log`` are written into the results folder of a calibration: :class:`~hydroBayesCal.hysim.HydroSimulations` calls :func:`setup_file_logging` with its ``asr_dir`` as soon as that folder exists, so the log lies next to the results it explains and several calibration cases keep their logs apart.
"""
import os
import logging
from datetime import datetime

# setup logging
info_formatter = logging.Formatter("%(asctime)s - %(message)s")
warn_formatter = logging.Formatter("WARNING [%(asctime)s]: %(message)s")
error_formatter = logging.Formatter("ERROR [%(asctime)s]: %(message)s")
logger = logging.getLogger("HydroBayesCal")
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

#: File handlers installed by :func:`setup_file_logging`, per logger.
_FILE_HANDLERS = {}

#: (logger, file name, level, formatter) of the three log files.
_LOG_FILES = (
    (logger, "logfile.log", logging.INFO, info_formatter),
    (logger_warn, "warnings.log", logging.WARNING, warn_formatter),
    (logger_error, "errors.log", logging.ERROR, error_formatter),
)


def setup_file_logging(log_dir):
    """Write the log files into ``log_dir`` from now on.

    The files are appended to, so a restart with ``only_bal_mode`` keeps the log of the runs it restarts from; every call that changes the folder marks the start of a session in ``logfile.log``. Files of an earlier call are closed, so the messages of one process go to one folder and are not written twice. Calling it again with the same folder changes nothing.

    Parameters
    ----------
    log_dir : str
        Folder of the log files, normally the ``asr_dir`` of the model.

    Returns
    -------
    str
        Absolute path of ``log_dir``.
    """
    log_dir = os.path.abspath(log_dir)
    current = _FILE_HANDLERS.get(logger.name)
    if current is not None and os.path.dirname(current.baseFilename) == log_dir:
        return log_dir
    os.makedirs(log_dir, exist_ok=True)

    for target, file_name, level, formatter in _LOG_FILES:
        previous = _FILE_HANDLERS.pop(target.name, None)
        if previous is not None:
            target.removeHandler(previous)
            previous.close()
        handler = logging.FileHandler(os.path.join(log_dir, file_name), "a")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        target.addHandler(handler)
        _FILE_HANDLERS[target.name] = handler

    logger.info(f"---- HydroBayesCal session of {datetime.now():%Y-%m-%d %H:%M:%S}, log files in {log_dir} ----")
    return log_dir
