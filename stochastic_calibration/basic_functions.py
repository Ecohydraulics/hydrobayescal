"""fundamental Python functions"""
from action_logger import Logger


def str2seq(list_like_string, separator=",", return_type="tuple"):
    """Convert a list-like string into a tuple or list based on a separator such as comma or semi-column

    :param str list_like_string: string to convert
    :param str separator: separator to use
    :param str return_type: defines if a list or tuple is returned (default: tuple)
    :return: list or tuple
    """
    seq = []
    for number in list_like_string.split(separator):
        try:
            seq.append(float(number))
        except ValueError:
            print("WARNING: Could not interpret user parameter value range definition (%s)" % number)
            print("         This Warning will probably cause an ERROR later in the script.")
    if "tuple" in return_type:
        return tuple(seq)
    else:
        return seq


def log_actions(func):
    def wrapper(*args, **kwargs):
        logger = Logger("logfile")
        func(*args, **kwargs)
        logger.logging_stop()
    return wrapper
