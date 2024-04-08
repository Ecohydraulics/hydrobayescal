import logging

def start_logging():
    logging.basicConfig(filename="logfile.log", format="[%(asctime)s] %(message)s",
                        filemode="w", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

def log_actions(func):
    def wrapper(*args, **kwargs):
        start_logging()
        result=func(*args, **kwargs)
        logging.shutdown()
        return result
    return wrapper
