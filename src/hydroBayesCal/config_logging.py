"""
Setup logging
"""
import os
import logging

def setup_logging():
    # directories
    script_dir = os.getcwd()

    # Set up logging
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger("HydroBayesCal")
    logger.setLevel(logging.INFO)

    # Check if handlers are already added to the logger
    if not logger.handlers:
        # Create console handler and set level to INFO
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Create file handler for all levels and set mode to append
        file_handler = logging.FileHandler(os.path.join(script_dir, "logfile.log"), mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)