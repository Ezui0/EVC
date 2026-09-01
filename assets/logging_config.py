"""Logging configuration setup for various libraries and modules."""

import logging
import os
import warnings


def configure_logging(enable_configure_logging=True, global_logger=False, logging_level="WARNING"):
    """
    This function sets logging levels for various libraries and modules
    to reduce the amount of printed messages and improve log readability.

    Parameters:
    - enable_configure_logging (bool, optional):
        Master switch for the entire logging configuration.
        If set to False, the function will not perform any actions.
      Default: True

    - global_logger (bool, optional):
        Flag to enable or disable the global logger.
        If True, configures the logging level for all loggers.
        If False, configures the logging level only for the listed libraries.
      Default: False

    - logging_level (str, optional):
        Custom logging level.
        Must be one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        If an invalid value is given, the default value "WARNING" is used.
      Default: "WARNING"

    Logging levels:
    - 0 | DEBUG: Detailed information, usually of interest only when debugging problems.
    - 1 | INFO: Confirmation that everything works as expected.
    - 2 | WARNING: An indication that something unexpected happened, or that a problem
               may occur in the near future (e.g. 'disk space running low').
               The software is still working as expected.
    - 3 | ERROR: Due to a more serious problem, the software has been unable to
             perform some functions.
    - 4 | CRITICAL: Indicates that the program itself may be unable to continue running.

    In this case we set the logging level to WARNING for all libraries and modules,
    so that DEBUG and INFO level messages are ignored.
    """

    if enable_configure_logging:
        # ===== Set environment variables for dependencies ===== #
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

        # ===== Handle system warnings ===== #
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Parse the logging level from the string
        level = getattr(logging, logging_level, logging.WARNING)

        # ===== Configure third-party library loggers ===== #
        if global_logger:
            logging.basicConfig(level=level)
        else:
            logging.getLogger("pydub").setLevel(level)
            logging.getLogger("numba").setLevel(level)
            logging.getLogger("faiss").setLevel(level)
            logging.getLogger("torio").setLevel(level)
            logging.getLogger("httpx").setLevel(level)
            logging.getLogger("urllib3").setLevel(level)
            logging.getLogger("fairseq").setLevel(level)
            logging.getLogger("asyncio").setLevel(level)
            logging.getLogger("httpcore").setLevel(level)
            logging.getLogger("matplotlib").setLevel(level)
            logging.getLogger("onnx2torch").setLevel(level)
            logging.getLogger("python_multipart").setLevel(level)


"""
Example of using the configure_logging function in the main file:

1. With full parameters:
from logging_config import configure_logging
configure_logging(enable_configure_logging=True, global_logger=False, logging_level="DEBUG")

2. With shortened parameters (using default values for named arguments):
from logging_config import configure_logging
configure_logging(True, False, "DEBUG")

3. With default parameters (if no special configuration is required):
from logging_config import configure_logging
configure_logging()
"""
