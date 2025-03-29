import logging
import sys


def setup_logging(log_file: str = 'simulation.log', log_level: int = logging.INFO, console: bool = False) -> logging.Logger:
    """
    Set up a dedicated logger for simulation with file logging and optional console logging.

    Parameters:
        log_file (str): The file to which logs will be written.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        console (bool): If True, also log messages to the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger('simulation')
    # Check if the logger already has handlers to avoid duplicate logging.
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.info("Logger configured. Logging to file: %s", log_file)
    return logger
