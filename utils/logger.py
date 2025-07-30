import logging
import os

def setup_logger(log_file: str, logger_name: str = None):
    """
    Sets up a logger that logs to both a file and the console.
    Args:
        log_file (str): Path to the log file.
        logger_name (str, optional): Name of the logger. If None, uses root logger.
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers if already set up
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger