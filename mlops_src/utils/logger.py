# mlops_src/utils/logger.py
import logging, os
from datetime import datetime

# One master log file per CI / local run
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
MASTER_LOG_FILE = os.path.join(LOG_DIR, f"run_{RUN_ID}.log")

def get_logger(name: str, log_file: str):
    """
    Returns a logger writing to:
      1. its own file  (module-specific)
      2. master run log
      3. stdout  (for CI visibility)
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:  # avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # File for this module
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)


    # Console output (CI friendly)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
