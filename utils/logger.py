import logging
import sys
from pathlib import Path

# Setup logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name="thyroid_app"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Consistent format: Time - Level - Module - Message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s', datefmt='%H:%M:%S')

        # Console Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # File Handler
        fh = logging.FileHandler(LOG_DIR / "app.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

# Global default logger
logger = setup_logger()