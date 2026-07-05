import logging
import sys

def setup_logger(name="lema", level=logging.INFO):
    """Sets up a centralized logger for the LEMA framework."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler with a specific format
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
    return logger

# Default framework-wide logger
logger = setup_logger()
