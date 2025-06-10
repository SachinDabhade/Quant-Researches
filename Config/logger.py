import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def get_logger(name):
    """
    Get a logger instance with the specified name.
    
    Parameters:
    - name: The name of the logger (usually __name__)
    
    Returns:
    - Logger instance
    """
    # Create a logger
    logger = logging.getLogger(name)
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Check if logger already has handlers to avoid duplicates
    if not logger.handlers:
        # Create a file handler
        log_file = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger