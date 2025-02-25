import logging
import os

def get_logger(name):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger(name)
    
    # Only configure if handler not already added
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(os.path.join(log_dir, 'anemonefish_model.log'))
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(fh)
    
    return logger