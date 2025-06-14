"""
Logging utility for anemonefish acoustic analysis.

This module provides a consistent logging interface with customizable
formatting, file output, and log levels.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import inspect


def get_logger(name=None, level=logging.INFO, workspace_root=None):
    """
    Get a configured logger that automatically determines log file location based on the calling script.
    
    Args:
        name (str, optional): Logger name. If None, uses the calling script's name.
        level (int): Logging level (default: logging.INFO)
        workspace_root (str, optional): Path to workspace root. If None, attempts to auto-detect.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get the calling script's information
    frame = inspect.currentframe()
    try:
        # Go up the call stack to find the actual calling script
        caller_frame = frame.f_back
        while caller_frame:
            caller_filename = caller_frame.f_code.co_filename
            # Skip if it's this logger module itself
            if not caller_filename.endswith('logger.py'):
                break
            caller_frame = caller_frame.f_back
        
        if caller_frame is None:
            caller_filename = __file__
    finally:
        del frame
    
    # Determine workspace root if not provided
    if workspace_root is None:
        # Try to find workspace root by looking for common indicators
        current_path = Path(caller_filename).resolve()
        workspace_root = None
        
        # Look for workspace indicators going up the directory tree
        for parent in current_path.parents:
            if (parent / 'setup.py').exists() or (parent / 'requirements.txt').exists():
                workspace_root = str(parent)
                break
        
        # Fallback: use current working directory
        if workspace_root is None:
            workspace_root = os.getcwd()
    
    workspace_root = Path(workspace_root)
    caller_path = Path(caller_filename).resolve()
    
    # Determine script name and relative path from workspace
    try:
        relative_path = caller_path.relative_to(workspace_root)
        script_name = caller_path.stem  # filename without extension
        
        # Determine log subdirectory based on script location
        if 'scripts' in relative_path.parts:
            log_subdir = 'logs/scripts'
        elif 'notebooks' in relative_path.parts or caller_filename.endswith('.ipynb'):
            log_subdir = 'logs/notebooks'
        elif 'src' in relative_path.parts:
            log_subdir = 'logs/src'
        else:
            # Check if this looks like a Jupyter notebook (numeric script name)
            if script_name.isdigit():
                log_subdir = 'logs/notebooks'
            else:
                log_subdir = 'logs/other'
            
    except ValueError:
        # If we can't determine relative path, use fallback
        script_name = Path(caller_filename).stem
        # Check if this looks like a Jupyter notebook
        if script_name.isdigit():
            log_subdir = 'logs/notebooks'
        else:
            log_subdir = 'logs/other'
    
    # Use provided name or default to script name
    logger_name = name or script_name
    
    # Create log directory
    log_dir = workspace_root / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename - use name parameter if provided, otherwise script name
    if name:
        log_filename = f"{name}.log"
    else:
        log_filename = f"{script_name}.log"
    log_filepath = log_dir / log_filename
    
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the setup info
    logger.info(f"Logger initialized for {logger_name}")
    logger.info(f"Log file: {log_filepath}")
    
    return logger


def setup_basic_logging(level=logging.INFO):
    """
    Setup basic logging configuration as a fallback.
    
    Args:
        level (int): Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Convenience function to get a logger with default settings
def get_default_logger():
    """Get a logger with default settings for the calling script."""
    return get_logger()

