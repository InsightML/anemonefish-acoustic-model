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

import json
import traceback
from collections import deque


class ErrorSummaryHandler(logging.Handler):
    """Custom handler that captures ERROR/CRITICAL messages to a JSON summary file."""
    
    def __init__(self, error_summary_path, max_errors=5, context_lines=20):
        super().__init__()
        self.error_summary_path = Path(error_summary_path)
        self.max_errors = max_errors
        self.context_buffer = deque(maxlen=context_lines)
        self.error_summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty summary file if it doesn't exist
        if not self.error_summary_path.exists():
            self._write_summary({"errors": []})
    
    def emit(self, record):
        """Capture ERROR/CRITICAL messages and update error summary."""
        if record.levelno >= logging.ERROR:
            try:
                # Format the error entry
                error_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                    "level": record.levelname,
                    "logger_name": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line_number": record.lineno,
                    "context_lines": list(self.context_buffer),
                }
                
                # Add traceback if available
                if record.exc_info:
                    error_entry["traceback"] = traceback.format_exception(*record.exc_info)
                
                # Read existing summary
                try:
                    with open(self.error_summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    summary = {"errors": []}
                
                # Add new error and keep only the last max_errors
                summary["errors"].insert(0, error_entry)
                summary["errors"] = summary["errors"][:self.max_errors]
                summary["latest_error"] = error_entry
                
                # Write updated summary
                self._write_summary(summary)
                
            except Exception:
                # Don't let logging errors break the application
                pass
    
    def _write_summary(self, summary):
        """Write summary to JSON file."""
        with open(self.error_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def add_context_line(self, line):
        """Add a line to the context buffer."""
        self.context_buffer.append(line)


class ContextCapturingHandler(logging.Handler):
    """Handler that captures all log messages for context in error summaries."""
    
    def __init__(self, error_handler):
        super().__init__()
        self.error_handler = error_handler
    
    def emit(self, record):
        """Capture all log messages for context."""
        try:
            formatted_message = self.format(record)
            self.error_handler.add_context_line(formatted_message)
        except Exception:
            pass


def get_logger(name=None, level=logging.INFO, workspace_root=None):
    """
    Get a configured logger with monolith logging and error summary tracking.
    
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
    
    # Determine script name
    script_name = caller_path.stem  # filename without extension
    
    # Use provided name or default to script name
    logger_name = name or script_name
    
    # Create logs directory
    log_dir = workspace_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Monolith log file
    log_filepath = log_dir / 'main.log'
    
    # Error summary file
    error_summary_path = log_dir / 'error_summary.json'
    
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Error summary handler (must be created first)
    error_handler = ErrorSummaryHandler(error_summary_path)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    # Context capturing handler (captures all messages for error context)
    context_handler = ContextCapturingHandler(error_handler)
    context_handler.setLevel(level)
    context_handler.setFormatter(file_formatter)
    logger.addHandler(context_handler)
    
    # File handler (monolith log)
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
    logger.info(f"Monolith log file: {log_filepath}")
    logger.info(f"Error summary file: {error_summary_path}")
    
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

