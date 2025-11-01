"""
Configuration utilities for anemonefish acoustics.

This module provides functionality for loading, validating, and working with
YAML configuration files for training and evaluation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save
    output_path : str
        Path to save the configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)