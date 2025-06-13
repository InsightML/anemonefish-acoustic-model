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


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration dictionary
    override_config : Dict[str, Any]
        Override configuration dictionary
    
    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    def recursive_merge(d1, d2):
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                recursive_merge(d1[k], v)
            else:
                d1[k] = v
    
    recursive_merge(merged_config, override_config)
    return merged_config


def create_default_binary_classifier_config() -> Dict[str, Any]:
    """
    Create a default configuration for the binary classifier.
    
    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary
    """
    config = {
        # Data settings
        'data': {
            'processed_wavs_dir': 'data/processed_wavs',
            'noise_dir': 'data/noise',
            'noise_chunked_dir': 'data/noise_chunked',
            'cache_dir': 'data/cache',
            'augmented_dir': 'data/augmented_wavs',
            'sample_rate': 8000,
            'test_size': 0.2,
            'validation_size': 0.15,
            'random_state': 42,
            'balance_ratio': 1.0
        },
        
        # Preprocessing settings
        'preprocessing': {
            'feature_type': 'spectrogram',  # Options: 'mfcc', 'spectral_contrast', 'spectrogram'
            'n_mfcc': 14,
            'n_fft': 2048,
            'hop_length': 512,
            'fmin': 0.0,
            'fmax': 2000.0,
            'frame_length': 2048
        },
        
        # Augmentation settings
        'augmentation': {
            'use_augmentation': True,
            'augmentation_factor': 3,
            'use_noise_addition': True,
            'time_stretch': {
                'enabled': True,
                'rate_range': [0.8, 1.2]
            },
            'pitch_shift': {
                'enabled': True,
                'n_steps_range': [-3.0, 3.0]
            },
            'time_shift': {
                'enabled': True,
                'shift_range': [-0.25, 0.25]
            },
            'volume_perturbation': {
                'enabled': True,
                'gain_range': [0.7, 1.3]
            },
            'frequency_mask': {
                'enabled': True,
                'max_mask_width': 50,
                'n_masks': 1,
                'fmin': 75,
                'fmax': 1800
            },
            'time_mask': {
                'enabled': True,
                'mask_ratio_range': [0.05, 0.15],
                'n_masks': 1
            },
            'simulate_multipath': {
                'enabled': True,
                'n_reflections_range': [1, 3],
                'delay_range': [0.005, 0.02],
                'decay_factor_range': [0.2, 0.5]
            }
        },
        
        # Model architecture settings
        'model': {
            'input_channels': 1,
            'freq_bins': 64,
            'conv_channels': [16, 32, 64],
            'fc_sizes': [128]
        },
        
        # Training settings
        'training': {
            'batch_size': 32,
            'num_epochs': 30,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'early_stopping_patience': 5,
            'device': 'cuda'  # Options: 'cuda', 'cpu'
        },
        
        # Logging and checkpoint settings
        'output': {
            'experiment_name': 'binary_classifier',
            'log_dir': 'logs/experiments',
            'checkpoints_dir': 'checkpoints',
            'save_best_only': True,
            'visualize_data': True,
            'visualize_model': True,
            'log_level': 'INFO'
        },
        
        # MLflow settings
        'mlflow': {
            'use_mlflow': False,
            'tracking_uri': 'mlruns',
            'experiment_name': 'binary_classifier'
        }
    }
    
    return config


def write_default_config(output_path: str, overwrite: bool = False) -> None:
    """
    Write the default binary classifier configuration to a file.
    
    Parameters
    ----------
    output_path : str
        Path to write the configuration file
    overwrite : bool, optional
        Whether to overwrite an existing file, by default False
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"Config file already exists at {output_path}. Use overwrite=True to replace it.")
        return
    
    config = create_default_binary_classifier_config()
    save_config(config, output_path)
    print(f"Default configuration written to {output_path}") 