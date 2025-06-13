"""
Utility modules for the anemonefish acoustics package.
"""

from .logger import get_logger, setup_basic_logging, get_default_logger
from .config import (
    load_config, 
    save_config, 
    merge_configs, 
    create_default_binary_classifier_config,
    write_default_config
)
from .visualization import (
    plot_training_history,
    plot_audio_waveform,
    plot_spectrogram,
    plot_feature_comparison,
    plot_confusion_matrix,
    plot_prediction_samples
)

__all__ = [
    # Logger utilities
    'setup_logger', 
    'TrainingLogger',
    
    # Config utilities
    'load_config',
    'save_config',
    'merge_configs',
    'create_default_binary_classifier_config',
    'write_default_config',
    
    # Visualization utilities
    'plot_training_history',
    'plot_audio_waveform',
    'plot_spectrogram',
    'plot_feature_comparison',
    'plot_confusion_matrix',
    'plot_prediction_samples'
]
