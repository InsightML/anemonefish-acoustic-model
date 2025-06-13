"""
Anemonefish Acoustics Models

This package contains models for detecting and classifying anemonefish sounds.
"""

from .binary_classifier import CNNAnemonefishBinaryClassifier
from .crnn_classifier import (
    CRNNAnemonefishBinaryClassifier,
    AttentionLayer,
    train_model,
    evaluate_model,
    save_model,
    load_model
)

__all__ = [
    'CNNAnemonefishBinaryClassifier',
    'CRNNAnemonefishBinaryClassifier',
    'AttentionLayer',
    'train_model',
    'evaluate_model',
    'save_model',
    'load_model'
]
