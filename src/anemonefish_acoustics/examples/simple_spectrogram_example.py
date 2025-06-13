#!/usr/bin/env python
"""
Simple example script demonstrating spectrogram-based preprocessing for anemonefish call detection.

This script shows how to:
1. Analyze audio lengths to determine optimal standardization parameters
2. Preprocess audio using the stretch/squash method
3. Extract mel spectrogram features
4. Prepare the data for a CNN model

Author: Anemonefish Acoustics Team
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import from anemonefish_acoustics package
from anemonefish_acoustics.data_processing import DatasetBuilder
from anemonefish_acoustics.data_processing import PREPROCESS_STRETCH_SQUASH, PREPROCESS_CROP_PAD

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set directories (adjust these paths to match your setup)
DATA_DIR = 'data'
PROCESSED_WAVS_DIR = os.path.join(DATA_DIR, 'processed_wavs')
NOISE_DIR = os.path.join(DATA_DIR, 'noise')
NOISE_CHUNKED_DIR = os.path.join(DATA_DIR, 'noise_chunked')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    """Run the simple spectrogram preprocessing example."""
    print("\n===== Anemonefish Call Detection with Spectrograms =====\n")
    
    # Part 1: Stretch/Squash Method
    print("\n----- Stretch/Squash Method -----\n")
    
    # Create dataset builder for mel spectrograms using stretch/squash method
    print("Initializing DatasetBuilder with mel spectrograms (stretch/squash method)...")
    dataset_builder_stretch = DatasetBuilder(
        processed_wavs_dir=PROCESSED_WAVS_DIR,
        noise_dir=NOISE_DIR,
        noise_chunked_dir=NOISE_CHUNKED_DIR,
        feature_type='mel_spectrogram',  
        preprocess_method=PREPROCESS_STRETCH_SQUASH,
        standard_length_sec=1.0
    )
    
    # Analyze audio lengths to determine optimal parameters
    print("\nAnalyzing anemonefish call lengths...")
    stats = dataset_builder_stretch.analyze_anemonefish_call_lengths()
    
    # If mean is significantly different from standard length, suggest adjusting
    if abs(stats['mean'] - dataset_builder_stretch.standard_length_sec) > 0.3:
        print(f"\nNote: Based on audio statistics, you might want to adjust standard_length_sec to {stats['mean']:.2f} seconds")
    
    # Prepare dataset with stretch/squash method - use a small subset for demonstration
    print("\nPreparing dataset with stretch/squash method...")
    X_train_stretch, X_test_stretch, y_train_stretch, y_test_stretch = dataset_builder_stretch.prepare_dataset_with_augmentation(
        test_size=0.2,
        use_augmentation=False,  # Set to False for faster processing
        balance_ratio=1.0,
        random_state=42
    )
    
    # Visualize features
    print("\nVisualizing mel spectrograms (stretch/squash method)...")
    dataset_builder_stretch.visualize_features(num_samples=2)
    
    # Prepare data for CNN model
    print("\nPreparing data for CNN model...")
    X_train_cnn, X_test_cnn, y_train, y_test = dataset_builder_stretch.prepare_data_for_model(
        X_train_stretch, X_test_stretch, y_train_stretch, y_test_stretch, 
        model_type='cnn'
    )
    
    # Print shapes
    print(f"\nTraining data shape: {X_train_cnn.shape}")
    print(f"Testing data shape: {X_test_cnn.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    
    # Part 2: Crop/Pad Method
    print("\n----- Crop/Pad Method -----\n")
    
    # Create dataset builder for mel spectrograms using crop/pad method
    print("Initializing DatasetBuilder with mel spectrograms (crop/pad method)...")
    dataset_builder_crop = DatasetBuilder(
        processed_wavs_dir=PROCESSED_WAVS_DIR,
        noise_dir=NOISE_DIR,
        noise_chunked_dir=NOISE_CHUNKED_DIR,
        feature_type='mel_spectrogram',  
        preprocess_method=PREPROCESS_CROP_PAD,
        standard_length_sec=1.0
    )
    
    # Prepare dataset with crop/pad method - use a small subset for demonstration
    print("\nPreparing dataset with crop/pad method...")
    print("Note: This method may generate more training examples from longer recordings")
    X_train_crop, X_test_crop, y_train_crop, y_test_crop = dataset_builder_crop.prepare_dataset_with_augmentation(
        test_size=0.2,
        use_augmentation=False,  # Set to False for faster processing
        balance_ratio=1.0,
        random_state=42
    )
    
    # Visualize features
    print("\nVisualizing mel spectrograms (crop/pad method)...")
    dataset_builder_crop.visualize_features(num_samples=2)
    
    # Prepare data for CNN model
    print("\nPreparing data for CNN model...")
    X_train_cnn_crop, X_test_cnn_crop, y_train_crop, y_test_crop = dataset_builder_crop.prepare_data_for_model(
        X_train_crop, X_test_crop, y_train_crop, y_test_crop,
        model_type='cnn'
    )
    
    # Print shapes - may have more examples than stretch/squash method
    print(f"\nTraining data shape: {X_train_cnn_crop.shape}")
    print(f"Testing data shape: {X_test_cnn_crop.shape}")
    print(f"Training labels shape: {y_train_crop.shape}")
    print(f"Testing labels shape: {y_test_crop.shape}")
    
    print("""
===== Comparison: Crop/Pad Method vs Stretch/Squash Method =====

For short anemonefish calls:
- Crop/Pad: Pads with zeros to reach standard length
- Stretch/Squash: Stretches audio to standard length

For long anemonefish calls:
- Crop/Pad: Creates multiple overlapping segments
- Stretch/Squash: Compresses audio to standard length

Choose the approach that works best for your dataset!
    """)
    
    print("\n===== Example completed successfully =====")

if __name__ == "__main__":
    main() 