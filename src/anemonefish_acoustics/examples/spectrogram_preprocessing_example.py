#!/usr/bin/env python
"""
Example script demonstrating spectrogram-based preprocessing for anemonefish call detection.

This script shows how to:
1. Analyze audio lengths to determine optimal standardization parameters
2. Preprocess audio using both stretch/squash and crop/pad methods
3. Extract mel spectrogram features
4. Train a CNN classifier on these features
5. Evaluate and visualize the results

Author: Anemonefish Acoustics Team
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import from anemonefish_acoustics package
from anemonefish_acoustics.data_processing import DatasetBuilder
from anemonefish_acoustics.models import CNNAnemonefishBinaryClassifier, train_model, evaluate_model
from anemonefish_acoustics.models import plot_training_history, plot_roc_curve, plot_confusion_matrix

# Set random seeds for reproducibility
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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    """Run the spectrogram preprocessing example."""
    print("\n===== Anemonefish Call Detection with Spectrograms =====\n")
    
    # Create dataset builder for mel spectrograms using stretch/squash method
    print("Initializing DatasetBuilder with mel spectrograms (stretch/squash method)...")
    dataset_builder_stretch = DatasetBuilder(
        processed_wavs_dir=PROCESSED_WAVS_DIR,
        noise_dir=NOISE_DIR,
        noise_chunked_dir=NOISE_CHUNKED_DIR,
        feature_type='mel_spectrogram',  
        preprocess_method='stretch_squash',
        standard_length_sec=1.0
    )
    
    # Analyze audio lengths to determine optimal parameters
    print("\nAnalyzing anemonefish call lengths...")
    stats = dataset_builder_stretch.analyze_anemonefish_call_lengths()
    
    # If mean is significantly different from standard length, suggest adjusting
    if abs(stats['mean'] - dataset_builder_stretch.standard_length_sec) > 0.3:
        print(f"\nNote: Based on audio statistics, you might want to adjust standard_length_sec to {stats['mean']:.2f} seconds")
    
    # Prepare dataset with stretch/squash method
    print("\nPreparing dataset with stretch/squash method...")
    X_train_stretch, X_test_stretch, y_train_stretch, y_test_stretch = dataset_builder_stretch.prepare_dataset_with_augmentation(
        test_size=0.2,
        use_augmentation=True,
        balance_ratio=1.0,
        random_state=42
    )
    
    # Visualize features
    print("\nVisualizing mel spectrograms (stretch/squash method)...")
    dataset_builder_stretch.visualize_features(num_samples=3)
    
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
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_cnn).float(), 
        torch.from_numpy(y_train).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_cnn).float(), 
        torch.from_numpy(y_test).float()
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    # Get the number of frequency bins from the data
    input_channels = X_train_cnn.shape[1]  # Number of channels (should be 1)
    freq_bins = X_train_cnn.shape[2]       # Number of frequency bins
    
    print(f"\nInitializing CNN model with {freq_bins} frequency bins...")
    model = CNNAnemonefishBinaryClassifier(
        input_channels=input_channels,
        freq_bins=freq_bins,
        conv_channels=(16, 32, 64),
        fc_sizes=(128,)
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nTraining the model...")
    start_time = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\nEvaluating the model...")
    metrics = evaluate_model(model, test_loader, device=device)
    
    # Print metrics
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        history, 
        save_path=os.path.join(RESULTS_DIR, 'training_history_mel_spectrogram.png')
    )
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(
        model, 
        test_loader, 
        device=device,
        save_path=os.path.join(RESULTS_DIR, 'roc_curve_mel_spectrogram.png')
    )
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        model, 
        test_loader, 
        device=device,
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_mel_spectrogram.png')
    )
    
    # Save the model
    print("\nSaving the model...")
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'cnn_model_mel_spectrogram.pt'))
    
    print("\n===== Example completed successfully =====")

if __name__ == "__main__":
    main() 