#!/usr/bin/env python
"""
Example script demonstrating the prediction pipeline for long audio files.

This script shows how to:
1. Load a trained binary classifier model
2. Set up the prediction pipeline
3. Process a long audio file and extract timestamps where anemonefish sounds are detected
4. Visualize and save the results

Author: Anemonefish Acoustics Team
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import from anemonefish_acoustics package
from anemonefish_acoustics.data_processing import PredictionPipeline
from anemonefish_acoustics.models import CNNAnemonefishBinaryClassifier

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Default directories (adjust these paths to match your setup)
DATA_DIR = 'data'
MODELS_DIR = os.path.join(DATA_DIR, 'models')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio_samples')
RESULTS_DIR = os.path.join(DATA_DIR, 'prediction_results')


def setup_model(model_path=None):
    """
    Set up and load a trained binary classifier model.
    
    Parameters
    ----------
    model_path : str, optional
        Path to saved model weights, by default None
        
    Returns
    -------
    CNNAnemonefishBinaryClassifier
        Loaded model
    """
    # Create model with default architecture
    model = CNNAnemonefishBinaryClassifier(
        input_channels=1,  # Single channel for audio
        freq_bins=64,  # Number of frequency bins in mel spectrogram
        conv_channels=(16, 32, 64),  # Channels in each conv layer
        fc_sizes=(128,)  # Size of fully connected layer
    )
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Warning: No model weights provided. Using untrained model.")
    
    return model


def main():
    """Run the prediction pipeline example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Anemonefish prediction pipeline example")
    parser.add_argument("--model", type=str, help="Path to saved model weights")
    parser.add_argument("--audio", type=str, help="Path to audio file for prediction")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--window", type=float, default=0.6, help="Window length in seconds")
    parser.add_argument("--hop", type=float, default=0.2, help="Hop length in seconds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for merging")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    
    args = parser.parse_args()
    
    # Setup directories and paths
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get model path
    model_path = args.model
    if not model_path:
        model_path = os.path.join(MODELS_DIR, "binary_classifier.pt")
        print(f"No model path provided, using default: {model_path}")
    
    # Get audio path
    audio_path = args.audio
    if not audio_path:
        # Try to find some audio files
        audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
        if audio_files:
            audio_path = os.path.join(AUDIO_DIR, audio_files[0])
            print(f"No audio path provided, using first found: {audio_path}")
        else:
            print("Error: No audio path provided and no audio files found in default directory.")
            return
    
    # Get output path
    output_dir = args.output
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"prediction_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"No output path provided, using: {output_dir}")
    
    # Setup model
    model = setup_model(model_path)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup prediction pipeline
    pipeline = PredictionPipeline(
        model=model,
        window_length_sec=args.window,
        hop_length_sec=args.hop,
        threshold=args.threshold,
        iou_threshold=args.iou
    )
    
    print("\n===== Anemonefish Sound Detection on Long Audio Files =====\n")
    print(f"Processing file: {audio_path}")
    print(f"Window length: {args.window} seconds")
    print(f"Hop length: {args.hop} seconds")
    print(f"Prediction threshold: {args.threshold}")
    print(f"IoU threshold: {args.iou}")
    
    # Run prediction
    timestamps = pipeline.predict_file(audio_path, visualize=args.visualize)
    
    # Print results
    print("\nDetection results:")
    for i, (start, end) in enumerate(timestamps):
        duration = end - start
        print(f"  {i+1}. From {start:.2f}s to {end:.2f}s (duration: {duration:.2f}s)")
    
    # Save results to CSV
    output_csv = os.path.join(output_dir, f"detections_{Path(audio_path).stem}_{timestamp}.csv")
    pipeline.save_predictions_to_csv(timestamps, output_csv)
    
    # Create a simple plot of detections and save it
    if timestamps:
        plt.figure(figsize=(15, 5))
        plt.title(f"Anemonefish Sound Detections - {Path(audio_path).name}")
        plt.xlabel("Time (seconds)")
        
        # Plot detections as horizontal bars
        for i, (start, end) in enumerate(timestamps):
            plt.barh(y=0, width=end-start, left=start, height=0.5, color='green', alpha=0.6)
            plt.text(start + (end-start)/2, 0, f"{i+1}", ha='center', va='center')
        
        plt.yticks([])
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save the plot
        output_plot = os.path.join(output_dir, f"detections_plot_{Path(audio_path).stem}_{timestamp}.png")
        plt.savefig(output_plot)
        print(f"\nSaved detection plot to {output_plot}")
        
        if args.visualize:
            plt.show()
    
    print("\n===== Prediction completed successfully =====")


if __name__ == "__main__":
    main() 