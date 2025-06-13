#!/usr/bin/env python
"""
Script to augment anemonefish acoustic data for training models.

This script takes anemonefish call recordings and creates augmented versions
using various acoustic transformations. The augmented data can be used to
increase training dataset size and improve model generalization.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import sys

from anemonefish_acoustics.data_processing.data_preprocessing import DatasetBuilder
from anemonefish_acoustics.data_processing.data_augmentation import AudioAugmenter, DataAugmentationPipeline

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Augment anemonefish acoustic data for training models"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="data/processed_wavs",
        help="Directory containing processed anemonefish calls (default: data/processed_wavs)"
    )
    
    parser.add_argument(
        "--noise-dir", 
        type=str, 
        default="data/noise_chunked",
        help="Directory containing noise files (default: data/noise_chunked)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/augmented_wavs",
        help="Directory to save augmented files (default: data/augmented_wavs)"
    )
    
    parser.add_argument(
        "--augmentation-factor", 
        type=int, 
        default=5,
        help="Number of augmented versions to create per original file (default: 5)"
    )
    
    parser.add_argument(
        "--no-noise-addition", 
        action="store_true",
        help="Disable noise addition augmentation"
    )
    
    parser.add_argument(
        "--visualize-examples", 
        action="store_true",
        help="Visualize some augmentation examples"
    )
    
    parser.add_argument(
        "--sample-rate", 
        type=int, 
        default=8000,
        help="Audio sample rate in Hz (default: 8000)"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()

def visualize_augmentations(original_file, sample_rate, output_dir):
    """Visualize different augmentation types on a sample file."""
    # Create augmenter
    augmenter = AudioAugmenter(sr=sample_rate, seed=42)
    
    # Load sample file
    audio, sr = librosa.load(original_file, sr=sample_rate)
    
    # Define augmentations to try
    augmentation_configs = [
        ("Time Stretch", lambda a: augmenter.time_stretch(a.copy())),
        ("Pitch Shift", lambda a: augmenter.pitch_shift(a.copy())),
        ("Time Shift", lambda a: augmenter.time_shift(a.copy())),
        ("Volume Perturbation", lambda a: augmenter.volume_perturbation(a.copy())),
        ("Frequency Mask", lambda a: augmenter.frequency_mask(a.copy(), fmin=100, fmax=1000)),  # Use safer values
        ("Time Mask", lambda a: augmenter.time_mask(a.copy())),
        ("Multipath Simulation", lambda a: augmenter.simulate_multipath(a.copy()))
    ]
    
    # Apply each augmentation, handling errors
    augmentations = []
    for title, augment_fn in augmentation_configs:
        try:
            augmented = augment_fn(audio)
            augmentations.append((title, augmented))
        except Exception as e:
            print(f"Warning: Could not create {title} augmentation: {e}")
            # Skip this augmentation rather than using a fallback
    
    # Set up the plot - dynamically calculate the number of rows needed
    # Include 1 row for the original audio + 1 row per augmentation type
    # Two columns: waveform and spectrogram
    n_augmentations = len(augmentations)
    n_rows = n_augmentations + 1  # +1 for original audio
    
    # Create a figure with the right size
    plt.figure(figsize=(15, 3 * n_rows))
    
    # Plot original audio
    plt.subplot(n_rows, 2, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Original Audio")
    
    # Plot spectrogram of original audio
    plt.subplot(n_rows, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title("Original Spectrogram")
    
    # Plot each augmentation
    for i, (title, augmented) in enumerate(augmentations):
        # Plot waveform (i+1 because we already plotted the original)
        plt.subplot(n_rows, 2, (i+1)*2+1)
        librosa.display.waveshow(augmented, sr=sr)
        plt.title(f"{title} Waveform")
        
        # Plot spectrogram
        plt.subplot(n_rows, 2, (i+1)*2+2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(augmented)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title(f"{title} Spectrogram")
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "augmentation_examples.png"))
    plt.close()

def main():
    """Main function to run the data augmentation."""
    args = parse_args()
    
    # Create DatasetBuilder
    builder = DatasetBuilder(
        processed_wavs_dir=args.input_dir,
        noise_chunked_dir=args.noise_dir,
        augmented_dir=args.output_dir,
        sr=args.sample_rate
    )
    
    # Augment data
    print(f"Augmenting anemonefish data with factor {args.augmentation_factor}...")
    augmented_files = builder.augment_anemonefish_data(
        augmentation_factor=args.augmentation_factor,
        use_noise_addition=not args.no_noise_addition,
        random_seed=args.random_seed
    )
    
    print(f"Successfully created {len(augmented_files)} augmented files in {args.output_dir}")
    
    # Visualize examples if requested
    if args.visualize_examples:
        print("Generating visualization of augmentation examples...")
        # Use first original file for visualization
        original_files = builder.list_anemonefish_files()
        if original_files:
            visualize_augmentations(
                original_files[0], 
                args.sample_rate,
                args.output_dir
            )
            print(f"Visualization saved to {os.path.join(args.output_dir, 'augmentation_examples.png')}")
        else:
            print("No original files found for visualization")
    
    print("Done!")

if __name__ == "__main__":
    main() 