"""
Prediction pipeline for anemonefish sound detection.

This module provides functionality to process audio files of arbitrary length,
make predictions using a trained binary classifier model, and generate
timestamps for detected anemonefish sounds.
"""

import os
import numpy as np
import torch
import librosa
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Import from the same package
from .data_preprocessing import AudioProcessor, SAMPLE_RATE, STANDARD_LENGTH_SEC, SLIDING_WINDOW_HOP
from .data_preprocessing import PREPROCESS_CROP_PAD, PREPROCESS_STRETCH_SQUASH


class PredictionPipeline:
    """
    Pipeline for predicting anemonefish sounds in audio files of arbitrary length.
    
    This class handles loading audio, breaking it into overlapping windows,
    making predictions with a trained model, and merging the overlapping
    predictions to provide timestamps of detected sounds.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 sample_rate: int = SAMPLE_RATE,
                 window_length_sec: float = STANDARD_LENGTH_SEC,
                 hop_length_sec: float = SLIDING_WINDOW_HOP,
                 threshold: float = 0.5,
                 iou_threshold: float = 0.25,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the prediction pipeline.
        
        Parameters
        ----------
        model : torch.nn.Module
            Trained binary classifier model
        sample_rate : int, optional
            Sample rate for audio processing, by default SAMPLE_RATE
        window_length_sec : float, optional
            Length of each window in seconds, by default STANDARD_LENGTH_SEC
        hop_length_sec : float, optional
            Hop length between windows in seconds, by default SLIDING_WINDOW_HOP
        threshold : float, optional
            Probability threshold for positive prediction, by default 0.5
        iou_threshold : float, optional
            Threshold for merging overlapping predictions, by default 0.25
        device : str, optional
            Device to run model on, by default 'cuda' if available else 'cpu'
        """
        self.model = model
        self.model.to(device)
        self.model.eval()
        
        self.sr = sample_rate
        self.window_length_sec = window_length_sec
        self.hop_length_sec = hop_length_sec
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Convert to samples
        self.window_length_samples = int(self.window_length_sec * self.sr)
        self.hop_length_samples = int(self.hop_length_sec * self.sr)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        Parameters
        ----------
        file_path : str
            Path to audio file
            
        Returns
        -------
        Tuple[np.ndarray, int]
            Audio data and sample rate
        """
        return AudioProcessor.load_audio(file_path, self.sr)
    
    def create_overlapping_windows(self, audio_data: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Create overlapping windows from audio data.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data
            
        Returns
        -------
        List[Tuple[np.ndarray, int]]
            List of tuples containing (audio segment, start sample index)
        """
        segments = []
        
        # Create overlapping windows
        for start_sample in range(0, len(audio_data) - self.window_length_samples + 1, self.hop_length_samples):
            segment = audio_data[start_sample:start_sample + self.window_length_samples]
            
            # If segment is shorter than window length, pad it
            if len(segment) < self.window_length_samples:
                segment = np.pad(
                    segment, 
                    (0, self.window_length_samples - len(segment)), 
                    mode='constant'
                )
            
            segments.append((segment, start_sample))
        
        # Handle case where audio is shorter than window length
        if len(segments) == 0:
            padded_audio = np.pad(
                audio_data, 
                (0, self.window_length_samples - len(audio_data)), 
                mode='constant'
            )
            segments.append((padded_audio, 0))
        
        return segments
    
    def preprocess_segment(self, segment: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio segment for model input.
        
        Parameters
        ----------
        segment : np.ndarray
            Audio segment
            
        Returns
        -------
        torch.Tensor
            Preprocessed audio as model input
        """
        # Normalize audio
        segment = AudioProcessor.normalize_audio(segment)
        
        # Extract features (use the same processing as during training)
        mel_spec = AudioProcessor.extract_mel_spectrogram(segment, self.sr)
        
        # Convert to log mel spectrogram
        log_mel_spec = AudioProcessor.log_mel_spectrogram(mel_spec)
        
        # Add channel dimension for CNN input (batch_size, channels, height, width)
        x = torch.tensor(log_mel_spec).unsqueeze(0).unsqueeze(0).float()
        
        return x.to(self.device)
    
    def predict_segment(self, segment: np.ndarray) -> float:
        """
        Make prediction for a single audio segment.
        
        Parameters
        ----------
        segment : np.ndarray
            Audio segment
            
        Returns
        -------
        float
            Prediction probability (0-1)
        """
        # Preprocess segment
        x = self.preprocess_segment(segment)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(x)
        
        # Return as float
        return prediction.item()
    
    def calculate_iou(self, box1: Tuple[int, int], box2: Tuple[int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) for two time intervals.
        
        Parameters
        ----------
        box1 : Tuple[int, int]
            First interval (start, end)
        box2 : Tuple[int, int]
            Second interval (start, end)
            
        Returns
        -------
        float
            IoU value (0-1)
        """
        # Get coordinates
        x1_1, x2_1 = box1
        x1_2, x2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        x2_i = min(x2_1, x2_2)
        
        if x2_i <= x1_i:
            return 0.0
        
        intersection = x2_i - x1_i
        
        # Calculate union
        union = (x2_1 - x1_1) + (x2_2 - x1_2) - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        return iou
    
    def merge_overlapping_predictions(self, detections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping predictions using IoU.
        
        Parameters
        ----------
        detections : List[Tuple[int, int]]
            List of detection intervals (start, end)
            
        Returns
        -------
        List[Tuple[int, int]]
            List of merged intervals
        """
        if not detections:
            return []
        
        # Sort by start time
        detections.sort(key=lambda x: x[0])
        
        merged = [detections[0]]
        
        for current in detections[1:]:
            previous = merged[-1]
            
            # Calculate IoU
            iou = self.calculate_iou(previous, current)
            
            # If IoU is above threshold, merge them
            if iou >= self.iou_threshold:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def samples_to_timestamps(self, intervals: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Convert sample intervals to time intervals.
        
        Parameters
        ----------
        intervals : List[Tuple[int, int]]
            List of sample intervals (start, end)
            
        Returns
        -------
        List[Tuple[float, float]]
            List of time intervals (start, end) in seconds
        """
        return [(start / self.sr, end / self.sr) for start, end in intervals]
    
    def predict_file(self, file_path: str, visualize: bool = False) -> List[Tuple[float, float]]:
        """
        Process an audio file and return timestamps of detected anemonefish sounds.
        
        Parameters
        ----------
        file_path : str
            Path to audio file
        visualize : bool, optional
            Whether to visualize the predictions, by default False
            
        Returns
        -------
        List[Tuple[float, float]]
            List of time intervals (start, end) in seconds
        """
        print(f"Processing file: {file_path}")
        
        # Load audio file
        audio_data, sr = self.load_audio(file_path)
        
        # Create overlapping windows
        print("Creating overlapping windows...")
        window_segments = self.create_overlapping_windows(audio_data)
        
        # Process each window
        print("Making predictions on windows...")
        predictions = []
        
        for segment, start_sample in tqdm(window_segments):
            # Get prediction
            prob = self.predict_segment(segment)
            
            # Store prediction and window info
            predictions.append({
                'start_sample': start_sample,
                'end_sample': start_sample + self.window_length_samples,
                'probability': prob,
                'predicted_class': 1 if prob >= self.threshold else 0
            })
        
        # Convert to DataFrame for easier analysis
        predictions_df = pd.DataFrame(predictions)
        
        # Extract positive predictions
        positive_predictions = predictions_df[predictions_df['predicted_class'] == 1]
        
        # Extract start and end samples for positives
        detections = [(row['start_sample'], row['end_sample']) 
                     for _, row in positive_predictions.iterrows()]
        
        # Merge overlapping predictions
        print("Merging overlapping predictions...")
        merged_detections = self.merge_overlapping_predictions(detections)
        
        # Convert to timestamps
        timestamps = self.samples_to_timestamps(merged_detections)
        
        # Visualize if requested
        if visualize:
            self.visualize_predictions(audio_data, predictions_df, merged_detections)
        
        return timestamps
    
    def visualize_predictions(self, audio_data: np.ndarray, 
                            predictions_df: pd.DataFrame,
                            merged_detections: List[Tuple[int, int]]) -> None:
        """
        Visualize the audio waveform, window predictions, and merged detections.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data
        predictions_df : pd.DataFrame
            DataFrame containing window predictions
        merged_detections : List[Tuple[int, int]]
            List of merged detection intervals
        """
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot audio waveform
        plt.subplot(3, 1, 1)
        plt.title("Audio Waveform")
        plt.plot(np.arange(len(audio_data)) / self.sr, audio_data)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Plot window predictions
        plt.subplot(3, 1, 2)
        plt.title("Window Predictions")
        
        # Convert samples to time
        predictions_df['start_time'] = predictions_df['start_sample'] / self.sr
        predictions_df['end_time'] = predictions_df['end_sample'] / self.sr
        
        # Plot prediction probabilities
        for _, row in predictions_df.iterrows():
            x = (row['start_time'] + row['end_time']) / 2
            y = row['probability']
            color = 'red' if row['predicted_class'] == 1 else 'blue'
            plt.scatter(x, y, color=color, alpha=0.5)
            
            # Draw window
            rect = patches.Rectangle(
                (row['start_time'], 0), 
                row['end_time'] - row['start_time'], 
                1,
                linewidth=1, 
                edgecolor=color, 
                facecolor=color, 
                alpha=0.1
            )
            plt.gca().add_patch(rect)
        
        plt.axhline(y=self.threshold, color='r', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Probability")
        plt.ylim(0, 1.1)
        
        # Plot merged detections
        plt.subplot(3, 1, 3)
        plt.title("Merged Detections")
        
        # No audio data to plot here, just intervals
        plt.plot(np.arange(len(audio_data)) / self.sr, np.zeros_like(audio_data), color='gray')
        
        # Draw detection intervals
        for start, end in merged_detections:
            start_time = start / self.sr
            end_time = end / self.sr
            rect = patches.Rectangle(
                (start_time, -0.5), 
                end_time - start_time, 
                1,
                linewidth=1, 
                edgecolor='green', 
                facecolor='green', 
                alpha=0.5
            )
            plt.gca().add_patch(rect)
        
        plt.xlabel("Time (s)")
        plt.ylim(-0.6, 0.6)
        
        plt.tight_layout()
        plt.show()
    
    def save_predictions_to_csv(self, timestamps: List[Tuple[float, float]], output_path: str) -> None:
        """
        Save prediction timestamps to CSV file.
        
        Parameters
        ----------
        timestamps : List[Tuple[float, float]]
            List of time intervals (start, end) in seconds
        output_path : str
            Path to save the CSV file
        """
        # Create DataFrame
        df = pd.DataFrame(timestamps, columns=['start_time', 'end_time'])
        
        # Add duration column
        df['duration'] = df['end_time'] - df['start_time']
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # This code will only run when the file is executed directly, not when imported
    print("This module provides the PredictionPipeline class for anemonefish sound detection.")
    print("Import and use it in your own scripts.") 