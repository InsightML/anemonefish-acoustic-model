"""
Audio preprocessing module for inference pipeline.

This module provides functionality for loading, segmenting, and converting
audio to spectrograms using the same parameters as the training pipeline.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path


class AudioPreprocessor:
    """
    Audio preprocessor for inference pipeline.
    
    Ensures consistency with training parameters:
    - FMAX=2000Hz
    - N_FFT=1024
    - HOP_LENGTH=256
    - Output size: 256x256 pixels
    - Normalization: [0, 1] by dividing by 255.0 (NO ImageNet normalization)
    """
    
    def __init__(
        self,
        fmax: int = 2000,
        n_fft: int = 1024,
        hop_length: int = 256,
        width_pixels: int = 256,
        height_pixels: int = 256,
        target_sr: Optional[int] = None
    ):
        """
        Initialize AudioPreprocessor with configuration.
        
        Parameters
        ----------
        fmax : int
            Maximum frequency for spectrogram (Hz). Default: 2000
        n_fft : int
            FFT window size. Default: 1024
        hop_length : int
            Hop length for STFT. Default: 256
        width_pixels : int
            Output spectrogram width in pixels. Default: 256
        height_pixels : int
            Output spectrogram height in pixels. Default: 256
        target_sr : int, optional
            Target sampling rate. If None, uses original sample rate.
        """
        self.fmax = fmax
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
        self.target_sr = target_sr
    
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
        """
        Load and preprocess audio file.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        audio_data : np.ndarray or None
            Audio data array (1D)
        sample_rate : int or None
            Sample rate of loaded audio
        duration : float or None
            Duration in seconds
        """
        try:
            # Load audio using librosa (matches training pipeline)
            audio_data, sample_rate = librosa.load(audio_path, sr=self.target_sr)
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            duration = len(audio_data) / sample_rate
            
            return audio_data, sample_rate, duration
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None, None, None
    
    def segment_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_duration: float = 1.0,
        stride_duration: float = 0.4
    ) -> List[Tuple[int, int, float, float]]:
        """
        Generate sliding window segments for audio processing.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data array
        sample_rate : int
            Sample rate
        window_duration : float
            Window size in seconds. Default: 1.0
        stride_duration : float
            Stride size in seconds. Default: 0.4 (as per training: 0.4s windows)
            
        Returns
        -------
        windows : List[Tuple[int, int, float, float]]
            List of (start_sample, end_sample, start_time, end_time) tuples
        """
        window_samples = int(window_duration * sample_rate)
        stride_samples = int(stride_duration * sample_rate)
        total_samples = len(audio_data)
        
        windows = []
        
        start_sample = 0
        while start_sample + window_samples <= total_samples:
            end_sample = start_sample + window_samples
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            windows.append((start_sample, end_sample, start_time, end_time))
            start_sample += stride_samples
        
        return windows
    
    def create_spectrogram(
        self,
        audio_segment: np.ndarray,
        sample_rate: int,
        return_image: bool = False
    ) -> Optional[np.ndarray]:
        """
        Create spectrogram from audio segment matching training pipeline.
        
        Parameters
        ----------
        audio_segment : np.ndarray
            1D audio array
        sample_rate : int
            Sample rate
        return_image : bool
            If True, returns image array (H, W, C) suitable for model input.
            If False, returns raw spectrogram data.
            Default: False (for Lambda/cloud environments, set to True)
            
        Returns
        -------
        spectrogram : np.ndarray or None
            If return_image=True: Preprocessed spectrogram (H, W, C) ready for model.
            If return_image=False: Raw spectrogram data (for saving as PNG).
        """
        try:
            # Compute STFT (matches training)
            D = librosa.stft(
                audio_segment,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to dB scale (matches training)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            if return_image:
                # Create figure for rendering (matches training approach)
                dpi = 100
                width_inches = self.width_pixels / dpi
                height_inches = self.height_pixels / dpi
                
                fig, ax = plt.subplots(1, figsize=(width_inches, height_inches), dpi=dpi)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                # Plot spectrogram
                librosa.display.specshow(
                    S_db,
                    sr=sample_rate,
                    hop_length=self.hop_length,
                    x_axis=None,
                    y_axis=None,
                    fmax=self.fmax,
                    ax=ax
                )
                
                # Handle frequency limit
                num_frequency_bins = S_db.shape[0]
                if self.fmax is not None and sample_rate is not None:
                    fmax_bin = int(self.fmax / (sample_rate / 2.0) * num_frequency_bins)
                    if fmax_bin < num_frequency_bins:
                        ax.set_ylim(0, fmax_bin)
                
                ax.axis('off')
                
                # Convert to array (fix for newer matplotlib versions)
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Convert RGBA to RGB
                buf = buf[:, :, :3]
                plt.close(fig)
                
                # Resize to exact dimensions
                spectrogram = cv2.resize(buf, (self.width_pixels, self.height_pixels))
                
                # Normalize to [0, 1] by dividing by 255.0 (matches training - NO ImageNet normalization)
                # Training: image = tf.cast(image, tf.float32) / 255.0
                spectrogram = spectrogram.astype(np.float32) / 255.0
                
                return spectrogram
            else:
                # Return raw spectrogram data for saving as PNG (training pipeline approach)
                dpi = 100
                width_inches = self.width_pixels / dpi
                height_inches = self.height_pixels / dpi
                
                fig, ax = plt.subplots(1, figsize=(width_inches, height_inches), dpi=dpi)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                # Plot spectrogram
                librosa.display.specshow(
                    S_db,
                    sr=sample_rate,
                    hop_length=self.hop_length,
                    x_axis=None,
                    y_axis=None,
                    fmax=self.fmax,
                    ax=ax
                )
                
                # Handle frequency limit
                num_frequency_bins = S_db.shape[0]
                if self.fmax is not None and sample_rate is not None:
                    fmax_bin = int(self.fmax / (sample_rate / 2.0) * num_frequency_bins)
                    if fmax_bin < num_frequency_bins:
                        ax.set_ylim(0, fmax_bin)
                
                ax.axis('off')
                
                # Return the figure/axes for saving
                return (fig, ax)
            
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            return None
    
    def process_audio_to_spectrograms(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        windows: List[Tuple[int, int, float, float]],
        return_images: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Process all windows to create spectrograms for model input.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Full audio array
        sample_rate : int
            Sample rate
        windows : List[Tuple[int, int, float, float]]
            List of window parameters from segment_audio()
        return_images : bool
            If True, returns preprocessed image arrays. If False, returns figures.
            Default: True for inference.
            
        Returns
        -------
        spectrograms : np.ndarray
            Array of spectrograms (N, H, W, C) if return_images=True
        timestamps : List[Tuple[float, float]]
            List of (start_time, end_time) for each window
        """
        spectrograms = []
        timestamps = []
        
        for start_sample, end_sample, start_time, end_time in windows:
            # Extract audio segment
            audio_segment = audio_data[start_sample:end_sample]
            
            # Create spectrogram
            spectrogram = self.create_spectrogram(audio_segment, sample_rate, return_image=return_images)
            
            if spectrogram is not None:
                if return_images:
                    # spectrogram is already a numpy array (H, W, C)
                    spectrograms.append(spectrogram)
                else:
                    # spectrogram is (fig, ax) tuple - save to temp file or skip for now
                    # For inference pipeline, we always use return_images=True
                    fig, ax = spectrogram
                    # Save would be handled by caller if needed
                    plt.close(fig)
                timestamps.append((start_time, end_time))
        
        if spectrograms:
            spectrograms = np.array(spectrograms)
        else:
            spectrograms = np.array([])
        
        return spectrograms, timestamps
