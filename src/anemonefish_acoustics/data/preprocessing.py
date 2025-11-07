"""
Audio Preprocessing Pipeline

This module contains the preprocessing pipeline for converting raw audio into spectrograms
ready for model inference or training.

=== AUDIO FUNDAMENTALS ===

Sample Rate:
    The number of audio samples captured per second, measured in Hz (Hertz).
    Common sample rates: 8000 Hz (telephony), 16000 Hz (wideband), 44100 Hz (CD quality).
    
    Example: At 8000 Hz sample rate:
        - 1 second of audio = 8000 samples
        - 0.4 seconds = 3200 samples
        - 0.2 seconds = 1600 samples
        - 0.1 seconds = 800 samples

Time to Samples Conversion:
    samples = duration_seconds * sample_rate
    
    Example at 8000 Hz:
        0.4s * 8000 = 3200 samples
        0.2s * 8000 = 1600 samples

=== WINDOWING CONCEPTS ===

Window (or Frame):
    A short segment of audio extracted for analysis. Instead of analyzing entire audio files,
    we break them into smaller overlapping windows. This allows us to capture temporal patterns.
    
    Example: A 5-second audio file at 8000 Hz = 40,000 samples
        Window size: 0.4s = 3200 samples
        Each window contains 3200 consecutive audio samples

Stride (or Hop or Slide):
    The distance (in samples) between the start of consecutive windows. Smaller strides create
    more overlap between windows, capturing more temporal detail but creating more data.
    
    Example with 0.4s windows and 0.2s stride:
        Window 0: samples 0-3199    (0.0s - 0.4s)
        Window 1: samples 1600-4799 (0.2s - 0.6s)  ← moved forward 1600 samples
        Window 2: samples 3200-6399 (0.4s - 0.8s)  ← moved forward 1600 samples
    

=== SPECTROGRAM CONCEPTS ===

Short-Time Fourier Transform (STFT):
    Converts time-domain audio signal into frequency-domain representation by applying
    Fourier Transform to overlapping windows. Shows which frequencies are present at each
    time point.
    
    Key parameters:
        - n_fft: FFT window size (e.g., 1024). Larger = better frequency resolution
        - hop_length: Distance between STFT windows (e.g., 256). Usually n_fft/4
        - fmax: Maximum frequency of interest (e.g., 2000 Hz for fish sounds)

Spectrogram:
    A 2D representation of audio: frequency (y-axis) vs time (x-axis), with intensity
    showing amplitude. Each audio window generates one spectrogram.
    
    Shape: (frequency_bins, time_frames)
        - frequency_bins: Number of frequency bands analyzed (depends on n_fft and fmax)
        - time_frames: Number of STFT windows that fit in the audio window

=== PIPELINE SUMMARY ===

For inference on a 5-second audio file (8000 Hz, window=0.4s, stride=0.2s):
    1. Load audio: 5s * 8000 = 40,000 samples
    2. Extract windows: Creates ~24 windows of 3200 samples each
    3. Generate spectrograms: Converts each window to frequency representation
    4. Output: (24, freq_bins, time_frames) array ready for model
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

import soundfile as sf

def extract_windows(audio_file,
                    window_duration_s, 
                    slide_duration_s, 
                    sample_rate):
    """
    Extracts overlapping windows from audio data using vectorized slicing.
    
    Implementation: Uses sliding_window_view to create all 1-sample-stride windows,
    then slices to select windows at the desired stride interval.

    Args:
        audio_file (numpy.ndarray): Audio samples to extract windows from.
        window_duration_s (float): Duration of each window in seconds.
        slide_duration_s (float): Stride between consecutive windows in seconds.
        sample_rate (int): Sample rate of the audio data in Hz.
        
    Returns:
        numpy.ndarray: Shape (num_windows, window_samples). Each row is one window.
    """
    window_samples = int(window_duration_s * sample_rate)
    slide_samples = int(slide_duration_s * sample_rate)

    # Create sliding windows (by default, slides by 1 sample at a time)
    all_windows = sliding_window_view(audio_file, window_shape=window_samples)
    
    # Slice to get windows at the desired stride (every slide_samples positions)
    windows = all_windows[::slide_samples]
    
    return windows

def create_spectrogram(audio_data, sr_target=8000, n_fft=512, hop_length=256, fmax=2000):
    """
    Converts audio window to frequency-domain spectrogram using STFT.
    
    Implementation: Applies Short-Time Fourier Transform, converts to decibels,
    and crops to maximum frequency of interest.

    Args:
        audio_data (numpy.ndarray): Audio samples for one window.
        sr_target (int): Sample rate of the audio data in Hz.
        n_fft (int): FFT window size for STFT.
        hop_length (int): Number of samples between successive STFT windows.
        fmax (int): Maximum frequency in Hz. Crops spectrogram to this frequency.
    
    Returns:
        numpy.ndarray: Spectrogram in decibels, shape (frequency_bins, time_frames).
    """
    # Compute Short-Time Fourier Transform (STFT)
    # The STFT will consider frequencies up to sr/2.
    # We are interested in fmax, so ensure sr is adequate.
    # If sr_target is set (e.g., 4000Hz for fmax=2000Hz), this is fine.
    D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)

    # Convert amplitude to decibels
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # If fmax is specified, crop the spectrogram to only include frequencies up to fmax
    if fmax is not None and sr_target is not None:
        # Calculate the bin index corresponding to fmax
        num_frequency_bins = S_db.shape[0]
        fmax_bin = int(fmax / (sr_target / 2.0) * num_frequency_bins)
        if fmax_bin < num_frequency_bins:
            S_db = S_db[:fmax_bin, :]

    return S_db


def preprocess_audio_for_inference(audio_buffer, 
                                   window_duration_s=0.4,
                                   slide_duration_s=0.2,
                                   sr_target=8000,
                                   n_fft=1024,
                                   hop_length=None,
                                   fmax=2000,
                                   logger=None):
    """
    Complete preprocessing pipeline: audio buffer → spectrograms ready for model inference.
    
    Pipeline: Load audio → Extract overlapping windows → Convert each to spectrogram.
    All parameters should match training configuration for consistent results.
    
    Args:
        audio_buffer: File-like object or path containing audio data.
        window_duration_s (float): Duration of each window in seconds.
        slide_duration_s (float): Stride between windows in seconds.
        sr_target (int): Sample rate in Hz.
        n_fft (int): FFT window size for spectrograms.
        hop_length (int, optional): STFT hop length. Defaults to n_fft // 4.
        fmax (int): Maximum frequency in Hz for spectrograms.
        logger: Logger instance for logging progress.
    
    Returns:
        numpy.ndarray: Shape (num_windows, freq_bins, time_frames). Batch of spectrograms.
    """
    if hop_length is None:
        hop_length = n_fft // 4 # Default hop length is 1/4 of n_fft
    
    # 1. Load audio data using soundfile
    audio_data, sample_rate = sf.read(audio_buffer)
    
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # 2. Extract windows
    windows = extract_windows(audio_data, window_duration_s, slide_duration_s, sample_rate)
    logger.info(f"Windows shape: {windows.shape}")
    
    # 3. Convert each window to spectrogram
    spectrograms = []
    for window in windows:
        spec = create_spectrogram(window, n_fft=n_fft, 
                                  hop_length=hop_length, fmax=fmax, sr_target=sr_target)
        spectrograms.append(spec)
    # Stack into array: (num_windows, freq_bins, time_frames)
    spectrograms = np.array(spectrograms)
    logger.info(f"Spectrograms shape: {spectrograms.shape}")
    
    return spectrograms