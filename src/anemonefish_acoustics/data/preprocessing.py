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

Training Data:
    | Segment Type | Segment Duration | Action |
    |--------------|------------------|--------|
    | **Anemonefish/Biological** | < 0.4s | ✅ Pad with zeros to 0.4s → Extract 1 window |
    | **Anemonefish/Biological** | ≥ 0.4s | ✅ Extract multiple windows with sliding window |
    | **Noise** | < 0.4s | ✅ Skip (plenty of data from longer segments) |
    | **Noise** | ≥ 0.4s | ✅ Extract multiple windows + random shortening/padding |

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
import pandas as pd
import soundfile as sf
import os

from anemonefish_acoustics.utils.logger import get_logger
logging = get_logger(__name__)

def parse_audacity_labels(label_file_path):
    """
    Parses an Audacity label file (TXT format, tab-separated values).
    
    Audacity label files contain time-based annotations with three columns:
    - Column 1: Start time in seconds (float)
    - Column 2: End time in seconds (float)  
    - Column 3: Label text (string, optional)
    
    Args:
        label_file_path (str): Path to the Audacity label file (.txt)
    
    Returns:
        list: List of tuples representing labeled segments with their text:
              [(start_time1, end_time1, label_text1), (start_time2, end_time2, label_text2), ...]
              Segments are sorted by start time in ascending order.
              
    Notes:
        - Invalid segments (start_time >= end_time) are skipped with a warning
        - Missing label text defaults to empty string
        - File parsing errors are logged but don't raise exceptions
        - Returns empty list if file cannot be parsed
    
    Example:
        >>> segments = parse_audacity_labels("annotations.txt")
        >>> print(segments)
        [(0.5, 1.2, "anemonefish"), (2.1, 3.0, "biological"), (4.5, 5.8, "")]
    """
    labeled_segments = []
    try:
        # Read with tab delimiter, no header, read all three columns
        df = pd.read_csv(label_file_path, sep='\t', header=None, float_precision='round_trip')
        for index, row in df.iterrows():
            start_time = float(row[0])
            end_time = float(row[1])
            # Extract label text (third column), default to empty string if not present
            label_text = str(row[2]).strip() if len(row) > 2 and pd.notna(row[2]) else ""
            
            if start_time < end_time: # Basic validation
                 labeled_segments.append((start_time, end_time, label_text))
            else:
                logging.warning(f"Skipping invalid segment in {label_file_path}: start_time {start_time} >= end_time {end_time}")

        # Sort segments by start time
        labeled_segments.sort(key=lambda x: x[0])
        logging.info(f"Parsed {len(labeled_segments)} segments from {label_file_path}")
    except FileNotFoundError:
        logging.error(f"Label file not found: {label_file_path}")
    except pd.errors.EmptyDataError:
        logging.warning(f"Label file is empty: {label_file_path}")
    except Exception as e:
        logging.error(f"Error parsing label file {label_file_path}: {e}")
    return labeled_segments


def classify_labeled_segments(labeled_segments_with_text, classes):
    """
    Classifies labeled segments into their respective classes based on label text matching.
    
    Args:
        labeled_segments_with_text (list): List of tuples [(start, end, label_text), ...]
        classes (list): List of class names from config (e.g., ["noise", "anemonefish", "biological"])
    
    Returns:
        dict: Dictionary mapping class names to segment lists (without text)
              {class_name: [(start1, end1), (start2, end2), ...]}
    
    Classification Logic:
        - 'noise': Always represents unlabeled regions (handled separately, not from annotations)
        - 'anemonefish': Default class for all labeled segments that don't match other classes
        - Other classes: Exact match (case-sensitive) on label text
    """
    # Initialize result dictionary for all non-noise classes
    classified_segments = {}
    for class_name in classes:
        if class_name != 'noise':  # Noise is handled separately (unlabeled regions)
            classified_segments[class_name] = []
    
    # Get list of specific class names to match (excluding 'noise' and 'anemonefish')
    specific_classes = [c for c in classes if c not in ['noise', 'anemonefish']]
    
    # Classify each labeled segment
    for start_time, end_time, label_text in labeled_segments_with_text:
        matched = False
        
        # Try to match with specific classes (exact match, case-sensitive)
        for specific_class in specific_classes:
            if label_text == specific_class:
                classified_segments[specific_class].append((start_time, end_time))
                matched = True
                break
        
        # If not matched with any specific class, assign to 'anemonefish' (default target class)
        if not matched and 'anemonefish' in classified_segments:
            classified_segments['anemonefish'].append((start_time, end_time))
    
    # Log classification results
    for class_name, segments in classified_segments.items():
        if segments:
            logging.info(f"  Classified {len(segments)} segments as '{class_name}'")
    
    return classified_segments

def get_noise_segments(total_duration_seconds, labeled_segments, min_segment_len_seconds):
    """
    Identifies noise segments in an audio file given its total duration and labeled (non-noise) segments.
    
    Args:
        total_duration_seconds (float): Total duration of the audio file.
        labeled_segments (list of tuples): Sorted list of (start, end) or (start, end, class) times 
            for labeled regions. Function handles both formats.
        min_segment_len_seconds (float): Minimum duration for a segment to be considered noise.
            Segments shorter than this will be ignored.
    
    Returns:
        list of tuples: [(noise_start1, noise_end1), ...] for noise regions.
    """
    noise_segments = []
    current_time = 0.0
    
    # Normalize labeled_segments to (start, end) format (handle both 2-tuple and 3-tuple)
    normalized_segments = []
    for seg in labeled_segments:
        if len(seg) >= 2:
            normalized_segments.append((seg[0], seg[1]))
    
    labeled_segments = normalized_segments

    # If no labeled segments, the whole file is noise
    if not labeled_segments:
        if total_duration_seconds >= min_segment_len_seconds:
            noise_segments.append((0.0, total_duration_seconds))
            logging.info(f"No labels found. Entire duration {total_duration_seconds:.2f}s considered noise for segmentation.")
        else:
            logging.info(f"No labels found. Entire duration {total_duration_seconds:.2f}s is less than min_segment_len_seconds {min_segment_len_seconds:.2f}s. No noise segments generated.")
        return noise_segments

    # Process segment from start of tape to the first label
    first_label_start = labeled_segments[0][0]
    if first_label_start > current_time:
        duration = first_label_start - current_time
        if duration >= min_segment_len_seconds:
            noise_segments.append((current_time, first_label_start))
        # else: logging.debug(f"Initial noise segment from {current_time:.2f} to {first_label_start:.2f} (duration {duration:.2f}s) too short.")
    current_time = max(current_time, labeled_segments[0][1]) # Move current time to end of first label

    # Process segments between labels
    for i in range(len(labeled_segments) - 1):
        end_current_label = labeled_segments[i][1]
        start_next_label = labeled_segments[i+1][0]
        
        # Ensure current_time is at least at the end of the current label before looking for a gap
        current_time = max(current_time, end_current_label) 

        if start_next_label > current_time: # If there's a gap
            duration = start_next_label - current_time
            if duration >= min_segment_len_seconds:
                noise_segments.append((current_time, start_next_label))
            # else: logging.debug(f"Noise segment between labels (from {current_time:.2f} to {start_next_label:.2f}, duration {duration:.2f}s) too short.")
        current_time = max(current_time, labeled_segments[i+1][1]) # Move current time to end of next label

    # Process segment from the end of the last label to the end of the file
    last_label_end = labeled_segments[-1][1]
    current_time = max(current_time, last_label_end) # Ensure current_time is at least at the end of the last label
    
    if total_duration_seconds > current_time:
        duration = total_duration_seconds - current_time
        if duration >= min_segment_len_seconds:
            noise_segments.append((current_time, total_duration_seconds))
        # else: logging.debug(f"Final noise segment (from {current_time:.2f} to {total_duration_seconds:.2f}, duration {duration:.2f}s) too short.")
            
    if noise_segments:
        logging.info(f"Identified {len(noise_segments)} noise segments meeting minimum duration of {min_segment_len_seconds:.2f}s.")
    else:
        logging.info(f"No noise segments meeting minimum duration of {min_segment_len_seconds:.2f}s were identified.")
    return noise_segments

def segment_audio_data(audio_data, segments, sample_rate, class_name=None):
    """
    Segments audio data using a list of tuples (start, end) and returns either a list of audio segments
    or a dictionary mapping class name to list of audio segments.
    
    Args:
        audio_data (numpy.ndarray): Audio data to segment.
        segments (list of tuples): List of tuples (start_time, end_time) in seconds for segments to extract.
        class_name (str, optional): Name of the class to segment. If None, returns list directly.
        sample_rate (int): Sample rate of the audio data in Hz.
        
    Returns:
        list or dict: If class_name is None, returns list of audio data segments.
                     If class_name is provided, returns dictionary mapping class name to list of segments.
                     Format: [segment1_array, segment2_array, segment3_array] or 
                            {class_name: [segment1_array, segment2_array, segment3_array]}
    """
    audio_segments = []
    
    for start, end in segments:
        # Convert time in seconds to sample indices
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        
        # Extract the audio segment
        segment = audio_data[start_idx:end_idx]
        audio_segments.append(segment)
    
    if class_name is None:
        return audio_segments
    else:
        return {class_name: audio_segments}

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

def create_spectrogram(audio_data, sr_target=8000, n_fft=512, hop_length=256, fmax=2000, normalize=False):
    """
    Converts audio window to frequency-domain spectrogram using STFT.
    
    Implementation: Applies Short-Time Fourier Transform, converts to decibels,
    crops to maximum frequency of interest, and optionally applies min-max normalization.

    Args:
        audio_data (numpy.ndarray): Audio samples for one window.
        sr_target (int): Sample rate of the audio data in Hz.
        n_fft (int): FFT window size for STFT.
        hop_length (int): Number of samples between successive STFT windows.
        fmax (int): Maximum frequency in Hz. Crops spectrogram to this frequency.
        normalize (bool): If True, applies min-max normalization to scale values to [0, 1] range.
    
    Returns:
        numpy.ndarray: Spectrogram in decibels, shape (frequency_bins, time_frames).
                      If normalize=True, values are scaled to [0, 1] range.
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

    # Apply min-max normalization if requested
    if normalize:
        S_min = S_db.min()
        S_max = S_db.max()
        if S_max > S_min:  # Avoid division by zero
            S_db = (S_db - S_min) / (S_max - S_min)
        else:
            S_db = np.zeros_like(S_db)  # Handle edge case where all values are the same

    return S_db


def pad_segment_to_window_size(audio_segment, window_samples):
    """
    Pads an audio segment with zeros to match the window size if it's shorter.
    Used for anemonefish/biological classes when segments are shorter than the window.
    
    Args:
        audio_segment (numpy.ndarray): Audio segment data
        window_samples (int): Target window size in samples
    
    Returns:
        numpy.ndarray: Padded audio segment of length window_samples
    """
    if len(audio_segment) >= window_samples:
        return audio_segment[:window_samples]
    
    # Calculate padding needed
    num_padding_samples = window_samples - len(audio_segment)
    
    # Pad with zeros at the end
    padded_segment = np.pad(audio_segment, (0, num_padding_samples), 'constant', constant_values=0.0)
    
    return padded_segment


def randomly_shorten_and_pad_segment(audio_segment, window_samples, sample_rate, 
                                   min_duration_s=0.1, max_duration_s=0.35,
                                   padding_ratio=0.3):
    """
    Randomly shortens and pads noise segments to balance the dataset and prevent
    duration-based bias. Applied to a proportion of noise windows.
    
    Args:
        audio_segment (numpy.ndarray): Audio segment data
        window_samples (int): Target window size in samples
        sample_rate (int): Sample rate in Hz
        min_duration_s (float): Minimum duration for shortened segments
        max_duration_s (float): Maximum duration for shortened segments
        padding_ratio (float): Proportion of segments to apply shortening to
    
    Returns:
        tuple: (processed_segment, was_padded) - audio data and whether padding was applied
    """
    import random
    
    # Only apply to a proportion of segments
    if random.random() > padding_ratio:
        return audio_segment[:window_samples], False
    
    # Random duration within range
    random_duration = random.uniform(min_duration_s, min(max_duration_s, len(audio_segment) / sample_rate))
    random_samples = int(random_duration * sample_rate)
    
    # Truncate to random duration
    shortened_segment = audio_segment[:random_samples]
    
    # Pad to window size
    padded_segment = pad_segment_to_window_size(shortened_segment, window_samples)
    
    return padded_segment, True


def process_segments_to_spectrograms(segments_audio_data, class_name, 
                                   window_duration_s, slide_duration_s, sample_rate,
                                   n_fft, hop_length, fmax, sr_target,
                                   noise_padding_params=None):
    """
    Process audio segments into spectrograms with appropriate padding based on class.
    
    Args:
        segments_audio_data (list): List of audio segments (numpy arrays)
        class_name (str): Name of the class being processed
        window_duration_s (float): Window size in seconds
        slide_duration_s (float): Slide/hop size in seconds
        sample_rate (int): Sample rate of audio data
        n_fft (int): FFT size for spectrogram
        hop_length (int): Hop length for STFT
        fmax (int): Maximum frequency for spectrogram
        sr_target (int): Target sample rate
        noise_padding_params (dict): Parameters for noise padding (min_duration, max_duration, ratio)
    
    Returns:
        list: List of spectrograms (numpy arrays)
    """
    window_samples = int(window_duration_s * sample_rate)
    spectrograms = []
    
    # Default noise padding parameters if not provided
    if noise_padding_params is None:
        noise_padding_params = {
            'min_duration_s': 0.1,
            'max_duration_s': 0.35,
            'padding_ratio': 0.3
        }
    
    for segment in segments_audio_data:
        segment_duration_s = len(segment) / sample_rate
        
        # Handle short segments based on class
        if segment_duration_s < window_duration_s:
            if class_name != 'noise':
                # For anemonefish/biological: pad to window size
                padded_segment = pad_segment_to_window_size(segment, window_samples)
                # Convert padded segment to spectrogram
                spec = create_spectrogram(padded_segment, sr_target=sr_target, 
                                        n_fft=n_fft, hop_length=hop_length, fmax=fmax, normalize=True)
                spectrograms.append(spec)
            else:
                # For noise: skip short segments (can't extract windows from them)
                # This is fine since there's plenty of noise data from longer segments
                continue
        else:
            # Regular sliding window extraction for segments >= window size
            windows = extract_windows(segment, window_duration_s, slide_duration_s, sample_rate)
            
            # For noise class, apply random shortening to some windows
            if class_name == 'noise':
                for window in windows:
                    processed_window, was_padded = randomly_shorten_and_pad_segment(
                        window, window_samples, sample_rate,
                        min_duration_s=noise_padding_params['min_duration_s'],
                        max_duration_s=noise_padding_params['max_duration_s'],
                        padding_ratio=noise_padding_params['padding_ratio']
                    )
                    spec = create_spectrogram(processed_window, sr_target=sr_target,
                                            n_fft=n_fft, hop_length=hop_length, fmax=fmax, normalize=True)
                    spectrograms.append(spec)
            else:
                # For other classes, just convert windows to spectrograms
                for window in windows:
                    spec = create_spectrogram(window, sr_target=sr_target,
                                            n_fft=n_fft, hop_length=hop_length, fmax=fmax, normalize=True)
                    spectrograms.append(spec)
    
    return spectrograms


def format_training_data(audio_data_segments, classes):
    """
    Format segmented spectrogram data into training-ready arrays.
    
    Args:
        audio_data_segments (dict): Dictionary mapping class names to lists of spectrograms
                                   {class_name: [spectrogram1, spectrogram2, ...]}
        classes (list): List of class names in order (e.g., ['noise', 'anemonefish', 'biological'])
    
    Returns:
        tuple: (X, y, class_mappings) where:
            - X: numpy array of shape (N, freq_bins, time_frames) containing all spectrograms
            - y: numpy array of shape (N, num_classes) containing one-hot encoded labels
            - class_mappings: dict mapping class indices to class names {0: 'noise', 1: 'anemonefish', ...}
    """
    # Initialize lists to collect data
    X_list = []
    y_list = []
    
    # Create class mappings
    class_mappings = {i: class_name for i, class_name in enumerate(classes)}
    class_to_index = {class_name: i for i, class_name in enumerate(classes)}
    num_classes = len(classes)
    
    # Process each class
    for class_name in classes:
        if class_name not in audio_data_segments:
            logging.warning(f"Class '{class_name}' not found in audio_data_segments")
            continue
            
        spectrograms = audio_data_segments[class_name]
        class_index = class_to_index[class_name]
        
        # Create one-hot encoded label for this class
        one_hot = np.zeros(num_classes)
        one_hot[class_index] = 1
        
        # Add spectrograms and labels
        for spectrogram in spectrograms:
            X_list.append(spectrogram)
            y_list.append(one_hot)
    
    # Convert to numpy arrays
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        
        logging.info(f"Formatted training data: X shape = {X.shape}, y shape = {y.shape}")
        for i, class_name in enumerate(classes):
            count = np.sum(y[:, i])
            logging.info(f"  Class '{class_name}' (index {i}): {int(count)} samples")
    else:
        # Return empty arrays if no data
        X = np.array([])
        y = np.array([])
        logging.warning("No training data to format")
    
    return X, y, class_mappings


def process_single_audio_file_for_training(audio_path, 
                                          annotation_path,
                                          window_duration_s=0.4,
                                          slide_duration_s=0.2,
                                          sr_target=8000,
                                          n_fft=1024,
                                          hop_length=None,
                                          fmax=2000,
                                          classes=['noise', 'anemonefish', 'biological'],
                                          min_segment_len_seconds=0.1,
                                          noise_padding_params=None):
    """
    Process a single audio file and its annotation into spectrograms by class.
    
    Pipeline:
    1. Read audio file and load class annotations from Audacity label file
    2. Split audio into labeled segments (by class) and noise segments (unlabeled regions)
    3. Segment audio data by classes
    4. Create sliding window spectrograms from segments
    
    Args:
        audio_path: Path to audio file (.wav)
        annotation_path: Path to annotation file (.txt in Audacity label format)
        window_duration_s: Window size in seconds (default 0.4)
        slide_duration_s: Stride between windows in seconds (default 0.2)
        sr_target: Target sample rate in Hz (default 8000)
        n_fft: FFT size for spectrogram (default 1024)
        hop_length: STFT hop length (default n_fft//4)
        fmax: Maximum frequency for spectrogram (default 2000)
        classes: List of class names (default ['noise', 'anemonefish', 'biological'])
        min_segment_len_seconds: Minimum segment duration (default 0.1)
        noise_padding_params: Dict with noise padding parameters
    
    Returns:
        dict: {class_name: [spectrograms]} for this file
    """
    if hop_length is None:
        hop_length = n_fft // 4 # Default hop length is 1/4 of n_fft
    
    # 1. Load audio data using soundfile
    audio_data, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Load classes data from txt
    classes_segments = parse_audacity_labels(annotation_path)

    # 2. Split audio file into classes and noise segments
    total_duration_seconds = len(audio_data) / sample_rate
    noise_segments = get_noise_segments(total_duration_seconds, classes_segments, min_segment_len_seconds)
    classes_segments_dict = classify_labeled_segments(classes_segments, classes)

    # Append noise class to the dict
    classes_segments_dict['noise'] = noise_segments

    # 3. Segment audio file data by classes and noise
    audio_data_segments = {}

    for class_name, segments in classes_segments_dict.items():
        segments_audio_data = segment_audio_data(audio_data, segments, sample_rate)
        logging.info(f"Segmented {len(segments_audio_data)} segments for class {class_name}")
        
        # 4. Process segments to spectrograms
        spectrograms = process_segments_to_spectrograms(
            segments_audio_data, 
            class_name,
            window_duration_s, 
            slide_duration_s, 
            sample_rate,
            n_fft, 
            hop_length, 
            fmax, 
            sr_target,
            noise_padding_params=noise_padding_params
        )
        
        audio_data_segments[class_name] = spectrograms
        logging.info(f"Created {len(spectrograms)} spectrograms for class {class_name}")
    
    return audio_data_segments


def preprocess_audio_for_training(audio_dir,
                                  annotations_dir,
                                  window_duration_s=0.4,
                                  slide_duration_s=0.2,
                                  sr_target=8000,
                                  n_fft=1024,
                                  hop_length=None,
                                  fmax=2000,
                                  logger=None,
                                  classes=['noise', 'anemonefish', 'biological'],
                                  min_segment_len_seconds=0.1,
                                  noise_padding_params=None
                                  ):
    """
    Preprocess multiple audio files from directories for training by aggregating spectrograms across all files.
    
    This function processes all .wav files in the audio directory that have matching .txt annotation files
    in the annotations directory. Files are matched by identical basenames (e.g., recording1.wav → recording1.txt).
    
    Pipeline:
    1. List all .wav files in audio_dir (excluding hidden files)
    2. For each audio file with a matching annotation file:
       a. Process the file pair using process_single_audio_file_for_training()
       b. Accumulate spectrograms by class across all files
    3. Format all accumulated data into training-ready arrays with one-hot encoded labels
    
    Directory Structure Expected:
    - audio_dir/: Contains .wav files (e.g., recording1.wav, recording2.wav, ...)
    - annotations_dir/: Contains .txt files (e.g., recording1.txt, recording2.txt, ...)
    
    Example:
        If file1 has ClassA=10 specs, ClassB=5 specs and file2 has ClassB=15 specs, ClassC=10 specs,
        the result will contain: ClassA=10, ClassB=20, ClassC=10 total spectrograms.
    
    Args:
        audio_dir: Directory path containing .wav audio files
        annotations_dir: Directory path containing .txt annotation files (Audacity label format)
        window_duration_s: Window size in seconds (default 0.4)
        slide_duration_s: Stride between windows in seconds (default 0.2)
        sr_target: Target sample rate in Hz (default 8000)
        n_fft: FFT size for spectrogram (default 1024)
        hop_length: STFT hop length (default n_fft//4)
        fmax: Maximum frequency for spectrogram (default 2000)
        logger: Logger instance
        classes: List of class names (default ['noise', 'anemonefish', 'biological'])
        min_segment_len_seconds: Minimum segment duration (default 0.1)
        noise_padding_params: Dict with noise padding parameters:
            - min_duration_s: min duration for shortened noise (default 0.1)
            - max_duration_s: max duration for shortened noise (default 0.35)
            - padding_ratio: proportion of noise windows to shorten (default 0.3)
    
    Returns:
        tuple: (X, y, class_mappings) where:
            - X: numpy array of shape (N, freq_bins, time_frames) containing all spectrograms from all files
            - y: numpy array of shape (N, num_classes) containing one-hot encoded labels
            - class_mappings: dict mapping class indices to class names {0: 'noise', 1: 'anemonefish', ...}
            
    Notes:
        - Files without matching annotations are skipped with a warning
        - Files that fail to process are skipped with an error log
        - Empty directories return empty arrays
    """
    if hop_length is None:
        hop_length = n_fft // 4 # Default hop length is 1/4 of n_fft
    
    # 1. List all .wav files from audio_dir (exclude hidden/macOS files starting with .)
    audio_files = [f for f in os.listdir(audio_dir) 
                   if f.endswith('.wav') and not f.startswith('.')]
    audio_files.sort()  # Sort for consistent ordering
    
    logging.info(f"Found {len(audio_files)} audio files in {audio_dir}")
    
    if not audio_files:
        logging.warning("No .wav files found in audio directory")
        return np.array([]), np.array([]), {}
    
    # Initialize accumulator dict for all files
    accumulated_spectrograms = {class_name: [] for class_name in classes}
    
    # 2. Process each audio file with its matching annotation
    processed_count = 0
    skipped_count = 0
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Match by identical basename: audio1.wav -> audio1.txt
        basename = os.path.splitext(audio_file)[0]
        annotation_path = os.path.join(annotations_dir, f"{basename}.txt")
        
        if not os.path.exists(annotation_path):
            logging.warning(f"No annotation found for {audio_file}, skipping")
            skipped_count += 1
            continue
        
        logging.info(f"Processing file pair: {audio_file} + {basename}.txt")
        
        # Process this audio+annotation pair
        try:
            file_spectrograms = process_single_audio_file_for_training(
                audio_path, annotation_path, 
                window_duration_s, slide_duration_s, sr_target,
                n_fft, hop_length, fmax, classes,
                min_segment_len_seconds, noise_padding_params
            )
            
            # Accumulate spectrograms by class across all files
            for class_name, specs in file_spectrograms.items():
                accumulated_spectrograms[class_name].extend(specs)
            
            # Log statistics for this file
            file_total = sum(len(specs) for specs in file_spectrograms.values())
            logging.info(f"Processed {audio_file}: added {file_total} spectrograms")
            for class_name, specs in file_spectrograms.items():
                if specs:
                    logging.info(f"  - {class_name}: {len(specs)} spectrograms")
            
            processed_count += 1
            
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {str(e)}")
            skipped_count += 1
            continue
    
    # Log final statistics
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Processed: {processed_count} file pairs")
    logging.info(f"Skipped: {skipped_count} files")
    
    # Log accumulated spectrograms per class
    total_spectrograms = 0
    for class_name in classes:
        count = len(accumulated_spectrograms[class_name])
        total_spectrograms += count
        logging.info(f"{class_name}: {count} total spectrograms")
    logging.info(f"Total spectrograms: {total_spectrograms}")
    
    # 3. Format all accumulated data into training arrays
    X, y, class_mappings = format_training_data(accumulated_spectrograms, classes)

    # Add channel dimension and transpose for Conv2D layers (batch, time, freq) -> (batch, freq, time, 1)
    X = np.transpose(X, (0, 2, 1))
    X = np.expand_dims(X, axis=-1)
    logging.info(f"X shape after adding channel dimension and transposing for Conv2D layers: {X.shape}")
    return X, y, class_mappings

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
                                  hop_length=hop_length, fmax=fmax, sr_target=sr_target, normalize=True)
        spectrograms.append(spec)
    # Stack into array: (num_windows, freq_bins, time_frames)
    spectrograms = np.array(spectrograms)
    logger.info(f"Spectrograms shape: {spectrograms.shape}")
    
    return spectrograms