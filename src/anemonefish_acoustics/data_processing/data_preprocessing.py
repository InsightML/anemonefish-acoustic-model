import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import the new data_augmentation module
from .data_augmentation import AudioAugmenter, DataAugmentationPipeline

# Constants
SAMPLE_RATE = 8000  # Reduced from 44100 Hz to 8000 Hz for 0-2000 Hz signals
CHUNK_DURATION = 5.0  # 5 second chunks for noise files (increased from 1.0)
MAX_SIGNAL_VALUE = 2000.0  # For 16-bit audio normalization

# Feature extraction constants
N_MFCC = 14  # Number of MFCC coefficients to extract
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
FMIN = 0.0  # Minimum frequency for analysis (Hz)
FMAX = 2000.0  # Maximum frequency for analysis (Hz)
FRAME_LENGTH = 2048  # Frame length for RMS calculation

# New spectrogram constants
SPEC_N_FFT = 1024  # FFT window size for spectrogram
SPEC_HOP_LENGTH = 256  # Hop length for spectrogram
SPEC_WIN_LENGTH = 1024  # Window length for spectrogram
SPEC_WINDOW = 'hann'  # Window type for spectrogram
SPEC_POWER = 2.0  # Power for spectrogram (2.0 = power spectrogram)
SPEC_N_MELS = 64  # Number of mel bands to generate
SPEC_FMIN = 0.0  # Minimum frequency for mel bands
SPEC_FMAX = 2000.0  # Maximum frequency for mel bands

# Constants for handling variable lengths
STANDARD_LENGTH_SEC = 0.6  # Standard length for audio files in seconds
STANDARD_LENGTH_SAMPLES = int(STANDARD_LENGTH_SEC * SAMPLE_RATE)  # Standard length in samples
SLIDING_WINDOW_HOP = 0.2  # Hop size for sliding window in seconds (for crop method)

# Preprocessing methods
PREPROCESS_STRETCH_SQUASH = 'stretch_squash'  # Stretch or squash to standard length
PREPROCESS_CROP_PAD = 'crop_pad'  # Crop or pad to standard length


class AnemoneMetadataParser:
    """Parser for extracting metadata from anemonefish call filenames."""
    
    @staticmethod
    def parse_filename(filename: str) -> Dict[str, Any]:
        """Parse an anemonefish call filename and extract the metadata.
        
        The filename format is:
        target-REEF-TIMEOFDAY-BREEDING-RANK-INTERACTIONWITH_CATEGORY_ID_YEAR_TAPENAME_STARTTIMEMS_ENDTIMEMS.wav
        
        Example:
        territorial-Lui-A-NR-R2-Cooperation_B27_2023_20230320-030015-Lui-B27-A-NR-withlabels-aup_30472_30804.wav
        
        Which breaks down as:
        target = territorial
        REEF = Lui
        TIMEOFDAY = A
        BREEDING = NR
        RANK = R2
        INTERACTIONWITH = none
        CATEGORY = Cooperation
        ID = B27
        YEAR = 2023
        TAPENAME = 20230320-030015-Lui-B27-A-NR-withlabels-aup
        STARTTIMEMS = 30472
        ENDTIMEMS = 30804
        
        Note: 
        - Fields are separated by '-' until first '_'
        - After first '_', fields are separated by '_'
        - INTERACTIONWITH may be empty, in which case the sequence is: RANK_CATEGORY
        - INTERACTIONWITH should only include ranks like R1, R2, R3, not "Cooperation" etc.
        
        Args:
            filename: The filename to parse
            
        Returns:
            A dictionary containing the extracted metadata
        """
        # Remove file extension
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Initialize metadata dictionary
        metadata = {"original_filename": base_name}
        
        try:
            # First, split on the first underscore to separate the front section
            if '_' in name_without_ext:
                front_section, rest_section = name_without_ext.split('_', 1)
                
                # The front section is separated by hyphens
                front_parts = front_section.split('-')
                
                # Extract the fixed position metadata from the front section
                metadata["behavior_type"] = front_parts[0] if len(front_parts) > 0 else None
                metadata["reef"] = front_parts[1] if len(front_parts) > 1 else None
                metadata["time_of_day"] = front_parts[2] if len(front_parts) > 2 else None
                metadata["breeding"] = front_parts[3] if len(front_parts) > 3 else None
                metadata["rank"] = front_parts[4] if len(front_parts) > 4 else None
                
                # Check if there's an INTERACTIONWITH field (it would be at index 5)
                if len(front_parts) > 5:
                    # Only consider it INTERACTIONWITH if it follows the pattern Rx (e.g., R1, R2)
                    if front_parts[5].startswith('R') and len(front_parts[5]) > 1 and front_parts[5][1:].isdigit():
                        metadata["interaction_with"] = front_parts[5]
                        # In this case, the last part of front_parts is the CATEGORY
                        if len(front_parts) > 6:
                            metadata["category"] = front_parts[6]
                        else:
                            # If there's no explicit category, use the first part of rest_section
                            metadata["category"] = rest_section.split('_')[0]
                    else:
                        metadata["interaction_with"] = None
                        # If no interaction_with, the last part of front_parts is the CATEGORY
                        metadata["category"] = front_parts[5]
                else:
                    metadata["interaction_with"] = None
                    # If front_parts doesn't have enough elements, use the first part of rest_section
                    metadata["category"] = rest_section.split('_')[0]
                
                # Now handle the rest of the section
                rest_parts = rest_section.split('_')
                
                # Find the timestamps (last two consecutive numeric parts)
                time_indices = []
                for i, part in enumerate(rest_parts):
                    if part.isdigit():
                        time_indices.append(i)
                
                # Extract timestamps
                if len(time_indices) >= 2:
                    # Find the last two consecutive indices
                    for i in range(len(time_indices) - 1):
                        if time_indices[i + 1] - time_indices[i] == 1:
                            start_time_idx = time_indices[i]
                            end_time_idx = time_indices[i + 1]
                            
                            metadata["start_time_ms"] = int(rest_parts[start_time_idx])
                            metadata["end_time_ms"] = int(rest_parts[end_time_idx])
                            metadata["duration_ms"] = metadata["end_time_ms"] - metadata["start_time_ms"]
                            break
                    else:
                        # If no consecutive indices found, use the last two
                        start_time_idx = time_indices[-2]
                        end_time_idx = time_indices[-1]
                        
                        metadata["start_time_ms"] = int(rest_parts[start_time_idx])
                        metadata["end_time_ms"] = int(rest_parts[end_time_idx])
                        metadata["duration_ms"] = metadata["end_time_ms"] - metadata["start_time_ms"]
                else:
                    metadata["start_time_ms"] = None
                    metadata["end_time_ms"] = None
                    metadata["duration_ms"] = None
                    start_time_idx = None
                    end_time_idx = None
                
                # Based on the example, ID and YEAR are the first two parts of rest_section
                # But we need to handle the case where CATEGORY is part of rest_section
                if metadata["category"] == rest_parts[0]:
                    # CATEGORY is the first part of rest_section
                    metadata["id"] = rest_parts[1] if len(rest_parts) > 1 else None
                    metadata["year"] = rest_parts[2] if len(rest_parts) > 2 else None
                    
                    # TAPENAME is everything between YEAR and START_TIME_MS
                    if len(rest_parts) > 3 and start_time_idx is not None and start_time_idx > 3:
                        metadata["tapename"] = "_".join(rest_parts[3:start_time_idx])
                    else:
                        metadata["tapename"] = None
                else:
                    # CATEGORY is not part of rest_section (it's in front_parts)
                    metadata["id"] = rest_parts[0] if len(rest_parts) > 0 else None
                    metadata["year"] = rest_parts[1] if len(rest_parts) > 1 else None
                    
                    # TAPENAME is everything between YEAR and START_TIME_MS
                    if len(rest_parts) > 2 and start_time_idx is not None and start_time_idx > 2:
                        metadata["tapename"] = "_".join(rest_parts[2:start_time_idx])
                    else:
                        metadata["tapename"] = None
                
                # Special case handling for the example provided
                if metadata["category"] in ["B27", "B58", "B64", "B10"] and metadata["id"] == "2023":
                    # This is likely a case where CATEGORY is actually ID
                    # and ID is actually YEAR
                    # and YEAR is actually the start of TAPENAME
                    temp_id = metadata["category"]
                    temp_year = metadata["id"]
                    
                    metadata["category"] = front_parts[-1] if len(front_parts) > 5 else None
                    metadata["id"] = temp_id
                    metadata["year"] = temp_year
                    
                    # Adjust TAPENAME accordingly
                    if start_time_idx is not None:
                        metadata["tapename"] = "_".join(rest_parts[2:start_time_idx])
            else:
                # If there's no underscore, just split by hyphen
                parts = name_without_ext.split('-')
                metadata["behavior_type"] = parts[0] if len(parts) > 0 else None
                metadata["reef"] = parts[1] if len(parts) > 1 else None
                metadata["time_of_day"] = parts[2] if len(parts) > 2 else None
                metadata["breeding"] = parts[3] if len(parts) > 3 else None
                metadata["rank"] = parts[4] if len(parts) > 4 else None
                
                # Check if there's an INTERACTIONWITH field
                if len(parts) > 5 and parts[5].startswith('R') and len(parts[5]) > 1 and parts[5][1:].isdigit():
                    metadata["interaction_with"] = parts[5]
                else:
                    metadata["interaction_with"] = None
                
                # Other fields would be missing in this case
                metadata["category"] = None
                metadata["id"] = None
                metadata["year"] = None
                metadata["tapename"] = None
                metadata["start_time_ms"] = None
                metadata["end_time_ms"] = None
                metadata["duration_ms"] = None
                
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
        
        return metadata


class AudioProcessor:
    """Class for processing audio files and extracting features."""
    
    @staticmethod
    def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
        """Load an audio file and return the audio data and sample rate.
        
        Args:
            file_path: Path to the audio file
            sr: Target sample rate for resampling
            
        Returns:
            Tuple containing the audio data and sample rate
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return np.array([]), sr
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to have values between -1 and 1.
        
        Args:
            audio_data: The audio data to normalize
            
        Returns:
            Normalized audio data
        """
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        return audio_data
    
    @staticmethod
    def extract_mfcc(audio_data: np.ndarray, 
                     sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extract Mel-frequency cepstral coefficients from audio data.
        
        Args:
            audio_data: The audio data
            sr: Sample rate
            
        Returns:
            Array of MFCCs
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sr, 
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                fmin=FMIN,
                fmax=FMAX
            )
            return mfccs
        except Exception as e:
            print(f"Error extracting MFCCs: {e}")
            return np.array([])
    
    @staticmethod
    def extract_spectrogram(audio_data: np.ndarray, 
                           sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extract power spectrogram from audio data.
        
        Args:
            audio_data: The audio data
            sr: Sample rate
            
        Returns:
            Power spectrogram (amplitude squared)
        """
        try:
            # Compute Short-Time Fourier Transform (STFT)
            stft = librosa.stft(
                y=audio_data,
                n_fft=SPEC_N_FFT,
                hop_length=SPEC_HOP_LENGTH,
                win_length=SPEC_WIN_LENGTH,
                window=SPEC_WINDOW
            )
            
            # Convert to power spectrogram
            spectrogram = np.abs(stft) ** SPEC_POWER
            
            return spectrogram
        except Exception as e:
            print(f"Error extracting spectrogram: {e}")
            return np.array([])
    
    @staticmethod
    def extract_mel_spectrogram(audio_data: np.ndarray, 
                               sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extract mel spectrogram from audio data.
        
        Args:
            audio_data: The audio data
            sr: Sample rate
            
        Returns:
            Mel spectrogram
        """
        try:
            # Compute mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sr,
                n_fft=SPEC_N_FFT,
                hop_length=SPEC_HOP_LENGTH,
                win_length=SPEC_WIN_LENGTH,
                window=SPEC_WINDOW,
                n_mels=SPEC_N_MELS,
                fmin=SPEC_FMIN,
                fmax=SPEC_FMAX,
                power=SPEC_POWER
            )
            
            return mel_spectrogram
        except Exception as e:
            print(f"Error extracting mel spectrogram: {e}")
            return np.array([])
    
    @staticmethod
    def log_mel_spectrogram(mel_spectrogram: np.ndarray, 
                           ref: float = 1.0, 
                           amin: float = 1e-10, 
                           top_db: float = 80.0) -> np.ndarray:
        """Convert mel spectrogram to log-mel spectrogram (in dB).
        
        Args:
            mel_spectrogram: The mel spectrogram
            ref: Reference value for converting to dB
            amin: Minimum value to avoid log(0)
            top_db: Maximum dB value
            
        Returns:
            Log-mel spectrogram
        """
        try:
            # Convert to dB
            log_mel_spec = librosa.power_to_db(
                mel_spectrogram,
                ref=ref,
                amin=amin,
                top_db=top_db
            )
            
            return log_mel_spec
        except Exception as e:
            print(f"Error converting to log-mel spectrogram: {e}")
            return np.array([])
    
    @staticmethod
    def standardize_audio_length(audio_data: np.ndarray, 
                                sr: int = SAMPLE_RATE,
                                target_length_sec: float = STANDARD_LENGTH_SEC,
                                method: str = PREPROCESS_STRETCH_SQUASH) -> np.ndarray:
        """Standardize audio length to a target duration.
        
        Args:
            audio_data: The audio data
            sr: Sample rate
            target_length_sec: Target length in seconds
            method: Method to use ('stretch_squash' or 'crop_pad')
            
        Returns:
            Standardized audio data
        """
        # Calculate target length in samples
        target_length_samples = int(target_length_sec * sr)
        
        # If audio is already the target length, return as is
        if len(audio_data) == target_length_samples:
            return audio_data
        
        if method == PREPROCESS_STRETCH_SQUASH:
            # Use librosa's time stretch to resize audio
            if len(audio_data) > target_length_samples:
                # Squeeze (speed up)
                rate = len(audio_data) / target_length_samples
                resized_audio = librosa.effects.time_stretch(audio_data, rate=rate)
                
                # Ensure exact length
                if len(resized_audio) > target_length_samples:
                    resized_audio = resized_audio[:target_length_samples]
                elif len(resized_audio) < target_length_samples:
                    # Unlikely, but pad if needed
                    resized_audio = np.pad(
                        resized_audio, 
                        (0, target_length_samples - len(resized_audio)), 
                        mode='constant'
                    )
                
                return resized_audio
            else:
                # Stretch (slow down)
                rate = len(audio_data) / target_length_samples
                resized_audio = librosa.effects.time_stretch(audio_data, rate=rate)
                
                # Ensure exact length
                if len(resized_audio) > target_length_samples:
                    resized_audio = resized_audio[:target_length_samples]
                elif len(resized_audio) < target_length_samples:
                    # Pad if needed
                    resized_audio = np.pad(
                        resized_audio, 
                        (0, target_length_samples - len(resized_audio)), 
                        mode='constant'
                    )
                
                return resized_audio
                
        elif method == PREPROCESS_CROP_PAD:
            if len(audio_data) > target_length_samples:
                # Crop from the center
                start = (len(audio_data) - target_length_samples) // 2
                return audio_data[start:start + target_length_samples]
            else:
                # Pad with zeros
                return np.pad(
                    audio_data, 
                    (0, target_length_samples - len(audio_data)), 
                    mode='constant'
                )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def get_multiple_segments(audio_data: np.ndarray, 
                             sr: int = SAMPLE_RATE,
                             segment_length_sec: float = STANDARD_LENGTH_SEC,
                             hop_length_sec: float = SLIDING_WINDOW_HOP) -> List[np.ndarray]:
        """Get multiple segments from audio data using sliding window.
        
        Args:
            audio_data: The audio data
            sr: Sample rate
            segment_length_sec: Segment length in seconds
            hop_length_sec: Hop length in seconds
            
        Returns:
            List of audio segments
        """
        # Calculate lengths in samples
        segment_length_samples = int(segment_length_sec * sr)
        hop_length_samples = int(hop_length_sec * sr)
        
        # If audio is shorter than segment length, just return padded audio
        if len(audio_data) <= segment_length_samples:
            padded_audio = np.pad(
                audio_data, 
                (0, segment_length_samples - len(audio_data)), 
                mode='constant'
            )
            return [padded_audio]
        
        # Extract segments with sliding window
        segments = []
        for start in range(0, max(1, len(audio_data) - segment_length_samples + 1), hop_length_samples):
            segment = audio_data[start:start + segment_length_samples]
            
            # Pad last segment if needed
            if len(segment) < segment_length_samples:
                segment = np.pad(
                    segment, 
                    (0, segment_length_samples - len(segment)), 
                    mode='constant'
                )
            
            segments.append(segment)
        
        return segments
    
    @staticmethod
    def analyze_audio_lengths(audio_files: List[str], 
                             sr: int = SAMPLE_RATE) -> Dict[str, float]:
        """Analyze audio lengths of a list of files.
        
        Args:
            audio_files: List of audio file paths
            sr: Sample rate for loading audio
            
        Returns:
            Dictionary with statistics about audio lengths
        """
        lengths_seconds = []
        
        for file_path in tqdm(audio_files, desc="Analyzing audio lengths"):
            try:
                audio_data, _ = AudioProcessor.load_audio(file_path, sr)
                length_seconds = len(audio_data) / sr
                lengths_seconds.append(length_seconds)
            except Exception as e:
                print(f"Error analyzing audio file {file_path}: {e}")
        
        if not lengths_seconds:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'percentile_25': 0,
                'percentile_75': 0,
                'percentile_95': 0
            }
        
        # Calculate statistics
        lengths_array = np.array(lengths_seconds)
        stats = {
            'count': len(lengths_seconds),
            'min': np.min(lengths_array),
            'max': np.max(lengths_array),
            'mean': np.mean(lengths_array),
            'median': np.median(lengths_array),
            'std': np.std(lengths_array),
            'percentile_25': np.percentile(lengths_array, 25),
            'percentile_75': np.percentile(lengths_array, 75),
            'percentile_95': np.percentile(lengths_array, 95)
        }
        
        return stats
    
    @staticmethod
    def extract_features(audio_data: np.ndarray, 
                        sr: int = SAMPLE_RATE,
                        feature_type: str = 'mfcc') -> Dict[str, np.ndarray]:
        """Extract multiple audio features from audio data.
        
        Args:
            audio_data: The audio data
            sr: Sample rate
            feature_type: Type of features to extract ('mfcc', 'spectrogram', 'mel_spectrogram')
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # Ensure n_fft is not larger than the audio length
        if len(audio_data) < N_FFT:
            n_fft = 2 ** int(np.floor(np.log2(len(audio_data))))
            if n_fft < 32:  # Set a minimum FFT size to avoid too small FFT windows
                n_fft = 32
        else:
            n_fft = N_FFT
        
        # Extract features based on feature_type
        if feature_type == 'mfcc' or feature_type == 'all':
            features['mfcc'] = AudioProcessor.extract_mfcc(audio_data, sr)
            
            # Use a positive value for fmin with spectral_contrast
            features['spectral_contrast'] = AudioProcessor.extract_spectral_contrast(audio_data, sr, n_bands=6)
            
            features['chroma'] = AudioProcessor.extract_chroma(audio_data, sr)
            
            features['rms'] = AudioProcessor.extract_rms(audio_data)
        
        if feature_type == 'spectrogram' or feature_type == 'all':
            features['spectrogram'] = AudioProcessor.extract_spectrogram(audio_data, sr)
        
        if feature_type == 'mel_spectrogram' or feature_type == 'all':
            mel_spec = AudioProcessor.extract_mel_spectrogram(audio_data, sr)
            features['mel_spectrogram'] = mel_spec
            
            # Also add log-mel spectrogram
            features['log_mel_spectrogram'] = AudioProcessor.log_mel_spectrogram(mel_spec)
        
        return features
    
    @staticmethod
    def save_audio_chunk(audio_data: np.ndarray, 
                         file_path: str, 
                         sr: int = SAMPLE_RATE) -> None:
        """Save audio data to a file.
        
        Args:
            audio_data: The audio data to save
            file_path: Path where to save the audio
            sr: Sample rate
        """
        try:
            sf.write(file_path, audio_data, sr)
        except Exception as e:
            print(f"Error saving audio chunk to {file_path}: {e}")
    
    @staticmethod
    def downsample_audio_file(input_file_path: str, 
                             output_file_path: str, 
                             target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
        """Downsample an audio file to the target sample rate.
        
        Args:
            input_file_path: Path to the input audio file
            output_file_path: Path where to save the downsampled audio
            target_sr: Target sample rate
            
        Returns:
            Tuple containing the downsampled audio data and sample rate
        """
        try:
            # Load with original SR
            audio_data, _ = librosa.load(input_file_path, sr=None)
            
            # Resample
            downsampled_audio = librosa.resample(
                audio_data, 
                orig_sr=librosa.get_samplerate(input_file_path), 
                target_sr=target_sr
            )
            
            # Save the downsampled audio
            AudioProcessor.save_audio_chunk(downsampled_audio, output_file_path, target_sr)
            
            return downsampled_audio, target_sr
        except Exception as e:
            print(f"Error downsampling audio file {input_file_path}: {e}")
            return np.array([]), target_sr


class DatasetBuilder:
    """Class for building datasets from audio files."""
    
    def __init__(self, 
                 processed_wavs_dir: str = 'data/processed_wavs',
                 noise_dir: str = 'data/noise',
                 noise_chunked_dir: str = 'data/noise_chunked',
                 cache_dir: str = 'data/cache',
                 augmented_dir: str = 'data/augmented_wavs',
                 sr: int = SAMPLE_RATE,
                 feature_type: str = 'mel_spectrogram',  # Changed default from 'mfcc' to 'mel_spectrogram'
                 preprocess_method: str = PREPROCESS_STRETCH_SQUASH,
                 standard_length_sec: float = STANDARD_LENGTH_SEC):
        """Initialize the DatasetBuilder.
        
        Args:
            processed_wavs_dir: Directory containing processed anemonefish calls
            noise_dir: Directory containing noise files
            noise_chunked_dir: Directory containing chunked noise files
            cache_dir: Directory for caching extracted features
            augmented_dir: Directory for saving augmented audio files
            sr: Sample rate for audio processing
            feature_type: Type of features to use ('mfcc', 'spectrogram', 'mel_spectrogram')
            preprocess_method: Method for handling variable length audio ('stretch_squash' or 'crop_pad')
            standard_length_sec: Standard length for audio files in seconds
        """
        self.processed_wavs_dir = processed_wavs_dir
        self.noise_dir = noise_dir
        self.noise_chunked_dir = noise_chunked_dir
        self.cache_dir = cache_dir
        self.augmented_dir = augmented_dir
        self.sr = sr
        self.feature_type = feature_type
        self.preprocess_method = preprocess_method
        self.standard_length_sec = standard_length_sec
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create augmented directory if it doesn't exist
        os.makedirs(self.augmented_dir, exist_ok=True)
        
    def list_anemonefish_files(self) -> List[str]:
        """List all anemonefish call files.
        
        Returns:
            List of file paths
        """
        return [os.path.join(self.processed_wavs_dir, f) 
                for f in os.listdir(self.processed_wavs_dir) 
                if f.endswith('.wav') and not (f.startswith('.') or f.startswith('._'))]
    
    def list_noise_files(self, use_chunked: bool = True) -> List[str]:
        """List all noise files.
        
        Args:
            use_chunked: Whether to use chunked noise files
            
        Returns:
            List of file paths
        """
        if use_chunked and os.path.exists(self.noise_chunked_dir):
            return [os.path.join(self.noise_chunked_dir, f) 
                    for f in os.listdir(self.noise_chunked_dir) 
                    if f.endswith('.wav') and not (f.startswith('.') or f.startswith('._'))]
        else:
            return [os.path.join(self.noise_dir, f) 
                    for f in os.listdir(self.noise_dir) 
                    if f.endswith('.wav') and not (f.startswith('.') or f.startswith('._'))]
    
    def analyze_anemonefish_call_lengths(self) -> Dict[str, float]:
        """Analyze the lengths of all anemonefish call files.
        
        Returns:
            Dictionary with statistics about audio lengths
        """
        anemonefish_files = self.list_anemonefish_files()
        stats = AudioProcessor.analyze_audio_lengths(anemonefish_files, self.sr)
        
        print("\nAnemonefish Call Length Statistics:")
        print(f"Count: {stats['count']}")
        print(f"Min: {stats['min']:.3f} seconds")
        print(f"Max: {stats['max']:.3f} seconds")
        print(f"Mean: {stats['mean']:.3f} seconds")
        print(f"Median: {stats['median']:.3f} seconds")
        print(f"Standard Deviation: {stats['std']:.3f} seconds")
        print(f"25th Percentile: {stats['percentile_25']:.3f} seconds")
        print(f"75th Percentile: {stats['percentile_75']:.3f} seconds")
        print(f"95th Percentile: {stats['percentile_95']:.3f} seconds\n")
        
        return stats
    
    def chunk_noise_files(self, 
                          chunk_duration: float = None, 
                          overlap: float = 0.5) -> None:
        """Chunk long noise files into smaller segments.
        
        Args:
            chunk_duration: Duration of each chunk in seconds (defaults to standard_length_sec)
            overlap: Overlap between consecutive chunks (0.0 to 1.0, as a fraction of chunk_duration)
        """
        # Use standard length if chunk_duration is not specified
        if chunk_duration is None:
            chunk_duration = self.standard_length_sec
            
        # Create chunked directory if it doesn't exist
        os.makedirs(self.noise_chunked_dir, exist_ok=True)
        
        # Clean the chunked directory first to avoid mixing different chunk sizes
        try:
            for old_file in os.listdir(self.noise_chunked_dir):
                old_path = os.path.join(self.noise_chunked_dir, old_file)
                if os.path.isfile(old_path) and old_file.endswith('.wav') and not (old_file.startswith('.') or old_file.startswith('._')):
                    try:
                        os.remove(old_path)
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not remove old chunk {old_path}: {e}")
                        # Continue with chunking anyway
        except Exception as e:
            print(f"Warning: Could not clean chunked directory: {e}")
            # Continue with chunking anyway
        
        # Get list of noise files
        noise_files = self.list_noise_files(use_chunked=False)
        
        for noise_file in tqdm(noise_files, desc="Chunking noise files"):
            try:
                # Load audio
                audio_data, sr = AudioProcessor.load_audio(noise_file, self.sr)
                
                # Skip if audio data is empty
                if len(audio_data) == 0:
                    continue
                
                # Calculate number of samples per chunk
                samples_per_chunk = int(chunk_duration * sr)
                
                # Calculate step size with overlap
                step_size = int(samples_per_chunk * (1 - overlap))
                
                # Get base filename without extension
                base_name = os.path.basename(noise_file)
                file_name_without_ext = os.path.splitext(base_name)[0]
                
                # Create chunks with overlap
                chunk_index = 1
                # Ensure we get at least one chunk even for shorter files
                max_start_idx = max(0, len(audio_data) - samples_per_chunk)
                for start_idx in range(0, max_start_idx + 1, step_size):
                    # Extract chunk
                    end_idx = min(start_idx + samples_per_chunk, len(audio_data))
                    chunk = audio_data[start_idx:end_idx]
                    
                    # Handle short chunks based on preprocessing method
                    if len(chunk) < samples_per_chunk:
                        if self.preprocess_method == PREPROCESS_STRETCH_SQUASH:
                            # Stretch to fill the chunk duration
                            chunk = AudioProcessor.standardize_audio_length(
                                chunk, sr, chunk_duration, PREPROCESS_STRETCH_SQUASH
                            )
                        else:  # PREPROCESS_CROP_PAD
                            # Pad with zeros if chunk is long enough to be meaningful
                            if len(chunk) >= samples_per_chunk * 0.5:
                                chunk = AudioProcessor.standardize_audio_length(
                                    chunk, sr, chunk_duration, PREPROCESS_CROP_PAD
                                )
                            else:
                                continue  # Skip if too short for padding
                    
                    # Normalize chunk
                    chunk = AudioProcessor.normalize_audio(chunk)
                    
                    # Create chunk filename
                    chunk_filename = f"{file_name_without_ext}_chunk_{chunk_index}.wav"
                    chunk_path = os.path.join(self.noise_chunked_dir, chunk_filename)
                    
                    # Save chunk
                    AudioProcessor.save_audio_chunk(chunk, chunk_path, sr)
                    
                    chunk_index += 1
                    
            except Exception as e:
                print(f"Error chunking noise file {noise_file}: {e}")
    
    def build_metadata_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame containing metadata for all anemonefish call files.
        
        Returns:
            DataFrame with metadata
        """
        metadata_list = []
        
        # Get list of anemonefish files
        anemonefish_files = self.list_anemonefish_files()
        
        for file_path in tqdm(anemonefish_files, desc="Extracting metadata"):
            try:
                # Get filename
                filename = os.path.basename(file_path)
                
                # Parse metadata
                metadata = AnemoneMetadataParser.parse_filename(filename)
                
                # Add file path
                metadata['file_path'] = file_path
                
                # Add to list
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"Error extracting metadata from {file_path}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(metadata_list)
        
        return df
    
    def preprocess_audio_files(self, 
                              file_list: List[str], 
                              label: int) -> List[Tuple[np.ndarray, int]]:
        """Preprocess a list of audio files, applying standardization.
        
        Args:
            file_list: List of audio file paths
            label: Label value for these files (0 for noise, 1 for anemonefish)
            
        Returns:
            List of tuples containing (audio_segment, label)
        """
        all_segments = []
        
        for file_path in tqdm(file_list, desc=f"Preprocessing audio files (label={label})"):
            try:
                # Load audio
                audio_data, sr = AudioProcessor.load_audio(file_path, self.sr)
                
                # Skip if audio data is empty
                if len(audio_data) == 0:
                    continue
                
                # Normalize audio
                audio_data = AudioProcessor.normalize_audio(audio_data)
                
                # Process audio based on method
                if self.preprocess_method == PREPROCESS_STRETCH_SQUASH:
                    # Stretch or squash to standard length
                    processed_audio = AudioProcessor.standardize_audio_length(
                        audio_data, sr, self.standard_length_sec, PREPROCESS_STRETCH_SQUASH
                    )
                    all_segments.append((processed_audio, label))
                else:  # PREPROCESS_CROP_PAD
                    # Get multiple segments using sliding window
                    segments = AudioProcessor.get_multiple_segments(
                        audio_data, sr, self.standard_length_sec, SLIDING_WINDOW_HOP
                    )
                    
                    # Add all segments
                    for segment in segments:
                        all_segments.append((segment, label))
                
            except Exception as e:
                print(f"Error preprocessing audio file {file_path}: {e}")
        
        return all_segments
    
    def extract_and_cache_features(self, 
                                   audio_segments: List[Tuple[np.ndarray, int]], 
                                   cache_prefix: str) -> Tuple[Dict[str, np.ndarray], List[int]]:
        """Extract features from preprocessed audio segments and cache them.
        
        Args:
            audio_segments: List of tuples containing (audio_segment, label)
            cache_prefix: Prefix for cache files
            
        Returns:
            Tuple containing feature dictionary and list of labels
        """
        # Check if cache exists
        cache_file = os.path.join(self.cache_dir, f"{cache_prefix}_{self.feature_type}_features.npz")
        labels_file = os.path.join(self.cache_dir, f"{cache_prefix}_{self.feature_type}_labels.npy")
        
        if os.path.exists(cache_file) and os.path.exists(labels_file):
            # Load from cache
            feature_data = np.load(cache_file, allow_pickle=True)
            features = {key: feature_data[key] for key in feature_data.files}
            labels = np.load(labels_file)
            
            return features, labels.tolist()
        
        # Extract features for all segments
        features = {}
        labels_list = []
        
        for audio_data, label in tqdm(audio_segments, desc=f"Extracting {self.feature_type} features"):
            # Extract features based on feature type
            segment_features = AudioProcessor.extract_features(audio_data, self.sr, self.feature_type)
            
            # Skip if any feature is empty
            if any(len(feat) == 0 for feat in segment_features.values()):
                continue
            
            # Initialize features dictionary if not already done
            if not features:
                features = {key: [] for key in segment_features.keys()}
            
            # Add features
            for key, value in segment_features.items():
                features[key].append(value)
            
            # Add label
            labels_list.append(label)
        
        # Convert lists to arrays
        for key in features:
            if features[key]:
                features[key] = np.array(features[key])
        
        # Save to cache
        np.savez(cache_file, **features)
        np.save(labels_file, np.array(labels_list))
        
        return features, labels_list
    
    def augment_anemonefish_data(self, 
                               augmentation_factor: int = 5, 
                               use_noise_addition: bool = True,
                               random_seed: Optional[int] = 42) -> List[str]:
        """Augment anemonefish call data to increase the dataset size.
        
        Args:
            augmentation_factor: Number of augmented copies to create per original file
            use_noise_addition: Whether to add noise as part of the augmentation
            random_seed: Random seed for reproducibility
            
        Returns:
            List of paths to augmented files
        """
        # Create augmenter with appropriate parameters for anemonefish sounds
        augmenter = AudioAugmenter(
            sr=self.sr,
            seed=random_seed,
            preserve_length=True
        )
        
        # Create augmentation pipeline
        noise_dir = self.noise_chunked_dir if use_noise_addition else None
        pipeline = DataAugmentationPipeline(
            augmenter=augmenter,
            noise_dir=noise_dir,
            sr=self.sr,
            output_dir=self.augmented_dir,
            augmentation_factor=augmentation_factor,
            random_seed=random_seed
        )
        
        # Get list of anemonefish files
        anemonefish_files = self.list_anemonefish_files()
        
        # Extract metadata for these files for potential use
        metadata_df = self.build_metadata_dataframe()
        
        # Filter metadata to include only files we're processing
        filenames = [os.path.basename(f) for f in anemonefish_files]
        metadata_list = [
            metadata_df[metadata_df['original_filename'] == fname].to_dict('records')[0]
            if len(metadata_df[metadata_df['original_filename'] == fname]) > 0
            else None
            for fname in filenames
        ]
        
        # Augment the dataset
        print(f"Augmenting {len(anemonefish_files)} anemonefish call files...")
        result = pipeline.augment_dataset(
            file_list=anemonefish_files,
            save=True,
            metadata_list=metadata_list
        )
        
        print(f"Created {len(result['augmented_files'])} augmented files")
        
        return result['augmented_files']
    
    def list_augmented_files(self) -> List[str]:
        """List all augmented anemonefish call files.
        
        Returns:
            List of file paths
        """
        if not os.path.exists(self.augmented_dir):
            return []
            
        return [os.path.join(self.augmented_dir, f) 
                for f in os.listdir(self.augmented_dir) 
                if f.endswith('.wav') and not (f.startswith('.') or f.startswith('._'))]
    
    def prepare_dataset_with_augmentation(self, 
                                         test_size: float = 0.2, 
                                         use_augmentation: bool = True,
                                         balance_ratio: float = 1.0,  # 1:1 ratio of noise:anemonefish
                                         random_state: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Prepare a dataset for training and testing with optional augmentation.
        
        Args:
            test_size: Proportion of data to use for testing
            use_augmentation: Whether to use augmented files in training
            balance_ratio: Ratio of noise to anemonefish samples (e.g., 2.0 means 2x noise examples)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple containing training features, testing features, training labels, and testing labels
        """
        # Get lists of files
        anemonefish_files = self.list_anemonefish_files()
        
        # Get augmented files if using augmentation
        augmented_files = []
        if use_augmentation:
            augmented_files = self.list_augmented_files()
            
            # Generate augmented files if none exist
            if not augmented_files:
                print("No augmented files found. Generating augmented data...")
                augmented_files = self.augment_anemonefish_data(
                    random_seed=random_state
                )
        
        # Combine original and augmented files
        all_anemonefish_files = anemonefish_files + augmented_files
        
        # Get noise files
        noise_files = self.list_noise_files(use_chunked=True)
        
        # If no chunked noise files exist, chunk them
        if not noise_files:
            print("No chunked noise files found. Chunking noise files...")
            self.chunk_noise_files()
            noise_files = self.list_noise_files(use_chunked=True)
        
        # Analyze anemonefish call lengths to better understand the data
        if not augmented_files:  # Only analyze original files
            stats = self.analyze_anemonefish_call_lengths()
            
            # Suggest an appropriate standard length based on statistics
            suggested_length = max(stats['median'], 1.0)  # At least 1 second
            if abs(suggested_length - self.standard_length_sec) > 0.3:  # If significantly different
                print(f"Note: Based on the audio statistics, a standard length of {suggested_length:.1f} seconds")
                print(f"might be more appropriate than the current {self.standard_length_sec:.1f} seconds.")
        
        # Determine number of noise files to use based on balance ratio
        num_anemonefish = len(all_anemonefish_files)
        num_noise_to_use = min(int(num_anemonefish * balance_ratio), len(noise_files))
        
        # Randomly select noise files according to the balance ratio
        random.seed(random_state)
        selected_noise_files = random.sample(noise_files, num_noise_to_use)
        
        print(f"Using {num_anemonefish} anemonefish files (including {len(augmented_files)} augmented) and {num_noise_to_use} noise files")
        
        # Clear the cache directory to ensure fresh feature extraction
        cache_dir = self.cache_dir
        os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists
        if os.path.exists(cache_dir):
            for cache_file in os.listdir(cache_dir):
                if cache_file.endswith(f"_{self.feature_type}_features.npz") or cache_file.endswith(f"_{self.feature_type}_labels.npy"):
                    cache_path = os.path.join(cache_dir, cache_file)
                    if os.path.isfile(cache_path):
                        try:
                            os.remove(cache_path)
                        except (OSError, FileNotFoundError) as e:
                            print(f"Warning: Could not remove cache file {cache_path}: {e}")
        
        # Preprocess audio files
        print("\nPreprocessing and extracting features...")
        anemonefish_segments = self.preprocess_audio_files(all_anemonefish_files, 1)
        noise_segments = self.preprocess_audio_files(selected_noise_files, 0)
        
        # Extract features
        anemonefish_features, anemonefish_labels = self.extract_and_cache_features(
            anemonefish_segments, "anemonefish_augmented" if use_augmentation else "anemonefish"
        )
        
        noise_features, noise_labels = self.extract_and_cache_features(
            noise_segments, "noise"
        )
        
        # Get primary feature type (e.g., 'mel_spectrogram' if that's what we're using)
        primary_feature = None
        for key in anemonefish_features:
            if self.feature_type in key:
                primary_feature = key
                break
        
        if primary_feature is None:
            primary_feature = list(anemonefish_features.keys())[0]
            print(f"Warning: Requested feature type '{self.feature_type}' not found. Using '{primary_feature}' instead.")
        
        # Create a single feature dictionary with just the primary feature
        features = {
            primary_feature: np.concatenate([
                anemonefish_features[primary_feature], 
                noise_features[primary_feature]
            ], axis=0)
        }
        
        labels = np.array(anemonefish_labels + noise_labels)
        
        # Print feature shape information
        print(f"\nFeature shape: {features[primary_feature].shape}")
        print(f"Number of samples: {len(labels)}")
        print(f"Number of positive samples (anemonefish): {sum(labels == 1)}")
        print(f"Number of negative samples (noise): {sum(labels == 0)}\n")
        
        # Split data into training and testing sets
        X_train = {}
        X_test = {}
        
        # Use stratified sampling to maintain class proportions
        indices = np.arange(len(labels))
        train_indices, test_indices, y_train, y_test = train_test_split(
            indices, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Apply indices to features
        for key in features:
            X_train[key] = features[key][train_indices]
            X_test[key] = features[key][test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def visualize_features(self, 
                          feature_type: str = None, 
                          num_samples: int = 5, 
                          random_state: int = 42) -> None:
        """Visualize features for a random sample of files.
        
        Args:
            feature_type: Type of feature to visualize (defaults to self.feature_type)
            num_samples: Number of samples to visualize
            random_state: Random seed for reproducibility
        """
        # Use instance feature type if not specified
        if feature_type is None:
            feature_type = self.feature_type
            
        # Set random seed
        random.seed(random_state)
        
        # Get lists of files
        anemonefish_files = self.list_anemonefish_files()
        noise_files = self.list_noise_files(use_chunked=True)
        
        # Select random files
        selected_anemonefish = random.sample(anemonefish_files, min(num_samples, len(anemonefish_files)))
        selected_noise = random.sample(noise_files, min(num_samples, len(noise_files)))
        
        # Set up plot with adaptable layout
        if num_samples == 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            axes = axes.reshape(2, 1)  # Ensure 2D shape
        else:
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        # Plot anemonefish features
        for i, file_path in enumerate(selected_anemonefish):
            # Load audio
            audio_data, sr = AudioProcessor.load_audio(file_path, self.sr)
            
            # Normalize audio
            audio_data = AudioProcessor.normalize_audio(audio_data)
            
            # Standardize audio length
            audio_data = AudioProcessor.standardize_audio_length(
                audio_data, sr, self.standard_length_sec, self.preprocess_method
            )
            
            # Extract features
            file_features = AudioProcessor.extract_features(audio_data, sr, feature_type)
            
            # Get feature to plot
            feature = None
            for key in file_features:
                if feature_type in key:
                    feature = file_features[key]
                    break
            
            if feature is None and file_features:
                # If exact match not found, use the first available feature
                feature = file_features[list(file_features.keys())[0]]
                feature_type = list(file_features.keys())[0]
            
            if feature is not None and len(feature) > 0:
                # For spectrograms, use log scale for better visualization
                if 'mel_spectrogram' in feature_type:
                    # Convert to dB if not already
                    if 'log' not in feature_type:
                        feature = librosa.power_to_db(feature, ref=np.max)
                
                # Different visualization based on feature type
                if feature_type == 'rms':
                    # For RMS, just plot the values directly
                    axes[0, i].plot(feature.T)
                    axes[0, i].set_title(f"Anemonefish {i+1} - RMS")
                    axes[0, i].set_xlabel('Time')
                    axes[0, i].set_ylabel('RMS Energy')
                else:
                    # For spectral features, use specshow
                    img = librosa.display.specshow(
                        feature, 
                        ax=axes[0, i], 
                        sr=sr, 
                        hop_length=SPEC_HOP_LENGTH,
                        x_axis='time',
                        y_axis='mel' if 'mel' in feature_type else 'log' if 'spectrogram' in feature_type else 'linear'
                    )
                    
                    # Add colorbar
                    if i == num_samples - 1:
                        fig.colorbar(img, ax=axes[0, i], format='%+2.0f dB' if 'log' in feature_type else '%+2.0f')
                    
                    axes[0, i].set_title(f"Anemonefish {i+1}")
        
        # Plot noise features
        for i, file_path in enumerate(selected_noise):
            # Load audio
            audio_data, sr = AudioProcessor.load_audio(file_path, self.sr)
            
            # Normalize audio
            audio_data = AudioProcessor.normalize_audio(audio_data)
            
            # Standardize audio length
            audio_data = AudioProcessor.standardize_audio_length(
                audio_data, sr, self.standard_length_sec, self.preprocess_method
            )
            
            # Extract features
            file_features = AudioProcessor.extract_features(audio_data, sr, feature_type)
            
            # Get feature to plot
            feature = None
            for key in file_features:
                if feature_type in key:
                    feature = file_features[key]
                    break
            
            if feature is None and file_features:
                # If exact match not found, use the first available feature
                feature = file_features[list(file_features.keys())[0]]
                feature_type = list(file_features.keys())[0]
            
            if feature is not None and len(feature) > 0:
                # For spectrograms, use log scale for better visualization
                if 'mel_spectrogram' in feature_type and 'log' not in feature_type:
                    feature = librosa.power_to_db(feature, ref=np.max)
                
                # Different visualization based on feature type
                if feature_type == 'rms':
                    # For RMS, just plot the values directly
                    axes[1, i].plot(feature.T)
                    axes[1, i].set_title(f"Noise {i+1} - RMS")
                    axes[1, i].set_xlabel('Time')
                    axes[1, i].set_ylabel('RMS Energy')
                else:
                    # For spectral features, use specshow
                    img = librosa.display.specshow(
                        feature, 
                        ax=axes[1, i], 
                        sr=sr, 
                        hop_length=SPEC_HOP_LENGTH,
                        x_axis='time',
                        y_axis='mel' if 'mel' in feature_type else 'log' if 'spectrogram' in feature_type else 'linear'
                    )
                    
                    # Add colorbar
                    if i == num_samples - 1:
                        fig.colorbar(img, ax=axes[1, i], format='%+2.0f dB' if 'log' in feature_type else '%+2.0f')
                    
                    axes[1, i].set_title(f"Noise {i+1}")
        
        plt.tight_layout()
        plt.show()
    
    def prepare_data_for_model(self, X_train: Dict[str, np.ndarray], 
                              X_test: Dict[str, np.ndarray], 
                              y_train: np.ndarray, 
                              y_test: np.ndarray, 
                              feature_type: str = None, 
                              model_type: str = 'cnn') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for different model architectures following standard conventions.
        
        Args:
            X_train: Dictionary of training features
            X_test: Dictionary of testing features
            y_train: Training labels
            y_test: Testing labels
            feature_type: Type of feature to use (defaults to self.feature_type)
            model_type: Type of model ('cnn', 'lstm', or 'mlp')
            
        Returns:
            Tuple containing prepared training data, testing data, training labels, and testing labels
        """
        # Use instance feature type if not specified
        if feature_type is None:
            feature_type = self.feature_type
            
        # Get the appropriate feature key from the dictionary
        feature_key = None
        for key in X_train:
            if feature_type in key:
                feature_key = key
                break
                
        if feature_key is None:
            # If no matching feature found, use the first available one
            feature_key = list(X_train.keys())[0]
            print(f"Warning: Requested feature '{feature_type}' not found. Using '{feature_key}' instead.")
        
        # For log-mel spectrograms, ensure the values are in dB
        if 'mel_spectrogram' in feature_key and 'log' not in feature_key:
            X_train_data = librosa.power_to_db(X_train[feature_key], ref=np.max)
            X_test_data = librosa.power_to_db(X_test[feature_key], ref=np.max)
        else:
            X_train_data = X_train[feature_key]
            X_test_data = X_test[feature_key]
        
        # Prepare data for different model types
        if model_type == 'cnn':
            # For CNN: (samples, channels, height, width) - PyTorch format
            # or (samples, height, width, channels) - TensorFlow format
            # We'll use PyTorch format as that's what the models are using
            
            # Data is currently (samples, freq_bins, time_frames)
            # Need to add channel dimension: (samples, 1, freq_bins, time_frames)
            X_train_data = X_train_data[:, np.newaxis, :, :]
            X_test_data = X_test_data[:, np.newaxis, :, :]
            
        elif model_type == 'lstm':
            # For LSTM: (samples, time_steps, features)
            # Need to transpose from (samples, freq_bins, time_frames) to (samples, time_frames, freq_bins)
            X_train_data = X_train_data.transpose(0, 2, 1)
            X_test_data = X_test_data.transpose(0, 2, 1)
        
        elif model_type == 'mlp':
            # For MLP, flatten everything except batch dimension
            X_train_data = X_train_data.reshape(X_train_data.shape[0], -1)
            X_test_data = X_test_data.reshape(X_test_data.shape[0], -1)
        
        elif model_type == 'crnn':
            # For CRNN: Same as CNN but might need specific shape adjustments
            # Data is currently (samples, freq_bins, time_frames)
            # Need to add channel dimension: (samples, 1, freq_bins, time_frames)
            X_train_data = X_train_data[:, np.newaxis, :, :]
            X_test_data = X_test_data[:, np.newaxis, :, :]
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Ensure labels are appropriate for binary classification
        # For binary classification in PyTorch, shape should be (samples,)
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
        
        return X_train_data, X_test_data, y_train, y_test
    
    def visualize_preprocessing_stages(self, 
                                    file_path: str, 
                                    output_dir: str,
                                    num_augmentations: int = 3) -> None:
        """Visualize all preprocessing stages for a given audio file.
        
        This method loads an audio file, applies all preprocessing steps,
        captures the intermediate results, and visualizes them.
        
        Args:
            file_path: Path to the audio file
            output_dir: Directory to save visualizations
            num_augmentations: Number of different augmentations to visualize
        """
        from anemonefish_acoustics.data_processing.data_augmentation import AudioAugmenter
        from anemonefish_acoustics.utils.visualization import plot_preprocessing_stages, plot_augmentation_comparison
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Dictionary to store audio at different stages
            audio_stages = {}
            feature_stages = {}
            
            # Step 1: Load original audio
            audio_data, sr = AudioProcessor.load_audio(file_path, self.sr)
            audio_stages["1_raw"] = audio_data.copy()
            
            # Step 2: Normalize audio
            normalized_audio = AudioProcessor.normalize_audio(audio_data)
            audio_stages["2_normalized"] = normalized_audio.copy()
            
            # Step 3: Standardize audio length
            processed_audio = AudioProcessor.standardize_audio_length(
                normalized_audio, sr, self.standard_length_sec, self.preprocess_method
            )
            audio_stages["3_standardized"] = processed_audio.copy()
            
            # Extract features for each stage
            # For raw audio (might be variable length)
            raw_features = AudioProcessor.extract_features(audio_stages["1_raw"], sr, self.feature_type)
            for key, value in raw_features.items():
                if key == self.feature_type or key == 'log_mel_spectrogram':  # Only use primary feature type
                    feature_stages[f"1_raw_{key}"] = value
            
            # For normalized audio (still variable length)
            norm_features = AudioProcessor.extract_features(audio_stages["2_normalized"], sr, self.feature_type)
            for key, value in norm_features.items():
                if key == self.feature_type or key == 'log_mel_spectrogram':  # Only use primary feature type
                    feature_stages[f"2_normalized_{key}"] = value
            
            # For standardized audio (fixed length)
            std_features = AudioProcessor.extract_features(audio_stages["3_standardized"], sr, self.feature_type)
            for key, value in std_features.items():
                if key == self.feature_type or key == 'log_mel_spectrogram':  # Only use primary feature type
                    feature_stages[f"3_standardized_{key}"] = value
            
            # Visualize preprocessing stages
            preprocessing_output_path = os.path.join(output_dir, "preprocessing_stages.png")
            plot_preprocessing_stages(
                audio_stages=audio_stages,
                feature_stages=feature_stages,
                sr=sr,
                hop_length=HOP_LENGTH,
                fmin=FMIN,
                fmax=FMAX,
                title=f"Preprocessing Stages: {os.path.basename(file_path)}",
                output_path=preprocessing_output_path,
                feature_type=self.feature_type,
                duration=self.standard_length_sec  # Pass the standard_length_sec as duration
            )
            
            # Visualize augmentations
            augmenter = AudioAugmenter(sr=self.sr)
            
            # Generate some random noise for the add_noise augmentation
            noise_data = np.random.randn(len(processed_audio))
            
            # Common augmentation types to visualize
            augmentation_types = [
                "time_stretch",
                "pitch_shift",
                "time_shift",
                "add_noise",
                "frequency_mask",
                "time_mask",
                "simulate_multipath"
            ]
            
            for i, aug_type in enumerate(augmentation_types[:num_augmentations]):
                try:
                    # Apply augmentation to standardized audio
                    if aug_type == "time_stretch":
                        aug_audio = augmenter.time_stretch(processed_audio.copy(), rate_range=(0.85, 0.85))
                    elif aug_type == "pitch_shift":
                        aug_audio = augmenter.pitch_shift(processed_audio.copy(), n_steps_range=(2.0, 2.0))
                    elif aug_type == "time_shift":
                        aug_audio = augmenter.time_shift(processed_audio.copy(), shift_range=(0.1, 0.1))
                    elif aug_type == "add_noise":
                        aug_audio = augmenter.add_noise(processed_audio.copy(), noise_data, snr_range=(15.0, 15.0))
                    elif aug_type == "frequency_mask":
                        aug_audio = augmenter.frequency_mask(processed_audio.copy(), max_mask_width=15, fmin=100, fmax=1000)
                    elif aug_type == "time_mask":
                        aug_audio = augmenter.time_mask(processed_audio.copy(), mask_ratio_range=(0.1, 0.1))
                    elif aug_type == "simulate_multipath":
                        aug_audio = augmenter.simulate_multipath(processed_audio.copy(), n_reflections_range=(2, 2))
                    else:
                        continue
                    
                    # Extract features from augmented audio
                    aug_features = AudioProcessor.extract_features(aug_audio, sr, self.feature_type)
                    
                    # Visualize original vs augmented audio and features
                    aug_output_path = os.path.join(output_dir, f"augmentation_{aug_type}.png")
                    plot_augmentation_comparison(
                        original_audio=processed_audio,
                        augmented_audio=aug_audio,
                        original_feature=std_features[self.feature_type],
                        augmented_feature=aug_features[self.feature_type],
                        augmentation_name=aug_type,
                        sr=sr,
                        hop_length=HOP_LENGTH,
                        fmin=FMIN,
                        fmax=FMAX,
                        output_path=aug_output_path,
                        feature_type=self.feature_type,
                        duration=self.standard_length_sec  # Pass the standard_length_sec as duration
                    )
                    
                except Exception as e:
                    print(f"Error visualizing augmentation {aug_type}: {e}")
            
            return preprocessing_output_path
            
        except Exception as e:
            print(f"Error visualizing preprocessing stages for {file_path}: {e}")
            return None 