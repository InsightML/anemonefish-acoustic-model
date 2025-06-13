import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import random
from pathlib import Path
import scipy.signal


class AudioAugmenter:
    """Class containing methods for augmenting acoustic data of anemonefish.
    
    This class provides various augmentation techniques specifically designed
    for underwater acoustic recordings of anemonefish, with parameters
    tuned for the frequency ranges and characteristics of these sounds.
    """
    
    def __init__(self, 
                 sr: int = 8000, 
                 seed: Optional[int] = None,
                 preserve_length: bool = True):
        """Initialize the AudioAugmenter.
        
        Args:
            sr: Sample rate of the audio data
            seed: Random seed for reproducibility
            preserve_length: Whether to preserve the original length when augmenting
        """
        self.sr = sr
        self.preserve_length = preserve_length
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def time_stretch(self, 
                     audio: np.ndarray, 
                     rate_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply time stretching to audio.
        
        Args:
            audio: Audio data to augment
            rate_range: Range of stretching rates (values < 1 stretch, values > 1 compress)
            
        Returns:
            Time-stretched audio data
        """
        # Choose random rate from the specified range
        rate = np.random.uniform(rate_range[0], rate_range[1])
        
        # Apply time stretching
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # If preserving length, we need to either trim or pad
        if self.preserve_length and len(stretched_audio) != len(audio):
            if len(stretched_audio) > len(audio):
                # Trim
                stretched_audio = stretched_audio[:len(audio)]
            else:
                # Pad with zeros
                padding = np.zeros(len(audio) - len(stretched_audio))
                stretched_audio = np.concatenate([stretched_audio, padding])
        
        return stretched_audio
    
    def time_shift(self, 
                   audio: np.ndarray, 
                   shift_range: Tuple[float, float] = (-0.25, 0.25)) -> np.ndarray:
        """Shift audio in time.
        
        Args:
            audio: Audio data to augment
            shift_range: Range of shift as a fraction of the total length
            
        Returns:
            Time-shifted audio data
        """
        # Calculate maximum shift in samples
        max_shift = int(len(audio) * max(abs(shift_range[0]), abs(shift_range[1])))
        
        # Choose random shift from the specified range
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        # Create output array
        shifted_audio = np.zeros_like(audio)
        
        if shift > 0:
            # Shift right
            shifted_audio[shift:] = audio[:-shift]
        elif shift < 0:
            # Shift left
            shifted_audio[:shift] = audio[-shift:]
        else:
            # No shift
            shifted_audio = audio.copy()
        
        return shifted_audio
    
    def pitch_shift(self, 
                    audio: np.ndarray, 
                    n_steps_range: Tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
        """Apply pitch shifting to audio.
        
        Args:
            audio: Audio data to augment
            n_steps_range: Range of pitch shift in semitones
            
        Returns:
            Pitch-shifted audio data
        """
        # Choose random number of semitones from the specified range
        n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
        
        # Apply pitch shifting
        shifted_audio = librosa.effects.pitch_shift(
            audio, 
            sr=self.sr, 
            n_steps=n_steps
        )
        
        # Ensure same length
        if self.preserve_length and len(shifted_audio) != len(audio):
            if len(shifted_audio) > len(audio):
                shifted_audio = shifted_audio[:len(audio)]
            else:
                padding = np.zeros(len(audio) - len(shifted_audio))
                shifted_audio = np.concatenate([shifted_audio, padding])
        
        return shifted_audio
    
    def volume_perturbation(self, 
                           audio: np.ndarray, 
                           gain_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Apply random volume change to audio.
        
        Args:
            audio: Audio data to augment
            gain_range: Range of gain factors to apply
            
        Returns:
            Volume-perturbed audio data
        """
        # Choose random gain from the specified range
        gain = np.random.uniform(gain_range[0], gain_range[1])
        
        # Apply gain
        augmented_audio = audio * gain
        
        # Clip to avoid distortion
        augmented_audio = np.clip(augmented_audio, -1.0, 1.0)
        
        return augmented_audio
    
    def add_noise(self, 
                 audio: np.ndarray, 
                 noise: np.ndarray, 
                 snr_range: Tuple[float, float] = (5.0, 15.0)) -> np.ndarray:
        """Add noise to audio at a specified SNR.
        
        Args:
            audio: Audio data to augment
            noise: Noise data to add
            snr_range: Range of signal-to-noise ratios in dB
            
        Returns:
            Noisy audio data
        """
        # Choose random SNR from the specified range
        snr = np.random.uniform(snr_range[0], snr_range[1])
        
        # Ensure noise is at least as long as audio
        if len(noise) < len(audio):
            # Pad noise with itself repeated
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
            noise = noise[:len(audio)]
        else:
            # Randomly select a segment of noise
            start = np.random.randint(0, len(noise) - len(audio) + 1)
            noise = noise[start:start + len(audio)]
        
        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Calculate noise scale factor for desired SNR
        if noise_power > 0:
            scale = np.sqrt(signal_power / (noise_power * 10 ** (snr / 10)))
            scaled_noise = noise * scale
        else:
            scaled_noise = noise
        
        # Add noise to signal
        noisy_audio = audio + scaled_noise
        
        # Normalize to avoid clipping
        if np.max(np.abs(noisy_audio)) > 1.0:
            noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
        
        return noisy_audio
    
    def frequency_mask(self, 
                      audio: np.ndarray, 
                      max_mask_width: int = 50, 
                      n_masks: int = 1,
                      fmin: int = 75,
                      fmax: int = 1800) -> np.ndarray:
        """Apply frequency masking by filtering out random frequency bands.
        
        Args:
            audio: Audio data to augment
            max_mask_width: Maximum mask width in Hz
            n_masks: Number of masks to apply
            fmin: Minimum frequency to consider for masking (Hz)
            fmax: Maximum frequency to consider for masking (Hz)
            
        Returns:
            Frequency-masked audio data
        """
        # Make a copy of the input audio
        masked_audio = audio.copy()
        
        # Calculate frequency resolution
        freq_range = fmax - fmin
        
        # Apply multiple masks
        for _ in range(n_masks):
            # Choose random mask width - ensure min_width is at least 1
            min_width = max(1, freq_range // 20)  # Avoid zero or negative values
            max_width = min(freq_range - 1, max_mask_width)  # Ensure max width is less than freq_range
            
            # Skip masking if we can't create a valid mask
            if min_width >= max_width:
                continue
                
            mask_width = np.random.randint(min_width, max_width + 1)
            
            # Choose random mask start frequency - ensure enough room for the mask
            max_start = fmax - mask_width
            
            # Skip masking if we can't create a valid mask
            if fmin >= max_start:
                continue
                
            mask_start = np.random.randint(fmin, max_start + 1)
            mask_end = mask_start + mask_width
            
            # Create bandstop filter
            b, a = scipy.signal.butter(
                N=4,
                Wn=[mask_start / (self.sr / 2), mask_end / (self.sr / 2)],
                btype='bandstop'
            )
            
            # Apply filter
            masked_audio = scipy.signal.filtfilt(b, a, masked_audio)
        
        return masked_audio
    
    def time_mask(self, 
                 audio: np.ndarray, 
                 mask_ratio_range: Tuple[float, float] = (0.05, 0.15),
                 n_masks: int = 1) -> np.ndarray:
        """Apply time masking by zeroing out random time segments.
        
        Args:
            audio: Audio data to augment
            mask_ratio_range: Range of mask lengths as fraction of total length
            n_masks: Number of masks to apply
            
        Returns:
            Time-masked audio data
        """
        # Make a copy of the input audio
        masked_audio = audio.copy()
        
        # Apply multiple masks
        for _ in range(n_masks):
            # Choose random mask length
            mask_ratio = np.random.uniform(mask_ratio_range[0], mask_ratio_range[1])
            mask_length = int(len(audio) * mask_ratio)
            
            # Choose random mask start
            mask_start = np.random.randint(0, len(audio) - mask_length + 1)
            
            # Apply mask (zero out the segment)
            masked_audio[mask_start:mask_start + mask_length] = 0
        
        return masked_audio
    
    def simulate_multipath(self, 
                          audio: np.ndarray, 
                          n_reflections_range: Tuple[int, int] = (1, 3),
                          delay_range: Tuple[float, float] = (0.005, 0.02),
                          decay_factor_range: Tuple[float, float] = (0.2, 0.5)) -> np.ndarray:
        """Simulate underwater multipath propagation by adding delayed and attenuated copies.
        
        Args:
            audio: Audio data to augment
            n_reflections_range: Range of number of reflections to add
            delay_range: Range of delay times in seconds
            decay_factor_range: Range of amplitude decay factors for each reflection
            
        Returns:
            Audio with simulated multipath effects
        """
        # Make a copy of the input audio
        result = audio.copy()
        
        # Choose random number of reflections
        n_reflections = np.random.randint(n_reflections_range[0], n_reflections_range[1] + 1)
        
        # Add each reflection
        for i in range(n_reflections):
            # Choose random delay
            delay_sec = np.random.uniform(delay_range[0], delay_range[1])
            delay_samples = int(delay_sec * self.sr)
            
            # Choose random decay factor
            decay = np.random.uniform(decay_factor_range[0], decay_factor_range[1])
            decay_factor = decay ** (i + 1)  # Progressive decay for later reflections
            
            # Create delayed signal
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] if delay_samples < len(audio) else audio[0:0]
            
            # Add to result with decay
            result += delayed * decay_factor
        
        # Normalize to avoid clipping
        if np.max(np.abs(result)) > 1.0:
            result = result / np.max(np.abs(result))
        
        return result
    
    def apply_random_augmentation(self, 
                                 audio: np.ndarray, 
                                 noise_pool: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """Apply a random combination of augmentations.
        
        Args:
            audio: Audio data to augment
            noise_pool: Pool of noise samples to use for adding noise
            
        Returns:
            Tuple containing augmented audio and list of applied augmentations
        """
        # List to keep track of applied augmentations
        applied_augmentations = []
        
        # Make a copy of the input audio
        augmented = audio.copy()
        
        # Decide which augmentations to apply (with probabilities)
        augmentations = [
            ('time_stretch', 0.5),
            ('time_shift', 0.7),
            ('pitch_shift', 0.6),
            ('volume_perturbation', 0.7),
            ('frequency_mask', 0.4),
            ('time_mask', 0.3),
            ('simulate_multipath', 0.4)
        ]
        
        # Add noise if noise pool is provided
        if noise_pool is not None and len(noise_pool) > 0:
            augmentations.append(('add_noise', 0.6))
        
        # Apply random augmentations
        for aug_name, prob in augmentations:
            if np.random.random() < prob:
                if aug_name == 'time_stretch':
                    augmented = self.time_stretch(augmented)
                    applied_augmentations.append('time_stretch')
                    
                elif aug_name == 'time_shift':
                    augmented = self.time_shift(augmented)
                    applied_augmentations.append('time_shift')
                    
                elif aug_name == 'pitch_shift':
                    augmented = self.pitch_shift(augmented)
                    applied_augmentations.append('pitch_shift')
                    
                elif aug_name == 'volume_perturbation':
                    augmented = self.volume_perturbation(augmented)
                    applied_augmentations.append('volume_perturbation')
                    
                elif aug_name == 'add_noise' and noise_pool is not None and len(noise_pool) > 0:
                    # Select random noise segment
                    noise_idx = np.random.randint(0, len(noise_pool))
                    noise = noise_pool[noise_idx]
                    augmented = self.add_noise(augmented, noise)
                    applied_augmentations.append('add_noise')
                    
                elif aug_name == 'frequency_mask':
                    augmented = self.frequency_mask(augmented)
                    applied_augmentations.append('frequency_mask')
                    
                elif aug_name == 'time_mask':
                    augmented = self.time_mask(augmented)
                    applied_augmentations.append('time_mask')
                    
                elif aug_name == 'simulate_multipath':
                    augmented = self.simulate_multipath(augmented)
                    applied_augmentations.append('simulate_multipath')
        
        return augmented, applied_augmentations


class DataAugmentationPipeline:
    """Pipeline for augmenting a dataset of audio files."""
    
    def __init__(self, 
                 augmenter: AudioAugmenter,
                 noise_dir: Optional[str] = None,
                 sr: int = 8000,
                 output_dir: Optional[str] = None,
                 augmentation_factor: int = 5,
                 random_seed: Optional[int] = None):
        """Initialize the DataAugmentationPipeline.
        
        Args:
            augmenter: AudioAugmenter instance to use for augmentation
            noise_dir: Directory containing noise audio files
            sr: Sample rate for audio processing
            output_dir: Directory for saving augmented files
            augmentation_factor: Number of augmented copies to create per original file
            random_seed: Random seed for reproducibility
        """
        self.augmenter = augmenter
        self.noise_dir = noise_dir
        self.sr = sr
        self.output_dir = output_dir
        self.augmentation_factor = augmentation_factor
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Cache for noise data
        self.noise_pool = None
        
        # Load noise data if provided
        if noise_dir is not None:
            self._load_noise_pool()
    
    def _load_noise_pool(self, max_files: int = 50) -> None:
        """Load a pool of noise samples for augmentation.
        
        Args:
            max_files: Maximum number of noise files to load
        """
        if self.noise_dir is None or not os.path.exists(self.noise_dir):
            print(f"Noise directory not found: {self.noise_dir}")
            return
        
        noise_files = [
            os.path.join(self.noise_dir, f) 
            for f in os.listdir(self.noise_dir)
            if f.endswith('.wav') and not (f.startswith('.') or f.startswith('._'))
        ]
        
        if not noise_files:
            print(f"No noise files found in {self.noise_dir}")
            return
        
        # Randomly select a subset if there are too many
        if len(noise_files) > max_files:
            noise_files = random.sample(noise_files, max_files)
        
        # Load noise files
        self.noise_pool = []
        for file_path in noise_files:
            try:
                audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
                if len(audio) > 0:
                    self.noise_pool.append(audio)
            except Exception as e:
                print(f"Error loading noise file {file_path}: {e}")
    
    def augment_file(self, 
                    file_path: str, 
                    save: bool = True,
                    metadata: Optional[Dict] = None) -> List[Tuple[np.ndarray, List[str], str]]:
        """Augment a single audio file.
        
        Args:
            file_path: Path to the audio file
            save: Whether to save the augmented files
            metadata: Optional metadata dictionary for the file
            
        Returns:
            List of tuples with (augmented_audio, applied_augmentations, output_path)
        """
        try:
            # Load audio
            audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
            
            # Skip if audio is empty
            if len(audio) == 0:
                print(f"Warning: Empty audio file {file_path}")
                return []
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # Get base filename for output
            base_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            
            # Result list
            results = []
            
            # Create augmented versions
            for i in range(self.augmentation_factor):
                # Apply random augmentations
                augmented, applied_augs = self.augmenter.apply_random_augmentation(audio, self.noise_pool)
                
                # Create output filename
                if save and self.output_dir is not None:
                    # Create output directory if it doesn't exist
                    os.makedirs(self.output_dir, exist_ok=True)
                    
                    # Generate output path
                    output_path = os.path.join(
                        self.output_dir, 
                        f"{file_name_without_ext}_aug_{i+1}_{'-'.join(applied_augs)}.wav"
                    )
                    
                    # Save augmented audio
                    sf.write(output_path, augmented, self.sr)
                else:
                    output_path = None
                
                # Add to results
                results.append((augmented, applied_augs, output_path))
            
            return results
            
        except Exception as e:
            print(f"Error augmenting file {file_path}: {e}")
            return []
    
    def augment_dataset(self, 
                       file_list: List[str], 
                       save: bool = True,
                       metadata_list: Optional[List[Dict]] = None) -> Dict[str, List]:
        """Augment a list of audio files.
        
        Args:
            file_list: List of audio file paths
            save: Whether to save the augmented files
            metadata_list: Optional list of metadata dictionaries for the files
            
        Returns:
            Dictionary with lists of augmented files and applied augmentations
        """
        if metadata_list is not None and len(metadata_list) != len(file_list):
            raise ValueError("metadata_list must have the same length as file_list")
        
        augmented_files = []
        all_augmentations = []
        
        for i, file_path in enumerate(file_list):
            metadata = metadata_list[i] if metadata_list is not None else None
            
            # Augment file
            results = self.augment_file(file_path, save, metadata)
            
            for augmented, applied_augs, output_path in results:
                if output_path is not None:
                    augmented_files.append(output_path)
                all_augmentations.append(applied_augs)
        
        return {
            'augmented_files': augmented_files,
            'applied_augmentations': all_augmentations
        }


# Utility function for mixing examples (mixup)
def mixup(audio1: np.ndarray, 
          audio2: np.ndarray, 
          alpha: float = 0.2) -> np.ndarray:
    """Mix two audio signals using mixup technique.
    
    Args:
        audio1: First audio signal
        audio2: Second audio signal
        alpha: Mixup parameter controlling the mixing ratio
        
    Returns:
        Mixed audio signal
    """
    # Ensure both audios have the same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Generate mixing ratio from beta distribution
    from numpy.random import beta
    ratio = beta(alpha, alpha)
    
    # Mix the signals
    mixed = ratio * audio1 + (1 - ratio) * audio2
    
    # Normalize to avoid clipping
    if np.max(np.abs(mixed)) > 1.0:
        mixed = mixed / np.max(np.abs(mixed))
    
    return mixed 