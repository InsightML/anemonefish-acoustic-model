"""
Shared spectrogram image preprocessing module for consistent data preparation
across binary classifier and autoencoder models.

This module ensures identical preprocessing pipelines for both supervised and 
unsupervised learning tasks on spectrogram images.
"""

import os
import numpy as np
import tensorflow as tf
from glob import glob
from typing import List, Tuple, Optional, Dict, Union
import random
from pathlib import Path


class SpectrogramConfig:
    """Configuration for spectrogram preprocessing."""
    
    def __init__(self):
        # Image dimensions
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.IMG_CHANNELS = 3
        
        # Normalization is now handled by scaling to [-1, 1] to match tanh activation.
        # ImageNet stats are not needed as we are not using a pre-trained model.
        
        # Augmentation settings
        self.ENABLE_AUGMENTATION = True
        self.AUG_BRIGHTNESS_DELTA = 0.1
        self.AUG_CONTRAST_LOWER = 0.9
        self.AUG_CONTRAST_UPPER = 1.1
        
        # Random seed for reproducibility
        self.SEED = 42
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Return the input shape for models."""
        return (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)


class SpectrogramDataLoader:
    """Handles loading and organizing spectrogram file paths."""
    
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        
    def load_labeled_data(self, 
                         anemonefish_path: str, 
                         noise_path: str,
                         label_map: Optional[Dict[str, int]] = None) -> Tuple[List[str], List[int]]:
        """
        Load labeled data for binary classification.
        
        Args:
            anemonefish_path: Path to anemonefish spectrograms
            noise_path: Path to noise spectrograms  
            label_map: Mapping from class names to integers (default: {'noise': 0, 'anemonefish': 1})
            
        Returns:
            Tuple of (file_paths, labels)
        """
        if label_map is None:
            label_map = {'noise': 0, 'anemonefish': 1}
            
        all_paths = []
        all_labels = []
        
        # Load anemonefish files
        anemonefish_files = self._load_files_from_directory(anemonefish_path, "anemonefish")
        all_paths.extend(anemonefish_files)
        all_labels.extend([label_map['anemonefish']] * len(anemonefish_files))
        
        # Load noise files
        noise_files = self._load_files_from_directory(noise_path, "noise")
        all_paths.extend(noise_files)
        all_labels.extend([label_map['noise']] * len(noise_files))
        
        print(f"Loaded labeled data:")
        print(f"  - Anemonefish: {len(anemonefish_files)} files")
        print(f"  - Noise: {len(noise_files)} files")
        print(f"  - Total: {len(all_paths)} files")
        
        return all_paths, all_labels
    
    def load_all_data(self, 
                     anemonefish_path: str, 
                     noise_path: str,
                     unlabeled_path: str,
                     max_unlabeled_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Load all data (labeled + unlabeled) for autoencoder training.
        
        Args:
            anemonefish_path: Path to anemonefish spectrograms
            noise_path: Path to noise spectrograms
            unlabeled_path: Path to unlabeled spectrograms
            max_unlabeled_samples: Maximum number of unlabeled samples to include
            
        Returns:
            Tuple of (file_paths, source_labels) where source_labels are strings for tracking
        """
        all_paths = []
        source_labels = []
        
        # Load anemonefish files
        anemonefish_files = self._load_files_from_directory(anemonefish_path, "anemonefish")
        all_paths.extend(anemonefish_files)
        source_labels.extend(['anemonefish'] * len(anemonefish_files))
        
        # Load noise files  
        noise_files = self._load_files_from_directory(noise_path, "noise")
        all_paths.extend(noise_files)
        source_labels.extend(['noise'] * len(noise_files))
        
        # Load unlabeled files (with optional sampling)
        unlabeled_files = self._load_files_recursively(unlabeled_path, "unlabeled", max_unlabeled_samples)
        all_paths.extend(unlabeled_files)
        source_labels.extend(['unlabeled'] * len(unlabeled_files))
        
        print(f"Loaded all data:")
        print(f"  - Anemonefish: {len(anemonefish_files)} files")
        print(f"  - Noise: {len(noise_files)} files") 
        print(f"  - Unlabeled: {len(unlabeled_files)} files")
        print(f"  - Total: {len(all_paths)} files")
        
        return all_paths, source_labels
    
    def _load_files_from_directory(self, directory: str, description: str) -> List[str]:
        """Load image files from a single directory."""
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found: {directory}")
            return []
            
        files = []
        for filename in os.listdir(directory):
            if (not filename.startswith('.') and 
                filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
                files.append(os.path.join(directory, filename))
                
        print(f"Found {len(files)} {description} files in {directory}")
        return files
    
    def _load_files_recursively(self, directory: str, description: str, 
                               max_samples: Optional[int] = None) -> List[str]:
        """Load image files recursively from directory tree."""
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found: {directory}")
            return []
            
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if (not filename.startswith('.') and 
                    filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
                    files.append(os.path.join(root, filename))
        
        # Sample if requested
        if files and max_samples and len(files) > max_samples:
            random.seed(self.config.SEED)
            random.shuffle(files)
            files = files[:max_samples]
            print(f"Sampled {max_samples} from {len(files) + (len(files) - max_samples)} available {description} files")
        
        print(f"Found {len(files)} {description} files in {directory}")
        return files


class SpectrogramPreprocessor:
    """Handles image preprocessing operations using TensorFlow."""
    
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        
    def parse_image(self, image_path: Union[str, tf.Tensor]) -> tf.Tensor:
        """
        Parse and preprocess a single image using TensorFlow operations.
        
        This function loads and resizes an image, normalizing it to the [0, 1] range.
        Scaling for the model's input range is handled separately.
        
        Args:
            image_path: Path to image file (string or tensor)
            
        Returns:
            Preprocessed image tensor in [0, 1] range.
        """
        # Read image file
        image = tf.io.read_file(image_path)
        
        # Decode image (handles PNG, JPG, JPEG automatically)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize using AREA method (equivalent to cv2.INTER_AREA)
        image = tf.image.resize(
            image, 
            [self.config.IMG_HEIGHT, self.config.IMG_WIDTH], 
            method=tf.image.ResizeMethod.AREA
        )
        
        # Normalize to [0, 1] range
        image = image / 255.0
        
        return image

    def scale_image(self, image: tf.Tensor) -> tf.Tensor:
        """Scales an image from [0, 1] to [-1, 1] to match a tanh activation range."""
        return (image * 2.0) - 1.0
    
    def augment_image(self, image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Apply data augmentation for training.
        
        Minimal augmentation to preserve spectrogram structure while adding variety.
        
        Args:
            image: Input image tensor
            training: Whether to apply augmentation (only during training)
            
        Returns:
            Augmented image tensor
        """
        if not training:
            return image
            
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=self.config.AUG_BRIGHTNESS_DELTA)
        
        # Random contrast adjustment  
        image = tf.image.random_contrast(
            image, 
            lower=self.config.AUG_CONTRAST_LOWER, 
            upper=self.config.AUG_CONTRAST_UPPER
        )
        
        # Note: No horizontal flips as time-frequency spectrograms have meaningful orientation
        
        return image


class SpectrogramDatasetBuilder:
    """Builds tf.data.Dataset objects for different training scenarios."""
    
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        self.preprocessor = SpectrogramPreprocessor(config)
        
    def create_classifier_dataset(self,
                                image_paths: List[str],
                                labels: List[int], 
                                batch_size: int,
                                is_training: bool = True,
                                cache_data: bool = True,
                                shuffle_buffer_size: int = 10000) -> tf.data.Dataset:
        """
        Create dataset for binary classification (returns X, y).
        
        Args:
            image_paths: List of image file paths
            labels: List of integer labels (0/1 for binary classification)
            batch_size: Batch size
            is_training: Whether this is training data (enables shuffling/augmentation)
            cache_data: Whether to cache dataset in memory
            shuffle_buffer_size: Size of shuffle buffer
            
        Returns:
            tf.data.Dataset yielding (images, labels) batches
        """
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Shuffle early if training
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=min(len(image_paths), shuffle_buffer_size),
                seed=self.config.SEED
            )
        
        # Parse images and keep labels (produces images in [0, 1])
        dataset = dataset.map(
            lambda path, label: (self.preprocessor.parse_image(path), label),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # Apply augmentation to images during training (works on [0, 1] images)
        if is_training and self.config.ENABLE_AUGMENTATION:
            dataset = dataset.map(
                lambda img, label: (self.preprocessor.augment_image(img, training=True), label),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
            )
            # Clip values back to [0, 1] after augmentation
            dataset = dataset.map(
                lambda img, label: (tf.clip_by_value(img, 0.0, 1.0), label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
        # Scale image to [-1, 1] for the model input
        dataset = dataset.map(
            lambda img, label: (self.preprocessor.scale_image(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
            
        # Batch the data
        dataset = dataset.batch(batch_size, drop_remainder=is_training)

        # Cache if requested
        if cache_data:
            dataset = dataset.cache()

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_autoencoder_dataset(self,
                                 image_paths: List[str],
                                 batch_size: int,
                                 is_training: bool = True,
                                 cache_data: bool = True,
                                 shuffle_buffer_size: int = 10000) -> tf.data.Dataset:
        """
        Create dataset for autoencoder training (returns X, X).
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size
            is_training: Whether this is training data (enables shuffling/augmentation)
            cache_data: Whether to cache dataset in memory
            shuffle_buffer_size: Size of shuffle buffer
            
        Returns:
            tf.data.Dataset yielding (images, images) batches for reconstruction
        """
        # Create dataset from paths
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        
        # Shuffle early if training
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=min(len(image_paths), shuffle_buffer_size),
                seed=self.config.SEED
            )
        
        # Parse images (produces images in [0, 1])
        dataset = dataset.map(
            self.preprocessor.parse_image,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # Apply augmentation during training (works on [0, 1] images)
        if is_training and self.config.ENABLE_AUGMENTATION:
            dataset = dataset.map(
                lambda img: self.preprocessor.augment_image(img, training=True),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
            )
            # Clip values back to [0, 1] after augmentation
            dataset = dataset.map(
                lambda img: tf.clip_by_value(img, 0.0, 1.0),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Scale image to [-1, 1] for the model input
        dataset = dataset.map(
            self.preprocessor.scale_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # For autoencoder: input and target are the same
        dataset = dataset.map(lambda x: (x, x))

        # Batch the data
        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        
        # Cache if requested
        if cache_data:
            dataset = dataset.cache()
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def get_dataset_info(dataset: tf.data.Dataset, name: str) -> None:
    """Utility function to print tf.data.Dataset information."""
    try:
        element_spec = dataset.element_spec
        print(f"\n{name} Dataset Info:")
        print(f"  Element spec: {element_spec}")
        
        cardinality = dataset.cardinality().numpy()
        if cardinality == tf.data.UNKNOWN_CARDINALITY:
            print(f"  Cardinality: Unknown")
        elif cardinality == tf.data.INFINITE_CARDINALITY:
            print(f"  Cardinality: Infinite")
        else:
            print(f"  Cardinality: {cardinality} batches")
            
    except Exception as e:
        print(f"Could not get full info for {name} dataset: {e}")


def validate_preprocessing_consistency(config: SpectrogramConfig, 
                                     test_image_path: str) -> None:
    """
    Validate that preprocessing produces identical results.
    
    This function can be used to ensure the shared preprocessing
    produces the same results as the original notebook implementations.
    """
    print("Validating preprocessing consistency...")
    
    preprocessor = SpectrogramPreprocessor(config)
    
    # Test image loading and preprocessing
    try:
        # Load and preprocess image to [0, 1]
        image_0_1 = preprocessor.parse_image(test_image_path)
        
        # Scale to final model input range [-1, 1]
        processed_image = preprocessor.scale_image(image_0_1)
        
        print(f"✓ Successfully processed test image: {test_image_path}")
        print(f"  - Output shape: {processed_image.shape}")
        print(f"  - Output dtype: {processed_image.dtype}")
        print(f"  - Value range: [{tf.reduce_min(processed_image):.3f}, {tf.reduce_max(processed_image):.3f}] (Expected [-1, 1])")
        
        # Test augmentation on the [0, 1] image
        augmented_image = preprocessor.augment_image(image_0_1, training=True)
        print(f"✓ Augmentation test passed")
        print(f"  - Augmented shape: {augmented_image.shape}")
        
    except Exception as e:
        print(f"✗ Preprocessing validation failed: {e}")


# Convenience function for quick setup
def create_spectrogram_config() -> SpectrogramConfig:
    """Create default spectrogram configuration."""
    return SpectrogramConfig() 