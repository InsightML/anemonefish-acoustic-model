"""Data processing package for anemonefish acoustics."""

from .data_preprocessing import (
    AnemoneMetadataParser,
    AudioProcessor,
    DatasetBuilder,
    PREPROCESS_STRETCH_SQUASH,
    PREPROCESS_CROP_PAD,
    STANDARD_LENGTH_SEC,
    SAMPLE_RATE,
    SPEC_N_FFT,
    SPEC_HOP_LENGTH,
    SPEC_N_MELS
)

from .data_augmentation import (
    AudioAugmenter,
    DataAugmentationPipeline,
    mixup
)

from .prediction_pipeline import (
    PredictionPipeline
)

from .spectrogram_preprocessing import (
    SpectrogramConfig,
    SpectrogramDataLoader,
    SpectrogramPreprocessor,
    SpectrogramDatasetBuilder,
    get_dataset_info,
    validate_preprocessing_consistency,
    create_spectrogram_config
)

__all__ = [
    'AnemoneMetadataParser',
    'AudioProcessor',
    'DatasetBuilder',
    'AudioAugmenter',
    'DataAugmentationPipeline',
    'PredictionPipeline',
    'mixup',
    'PREPROCESS_STRETCH_SQUASH',
    'PREPROCESS_CROP_PAD',
    'STANDARD_LENGTH_SEC',
    'SAMPLE_RATE',
    'SPEC_N_FFT',
    'SPEC_HOP_LENGTH',
    'SPEC_N_MELS',
    # Spectrogram preprocessing
    'SpectrogramConfig',
    'SpectrogramDataLoader',
    'SpectrogramPreprocessor',
    'SpectrogramDatasetBuilder',
    'get_dataset_info',
    'validate_preprocessing_consistency',
    'create_spectrogram_config'
]
