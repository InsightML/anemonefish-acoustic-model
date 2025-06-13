# Anemonefish Acoustics Data Preprocessing Guide

This document outlines the data preprocessing methodology for the Anemonefish Acoustics project, focusing on how audio data is prepared for machine learning models.

## Overview

Our preprocessing pipeline takes raw audio files of anemonefish calls and environmental noise, and transforms them into standardized spectrograms suitable for training deep learning models.

## New Features

We've recently transitioned from MFCC (Mel-frequency cepstral coefficients) to spectrogram-based features:

1. **Spectrograms**: Standard spectrograms created from STFT (Short-Time Fourier Transform)
2. **Mel Spectrograms**: Spectrograms mapped to the mel scale to better match human auditory perception
3. **Log-Mel Spectrograms**: Mel spectrograms converted to logarithmic scale (dB) for better visualization

These changes align better with how marine biologists analyze the data visually in tools like Audacity.

## Handling Variable Length Audio

Anemonefish calls have variable durations, which presents a challenge for training models that expect fixed-size inputs. We've implemented two strategies:

### 1. Stretch/Squash Method (`stretch_squash`)

- For audio shorter than the standard length (default: 1 second): Audio is stretched using time-stretching algorithms
- For audio longer than the standard length: Audio is compressed (squashed) to fit the target duration
- **Advantages**: Each call is preserved in its entirety; temporal relationships within the call are maintained proportionally
- **Disadvantages**: May introduce artifacts or distortions, especially with extreme stretching/squashing

### 2. Crop/Pad Method (`crop_pad`)

- For audio shorter than the standard length: Audio is padded with zeros to reach the target duration
- For audio longer than the standard length: Multiple overlapping segments are created using a sliding window approach
- **Advantages**: No distortion of the original audio content; can generate more training examples from longer recordings
- **Disadvantages**: Zero padding could create artifacts that the model might learn; calls might be truncated

## Recommended Settings

Based on analysis of typical anemonefish call durations:

- **Standard length**: 1.0 seconds (adjust based on your dataset's median call length)
- **Feature type**: `mel_spectrogram` (alternatives: `spectrogram`, `log_mel_spectrogram`)
- **Spectrogram parameters**:
  - n_fft = 1024 (FFT window size)
  - hop_length = 256 (hop size between frames)
  - n_mels = 64 (number of mel bands)
  - fmin = 0 Hz, fmax = 2000 Hz (frequency range)

## Usage Examples

```python
from anemonefish_acoustics.data_processing import DatasetBuilder

# Initialize with mel spectrograms and stretch/squash preprocessing
dataset_builder = DatasetBuilder(
    processed_wavs_dir='data/processed_wavs',
    noise_dir='data/noise',
    feature_type='mel_spectrogram',
    preprocess_method='stretch_squash',
    standard_length_sec=1.0
)

# Analyze anemonefish call lengths to determine optimal standard length
stats = dataset_builder.analyze_anemonefish_call_lengths()

# Prepare dataset with augmentation
X_train, X_test, y_train, y_test = dataset_builder.prepare_dataset_with_augmentation(
    test_size=0.2,
    use_augmentation=True, 
    balance_ratio=1.0
)

# Visualize features
dataset_builder.visualize_features(num_samples=3)

# Prepare data for CNN model
X_train_cnn, X_test_cnn, y_train, y_test = dataset_builder.prepare_data_for_model(
    X_train, X_test, y_train, y_test, model_type='cnn'
)
```

## Notes on Data Augmentation

When using the `crop_pad` method with zero padding, consider increasing time masking in data augmentation to prevent the model from overfitting on zero-padded regions. Alternatively, you can modify the model architecture to ignore or properly handle zero-padded regions.

For the `stretch_squash` method, you might want to increase time stretching/compression in data augmentation to make the model more robust to these transformations.

## Sliding Window for Prediction

For real-time or batch prediction on longer recordings, we recommend using a sliding window approach with the same window size as your `standard_length_sec` parameter. This ensures consistency between training and inference inputs.

## File Structure

- `data_preprocessing.py`: Main preprocessing module
- `data_preprocessing_demo.py`: Demo script showing how to use the preprocessing modules

## Data Organization

The data is organized into the following directories:

- `data/processed_wavs`: Contains processed anemonefish calls (Y=1)
- `data/noise`: Contains background noise recordings (Y=0)
- `data/noise_chunked`: Contains chunked noise files (created by the preprocessing pipeline)
- `data/cache`: Contains cached features and metadata (created by the preprocessing pipeline)

## Filename Convention

### Anemonefish Calls

Anemonefish call files follow this naming convention:
```
target-REEF-TIME_OF_DAY-BREEDING-RANK-INTERACTION_WITH-ID-YEAR-TAPENAME-START_TIME_MS-END_TIME_MS-NAME.wav
```

For example:
```
territorial-LL-A-NR-R1-Cooperation_B10_2023_LL-B10-A-NRaudiowithlabels_30105_31321.wav
```

Here:
- `territorial`: Behavior type (competition, territorial, submission, social, etc.)
- `LL`: Reef identifier
- `A`: Time of day (A = AM, P = PM, etc.)
- `NR`: Breeding status (R = Reproductive, NR = Non-Reproductive)
- `R1`: Rank of the fish (R1 = Rank 1, R2 = Rank 2, etc.)
- `Cooperation`: Interaction type
- `B10_2023`: ID and year
- `LL-B10-A-NRaudiowithlabels`: Tape name
- `30105_31321`: Start and end times in milliseconds
- `NAME`: Additional information (may be omitted)

### Noise Files

Noise files have simpler names such as:
```
divers_noise.wav
boat_noise_5.wav
parotfish_eating_background noise.wav
```

The preprocessing module chunks these longer recordings into segments comparable to anemonefish calls.

## Core Components

### AnemoneMetadataParser

Parses metadata from anemonefish call filenames.

```python
from data_preprocessing import AnemoneMetadataParser

metadata = AnemoneMetadataParser.parse_filename(filename)
```

### AudioProcessor

Handles loading, processing, and feature extraction from audio files.

```python
from data_preprocessing import AudioProcessor

# Load audio
audio_data, sr = AudioProcessor.load_audio(file_path)

# Normalize audio
audio_data = AudioProcessor.normalize_audio(audio_data)

# Extract features
features = AudioProcessor.extract_features(audio_data, sr)
```

### DatasetBuilder

Builds datasets from audio files for machine learning.

```python
from data_preprocessing import DatasetBuilder

# Initialize dataset builder
dataset_builder = DatasetBuilder(
    processed_wavs_dir='data/processed_wavs',
    noise_dir='data/noise',
    noise_chunked_dir='data/noise_chunked',
    cache_dir='data/cache'
)

# Chunk noise files
dataset_builder.chunk_noise_files()

# Build metadata dataframe
metadata_df = dataset_builder.build_metadata_dataframe()

# Prepare dataset
X_train, X_test, y_train, y_test = dataset_builder.prepare_dataset()
```

## Usage

You can use the demo script to explore the preprocessing pipeline:

```bash
# Extract metadata from anemonefish files
python data_preprocessing_demo.py --action metadata

# Chunk noise files
python data_preprocessing_demo.py --action chunk

# Extract features from a sample file
python data_preprocessing_demo.py --action features

# Prepare dataset
python data_preprocessing_demo.py --action dataset

# Visualize features
python data_preprocessing_demo.py --action visualize --feature_type mfcc --num_samples 5
```

## Required Libraries

- numpy
- pandas
- librosa
- soundfile
- matplotlib
- scikit-learn
- tqdm

Install these dependencies using:

```bash
pip install numpy pandas librosa soundfile matplotlib scikit-learn tqdm
``` 