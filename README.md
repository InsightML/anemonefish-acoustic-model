# Anemonefish Acoustics

A machine learning system for detecting and classifying anemonefish vocalizations in underwater acoustic recordings.

## Project Overview

This project aims to develop a custom machine learning model for:
1. Detecting anemonefish sounds in hydrophone recordings (binary classification)
2. Classifying the type of call (e.g., aggressive, submissive, territorial, etc.)

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Project Structure

### src/anemonefish_acoustics/
Source code for the anemonefish acoustics package:

- **data_processing/**: Code for preprocessing audio data, feature extraction, and data augmentation
- **models/**: Implementation of machine learning models
- **utils/**: Utility functions for the project
- **visualization/**: Tools for visualizing audio data, spectrograms, and model results

### data/
Raw and processed data files used in the project:

- **processed_wavs/**: Preprocessed WAV files of anemonefish calls
- **training_data/**: Data split into training, validation, and test sets
- **noise/**: Background noise samples for data augmentation

### notebooks/
Jupyter notebooks for exploratory data analysis and experimentation.

### models/
Saved machine learning models.

### scripts/
Standalone scripts for various tasks, such as data processing, training, and evaluation.

### configs/
Configuration files for the project, including settings and parameters.

### tests/
Unit and integration tests for the project code.

### docs/
Documentation files:
- Project overview and research
- Performance metrics and evaluation criteria
- Data description and labeling scheme

### logs/
Log files generated during model training and evaluation.

### results/
Output results, including plots, evaluation metrics, and reports.

## Target Performance Metrics

### Binary Model (Anemonefish Sound Detection)
- **Primary Metric**: Recall (target ≥ 90%)
- **Secondary Metric**: Precision (target ≥ 80%)

### Call-type Classification Model
- **Primary Metric**: F1-Score (macro-averaged) (target ≥ 75%)

## Noise Chunking Improvements

We've modified the noise chunking process to preserve the full temporal information in anemonefish calls during feature extraction and model training.

### Key Changes:

1. **Increased Chunk Duration**: Changed the noise chunk duration from 1.0 second to 5.0 seconds to ensure noise chunks are significantly longer than anemonefish calls.

2. **Added Overlapping Chunks**: Implemented an overlap parameter to create more training examples from the noise recordings.

3. **Improved Shape Handling**: Modified the dataset preparation process to preserve the complete temporal dimension of anemonefish calls while truncating the longer noise chunks.

### Benefits:

- **Preserves Acoustic Patterns**: All temporal information in anemonefish calls is now preserved, ensuring no important acoustic patterns are lost.

- **Consistent Data**: The dataset now has consistent dimensions for all features, with the time dimension set based on the anemonefish calls rather than the noise chunks.

- **Model Training Improvements**: The model can now learn from complete anemonefish call patterns instead of truncated versions.

### Feature Dimensions:

Before our changes:
- Anemonefish calls: (135, 13, 43) for MFCC features
- Noise chunks: (135, 13, 16) for MFCC features
- Result: Both truncated to (_, 13, 16) during feature concatenation

After our changes:
- Anemonefish calls: (135, 13, 43) for MFCC features
- Noise chunks: (13, 13, 9274) for MFCC features
- Result: Noise chunks truncated to (_, 13, 43) during feature concatenation, preserving all anemonefish data

### Issues Encountered:

We encountered file permission issues when trying to save new noise chunks. If you experience similar problems, make sure the noise_chunked directory has write permissions or try creating the directory in a location with appropriate access rights.
