# Anemonefish Acoustics Model Architecture

This document outlines the model architecture strategy for the Anemonefish Acoustics project. It provides a comprehensive overview of the approaches we're implementing to develop effective binary and call-type classifiers for anemonefish vocalizations.

## Overview

The project uses a two-stage approach:

1. **Self-supervised pre-training** using large unlabeled datasets (24-hour recordings)
2. **Supervised fine-tuning** using a smaller labeled dataset

This approach is designed to address the challenge of limited labeled data while leveraging larger amounts of unlabeled acoustic recordings.

## Primary Architecture: Convolutional Recurrent Neural Network (CRNN)

For the binary classification task (detecting whether an audio segment contains anemonefish calls), we've chosen a hybrid CRNN architecture that combines the strengths of both CNNs and RNNs.

### CRNN Architecture Details

```
Input Spectrograms/Features
       ↓
[CNN Feature Extractor]
       ↓
[Bidirectional LSTM/GRU]
       ↓
[Attention Mechanism]  (optional)
       ↓
[Dense Classification Layers]
       ↓
Binary Output (Anemonefish sound vs. Noise)
```

#### Why CRNN for Bioacoustics?

1. **CNN Component**: 
   - Extracts local spectral patterns from audio spectrograms
   - Effective at identifying frequency characteristics of anemonefish calls
   - Reduces dimensionality of the input data

2. **RNN Component**:
   - Captures temporal dynamics and patterns in the sequence of features
   - Models how spectral features evolve over time
   - Can handle variable-length inputs more naturally

3. **Combined Benefits**:
   - Better performance than pure CNN or RNN for audio classification tasks
   - Particularly suitable for bioacoustic signals that have both spectral and temporal patterns
   - More robust to variations in call patterns and environmental conditions

## Self-Supervised Pre-training Strategy

To leverage the available 24-hour unlabeled recordings, we'll implement self-supervised pre-training before fine-tuning on labeled data.

### Autoencoder Pre-training Approach

```
Input Spectrograms/Features
       ↓
[Encoder CNN]  ← (This becomes part of our final model)
       ↓
[Bottleneck/Latent Space]
       ↓
[Decoder CNN]
       ↓
Reconstructed Spectrograms/Features
```

#### Implementation Details

1. **Data Processing for Pre-training**:
   - Segment 24-hour recordings into short windows (0.5-1 second)
   - Extract consistent features (MFCCs, spectral contrast, chroma, RMS)
   - Create a large dataset of unlabeled audio segments

2. **Autoencoder Architecture**:
   - **Encoder**: CNN layers that compress the input into a latent representation
   - **Decoder**: CNN layers that reconstruct the original input from the latent space
   - **Training objective**: Minimize reconstruction error

3. **Pre-training Process**:
   - Train the autoencoder on unlabeled data
   - Extract and save the encoder part
   - Use the pre-trained encoder to initialize the CNN portion of the CRNN model

### Alternative: Contrastive Learning Approach

As an alternative to traditional autoencoders, we may explore contrastive learning approaches:

1. **Contrastive Predictive Coding (CPC)**:
   - Train a model to predict future time steps in audio features
   - Create representations that capture temporal dynamics

2. **SimCLR-style Contrastive Learning**:
   - Create "positive pairs" through data augmentation of the same audio clip
   - Train the model to maximize similarity between augmented versions of the same clip while minimizing similarity to other clips

## Model Architecture Comparison

| Architecture | Suitability | Difficulty | Strengths | Limitations |
|--------------|------------|------------|-----------|-------------|
| CNN | ★★★★☆ | ★★☆☆☆ | Good for spectral patterns, efficient | Fixed input size, weaker on temporal dynamics |
| RNN/LSTM | ★★★★☆ | ★★★☆☆ | Excellent for sequences, handles variable length | Slower training, may miss spectral patterns |
| CRNN | ★★★★★ | ★★★★☆ | Combines strengths of CNN & RNN, best for audio | More complex to optimize |
| Transformer | ★★★☆☆ | ★★★★★ | Models long-range dependencies | Requires large datasets, computationally expensive |
| Autoencoder Pre-training | ★★★★☆ | ★★★★☆ | Leverages unlabeled data, prevents overfitting | Adds complexity to the pipeline |

## Training Pipeline

The complete training pipeline will consist of:

1. **Data Processing**:
   - Preprocessing raw audio files
   - Feature extraction (MFCCs, spectral contrast, chroma, RMS)
   - Data augmentation to increase training set size

2. **Pre-training Phase**:
   - Train autoencoder on unlabeled data
   - Save encoder weights

3. **Fine-tuning Phase**:
   - Initialize CRNN with pre-trained encoder
   - Fine-tune on labeled data with appropriate learning rate scheduling
   - Implement early stopping to prevent overfitting

4. **Evaluation**:
   - Use precision, recall, F1-score, and accuracy metrics
   - Implement cross-validation for robust evaluation
   - Compare with baseline models (e.g., pure CNN, random forest)

## Future Directions

1. **Multi-task Learning**: Train a single model to perform both binary classification and call-type classification.

2. **Zero/Few-shot Learning**: Develop techniques to identify new call types with minimal labeled examples.

3. **Ensemble Methods**: Combine multiple model architectures for improved performance.

4. **Real-time Processing**: Optimize models for deployment on edge devices for real-time processing of hydrophone recordings.

## Technical Implementation Notes

- Framework: PyTorch for model development
- Input Features: MFCCs, spectral contrast, chroma, and RMS features
- Data Shape: 
  - For CNN: (batch_size, time_steps, features, channels)
  - For LSTM: (batch_size, time_steps, features)
  - For CRNN: CNN feature extraction followed by reshaping for LSTM
- Sample Rate: 8000 Hz (focused on 0-2000 Hz range for anemonefish vocalizations) 