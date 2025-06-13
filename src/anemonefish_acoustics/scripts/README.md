# Anemonefish Acoustics Training Scripts

This directory contains scripts for training machine learning models for anemonefish acoustics analysis.

## Binary Classifier Training

The `train_binary_classifier.py` script trains a binary classifier for detecting anemonefish sounds in acoustic recordings.

### Features

- **Configurable hyperparameters**: All hyperparameters can be adjusted via a YAML configuration file
- **Data augmentation**: Optional data augmentation with configurable parameters
- **Visualization**: Generates visualizations of training progress, data samples, and model evaluation
- **Experiment tracking**: Optional integration with MLflow for experiment tracking
- **Comprehensive logging**: Detailed logging of training progress, metrics, and model architecture
- **Model checkpointing**: Saves the best model during training for later use
- **Early stopping**: Prevents overfitting by stopping training when validation metrics stop improving

### Usage

```bash
# Create a default configuration file
python train_binary_classifier.py --create-config --config-output config/my_config.yaml

# Edit the configuration file as needed, then run training
python train_binary_classifier.py --config config/my_config.yaml

# Override specific configurations for a run
python train_binary_classifier.py --config config/base_config.yaml --override config/override.yaml
```

### Configuration

The configuration file is a YAML file with the following sections:

1. **Data settings**:
   - Paths to data directories
   - Sample rate
   - Test/validation split ratios
   - Random seed
   - Class balance ratio

2. **Preprocessing settings**:
   - Feature type (spectrogram, MFCC, etc.)
   - Parameters for feature extraction

3. **Augmentation settings**:
   - Enable/disable augmentation
   - Parameters for each augmentation type

4. **Model architecture settings**:
   - Input channels
   - Frequency bins
   - Convolutional layers
   - Fully-connected layers

5. **Training settings**:
   - Batch size
   - Number of epochs
   - Learning rate
   - Weight decay
   - Early stopping patience
   - Device (cuda/cpu)

6. **Logging and checkpoint settings**:
   - Experiment name
   - Log directory
   - Visualization settings
   - Log level

7. **MLflow settings**:
   - Enable/disable MLflow tracking
   - Tracking URI
   - Experiment name

### Example Configuration

```yaml
# Data settings
data:
  processed_wavs_dir: 'data/processed_wavs'
  noise_dir: 'data/noise'
  noise_chunked_dir: 'data/noise_chunked'
  cache_dir: 'data/cache'
  augmented_dir: 'data/augmented_wavs'
  sample_rate: 8000
  test_size: 0.2
  validation_size: 0.15
  random_state: 42
  balance_ratio: 1.0

# See full configuration example in config/binary_classifier_default.yaml
```

### Output Structure

The training script creates an experiment directory with the following structure:

```
logs/experiments/experiment_name_timestamp/
├── checkpoints/
│   └── best_model.pt
├── logs/
│   └── timestamp_training.experiment_name.log
├── models/
│   ├── binary_classifier.pt
│   ├── config.yaml
│   ├── metadata.json
│   └── model_architecture.json
└── visualizations/
    ├── confusion_matrix.png
    ├── neg_sample_*.png
    ├── pos_sample_*.png
    └── training_history.png
```

### Requirements

- PyTorch
- NumPy
- scikit-learn
- librosa
- matplotlib
- pyyaml
- tqdm
- (optional) MLflow

## Adding New Scripts

When adding new training scripts, follow these guidelines:

1. Create a new script file in this directory
2. Use the same configuration structure for consistency
3. Use the logging utilities from `anemonefish_acoustics.utils`
4. Add appropriate documentation and examples to this README 