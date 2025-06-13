#!/usr/bin/env python
"""
Training script for the anemonefish binary classifier.

This script loads acoustic data, performs preprocessing and augmentation,
trains the binary classifier model, and evaluates its performance.
Visualization of the training process and results are also generated.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
import random

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from anemonefish_acoustics.models.binary_classifier import CNNAnemonefishBinaryClassifier
from anemonefish_acoustics.data_processing.data_preprocessing import DatasetBuilder, AudioProcessor
from anemonefish_acoustics.data_processing.data_augmentation import AudioAugmenter, DataAugmentationPipeline
from anemonefish_acoustics.utils import (
    setup_logger,
    TrainingLogger,
    load_config,
    save_config,
    merge_configs,
    create_default_binary_classifier_config,
    write_default_config,
    plot_training_history,
    plot_audio_waveform,
    plot_spectrogram,
    plot_feature_comparison,
    plot_confusion_matrix,
    plot_prediction_samples
)

# Try to import MLflow if specified in config
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a binary classifier for anemonefish sound detection")
    parser.add_argument(
        "--config", type=str, default="config/binary_classifier_default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--override", type=str, default=None,
        help="Path to override configuration file"
    )
    parser.add_argument(
        "--create-config", action="store_true",
        help="Create a default configuration file"
    )
    parser.add_argument(
        "--config-output", type=str, default="config/binary_classifier_default.yaml",
        help="Path to save the default configuration file"
    )
    return parser.parse_args()


def prepare_data(config, logger):
    """
    Prepare data for training.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        Training, validation, and test data and labels
    """
    logger.info("Preparing data for training...")
    
    # Initialize DatasetBuilder
    data_config = config["data"]
    preproc_config = config["preprocessing"]
    
    # Add standard_length_sec to the preprocessing config if not present
    standard_length_sec = preproc_config.get("standard_length_sec", 0.6)  # Default to 0.6 if not specified
    
    logger.info(f"Using standard_length_sec: {standard_length_sec}")
    
    dataset_builder = DatasetBuilder(
        processed_wavs_dir=data_config["processed_wavs_dir"],
        noise_dir=data_config["noise_dir"],
        noise_chunked_dir=data_config["noise_chunked_dir"],
        cache_dir=data_config["cache_dir"],
        augmented_dir=data_config["augmented_dir"],
        sr=data_config["sample_rate"],
        feature_type=preproc_config["feature_type"],  # Pass the feature type explicitly
        standard_length_sec=standard_length_sec,  # Pass the standard_length_sec parameter
        preprocess_method=preproc_config["preprocess_method"]  # Pass the preprocessing method
    )
    
    # Prepare dataset with optional augmentation
    logger.info("Building dataset...")
    aug_config = config["augmentation"]
    
    # Log augmentation settings
    if aug_config["use_augmentation"]:
        logger.info(f"Data augmentation enabled (factor: {aug_config['augmentation_factor']})")
        for aug_type, settings in aug_config.items():
            if isinstance(settings, dict) and "enabled" in settings:
                if settings["enabled"]:
                    logger.info(f"  - {aug_type}: enabled with {settings}")
                else:
                    logger.info(f"  - {aug_type}: disabled")
    else:
        logger.info("Data augmentation disabled")
    
    # Prepare the dataset
    X_train_val, X_test, y_train_val, y_test = dataset_builder.prepare_dataset_with_augmentation(
        test_size=data_config["test_size"],
        use_augmentation=aug_config["use_augmentation"],
        balance_ratio=data_config["balance_ratio"],
        random_state=data_config["random_state"]
    )
    
    # Split train_val into train and validation
    total_samples = len(y_train_val)
    val_size = int(data_config["validation_size"] * total_samples / (1 - data_config["test_size"]))
    
    # Create indices for splitting
    indices = list(range(total_samples))
    np.random.seed(data_config["random_state"])
    np.random.shuffle(indices)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Split features and labels
    feature_type = preproc_config["feature_type"]
    
    logger.info(f"Using feature type: {feature_type}")
    
    # Prepare data for the model
    X_train_val_prepared, X_test_prepared, y_train_val, y_test = dataset_builder.prepare_data_for_model(
        X_train_val, X_test, y_train_val, y_test, 
        feature_type=feature_type, 
        model_type="cnn"
    )
    
    # Split train and validation
    X_train = X_train_val_prepared[train_indices]
    X_val = X_train_val_prepared[val_indices]
    y_train = y_train_val[train_indices]
    y_val = y_train_val[val_indices]
    
    # Log dataset shapes
    logger.info(f"Training set: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation set: {X_val.shape}, {y_val.shape}")
    logger.info(f"Test set: {X_test_prepared.shape}, {y_test.shape}")
    
    # Check class balance
    train_positives = np.sum(y_train)
    train_negatives = len(y_train) - train_positives
    val_positives = np.sum(y_val)
    val_negatives = len(y_val) - val_positives
    test_positives = np.sum(y_test)
    test_negatives = len(y_test) - test_positives
    
    logger.info(f"Class balance - Train: {train_positives} positive, {train_negatives} negative")
    logger.info(f"Class balance - Validation: {val_positives} positive, {val_negatives} negative")
    logger.info(f"Class balance - Test: {test_positives} positive, {test_negatives} negative")
    
    return X_train, X_val, X_test_prepared, y_train, y_val, y_test, dataset_builder


def visualize_preprocessing_stages(dataset_builder, config, experiment_dir):
    """
    Visualize preprocessing stages for sample audio files.
    
    Parameters
    ----------
    dataset_builder : DatasetBuilder
        Dataset builder instance
    config : dict
        Configuration dictionary
    experiment_dir : str
        Path to the experiment directory
    """
    # Create visualization directory for preprocessing stages
    preproc_vis_dir = os.path.join(experiment_dir, "visualizations", "preprocessing_stages")
    os.makedirs(preproc_vis_dir, exist_ok=True)
    
    # Get lists of files
    anemonefish_files = dataset_builder.list_anemonefish_files()
    noise_files = dataset_builder.list_noise_files(use_chunked=True)
    
    if not anemonefish_files:
        print("No anemonefish files found for preprocessing visualization")
        return
    
    if not noise_files:
        print("No noise files found for preprocessing visualization")
        return
    
    # Select a few samples from each class
    num_samples = min(3, len(anemonefish_files))
    random.seed(config["data"]["random_state"])
    selected_anemonefish = random.sample(anemonefish_files, num_samples)
    
    num_samples = min(3, len(noise_files))
    selected_noise = random.sample(noise_files, num_samples)
    
    # Visualize preprocessing stages for anemonefish samples
    for i, file_path in enumerate(selected_anemonefish):
        print(f"Visualizing preprocessing stages for anemonefish sample {i+1}")
        anemone_vis_dir = os.path.join(preproc_vis_dir, f"anemonefish_sample_{i+1}")
        dataset_builder.visualize_preprocessing_stages(
            file_path=file_path,
            output_dir=anemone_vis_dir,
            num_augmentations=min(5, 7)  # Visualize up to 5 augmentation types
        )
    
    # Visualize preprocessing stages for noise samples
    for i, file_path in enumerate(selected_noise):
        print(f"Visualizing preprocessing stages for noise sample {i+1}")
        noise_vis_dir = os.path.join(preproc_vis_dir, f"noise_sample_{i+1}")
        dataset_builder.visualize_preprocessing_stages(
            file_path=file_path,
            output_dir=noise_vis_dir,
            num_augmentations=min(5, 7)  # Visualize up to 5 augmentation types
        )


def visualize_data_samples(X_train, X_val, y_train, y_val, dataset_builder, config, experiment_dir=None):
    """
    Visualize data samples from training and validation sets.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data
    X_val : np.ndarray
        Validation data
    y_train : np.ndarray
        Training labels
    y_val : np.ndarray
        Validation labels
    dataset_builder : DatasetBuilder
        Dataset builder instance
    config : dict
        Configuration dictionary
    experiment_dir : str, optional
        Path to the experiment directory, if None, a new one will be created (for backward compatibility)
    """
    # Create visualization directory
    if experiment_dir is None:
        # For backward compatibility, create a new experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = config["output"]["experiment_name"]
        experiment_id = f"{experiment_name}_{timestamp}"
        log_dir = config["output"]["log_dir"]
        experiment_dir = os.path.join(log_dir, experiment_id)
        
    # Create data samples visualization directory
    vis_dir = os.path.join(experiment_dir, "visualizations", "data_samples")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualization of training samples
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    
    # Get feature type for proper visualization
    feature_type = config["preprocessing"]["feature_type"]
    
    # Get standard_length_sec for duration parameter
    standard_length_sec = config["preprocessing"].get("standard_length_sec", 0.6)
    
    if len(pos_indices) > 0:
        pos_idx = np.random.choice(pos_indices, min(5, len(pos_indices)), replace=False)
        for i, idx in enumerate(pos_idx):
            sample = X_train[idx].copy()
            
            # Remove any singleton dimensions
            if sample.ndim > 2:
                sample = np.squeeze(sample)
            
            if feature_type.lower() == 'mfcc':
                plt.figure(figsize=(10, 6))
                
                # For MFCC visualization, we want features on y-axis and time on x-axis
                # If shape is (time, features), transpose to (features, time)
                if sample.shape[0] > sample.shape[1]:  # If time dimension is larger than feature dimension
                    sample_display = sample.T
                else:
                    sample_display = sample
                
                # Create a proper visual representation of MFCCs with appropriate colormap
                im = plt.imshow(sample_display, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='viridis')
                plt.colorbar(im, format='%+2.0f')
                plt.title(f"Positive Sample {i+1} - MFCC Features")
                plt.xlabel('Time Frames')
                plt.ylabel('MFCC Coefficients')
                plt.tight_layout()
                
                output_path = os.path.join(vis_dir, f"pos_sample_{i+1}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Saved positive sample visualization to {output_path}")
            else:
                # For other feature types, use the existing visualization function
                plot_spectrogram(
                    sample, 
                    sr=config["data"]["sample_rate"],
                    hop_length=config["preprocessing"]["hop_length"],
                    fmin=config["preprocessing"]["fmin"],
                    fmax=config["preprocessing"]["fmax"],
                    title=f"Positive Sample {i+1}",
                    output_path=os.path.join(vis_dir, f"pos_sample_{i+1}.png"),
                    duration=standard_length_sec  # Pass the standard_length_sec as duration
                )
    
    if len(neg_indices) > 0:
        neg_idx = np.random.choice(neg_indices, min(5, len(neg_indices)), replace=False)
        for i, idx in enumerate(neg_idx):
            sample = X_train[idx].copy()
            
            # Remove any singleton dimensions
            if sample.ndim > 2:
                sample = np.squeeze(sample)
            
            if feature_type.lower() == 'mfcc':
                plt.figure(figsize=(10, 6))
                
                # For MFCC visualization, we want features on y-axis and time on x-axis
                # If shape is (time, features), transpose to (features, time)
                if sample.shape[0] > sample.shape[1]:  # If time dimension is larger than feature dimension
                    sample_display = sample.T
                else:
                    sample_display = sample
                
                # Create a proper visual representation of MFCCs with appropriate colormap
                im = plt.imshow(sample_display, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='viridis')
                plt.colorbar(im, format='%+2.0f')
                plt.title(f"Negative Sample {i+1} - MFCC Features")
                plt.xlabel('Time Frames')
                plt.ylabel('MFCC Coefficients')
                plt.tight_layout()
                
                output_path = os.path.join(vis_dir, f"neg_sample_{i+1}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Saved negative sample visualization to {output_path}")
            else:
                # For other feature types, use the existing visualization function
                plot_spectrogram(
                    sample, 
                    sr=config["data"]["sample_rate"],
                    hop_length=config["preprocessing"]["hop_length"],
                    fmin=config["preprocessing"]["fmin"],
                    fmax=config["preprocessing"]["fmax"],
                    title=f"Negative Sample {i+1}",
                    output_path=os.path.join(vis_dir, f"neg_sample_{i+1}.png"),
                    duration=standard_length_sec  # Pass the standard_length_sec as duration
                )


def create_model(config, logger):
    """
    Create a binary classifier model.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    CNNAnemonefishBinaryClassifier
        Binary classifier model
    """
    logger.info("Creating binary classifier model...")
    
    model_config = config["model"]
    model = CNNAnemonefishBinaryClassifier(
        input_channels=model_config["input_channels"],
        freq_bins=model_config["freq_bins"],
        conv_channels=model_config["conv_channels"],
        fc_sizes=model_config["fc_sizes"]
    )
    
    # Log model architecture
    model_summary = str(model)
    logger.info(f"Model architecture:\n{model_summary}")
    
    return model


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, config):
    """
    Create data loaders for training, validation, and testing.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data
    X_val : np.ndarray
        Validation data
    X_test : np.ndarray
        Test data
    y_train : np.ndarray
        Training labels
    y_val : np.ndarray
        Validation labels
    y_test : np.ndarray
        Test labels
    config : dict
        Configuration dictionary
        
    Returns
    -------
    tuple
        Training, validation, and test data loaders
    """
    # The data is already in the correct format from prepare_data_for_model: [batch, channels, height, width]
    # No need to transpose, just convert to PyTorch tensors
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_config = config["training"]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config["batch_size"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config["batch_size"], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_config["batch_size"], 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, config, logger):
    """
    Train the binary classifier model.
    
    Parameters
    ----------
    model : CNNAnemonefishBinaryClassifier
        Binary classifier model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    config : dict
        Configuration dictionary
    logger : TrainingLogger
        Logger instance
        
    Returns
    -------
    tuple
        Trained model and training history
    """
    logger.logger.info("Training model...")
    
    # Set device
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Set up training parameters
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["early_stopping_patience"]
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_recall': [],
        'val_precision': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(logger.experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float().view(-1, 1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == targets.view(-1, 1)).sum().item()
            train_total += targets.size(0)
        
        # Compute epoch statistics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_true_positives = 0
        val_predicted_positives = 0
        val_actual_positives = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float().view(-1, 1))
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == targets.view(-1, 1)).sum().item()
                val_total += targets.size(0)
                
                # Compute recall and precision metrics
                val_true_positives += ((predicted == 1) & (targets.view(-1, 1) == 1)).sum().item()
                val_predicted_positives += (predicted == 1).sum().item()
                val_actual_positives += (targets == 1).sum().item()
        
        # Compute epoch statistics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_recall = val_true_positives / val_actual_positives if val_actual_positives > 0 else 0
        val_precision = val_true_positives / val_predicted_positives if val_predicted_positives > 0 else 0
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        
        # Log epoch statistics
        logger.log_epoch(
            epoch + 1, 
            {'loss': train_loss, 'acc': train_acc}, 
            prefix='Train'
        )
        logger.log_epoch(
            epoch + 1, 
            {'loss': val_loss, 'acc': val_acc, 'recall': val_recall, 'precision': val_precision}, 
            prefix='Validation'
        )
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            
            # Save the best model
            best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.logger.info(f"Best model saved at epoch {epoch + 1}")
            
            # Log MLflow metrics if enabled
            if config["mlflow"]["use_mlflow"] and MLFLOW_AVAILABLE:
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("best_val_acc", val_acc)
                mlflow.log_metric("best_val_recall", val_recall)
                mlflow.log_metric("best_val_precision", val_precision)
                mlflow.log_artifact(best_model_path)
        else:
            epochs_without_improvement += 1
        
        # Log MLflow metrics if enabled
        if config["mlflow"]["use_mlflow"] and MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_recall": val_recall,
                "val_precision": val_precision
            }, step=epoch)
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.logger.info(f"Loaded best model from epoch {best_epoch + 1}")
    
    # Log training completion
    logger.log_training_complete(
        {
            'best_val_loss': best_val_loss,
            'best_val_acc': history['val_acc'][best_epoch],
            'best_val_recall': history['val_recall'][best_epoch],
            'best_val_precision': history['val_precision'][best_epoch]
        },
        training_time
    )
    
    # Plot training history
    history_plot_path = os.path.join(logger.experiment_dir, "visualizations", "training_history.png")
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    plot_training_history(history, output_path=history_plot_path)
    
    # Log MLflow artifacts if enabled
    if config["mlflow"]["use_mlflow"] and MLFLOW_AVAILABLE:
        mlflow.log_artifact(history_plot_path)
    
    return model, history


def evaluate_model(model, test_loader, config, logger):
    """
    Evaluate the trained model on the test set.
    
    Parameters
    ----------
    model : CNNAnemonefishBinaryClassifier
        Trained binary classifier model
    test_loader : DataLoader
        Test data loader
    config : dict
        Configuration dictionary
    logger : TrainingLogger
        Logger instance
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    logger.logger.info("Evaluating model on test set...")
    
    # Set device
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    
    # Set evaluation mode
    model.eval()
    
    # Initialize metrics
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []
    y_scores = []
    
    # Set up loss function
    criterion = nn.BCELoss()
    
    # Evaluate model
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float().view(-1, 1))
            
            # Update statistics
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == targets.view(-1, 1)).sum().item()
            test_total += targets.size(0)
            
            # Store targets and predictions for metrics
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    # Compute metrics
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    # Convert lists to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten()
    y_scores = np.array(y_scores).flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate precision, recall, and F1-score
    classification_metrics = classification_report(y_true, y_pred, output_dict=True)
    
    # Log metrics
    logger.logger.info(f"Test loss: {test_loss:.4f}")
    logger.logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.logger.info(f"Test precision: {classification_metrics['1.0']['precision']:.4f}")
    logger.logger.info(f"Test recall: {classification_metrics['1.0']['recall']:.4f}")
    logger.logger.info(f"Test F1-score: {classification_metrics['1.0']['f1-score']:.4f}")
    logger.logger.info(f"Confusion matrix:\n{cm}")
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(logger.experiment_dir, "visualizations", "confusion_matrix.png")
    plot_confusion_matrix(
        cm, 
        ["Background", "Anemonefish"], 
        title="Confusion Matrix",
        output_path=cm_plot_path
    )
    
    # Save evaluation metrics
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": classification_metrics['1.0']['precision'],
        "test_recall": classification_metrics['1.0']['recall'],
        "test_f1": classification_metrics['1.0']['f1-score'],
        "confusion_matrix": cm.tolist()
    }
    
    # Log MLflow metrics if enabled
    if config["mlflow"]["use_mlflow"] and MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": classification_metrics['1.0']['precision'],
            "test_recall": classification_metrics['1.0']['recall'],
            "test_f1": classification_metrics['1.0']['f1-score']
        })
        mlflow.log_artifact(cm_plot_path)
    
    return metrics


def save_model_and_metadata(model, config, metrics, history, logger):
    """
    Save the trained model and metadata.
    
    Parameters
    ----------
    model : CNNAnemonefishBinaryClassifier
        Trained binary classifier model
    config : dict
        Configuration dictionary
    metrics : dict
        Evaluation metrics
    history : dict
        Training history
    logger : TrainingLogger
        Logger instance
        
    Returns
    -------
    str
        Path to the saved model
    """
    logger.logger.info("Saving model and metadata...")
    
    # Create model directory
    models_dir = os.path.join(logger.experiment_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "binary_classifier.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save model architecture
    model_config_path = os.path.join(models_dir, "model_architecture.json")
    with open(model_config_path, 'w') as f:
        json.dump({
            "input_channels": config["model"]["input_channels"],
            "freq_bins": config["model"]["freq_bins"],
            "conv_channels": config["model"]["conv_channels"],
            "fc_sizes": config["model"]["fc_sizes"]
        }, f, indent=2)
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "history": {
            key: [float(value) for value in values] 
            for key, values in history.items()
        }
    }
    
    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save config
    config_path = os.path.join(models_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.logger.info(f"Model saved to {model_path}")
    logger.logger.info(f"Metadata saved to {metadata_path}")
    
    return model_path


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create default configuration file if requested
    if args.create_config:
        write_default_config(args.config_output, overwrite=True)
        print(f"Created default configuration file at {args.config_output}")
        return
    
    # Load configuration - fix the path resolution to use absolute path or relative to current directory
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        config_path = os.path.abspath(args.config)
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Creating default configuration file...")
        write_default_config(config_path, overwrite=True)
    
    config = load_config(config_path)
    
    # Load override configuration if provided
    if args.override and os.path.exists(args.override):
        override_config = load_config(args.override)
        config = merge_configs(config, override_config)
    
    # Set up experiment name with timestamp
    experiment_name = config["output"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"
    
    # Set up logger
    training_logger = TrainingLogger(
        experiment_name=experiment_name,
        log_level=config["output"]["log_level"],
        log_dir=config["output"]["log_dir"],
        show_module_info=True
    )
    
    # Log hyperparameters
    training_logger.log_hyperparameters(config)
    
    try:
        # Set up MLflow if enabled
        if config["mlflow"]["use_mlflow"]:
            if MLFLOW_AVAILABLE:
                mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
                mlflow.set_experiment(config["mlflow"]["experiment_name"])
                
                # Start MLflow run
                with mlflow.start_run(run_name=experiment_id):
                    mlflow.log_params({
                        "input_channels": config["model"]["input_channels"],
                        "freq_bins": config["model"]["freq_bins"],
                        "conv_channels": str(config["model"]["conv_channels"]),
                        "fc_sizes": str(config["model"]["fc_sizes"]),
                        "batch_size": config["training"]["batch_size"],
                        "learning_rate": config["training"]["learning_rate"],
                        "weight_decay": config["training"]["weight_decay"],
                        "use_augmentation": config["augmentation"]["use_augmentation"],
                        "feature_type": config["preprocessing"]["feature_type"]
                    })
                    
                    # Execute training pipeline
                    execute_training_pipeline(config, training_logger)
            else:
                training_logger.logger.warning("MLflow not available. Install mlflow package to enable tracking.")
                config["mlflow"]["use_mlflow"] = False
                execute_training_pipeline(config, training_logger)
        else:
            # Execute training pipeline without MLflow
            execute_training_pipeline(config, training_logger)
            
    except Exception as e:
        training_logger.logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


def execute_training_pipeline(config, training_logger):
    """
    Execute the training pipeline.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    training_logger : TrainingLogger
        Logger instance
    """
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, dataset_builder = prepare_data(config, training_logger.logger)
    
    # Visualize preprocessing stages if enabled (independent of data visualization)
    if config["output"].get("visualize_preprocessing", False):
        training_logger.logger.info("Visualizing preprocessing stages...")
        visualize_preprocessing_stages(dataset_builder, config, training_logger.experiment_dir)
    
    # Visualize data samples if enabled
    if config["output"]["visualize_data"]:
        training_logger.logger.info("Visualizing data samples...")
        # Pass the experiment directory from the logger to ensure consistency
        visualize_data_samples(X_train, X_val, y_train, y_val, dataset_builder, config, 
                              experiment_dir=training_logger.experiment_dir)
    
    # Create model
    model = create_model(config, training_logger.logger)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, config
    )
    
    # Train model
    trained_model, history = train_model(model, train_loader, val_loader, config, training_logger)
    
    # Evaluate model
    metrics = evaluate_model(trained_model, test_loader, config, training_logger)
    
    # Save model and metadata
    save_model_and_metadata(trained_model, config, metrics, history, training_logger)


if __name__ == "__main__":
    main() 