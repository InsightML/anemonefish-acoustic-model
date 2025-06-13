"""
CRNN-based binary classifier for anemonefish sound detection.

This module provides a PyTorch-based Convolutional Recurrent Neural Network (CRNN)
for detecting anemonefish sounds in hydrophone recordings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, List, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import pickle


class AttentionLayer(nn.Module):
    """
    Attention mechanism for focusing on the most relevant parts of the sequence.
    
    This implements a simple attention mechanism that computes a weighted sum
    of the RNN outputs based on learned attention weights.
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize the attention layer.
        
        Parameters
        ----------
        hidden_size : int
            Size of the hidden state from the RNN
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, rnn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the attention layer.
        
        Parameters
        ----------
        rnn_output : torch.Tensor
            Output from the RNN layer of shape (batch_size, seq_len, hidden_size)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - context_vector: Weighted sum of RNN outputs (batch_size, hidden_size)
            - attention_weights: Attention weights for each time step (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(rnn_output)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights to RNN outputs
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            rnn_output  # (batch_size, seq_len, hidden_size)
        )  # (batch_size, 1, hidden_size)
        
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights


class CRNNAnemonefishBinaryClassifier(nn.Module):
    """
    CRNN-based binary classifier for detecting anemonefish sounds in spectrograms.
    
    This model combines convolutional layers for feature extraction with
    recurrent layers for temporal modeling, followed by an optional attention
    mechanism and fully connected layers for classification.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        freq_bins: int = 64,
        conv_channels: Tuple[int, ...] = (16, 32, 64, 128),
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
        fc_sizes: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.5,
        use_attention: bool = True,
        rnn_type: str = 'lstm'
    ):
        """
        Initialize the CRNN binary classifier.
        
        Parameters
        ----------
        input_channels : int, optional
            Number of input channels, by default 1
        freq_bins : int, optional
            Number of frequency bins in the input spectrogram, by default 64
        conv_channels : Tuple[int, ...], optional
            Number of channels in each convolutional layer, by default (16, 32, 64, 128)
        rnn_hidden_size : int, optional
            Size of the hidden state in the RNN, by default 128
        rnn_num_layers : int, optional
            Number of RNN layers, by default 2
        rnn_bidirectional : bool, optional
            Whether to use bidirectional RNN, by default True
        fc_sizes : Tuple[int, ...], optional
            Size of each fully-connected layer, by default (128, 64)
        dropout_rate : float, optional
            Dropout rate for regularization, by default 0.5
        use_attention : bool, optional
            Whether to use attention mechanism, by default True
        rnn_type : str, optional
            Type of RNN to use ('lstm' or 'gru'), by default 'lstm'
        """
        super(CRNNAnemonefishBinaryClassifier, self).__init__()
        
        # Store all initialization parameters as instance attributes
        self.input_channels = input_channels
        self.freq_bins = freq_bins
        self.conv_channels = conv_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional
        self.fc_sizes = fc_sizes
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        
        # Determine the appropriate number of conv layers based on input dimensions
        # Each max pooling layer halves the dimension, so we need to ensure we don't end up with 0
        max_pooling_layers = min(
            int(math.log2(freq_bins)),  # Max layers based on freq dimension
            len(conv_channels)          # Max layers based on provided channels
        )
        
        # Ensure we have at least one layer
        max_pooling_layers = max(1, max_pooling_layers)
        
        # Use only the appropriate number of channels
        conv_channels = conv_channels[:max_pooling_layers]
        self.used_conv_channels = conv_channels  # Store the actually used channels
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
            )
            in_channels = out_channels
        
        # Calculate size after convolutions
        height = freq_bins
        for _ in range(len(conv_channels)):
            height = height // 2  # Max pooling with kernel_size=2
        
        # RNN layer
        self.rnn_input_size = conv_channels[-1] * height
        self.rnn_output_size = rnn_hidden_size * 2 if rnn_bidirectional else rnn_hidden_size
        
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                batch_first=True,
                dropout=dropout_rate if rnn_num_layers > 1 else 0
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=self.rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                batch_first=True,
                dropout=dropout_rate if rnn_num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'lstm' or 'gru'.")
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(self.rnn_output_size)
        
        # Fully-connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.rnn_output_size
        
        for out_features in fc_sizes:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, 1)
        
        # Print model structure info
        print(f"CRNN Model initialized with {len(conv_channels)} conv layers")
        print(f"Frequency dimension reduction: {freq_bins} -> {height}")
        print(f"RNN input size: {self.rnn_input_size}")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, freq_bins, time_frames)
            
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If use_attention is True:
                Tuple containing:
                - Output tensor of shape (batch_size, 1)
                - Attention weights of shape (batch_size, time_frames)
            Else:
                Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for RNN: (batch_size, channels, height, time) -> (batch_size, time, channels*height)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time, channels, height)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch_size, time, channels*height)
        
        # Apply RNN
        rnn_output, _ = self.rnn(x)  # (batch_size, time, hidden_size*num_directions)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            context_vector, attention_weights = self.attention(rnn_output)
            x = context_vector
        else:
            # Use the last output from the RNN
            x = rnn_output[:, -1, :]
        
        # Apply fully-connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Apply output layer
        x = self.output_layer(x)
        
        # Apply sigmoid activation for binary classification
        x = torch.sigmoid(x)
        
        if self.use_attention and attention_weights is not None:
            return x, attention_weights
        else:
            return x


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str = 'cuda',
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Train the CRNN binary classifier.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    num_epochs : int, optional
        Number of epochs to train for, by default 10
    device : str, optional
        Device to train on, by default 'cuda'
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
        Learning rate scheduler, by default None
    early_stopping_patience : Optional[int], optional
        Number of epochs to wait for improvement before stopping, by default None
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary containing training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_recall': [],
        'val_precision': [],
        'val_f1': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Train the model
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
            
            # Handle attention output if present
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            
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
        train_accuracy = train_correct / train_total
        
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
                
                # Handle attention output if present
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                
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
        val_accuracy = val_correct / val_total
        val_recall = val_true_positives / val_actual_positives if val_actual_positives > 0 else 0
        val_precision = val_true_positives / val_predicted_positives if val_predicted_positives > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        history['val_f1'].append(val_f1)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'  Val Recall: {val_recall:.4f}, Val Precision: {val_precision:.4f}, Val F1: {val_f1:.4f}')
        
        # Early stopping check
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate the CRNN binary classifier on a test set.
        
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : str, optional
        Device to evaluate on, by default 'cuda'
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    test_correct = 0
    test_total = 0
    test_true_positives = 0
    test_predicted_positives = 0
    test_actual_positives = 0
    
    # Store all predictions and targets for ROC curve
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle attention output if present
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            
            # Store outputs and targets for ROC curve
            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Calculate metrics
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == targets.view(-1, 1)).sum().item()
            test_total += targets.size(0)
            
            # Compute recall and precision metrics
            test_true_positives += ((predicted == 1) & (targets.view(-1, 1) == 1)).sum().item()
            test_predicted_positives += (predicted == 1).sum().item()
            test_actual_positives += (targets == 1).sum().item()
    
    # Compute metrics
    accuracy = test_correct / test_total
    recall = test_true_positives / test_actual_positives if test_actual_positives > 0 else 0
    precision = test_true_positives / test_predicted_positives if test_predicted_positives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'outputs': np.array(all_outputs),
        'targets': np.array(all_targets)
    }


def save_model(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save the model to disk.
    
    Parameters
    ----------
    model : nn.Module
        Model to save
    filepath : str
        Path to save the model to
    metadata : Optional[Dict], optional
        Additional metadata to save with the model, by default None
    """
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    # Save complete model configuration
    if isinstance(model, CRNNAnemonefishBinaryClassifier):
        # Get all initialization parameters
        model_config = {}
        # Check for all possible attributes that could be initialization parameters
        for attr in [
            'input_channels', 'freq_bins', 'conv_channels', 'rnn_hidden_size',
            'rnn_num_layers', 'rnn_bidirectional', 'fc_sizes', 'dropout_rate',
            'use_attention', 'rnn_type', 'rnn_input_size', 'rnn_output_size'
        ]:
            if hasattr(model, attr):
                model_config[attr] = getattr(model, attr)
        
        # Store the number of convolutional layers actually used
        if hasattr(model, 'conv_layers'):
            model_config['num_conv_layers'] = len(model.conv_layers)
        
        save_dict['model_config'] = model_config
    
    # Add metadata if provided
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    # Save model
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    filepath: str,
    device: str = 'cuda'
) -> Tuple[nn.Module, Optional[Dict]]:
    """
    Load a model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to load the model from
    device : str, optional
        Device to load the model to, by default 'cuda'
        
    Returns
    -------
    Tuple[nn.Module, Optional[Dict]]
        Tuple containing:
        - Loaded model
        - Metadata dictionary if available, otherwise None
    """
    # Import torch here to ensure it's available in this function scope
    import torch
    
    # Load save dictionary with appropriate settings for PyTorch 2.6+
    try:
        # First try with weights_only=False which might succeed but is less secure
        save_dict = torch.load(filepath, map_location=device, weights_only=False)
    except (RuntimeError, pickle.UnpicklingError) as e:
        print(f"Warning: Initial load attempt failed: {e}")
        try:
            # Try a more secure approach with explicit safe_globals
            import torch.serialization
            # Add numpy.core.multiarray._reconstruct to safe globals
            with torch.serialization.safe_globals(['numpy.core.multiarray._reconstruct']):
                save_dict = torch.load(filepath, map_location=device)
        except Exception as e2:
            print(f"Error: Failed to load model with safe globals: {e2}")
            # Last fallback, completely disable weights_only as a last resort
            try:
                print("Attempting to load with weights_only=False as last resort...")
                save_dict = torch.load(filepath, map_location=device, weights_only=False)
            except Exception as e3:
                raise RuntimeError(f"All attempts to load model failed: {e3}")
    
    # Check model class
    model_class = save_dict.get('model_class', 'CRNNAnemonefishBinaryClassifier')
    model_config = save_dict.get('model_config', {})
    
    # Initialize model based on class
    if model_class == 'CRNNAnemonefishBinaryClassifier':
        # Filter out non-initialization parameters
        init_params = {}
        for param in [
            'input_channels', 'freq_bins', 'conv_channels', 'rnn_hidden_size',
            'rnn_num_layers', 'rnn_bidirectional', 'fc_sizes', 'dropout_rate',
            'use_attention', 'rnn_type'
        ]:
            if param in model_config:
                init_params[param] = model_config[param]
        
        # Create model with exact same configuration
        model = CRNNAnemonefishBinaryClassifier(**init_params)
        
        # Load state dict
        try:
            model.load_state_dict(save_dict['model_state_dict'])
        except Exception as e:
            print(f"Warning: Error loading model state dict: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(save_dict['model_state_dict'], strict=False)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Return model and metadata
    return model, save_dict.get('metadata', None)


def test_load_model(
    filepath: str,
    test_input_shape: Tuple[int, ...] = (1, 1, 64, 43),
    device: str = 'cpu'
) -> None:
    """
    Test loading a model and running a basic prediction.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model file
    test_input_shape : Tuple[int, ...], optional
        Shape of test input tensor, by default (1, 1, 64, 43)
    device : str, optional
        Device to load the model to, by default 'cpu'
    """
    # Import torch here to ensure it's available in this function scope
    import torch
    
    try:
        # Load model
        print(f"Loading model from {filepath}...")
        model, metadata = load_model(filepath, device)
        
        # Print model configuration
        print("\nModel Configuration:")
        if isinstance(model, CRNNAnemonefishBinaryClassifier):
            for attr in [
                'input_channels', 'freq_bins', 'conv_channels', 'used_conv_channels',
                'rnn_hidden_size', 'rnn_num_layers', 'rnn_bidirectional', 
                'fc_sizes', 'dropout_rate', 'use_attention', 'rnn_type'
            ]:
                if hasattr(model, attr):
                    print(f"  {attr}: {getattr(model, attr)}")
        
        # Create test input with dimensions that match the loaded model
        if isinstance(model, CRNNAnemonefishBinaryClassifier):
            # Use the model's freq_bins attribute to create correct input shape
            # Format for CRNN input is (batch_size, channels, height, width)
            # where height is freq_bins and width is time steps
            correct_input_shape = (1, model.input_channels, model.freq_bins, test_input_shape[3] if len(test_input_shape) > 3 else 43)
            print(f"\nCreating test input with model-specific shape {correct_input_shape}...")
            x = torch.rand(correct_input_shape).to(device)
        else:
            # Fallback to provided shape
            print(f"\nCreating test input with shape {test_input_shape}...")
            x = torch.rand(test_input_shape).to(device)
        
        # Run prediction
        print("Running test prediction...")
        with torch.no_grad():
            output = model(x)
        
        # Print output shape
        if isinstance(output, tuple):
            print(f"Output shape: {output[0].shape}, Attention weights shape: {output[1].shape}")
        else:
            print(f"Output shape: {output.shape}")
        
        print("\nModel loaded and tested successfully!")
        
        # Return metadata if available
        if metadata:
            print("\nModel Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()


def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """Plot the training history.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary containing training history (loss, accuracy, etc.)
    save_path : str, optional
        Path where to save the plot, by default None
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 4)
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(model: nn.Module, 
                  test_loader: torch.utils.data.DataLoader,
                  device: str = 'cuda',
                  save_path: str = None) -> None:
    """Plot the ROC curve for the model.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader containing test data
    device : str, optional
        Device to run evaluation on, by default 'cuda'
    save_path : str, optional
        Path where to save the plot, by default None
    """
    model.eval()
    model = model.to(device)
    
    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []
    
    # Collect predictions and labels
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model outputs
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract predictions if attention is used
            
            # Store predictions and labels
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Compute ROC curve and area
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(model: nn.Module, 
                         test_loader: torch.utils.data.DataLoader,
                         device: str = 'cuda',
                         save_path: str = None) -> None:
    """Plot the confusion matrix for the model.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader containing test data
    device : str, optional
        Device to run evaluation on, by default 'cuda'
    save_path : str, optional
        Path where to save the plot, by default None
    """
    model.eval()
    model = model.to(device)
    
    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []
    
    # Collect predictions and labels
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model outputs
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract predictions if attention is used
            
            # Convert probabilities to binary predictions
            preds = (outputs > 0.5).float()
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Anemonefish', 'Anemonefish'],
                yticklabels=['Not Anemonefish', 'Anemonefish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show() 