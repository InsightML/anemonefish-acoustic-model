"""
Call-type classifier for anemonefish sounds.

This module provides a PyTorch-based multi-class classifier for identifying
different types of anemonefish calls (e.g., aggressive, submissive, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AnemonefishCallTypeClassifier(nn.Module):
    """
    Multi-class classifier for identifying different types of anemonefish calls.
    
    This CNN-based model takes spectrogram inputs and outputs probabilities for
    different call types.
    """
    
    def __init__(self, input_channels=1, freq_bins=64, num_classes=5, conv_channels=(16, 32, 64, 128), fc_sizes=(256, 128)):
        """
        Initialize the call-type classifier.
        
        Parameters
        ----------
        input_channels : int, optional
            Number of input channels, by default 1
        freq_bins : int, optional
            Number of frequency bins in the input spectrogram, by default 64 (reduced from 128 for 0-2000 Hz range)
        num_classes : int, optional
            Number of call types to classify, by default 5
        conv_channels : tuple, optional
            Number of channels in each convolutional layer, by default (16, 32, 64, 128)
        fc_sizes : tuple, optional
            Size of each fully-connected layer, by default (256, 128)
        """
        super(AnemonefishCallTypeClassifier, self).__init__()
        
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
        
        # Adaptive pooling to handle variable-length inputs
        self.adaptive_pool = nn.AdaptiveAvgPool2d((height, 1))
        
        # Fully-connected layers
        self.fc_layers = nn.ModuleList()
        in_features = conv_channels[-1] * height
        
        for out_features in fc_sizes:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                )
            )
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, freq_bins, time_frames)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes)
        """
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Apply adaptive pooling to handle variable time dimension
        x = self.adaptive_pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully-connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Apply output layer
        x = self.output_layer(x)
        
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Train the call-type classifier.
    
    Parameters
    ----------
    model : AnemonefishCallTypeClassifier
        Model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    num_epochs : int, optional
        Number of epochs to train for, by default 10
    device : str, optional
        Device to train on, by default 'cuda'
        
    Returns
    -------
    dict
        Dictionary containing training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1_macro': []
    }
    
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
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        # Compute epoch statistics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                # Store predictions and targets for F1 score computation
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Compute epoch statistics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Compute F1 score (macro-averaged)
        from sklearn.metrics import f1_score
        val_f1_macro = f1_score(val_targets, val_preds, average='macro')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Val F1 (macro): {val_f1_macro:.4f}')
    
    return history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the call-type classifier on a test set.
    
    Parameters
    ----------
    model : AnemonefishCallTypeClassifier
        Model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : str, optional
        Device to evaluate on, by default 'cuda'
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    test_correct = 0
    test_total = 0
    test_preds = []
    test_targets = []
    
    # Evaluate the model
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
            
            # Store predictions and targets for additional metrics
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(test_targets, test_preds)
    f1_macro = f1_score(test_targets, test_preds, average='macro')
    precision, recall, f1, support = precision_recall_fscore_support(test_targets, test_preds, average=None)
    conf_matrix = confusion_matrix(test_targets, test_preds)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'confusion_matrix': conf_matrix
    }
