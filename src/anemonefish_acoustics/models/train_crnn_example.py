#!/usr/bin/env python3
"""
Example script for training a CRNN model for anemonefish sound detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import importlib.util

# Add the project root to the Python path to facilitate imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Create dataloaders
def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Add channel dimension for BCE loss
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # Add channel dimension for BCE loss
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Import required modules using multiple fallback strategies
def import_dependencies():
    # Define the modules we need to import
    modules = {
        'DatasetBuilder': [
            'src.anemonefish_acoustics.data_processing.data_preprocessing',
            'anemonefish_acoustics.data_processing.data_preprocessing'
        ],
        'CRNN_modules': [
            'src.anemonefish_acoustics.models.crnn_classifier',
            'anemonefish_acoustics.models.crnn_classifier',
            'crnn_classifier'  # Local import
        ]
    }
    
    imported = {}
    
    # Try to import DatasetBuilder
    for module_path in modules['DatasetBuilder']:
        try:
            module = importlib.import_module(module_path)
            imported['DatasetBuilder'] = module.DatasetBuilder
            print(f"Successfully imported DatasetBuilder from {module_path}")
            break
        except (ImportError, AttributeError):
            continue
    
    # Try to import CRNN modules
    for module_path in modules['CRNN_modules']:
        try:
            module = importlib.import_module(module_path)
            imported['CRNNAnemonefishBinaryClassifier'] = module.CRNNAnemonefishBinaryClassifier
            imported['train_model'] = module.train_model
            imported['evaluate_model'] = module.evaluate_model
            imported['plot_training_history'] = module.plot_training_history
            imported['plot_roc_curve'] = module.plot_roc_curve
            imported['plot_confusion_matrix'] = module.plot_confusion_matrix
            imported['save_model'] = module.save_model
            imported['load_model'] = module.load_model
            imported['test_load_model'] = module.test_load_model
            print(f"Successfully imported CRNN modules from {module_path}")
            break
        except (ImportError, AttributeError) as e:
            print(f"Failed to import from {module_path}: {e}")
            continue
    
    # Check if we have all required modules
    required_modules = ['DatasetBuilder', 'CRNNAnemonefishBinaryClassifier', 
                      'train_model', 'evaluate_model',
                      'save_model', 'load_model']
    missing_modules = [module for module in required_modules if module not in imported]
    
    if missing_modules:
        raise ImportError(f"Could not import required modules: {', '.join(missing_modules)}")
    
    return imported


def main():
    """Main function."""
    # Import dependencies
    deps = import_dependencies()
    
    # Extract imported classes and functions
    DatasetBuilder = deps['DatasetBuilder']
    CRNNAnemonefishBinaryClassifier = deps['CRNNAnemonefishBinaryClassifier']
    train_model = deps['train_model']
    evaluate_model = deps['evaluate_model']
    plot_training_history = deps['plot_training_history'] if 'plot_training_history' in deps else None
    plot_roc_curve = deps['plot_roc_curve'] if 'plot_roc_curve' in deps else None
    plot_confusion_matrix = deps['plot_confusion_matrix'] if 'plot_confusion_matrix' in deps else None
    save_model = deps['save_model'] 
    load_model = deps['load_model']
    test_load_model = deps['test_load_model'] if 'test_load_model' in deps else None
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create DatasetBuilder
    dataset_builder = DatasetBuilder()
    
    # Prepare dataset with augmentation
    print("Preparing dataset...")
    X_train_dict, X_test_dict, y_train, y_test = dataset_builder.prepare_dataset_with_augmentation(
        test_size=0.2,
        use_augmentation=True,
        balance_ratio=1.0,
        random_state=42
    )
    
    # Prepare data for CRNN model (using combined features)
    print("Preparing data for CRNN model...")
    X_train, X_test, y_train, y_test = dataset_builder.prepare_data_for_model(
        X_train_dict, X_test_dict, y_train, y_test,
        feature_type='mfcc',  # Use just MFCC features to start with
        model_type='cnn'  # CRNN uses CNN-formatted input
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Fix the data shape for the model - our data is currently [batch, time, freq, channel]
    # We need to transpose to [batch, channel, freq, time]
    X_train = np.transpose(X_train, (0, 3, 2, 1))
    X_test = np.transpose(X_test, (0, 3, 2, 1))
    
    print(f"Transposed training data shape: {X_train.shape}")
    print(f"Transposed testing data shape: {X_test.shape}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Model saving path
    model_save_path = os.path.join(models_dir, 'crnn_binary_classifier.pt')
    
    # Check if we should train or load a saved model
    should_train_model = True
    if os.path.exists(model_save_path) and input(f"Model file {model_save_path} exists. Train new model? (y/n): ").lower() != 'y':
        should_train_model = False
    
    if should_train_model:
        # Create model
        model = CRNNAnemonefishBinaryClassifier(
            input_channels=X_train.shape[1],  # Number of input channels
            freq_bins=X_train.shape[2],  # Number of frequency bins
            conv_channels=(16, 32, 64, 128),
            rnn_hidden_size=128,
            rnn_num_layers=2,
            rnn_bidirectional=True,
            fc_sizes=(128, 64),
            dropout_rate=0.5,
            use_attention=True,
            rnn_type='lstm'
        )
        
        # Print model summary
        print(model)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Train model
        print("Training model...")
        history = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=test_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            num_epochs=30, 
            device=device,
            scheduler=scheduler,
            early_stopping_patience=10
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_metrics = evaluate_model(model, test_loader, device=device)
        
        print(f"Test metrics: {test_metrics}")
        
        # Save training history plot
        if plot_training_history:
            plot_training_history(history, save_path=os.path.join(models_dir, 'training_history.png'))
        
        # Plot ROC curve
        if plot_roc_curve:
            plot_roc_curve(model, test_loader, device=device, save_path=os.path.join(models_dir, 'roc_curve.png'))
        
        # Plot confusion matrix
        if plot_confusion_matrix:
            plot_confusion_matrix(model, test_loader, device=device, save_path=os.path.join(models_dir, 'confusion_matrix.png'))
        
        # Save model
        save_model(model, model_save_path, metadata={'test_metrics': test_metrics})
        
    # Test loading the saved model
    print("\n" + "="*50)
    print("Testing model loading functionality")
    print("="*50)
    
    # Get the expected input shape from the test data
    test_input_shape = (1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
    
    # Test loading the model
    if test_load_model:
        test_load_model(model_save_path, test_input_shape=test_input_shape, device=device)
    else:
        # Manual testing if test_load_model is not available
        print("test_load_model function not available, testing manually...")
        loaded_model, metadata = load_model(model_save_path, device)
        print(f"Model loaded successfully. Metadata: {metadata.keys() if metadata else None}")
    
    # If you want to test with real data
    if not should_train_model:
        print("\n" + "="*50)
        print("Testing loaded model on test data")
        print("="*50)
        
        # Load the model
        loaded_model, _ = load_model(model_save_path, device)
        
        # Evaluate the loaded model
        loaded_model_metrics = evaluate_model(loaded_model, test_loader, device=device)
        print(f"Loaded model test metrics: {loaded_model_metrics}")


if __name__ == '__main__':
    main() 