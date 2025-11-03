"""
Model inference module for batch processing and predictions.

This module provides functionality for loading models and running
batch predictions on spectrograms.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class ModelInference:
    """
    Model inference class for batch processing of spectrograms.
    
    Supports loading trained models and running batch predictions
    with confidence scores and timestamps for multi-class classification.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[tf.keras.Model] = None,
        classes: Optional[List[str]] = None
    ):
        """
        Initialize ModelInference.
        
        Parameters
        ----------
        model_path : str, optional
            Path to saved Keras model file. If provided, model will be loaded.
        model : tf.keras.Model, optional
            Pre-loaded Keras model. If provided, model_path is ignored.
        classes : List[str], optional
            List of class names in order. Default: ['noise', 'anemonefish', 'biological']
            Must match the order used during training.
        """
        self.model = None
        self.model_path = model_path
        self.classes = classes or ['noise', 'anemonefish', 'biological']
        self.num_classes = len(self.classes)
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained Keras model.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model file
            
        Returns
        -------
        bool
            True if model loaded successfully, False otherwise
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            print(f"Model loaded successfully from: {model_path}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            
            # Verify output shape matches expected number of classes
            output_shape = self.model.output_shape
            if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 1:
                model_num_classes = output_shape[-1]
                if model_num_classes != self.num_classes:
                    print(f"Warning: Model output has {model_num_classes} classes, but classes list has {self.num_classes}")
                    print(f"Using model output shape ({model_num_classes}) instead")
                    self.num_classes = model_num_classes
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False
    
    def predict_batch(
        self,
        spectrograms: np.ndarray,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """
        Run batch prediction on spectrograms.
        
        Parameters
        ----------
        spectrograms : np.ndarray
            Array of spectrograms with shape (N, H, W, C)
        batch_size : int
            Batch size for prediction. Default: 32
        verbose : int
            Verbosity mode for prediction (0, 1, or 2). Default: 0
            
        Returns
        -------
        predictions : np.ndarray
            Array of probability scores with shape (N, num_classes)
            Each row contains probabilities for all classes [noise, anemonefish, biological]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if len(spectrograms) == 0:
            return np.array([])
        
        try:
            # Predict in batches - returns (N, num_classes) for multi-class
            predictions = self.model.predict(spectrograms, batch_size=batch_size, verbose=verbose)
            
            # Ensure predictions are 2D: (N, num_classes)
            if predictions.ndim == 1:
                # If 1D, reshape to (N, 1) - handle binary case if needed
                predictions = predictions.reshape(-1, 1)
            elif predictions.ndim > 2:
                # Flatten extra dimensions if any
                predictions = predictions.reshape(predictions.shape[0], -1)
            
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.array([])
    
    def predict_with_timestamps(
        self,
        spectrograms: np.ndarray,
        timestamps: List[Tuple[float, float]],
        batch_size: int = 32,
        return_all_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run predictions and return results with timestamps.
        
        Parameters
        ----------
        spectrograms : np.ndarray
            Array of spectrograms
        timestamps : List[Tuple[float, float]]
            List of (start_time, end_time) for each spectrogram
        batch_size : int
            Batch size for prediction. Default: 32
        return_all_probabilities : bool
            If True, include probabilities for all classes in output. Default: False
            
        Returns
        -------
        predictions : List[Dict[str, Any]]
            List of prediction dictionaries with:
            - timestamp: str (e.g., "0.0-1.0s")
            - class: str (predicted class name)
            - confidence: float (probability of predicted class)
            - start_time: float
            - end_time: float
            - probabilities: Dict[str, float] (optional, if return_all_probabilities=True)
                Dictionary mapping class names to probabilities
        """
        if len(spectrograms) != len(timestamps):
            raise ValueError("Number of spectrograms must match number of timestamps")
        
        # Get probability scores - shape (N, num_classes)
        probabilities = self.predict_batch(spectrograms, batch_size=batch_size)
        
        if len(probabilities) == 0:
            return []
        
        # Get predicted class indices (argmax)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_confidences = np.max(probabilities, axis=1)
        
        predictions = []
        for i, (pred_idx, confidence, (start_time, end_time)) in enumerate(
            zip(predicted_indices, predicted_confidences, timestamps)
        ):
            # Ensure pred_idx is within class list bounds
            if pred_idx >= len(self.classes):
                print(f"Warning: Predicted class index {pred_idx} >= {len(self.classes)}, using 'unknown'")
                predicted_class = 'unknown'
            else:
                predicted_class = self.classes[pred_idx]
            
            pred_dict = {
                'timestamp': f"{start_time:.1f}-{end_time:.1f}s",
                'start_time': start_time,
                'end_time': end_time,
                'class': predicted_class,
                'confidence': float(confidence)
            }
            
            # Add all class probabilities if requested
            if return_all_probabilities:
                pred_dict['probabilities'] = {
                    class_name: float(prob)
                    for class_name, prob in zip(self.classes, probabilities[i])
                }
            
            predictions.append(pred_dict)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns
        -------
        info : Dict[str, Any]
            Dictionary with model information
        """
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_parameters': self.model.count_params()
        }
