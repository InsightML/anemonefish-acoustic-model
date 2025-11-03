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
    with confidence scores and timestamps.
    """
    
    def __init__(self, model_path: Optional[str] = None, model: Optional[tf.keras.Model] = None):
        """
        Initialize ModelInference.
        
        Parameters
        ----------
        model_path : str, optional
            Path to saved Keras model file. If provided, model will be loaded.
        model : tf.keras.Model, optional
            Pre-loaded Keras model. If provided, model_path is ignored.
        """
        self.model = None
        self.model_path = model_path
        
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
            Array of probability scores with shape (N,)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if len(spectrograms) == 0:
            return np.array([])
        
        try:
            # Predict in batches
            predictions = self.model.predict(spectrograms, batch_size=batch_size, verbose=verbose)
            
            # Flatten predictions (from (N, 1) to (N,))
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.array([])
    
    def predict_with_timestamps(
        self,
        spectrograms: np.ndarray,
        timestamps: List[Tuple[float, float]],
        batch_size: int = 32,
        probability_threshold: float = 0.5
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
        probability_threshold : float
            Threshold for binary classification. Default: 0.5
            
        Returns
        -------
        predictions : List[Dict[str, Any]]
            List of prediction dictionaries with:
            - timestamp: str (e.g., "0.0-1.0s")
            - class: str ("anemonefish" or "noise")
            - confidence: float
            - start_time: float
            - end_time: float
        """
        if len(spectrograms) != len(timestamps):
            raise ValueError("Number of spectrograms must match number of timestamps")
        
        # Get probability scores
        probabilities = self.predict_batch(spectrograms, batch_size=batch_size)
        
        predictions = []
        for i, (prob, (start_time, end_time)) in enumerate(zip(probabilities, timestamps)):
            predictions.append({
                'timestamp': f"{start_time:.1f}-{end_time:.1f}s",
                'start_time': start_time,
                'end_time': end_time,
                'class': 'anemonefish' if prob > probability_threshold else 'noise',
                'confidence': float(prob)
            })
        
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
