"""
Utility functions for inference pipeline.
"""

import json
from typing import Dict, Any, List, Optional
from scipy.signal import medfilt
import numpy as np


def smooth_predictions(predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply median filtering to smooth predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of prediction scores with shape (N, num_classes) or (N,)
    window_size : int
        Size of median filter window. Default: 5
        
    Returns
    -------
    smoothed_predictions : np.ndarray
        Smoothed prediction array with same shape as input
    """
    if len(predictions) == 0:
        return predictions
    
    # Apply median filter - handle both 1D and 2D arrays
    if predictions.ndim == 1:
        smoothed = medfilt(predictions, kernel_size=window_size)
    else:
        # For 2D arrays, apply median filter to each column separately
        smoothed = np.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            smoothed[:, i] = medfilt(predictions[:, i], kernel_size=window_size)
    
    return smoothed


def detect_events(
    predictions: np.ndarray,
    timestamps: List[tuple],
    predicted_classes: List[str],
    target_class: str = 'anemonefish',
    class_index_map: Optional[Dict[str, int]] = None,
    confidence_threshold: float = 0.5,
    min_event_duration: float = 0.2,
    min_gap_duration: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Detect events from prediction stream for multi-class classification.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of prediction probabilities with shape (N, num_classes) or (N,)
        If (N, num_classes), extracts probability for target_class
        If (N,), assumes these are probabilities for target_class
    timestamps : List[tuple]
        List of (start_time, end_time) for each prediction
    predicted_classes : List[str]
        List of predicted class names for each prediction
    target_class : str
        Class name to detect events for. Default: 'anemonefish'
    class_index_map : Dict[str, int], optional
        Mapping from class name to class index in predictions array.
        If None, will infer from predicted_classes and unique class names.
    confidence_threshold : float
        Minimum confidence for event detection. Default: 0.5
    min_event_duration : float
        Minimum duration for an event in seconds. Default: 0.2
    min_gap_duration : float
        Minimum gap between events in seconds. Default: 0.1
        
    Returns
    -------
    events : List[Dict[str, Any]]
        List of detected events with start/end times and confidence
    """
    if len(predictions) == 0:
        return []
    
    # Get confidences for target class
    if predictions.ndim == 1:
        # 1D: already probabilities for target class
        confidences = predictions
        is_target_class = predictions > confidence_threshold
    else:
        # 2D: (N, num_classes) - extract probability for target_class
        # Find target class index
        if class_index_map is None:
            # Infer from unique classes in predicted_classes
            unique_classes = list(set(predicted_classes))
            class_index_map = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        # Get target class index
        if target_class not in class_index_map:
            # Fallback: use first class that matches target_class in order
            # Default order: noise=0, anemonefish=1, biological=2
            default_map = {'noise': 0, 'anemonefish': 1, 'biological': 2}
            target_idx = default_map.get(target_class, 1)  # Default to anemonefish index
            print(f"Warning: {target_class} not in class_index_map, using default index {target_idx}")
        else:
            target_idx = class_index_map[target_class]
        
        # Extract probabilities for target class
        if target_idx >= predictions.shape[1]:
            print(f"Error: target class index {target_idx} >= num_classes {predictions.shape[1]}")
            return []
        
        confidences = predictions[:, target_idx]
        is_target_class = confidences > confidence_threshold
    
    # Find event boundaries
    events = []
    in_event = False
    event_start = None
    event_confidences = []
    
    for i, (is_event, (start_time, end_time)) in enumerate(zip(is_target_class, timestamps)):
        if is_event and not in_event:
            # Start of new event
            in_event = True
            event_start = start_time
            event_confidences = [confidences[i]]
            
        elif is_event and in_event:
            # Continue current event
            event_confidences.append(confidences[i])
            
        elif not is_event and in_event:
            # End of current event
            in_event = False
            event_end = timestamps[i-1][1] if i > 0 else end_time
            event_duration = event_end - event_start
            
            # Check minimum duration
            if event_duration >= min_event_duration:
                mean_confidence = np.mean(event_confidences)
                max_confidence = np.max(event_confidences)
                
                events.append({
                    'start_time': event_start,
                    'end_time': event_end,
                    'duration': event_duration,
                    'mean_confidence': float(mean_confidence),
                    'max_confidence': float(max_confidence)
                })
    
    # Handle case where file ends during an event
    if in_event and event_start is not None:
        event_end = timestamps[-1][1]
        event_duration = event_end - event_start
        
        if event_duration >= min_event_duration:
            mean_confidence = np.mean(event_confidences)
            max_confidence = np.max(event_confidences)
            
            events.append({
                'start_time': event_start,
                'end_time': event_end,
                'duration': event_duration,
                'mean_confidence': float(mean_confidence),
                'max_confidence': float(max_confidence)
            })
    
    # Merge events that are too close together
    if len(events) > 1:
        merged_events = []
        current_event = events[0]
        
        for next_event in events[1:]:
            gap = next_event['start_time'] - current_event['end_time']
            
            if gap < min_gap_duration:
                # Merge events
                current_event['end_time'] = next_event['end_time']
                current_event['duration'] = current_event['end_time'] - current_event['start_time']
                current_event['mean_confidence'] = (current_event['mean_confidence'] + next_event['mean_confidence']) / 2
                current_event['max_confidence'] = max(current_event['max_confidence'], next_event['max_confidence'])
            else:
                # Keep separate
                merged_events.append(current_event)
                current_event = next_event
        
        merged_events.append(current_event)
        events = merged_events
    
    return events


def format_response(
    predictions: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    audio_duration: float,
    processing_time: float,
    model_version: str = "v1.0"
) -> Dict[str, Any]:
    """
    Format prediction results as API response.
    
    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of all predictions with timestamps
    events : List[Dict[str, Any]]
        List of detected events
    audio_duration : float
        Duration of audio in seconds
    processing_time : float
        Processing time in seconds
    model_version : str
        Model version string. Default: "v1.0"
        
    Returns
    -------
    response : Dict[str, Any]
        Formatted API response dictionary
    """
    return {
        "predictions": predictions,
        "events": events,
        "metadata": {
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "model_version": model_version,
            "total_windows": len(predictions),
            "total_events": len(events)
        }
    }
