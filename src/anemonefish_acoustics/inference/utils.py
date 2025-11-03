"""
Utility functions for inference pipeline.
"""

import json
from typing import Dict, Any, List
from scipy.signal import medfilt
import numpy as np


def smooth_predictions(predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply median filtering to smooth predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of prediction scores
    window_size : int
        Size of median filter window. Default: 5
        
    Returns
    -------
    smoothed_predictions : np.ndarray
        Smoothed prediction array
    """
    if len(predictions) == 0:
        return predictions
    
    # Apply median filter
    smoothed = medfilt(predictions, kernel_size=window_size)
    
    return smoothed


def detect_events(
    predictions: np.ndarray,
    timestamps: List[tuple],
    probability_threshold: float = 0.5,
    min_event_duration: float = 0.2,
    min_gap_duration: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Detect events from prediction stream using thresholding and post-processing.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of prediction scores
    timestamps : List[tuple]
        List of (start_time, end_time) for each prediction
    probability_threshold : float
        Threshold for binary classification. Default: 0.5
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
    
    # Apply threshold
    binary_predictions = predictions > probability_threshold
    
    # Find event boundaries
    events = []
    in_event = False
    event_start = None
    event_confidences = []
    
    for i, (is_event, (start_time, end_time)) in enumerate(zip(binary_predictions, timestamps)):
        if is_event and not in_event:
            # Start of new event
            in_event = True
            event_start = start_time
            event_confidences = [predictions[i]]
            
        elif is_event and in_event:
            # Continue current event
            event_confidences.append(predictions[i])
            
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
