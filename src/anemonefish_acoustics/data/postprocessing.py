"""
Postprocessing module to take the numpy array prediction, that has the shape
(num_windows, num_classes), and convert into a variety of human friendly formats.

provides the data in Base64 encoded format for API handling.

Output format:
- JSON [{"start_time": float, "end_time": float, "class": str, "confidence": float}]
- Audacity labels (.txt)
- Raw numpy array (Base64 encoded)

"""

import base64
import numpy as np

CLASS_MAPPINGS = ["noise", "anemonefish", "biological"]
WINDOW_DURATION = 0.2  # seconds per window (from slide_size_seconds)

def _merge_consecutive_segments(prediction):
    """
    Helper function to merge consecutive windows with the same predicted class.
    
    Args:
        prediction: numpy array of shape (num_windows, num_classes) with probabilities
        
    Returns:
        List of segments: [{"start_time": float, "end_time": float, "class": str, "confidence": float}]
        Filters out noise class (index 0)
    """
    # Get predicted classes and confidences
    predicted_classes = np.argmax(prediction, axis=1)
    max_confidences = np.max(prediction, axis=1)
    
    segments = []
    current_segment = None
    
    for i, (pred_class, confidence) in enumerate(zip(predicted_classes, max_confidences)):
        # Skip noise class (index 0)
        if pred_class == 0:
            # If we were tracking a segment, close it
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
            continue
            
        # Calculate time for this window
        window_time = i * WINDOW_DURATION
        
        # Check if we need to start a new segment
        if current_segment is None or current_segment["class"] != CLASS_MAPPINGS[pred_class]:
            # Close previous segment if exists
            if current_segment is not None:
                segments.append(current_segment)
            
            # Start new segment
            current_segment = {
                "start_time": window_time,
                "end_time": window_time + WINDOW_DURATION,
                "class": CLASS_MAPPINGS[pred_class],
                "confidence": float(confidence),
                "confidence_sum": float(confidence),
                "window_count": 1
            }
        else:
            # Extend current segment
            current_segment["end_time"] = window_time + WINDOW_DURATION
            current_segment["confidence_sum"] += float(confidence)
            current_segment["window_count"] += 1
    
    # Don't forget the last segment
    if current_segment is not None:
        segments.append(current_segment)
    
    # Calculate average confidence for each segment
    for segment in segments:
        segment["confidence"] = segment["confidence_sum"] / segment["window_count"]
        # Remove temporary fields
        del segment["confidence_sum"]
        del segment["window_count"]
    
    return segments


def json_export(prediction):
    """
    Convert prediction (num_windows, num_classes) into a JSON format
    
    Returns:
        List of dictionaries with start_time, end_time, class, and confidence
    """
    return _merge_consecutive_segments(prediction)


def audacity_export(prediction):
    """
    Convert prediction (num_windows, num_classes) into an Audacity labels (.txt) format
    
    Returns:
        String in Audacity label format: "start_time\tend_time\tclass_label\n"
    """
    segments = _merge_consecutive_segments(prediction)
    
    # Format as Audacity labels
    lines = []
    for segment in segments:
        line = f"{segment['start_time']:.6f}\t{segment['end_time']:.6f}\t{segment['class']}"
        lines.append(line)
    
    return "\n".join(lines)


def raw_export(prediction):
    """
    Export the raw numpy array without any filtering or processing
    
    Args:
        prediction: numpy array of shape (num_windows, num_classes)
        
    Returns:
        The raw numpy array as-is
    """
    return prediction


def encoder(data):
    """
    Encode the data into a Base64 encoded string
    """
    if isinstance(data, np.ndarray):
        return base64.b64encode(data.tobytes()).decode('utf-8')
    elif isinstance(data, str):
        return base64.b64encode(data.encode('utf-8')).decode('utf-8')
    else:
        raise ValueError(f"Unsupported data type for encoding: {type(data)}")


def postprocess_prediction(prediction, logger=None):
    """
    Main postprocessing function that generates all export formats
    
    Args:
        prediction: numpy array of shape (num_windows, num_classes)
        logger: optional logger instance
        
    Returns:
        Dictionary with:
        - "json": List of segment dictionaries (not encoded)
        - "audacity": Base64 encoded Audacity label string
        - "raw": Base64 encoded numpy array
    """
    if logger:
        logger.info(f"Prediction shape: {prediction.shape}")
    
    # Generate exports
    json_result = json_export(prediction)
    audacity_result = audacity_export(prediction)
    raw_result = raw_export(prediction)
    
    # Encode appropriate formats
    audacity_encoded = encoder(audacity_result)
    raw_encoded = encoder(raw_result)
    
    return {
        "json": json_result,  # Return as native Python object
        "audacity": audacity_encoded,  # Base64 encoded string
        "raw": raw_encoded  # Base64 encoded numpy array
    }