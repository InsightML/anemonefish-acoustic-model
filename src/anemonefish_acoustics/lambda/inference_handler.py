"""
Lambda handler for inference API.

Handles multipart form uploads from frontend, processes audio through
preprocessing ? inference pipeline, and returns JSON response.
"""

import json
import os
import time
import base64
import tempfile
from typing import Dict, Any, Optional
import numpy as np

# Lambda runtime imports
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

# Local imports
import sys
sys.path.append('/opt/python')  # Lambda layer path
sys.path.append('/var/task')     # Lambda task path

from anemonefish_acoustics.inference import AudioPreprocessor, ModelInference
from anemonefish_acoustics.inference.utils import (
    smooth_predictions,
    detect_events,
    format_response
)


# Global variables for model and preprocessor (reused across invocations)
_model = None
_preprocessor = None
_config = None


def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns
    -------
    config : Dict[str, Any]
        Configuration dictionary
    """
    # Get classes from environment or use default
    classes_str = os.environ.get('MODEL_CLASSES', 'noise,anemonefish,biological')
    classes = [c.strip() for c in classes_str.split(',')]
    
    return {
        'spectrogram': {
            'fmax': int(os.environ.get('FMAX_HZ', 2000)),
            'n_fft': int(os.environ.get('N_FFT', 1024)),
            'hop_length': int(os.environ.get('HOP_LENGTH', 256)),
            'width_pixels': int(os.environ.get('WIDTH_PIXELS', 256)),
            'height_pixels': int(os.environ.get('HEIGHT_PIXELS', 256)),
            'target_sr': os.environ.get('TARGET_SR')
        },
        'window': {
            'window_duration': float(os.environ.get('WINDOW_DURATION', 0.4)),
            'stride_duration': float(os.environ.get('STRIDE_DURATION', 0.2))
        },
        'prediction': {
            'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
            'confidence_threshold': float(os.environ.get('CONFIDENCE_THRESHOLD', 0.5)),
            'min_event_duration': float(os.environ.get('MIN_EVENT_DURATION', 0.2)),
            'min_gap_duration': float(os.environ.get('MIN_GAP_DURATION', 0.1)),
            'smoothing_window': int(os.environ.get('SMOOTHING_WINDOW', 5)),
            'target_class': os.environ.get('TARGET_CLASS', 'anemonefish')  # Class to detect events for
        },
        'model': {
            'version': os.environ.get('MODEL_VERSION', 'v1.0'),
            'classes': classes
        },
        'aws': {
            's3_model_bucket': os.environ.get('S3_MODEL_BUCKET'),
            's3_model_key': os.environ.get('S3_MODEL_KEY'),
            's3_input_bucket': os.environ.get('S3_INPUT_BUCKET'),
            's3_output_bucket': os.environ.get('S3_OUTPUT_BUCKET')
        }
    }


def load_model_from_s3(bucket: str, key: str) -> Optional[str]:
    """
    Download model from S3 to local temp file.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    key : str
        S3 object key
        
    Returns
    -------
    local_path : str or None
        Path to downloaded model file
    """
    if boto3 is None:
        print("boto3 not available, cannot load from S3")
        return None
    
    try:
        s3 = boto3.client('s3')
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.keras')
        tmp_path = tmp_file.name
        tmp_file.close()
        
        s3.download_file(bucket, key, tmp_path)
        print(f"Model downloaded from s3://{bucket}/{key} to {tmp_path}")
        return tmp_path
    except ClientError as e:
        print(f"Error downloading model from S3: {e}")
        return None


def initialize_model():
    """
    Initialize model and preprocessor (called once per Lambda container).
    """
    global _model, _preprocessor, _config
    
    if _model is not None:
        return  # Already initialized
    
    _config = load_config()
    
    # Initialize preprocessor (NO ImageNet normalization - matches training)
    spec_config = _config['spectrogram']
    _preprocessor = AudioPreprocessor(
        fmax=spec_config['fmax'],
        n_fft=spec_config['n_fft'],
        hop_length=spec_config['hop_length'],
        width_pixels=spec_config['width_pixels'],
        height_pixels=spec_config['height_pixels'],
        target_sr=spec_config.get('target_sr')
    )
    
    # Load model
    model_path = None
    
    # Try S3 first
    aws_config = _config['aws']
    if aws_config.get('s3_model_bucket') and aws_config.get('s3_model_key'):
        model_path = load_model_from_s3(
            aws_config['s3_model_bucket'],
            aws_config['s3_model_key']
        )
    
    # Fallback to local path (for testing)
    if not model_path:
        model_path = os.environ.get('MODEL_LOCAL_PATH')
    
    if model_path and os.path.exists(model_path):
        # Initialize model with classes from config
        classes = _config['model'].get('classes', ['noise', 'anemonefish', 'biological'])
        _model = ModelInference(model_path=model_path, classes=classes)
        if _model.model is None:
            print("Failed to load model")
            _model = None
    else:
        print(f"Model path not found: {model_path}")
        _model = None


def decode_multipart_body(body: str, content_type: str) -> Dict[str, Any]:
    """
    Decode multipart form data from request body.
    
    Parameters
    ----------
    body : str
        Base64 encoded request body
    content_type : str
        Content-Type header value
        
    Returns
    -------
    form_data : Dict[str, Any]
        Dictionary with form fields and files
    """
    # For Lambda with API Gateway binary media, body might already be base64 decoded
    # This is a simplified parser - may need enhancement for production
    
    if content_type.startswith('multipart/form-data'):
        # Extract boundary
        boundary = content_type.split('boundary=')[1]
        
        # Decode body if base64
        try:
            body_bytes = base64.b64decode(body)
        except:
            body_bytes = body.encode() if isinstance(body, str) else body
        
        # Simple multipart parsing
        # Note: This is simplified - production should use a proper library
        parts = body_bytes.split(b'--' + boundary.encode())
        
        form_data = {}
        audio_data = None
        
        for part in parts:
            if b'Content-Disposition' not in part:
                continue
            
            # Extract field name
            if b'name="audio_file"' in part or b'name=\'audio_file\'' in part:
                # Extract audio data (everything after headers)
                header_end = part.find(b'\r\n\r\n')
                if header_end != -1:
                    audio_data = part[header_end + 4:].rstrip(b'\r\n')
            
            # Extract config JSON if present
            if b'name="config"' in part or b'name=\'config\'' in part:
                header_end = part.find(b'\r\n\r\n')
                if header_end != -1:
                    config_json = part[header_end + 4:].rstrip(b'\r\n').decode('utf-8')
                    form_data['config'] = json.loads(config_json)
        
        if audio_data:
            form_data['audio_data'] = audio_data
        
        return form_data
    
    return {}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler function for inference API.
    
    Parameters
    ----------
    event : Dict[str, Any]
        Lambda event containing API Gateway request
    context : Any
        Lambda context object
        
    Returns
    -------
    response : Dict[str, Any]
        API Gateway response with predictions
    """
    start_time = time.time()
    
    try:
        # Initialize model if not already done
        initialize_model()
        
        if _model is None or _model.model is None:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Model not loaded',
                    'message': 'Failed to initialize model'
                })
            }
        
        # Parse request - handle both API Gateway and direct Lambda invocation
        request_body = event
        
        # If this is an API Gateway request, parse the body
        if 'body' in event:
            if event.get('isBase64Encoded'):
                body = event.get('body', '')
            else:
                body = event.get('body', '')
            
            content_type = event.get('headers', {}).get('Content-Type', '') or \
                          event.get('headers', {}).get('content-type', '')
            
            # Parse request body
            try:
                request_body = json.loads(body) if isinstance(body, str) and body else event
            except:
                request_body = event
        
        audio_data = None
        tmp_audio_path = None
        
        # Method 1: S3 reference (for large files)
        if 's3_bucket' in request_body and 's3_key' in request_body:
            s3_bucket = request_body['s3_bucket']
            s3_key = request_body['s3_key']
            
            if boto3 is None:
                return {
                    'statusCode': 500,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'S3 not available', 'message': 'boto3 not installed'})
                }
            
            try:
                s3_client = boto3.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT_URL'))
                tmp_audio_path = f"/tmp/{os.path.basename(s3_key)}"
                s3_client.download_file(s3_bucket, s3_key, tmp_audio_path)
                
                with open(tmp_audio_path, 'rb') as f:
                    audio_data = f.read()
            except ClientError as e:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'S3 download failed', 'message': str(e)})
                }
        
        # Method 2: Handle multipart form data
        if not audio_data:
            form_data = decode_multipart_body(body, content_type)
            audio_data = form_data.get('audio_data')
        
        # Method 3: Handle direct base64 audio
        if not audio_data:
            if 'audio_base64' in request_body:
                audio_data = base64.b64decode(request_body['audio_base64'])
            elif 'audio_file' in request_body:
                audio_data = base64.b64decode(request_body['audio_file'])
        
        if not audio_data:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing audio data',
                    'message': 'No audio file found in request. Provide s3_bucket/s3_key or audio_file (base64)'
                })
            }
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            tmp_audio.write(audio_data)
            tmp_audio_path = tmp_audio.name
        
        try:
            # Load audio
            audio_data_array, sample_rate, duration = _preprocessor.load_audio(tmp_audio_path)
            
            if audio_data_array is None:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': 'Invalid audio file',
                        'message': 'Failed to load audio data'
                    })
                }
            
            # Generate sliding windows
            window_config = _config['window']
            windows = _preprocessor.segment_audio(
                audio_data_array,
                sample_rate,
                window_duration=window_config['window_duration'],
                stride_duration=window_config['stride_duration']
            )
            
            if not windows:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': 'Audio too short',
                        'message': 'No valid windows generated from audio'
                    })
                }
            
            # Create spectrograms
            spectrograms, timestamps = _preprocessor.process_audio_to_spectrograms(
                audio_data_array,
                sample_rate,
                windows,
                return_images=True
            )
            
            if len(spectrograms) == 0:
                return {
                    'statusCode': 500,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': 'Processing failed',
                        'message': 'No spectrograms created'
                    })
                }
            
            # Run predictions
            pred_config = _config['prediction']
            probabilities = _model.predict_batch(
                spectrograms,
                batch_size=pred_config['batch_size'],
                verbose=0
            )
            
            # Apply smoothing to probabilities (shape: N, num_classes)
            smoothed_probs = smooth_predictions(
                probabilities,
                pred_config['smoothing_window']
            )
            
            # Get predictions with timestamps (returns class predictions)
            predictions = _model.predict_with_timestamps(
                spectrograms,
                timestamps,
                batch_size=pred_config['batch_size'],
                return_all_probabilities=False
            )
            
            # Extract predicted class names for event detection
            predicted_classes = [pred['class'] for pred in predictions]
            
            # Create class index map from model classes
            class_index_map = {cls: idx for idx, cls in enumerate(_model.classes)}
            
            # Detect events for target class (e.g., anemonefish)
            events = detect_events(
                smoothed_probs,
                timestamps,
                predicted_classes=predicted_classes,
                target_class=pred_config.get('target_class', 'anemonefish'),
                class_index_map=class_index_map,
                confidence_threshold=pred_config.get('confidence_threshold', 0.5),
                min_event_duration=pred_config['min_event_duration'],
                min_gap_duration=pred_config['min_gap_duration']
            )
            
            # Format response
            processing_time = time.time() - start_time
            response_data = format_response(
                predictions,
                events,
                duration,
                processing_time,
                _config['model']['version']
            )
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response_data)
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_audio_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
