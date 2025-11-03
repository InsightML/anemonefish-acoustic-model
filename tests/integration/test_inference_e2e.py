"""
End-to-end integration tests for inference API.

Tests the complete pipeline from audio upload to prediction results.
"""

import json
import os
import base64
import pytest
import requests
import boto3
from pathlib import Path
import time


# Test configuration
LAMBDA_ENDPOINT = os.getenv("LAMBDA_ENDPOINT", "http://localhost:9000/2015-03-31/functions/function/invocations")
AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
S3_INPUT_BUCKET = os.getenv("S3_INPUT_BUCKET", "anemonefish-inference-input")
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET", "anemonefish-inference-output")


class TestInferenceE2E:
    """End-to-end tests for inference pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup S3 client for tests."""
        self.s3_client = boto3.client(
            's3',
            endpoint_url=AWS_ENDPOINT,
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1'
        )
        
    def test_lambda_health_check(self):
        """Test that Lambda endpoint is reachable."""
        # Simple health check - Lambda should respond even if handler fails
        try:
            response = requests.get(LAMBDA_ENDPOINT.replace('/2015-03-31/functions/function/invocations', ''))
            # Any response means Lambda is up
            assert response.status_code in [200, 404, 405]
        except requests.ConnectionError:
            pytest.fail("Lambda endpoint not reachable")
    
    def test_s3_buckets_exist(self):
        """Test that required S3 buckets exist."""
        buckets = self.s3_client.list_buckets()['Buckets']
        bucket_names = [b['Name'] for b in buckets]
        
        assert S3_INPUT_BUCKET in bucket_names, f"Input bucket {S3_INPUT_BUCKET} not found"
        assert S3_OUTPUT_BUCKET in bucket_names, f"Output bucket {S3_OUTPUT_BUCKET} not found"
    
    @pytest.mark.timeout(60)
    def test_inference_with_short_audio(self):
        """Test inference with a short audio file."""
        # Create a simple test audio file (1 second of silence as base64)
        # This is a minimal valid WAV file
        test_audio_b64 = self._create_test_wav_base64(duration_seconds=1.0)
        
        event = {
            "httpMethod": "POST",
            "path": "/inference/predict",
            "headers": {
                "Content-Type": "multipart/form-data"
            },
            "body": json.dumps({
                "audio_file": test_audio_b64,
                "config": {
                    "confidence_threshold": 0.5
                }
            }),
            "isBase64Encoded": False
        }
        
        response = self._invoke_lambda(event)
        
        # Validate response structure
        assert response['statusCode'] == 200, f"Expected 200, got {response['statusCode']}"
        
        body = json.loads(response['body'])
        assert 'predictions' in body
        assert 'metadata' in body
        assert isinstance(body['predictions'], list)
        assert 'audio_duration' in body['metadata']
        assert 'processing_time' in body['metadata']
        assert 'model_version' in body['metadata']
    
    @pytest.mark.timeout(120)
    def test_inference_with_s3_upload(self):
        """Test inference workflow using S3 upload."""
        # Upload test audio to S3
        test_audio_data = self._create_test_wav_bytes(duration_seconds=2.0)
        s3_key = "test_audio_" + str(int(time.time())) + ".wav"
        
        self.s3_client.put_object(
            Bucket=S3_INPUT_BUCKET,
            Key=s3_key,
            Body=test_audio_data
        )
        
        # Invoke Lambda with S3 reference
        event = {
            "httpMethod": "POST",
            "path": "/inference/predict",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "s3_bucket": S3_INPUT_BUCKET,
                "s3_key": s3_key
            }),
            "isBase64Encoded": False
        }
        
        response = self._invoke_lambda(event)
        
        # Validate response
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'predictions' in body
        
        # Cleanup
        self.s3_client.delete_object(Bucket=S3_INPUT_BUCKET, Key=s3_key)
    
    def test_inference_error_handling_no_audio(self):
        """Test error handling when no audio is provided."""
        event = {
            "httpMethod": "POST",
            "path": "/inference/predict",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({}),
            "isBase64Encoded": False
        }
        
        response = self._invoke_lambda(event)
        
        # Should return error
        assert response['statusCode'] in [400, 422, 500]
        body = json.loads(response['body'])
        assert 'error' in body or 'message' in body
    
    def test_inference_with_config_override(self):
        """Test that configuration can be overridden via request."""
        test_audio_b64 = self._create_test_wav_base64(duration_seconds=1.0)
        
        event = {
            "httpMethod": "POST",
            "path": "/inference/predict",
            "headers": {
                "Content-Type": "multipart/form-data"
            },
            "body": json.dumps({
                "audio_file": test_audio_b64,
                "config": {
                    "confidence_threshold": 0.7,
                    "batch_size": 16
                }
            }),
            "isBase64Encoded": False
        }
        
        response = self._invoke_lambda(event)
        assert response['statusCode'] == 200
    
    # Helper methods
    
    def _invoke_lambda(self, event):
        """Invoke Lambda function with event."""
        response = requests.post(
            LAMBDA_ENDPOINT,
            json=event,
            timeout=120
        )
        return response.json()
    
    def _create_test_wav_base64(self, duration_seconds=1.0):
        """Create a test WAV file as base64."""
        wav_bytes = self._create_test_wav_bytes(duration_seconds)
        return base64.b64encode(wav_bytes).decode('utf-8')
    
    def _create_test_wav_bytes(self, duration_seconds=1.0):
        """
        Create a minimal valid WAV file with silence.
        
        WAV format:
        - RIFF header (12 bytes)
        - fmt chunk (24 bytes)
        - data chunk (8 bytes + audio data)
        """
        import struct
        import numpy as np
        
        sample_rate = 48000
        num_channels = 1
        bits_per_sample = 16
        
        # Generate silence
        num_samples = int(sample_rate * duration_seconds)
        audio_data = np.zeros(num_samples, dtype=np.int16)
        audio_bytes = audio_data.tobytes()
        
        # WAV file structure
        data_size = len(audio_bytes)
        file_size = 36 + data_size
        
        wav = b''
        # RIFF header
        wav += b'RIFF'
        wav += struct.pack('<I', file_size)
        wav += b'WAVE'
        
        # fmt chunk
        wav += b'fmt '
        wav += struct.pack('<I', 16)  # chunk size
        wav += struct.pack('<H', 1)   # audio format (PCM)
        wav += struct.pack('<H', num_channels)
        wav += struct.pack('<I', sample_rate)
        wav += struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8)  # byte rate
        wav += struct.pack('<H', num_channels * bits_per_sample // 8)  # block align
        wav += struct.pack('<H', bits_per_sample)
        
        # data chunk
        wav += b'data'
        wav += struct.pack('<I', data_size)
        wav += audio_bytes
        
        return wav


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

