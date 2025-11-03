#!/usr/bin/env python3
"""
Simple API Gateway proxy for local testing with frontend.

This simulates API Gateway behavior, converting HTTP requests from the frontend
into Lambda invocations and returning the results.
"""

import json
import os
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from botocore.exceptions import ClientError
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
LAMBDA_ENDPOINT = os.getenv('LAMBDA_ENDPOINT', 'http://inference-lambda:8080/2015-03-31/functions/function/invocations')
S3_ENDPOINT = os.getenv('AWS_ENDPOINT_URL', 'http://localstack:4566')
S3_INPUT_BUCKET = os.getenv('S3_INPUT_BUCKET', 'anemonefish-inference-input')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'anemonefish-api-gateway-proxy'})


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Main prediction endpoint that mimics API Gateway.
    
    Accepts:
    - multipart/form-data with 'audio' file field
    - Large files are uploaded to S3, then Lambda is invoked with S3 reference
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'error': 'Missing audio file',
                'message': 'No audio file provided in request'
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'error': 'Empty filename',
                'message': 'No file selected'
            }), 400
        
        # Secure the filename
        filename = secure_filename(audio_file.filename)
        
        # Get file size
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        file_size_mb = file_size / (1024 * 1024)
        print(f"Received file: {filename} ({file_size_mb:.2f} MB)")
        
        # Upload to S3
        s3_key = f"uploads/{int(time.time())}_{filename}"
        print(f"Uploading to S3: s3://{S3_INPUT_BUCKET}/{s3_key}")
        
        s3_client.upload_fileobj(audio_file, S3_INPUT_BUCKET, s3_key)
        print(f"Upload complete")
        
        # Create Lambda event
        lambda_event = {
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
        
        # Invoke Lambda
        print(f"Invoking Lambda for processing...")
        start_time = time.time()
        
        lambda_response = requests.post(
            LAMBDA_ENDPOINT,
            json=lambda_event,
            timeout=900  # 15 minutes
        )
        
        duration = time.time() - start_time
        print(f"Lambda completed in {duration:.2f}s")
        
        lambda_result = lambda_response.json()
        
        # Extract body from Lambda response
        if lambda_result.get('statusCode') == 200:
            response_body = json.loads(lambda_result.get('body', '{}'))
            print(f"Success! {len(response_body.get('predictions', []))} predictions returned")
            return jsonify(response_body), 200
        else:
            error_body = json.loads(lambda_result.get('body', '{}'))
            print(f"Error: {error_body}")
            return jsonify(error_body), lambda_result.get('statusCode', 500)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("API Gateway Proxy Server")
    print("=" * 50)
    print(f"Lambda endpoint: {LAMBDA_ENDPOINT}")
    print(f"S3 endpoint: {S3_ENDPOINT}")
    print(f"S3 bucket: {S3_INPUT_BUCKET}")
    print(f"Listening on: http://0.0.0.0:8000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8000, debug=True)
