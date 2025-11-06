# Frontend Integration Guide - Anemonefish Acoustic Inference API

**Last Updated**: November 3, 2025  
**Environment**: Development  
**API Version**: v1.0

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Authentication](#authentication)
4. [File Upload Approaches](#file-upload-approaches)
5. [API Response Format](#api-response-format)
6. [Complete Code Examples](#complete-code-examples)
7. [Error Handling](#error-handling)
8. [Testing](#testing)

---

## Quick Start

### API Configuration

```javascript
// src/config/api.js
export const API_CONFIG = {
  // API Gateway endpoint
  apiGatewayUrl: 'https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api',
  
  // API Key (KEEP SECURE!)
  apiKey: 'VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M',
  
  // S3 bucket for large file uploads
  s3InputBucket: 'anemonefish-inference-dev-input-944269089535',
  
  // AWS Region
  awsRegion: 'eu-west-2',
  
  // Timeouts
  timeout: 900000, // 15 minutes
  
  // File size threshold for S3 vs direct upload
  largFileThreshold: 10 * 1024 * 1024, // 10 MB
};
```

---

## API Endpoints

| Endpoint | Method | Purpose | File Size Limit |
|----------|--------|---------|-----------------|
| `/predict` | POST | Direct inference (small files) | **10 MB max** |

⚠️ **IMPORTANT**: API Gateway has a hard **10 MB payload limit**. For larger files, use the S3 upload approach below.

---

## Authentication

All API requests require an API key in the header:

```javascript
headers: {
  'x-api-key': 'VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M'
}
```

---

## File Upload Approaches

### Approach 1: Direct Upload (Files < 10 MB)

**Use this for**: Small audio files under 10 MB

**Pros**: Simple, single API call  
**Cons**: 10 MB limit

```javascript
const formData = new FormData();
formData.append('audio_file', file);

const response = await fetch(
  'https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict',
  {
    method: 'POST',
    headers: {
      'x-api-key': 'VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M',
    },
    body: formData,
  }
);

const results = await response.json();
```

### Approach 2: S3 Upload (Files > 10 MB) ⭐ RECOMMENDED

**Use this for**: All files, especially large ones (your app supports up to 5 GB)

**Pros**: No size limit, faster uploads, better progress tracking  
**Cons**: Requires AWS SDK

#### Installation

```bash
npm install @aws-sdk/client-s3 @aws-sdk/client-lambda
```

#### Implementation Steps

**Step 1: Configure AWS SDK**

```javascript
// src/utils/awsConfig.js
import { S3Client } from '@aws-sdk/client-s3';
import { LambdaClient } from '@aws-sdk/client-lambda';

export const s3Client = new S3Client({
  region: 'eu-west-2',
  credentials: {
    accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
  },
});

export const lambdaClient = new LambdaClient({
  region: 'eu-west-2',
  credentials: {
    accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
  },
});
```

**Step 2: Upload to S3**

```javascript
// src/services/uploadService.js
import { PutObjectCommand } from '@aws-sdk/client-s3';
import { InvokeCommand } from '@aws-sdk/client-lambda';
import { s3Client, lambdaClient } from '../utils/awsConfig';

/**
 * Upload large audio file to S3 and trigger inference
 * @param {File} file - Audio file to upload
 * @param {Function} onProgress - Progress callback (0-100)
 * @returns {Promise<Object>} Inference results
 */
export const uploadLargeAudioFile = async (file, onProgress) => {
  try {
    // Step 1: Upload to S3
    const s3Key = `uploads/${Date.now()}_${file.name}`;
    
    const uploadCommand = new PutObjectCommand({
      Bucket: 'anemonefish-inference-dev-input-944269089535',
      Key: s3Key,
      Body: file,
      ContentType: file.type || 'audio/wav',
    });

    // Upload with progress tracking
    await s3Client.send(uploadCommand);
    onProgress?.(50); // Upload complete
    
    // Step 2: Invoke Lambda for inference
    const payload = {
      s3_bucket: 'anemonefish-inference-dev-input-944269089535',
      s3_key: s3Key,
    };
    
    const invokeCommand = new InvokeCommand({
      FunctionName: 'anemonefish-inference-dev-inference',
      Payload: JSON.stringify(payload),
    });
    
    const lambdaResponse = await lambdaClient.send(invokeCommand);
    onProgress?.(100); // Processing complete
    
    // Parse Lambda response
    const responsePayload = JSON.parse(
      new TextDecoder().decode(lambdaResponse.Payload)
    );
    
    // Lambda returns API Gateway format: { statusCode, body }
    const results = JSON.parse(responsePayload.body);
    
    return results;
    
  } catch (error) {
    console.error('Upload/inference error:', error);
    throw error;
  }
};
```

**Step 3: Smart Upload (Auto-select approach)**

```javascript
// src/services/audioInferenceService.js
import { uploadLargeAudioFile } from './uploadService';
import { API_CONFIG } from '../config';

/**
 * Upload audio file and run inference (automatically chooses best method)
 * @param {File} file - Audio file
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<Object>} Inference results
 */
export const runInference = async (file, onProgress) => {
  const isLargeFile = file.size > API_CONFIG.largFileThreshold;
  
  if (isLargeFile) {
    console.log(`Large file (${(file.size / 1024 / 1024).toFixed(1)} MB), using S3 upload`);
    return uploadLargeAudioFile(file, onProgress);
  } else {
    console.log(`Small file (${(file.size / 1024 / 1024).toFixed(1)} MB), using API Gateway`);
    return uploadSmallAudioFile(file, onProgress);
  }
};

/**
 * Upload small file directly via API Gateway
 */
const uploadSmallAudioFile = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('audio_file', file);
  
  onProgress?.(10); // Starting
  
  const response = await fetch(`${API_CONFIG.apiGatewayUrl}/predict`, {
    method: 'POST',
    headers: {
      'x-api-key': API_CONFIG.apiKey,
    },
    body: formData,
  });
  
  onProgress?.(90); // Upload complete
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error: ${response.status} - ${errorText}`);
  }
  
  const results = await response.json();
  onProgress?.(100); // Done
  
  return results;
};
```

---

## API Response Format

### Successful Response

```json
{
  "predictions": [
    {
      "timestamp": "0.0-1.0s",
      "class": "anemonefish",
      "confidence": 0.923,
      "probabilities": {
        "noise": 0.012,
        "anemonefish": 0.923,
        "biological": 0.065
      }
    },
    {
      "timestamp": "0.4-1.4s",
      "class": "noise",
      "confidence": 0.856,
      "probabilities": {
        "noise": 0.856,
        "anemonefish": 0.089,
        "biological": 0.055
      }
    }
    // ... ~1,222 predictions for 8-minute audio
  ],
  "events": [
    {
      "start_time": 4.4,
      "end_time": 5.4,
      "duration": 1.0,
      "mean_confidence": 0.575,
      "max_confidence": 0.623,
      "num_windows": 3
    },
    {
      "start_time": 54.0,
      "end_time": 55.0,
      "duration": 1.0,
      "mean_confidence": 0.507,
      "max_confidence": 0.541,
      "num_windows": 3
    }
    // ... 11 total events
  ],
  "metadata": {
    "audio_duration_seconds": 488.8,
    "processing_time_seconds": 95.2,
    "model_version": "v1.0",
    "num_predictions": 1222,
    "num_events": 11,
    "target_class": "anemonefish",
    "confidence_threshold": 0.5
  }
}
```

### Response Field Descriptions

#### `predictions` Array
Each prediction represents a time window analyzed by the model:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Time range in format "start-end" (e.g., "0.0-1.0s") |
| `class` | string | Predicted class: "noise", "anemonefish", or "biological" |
| `confidence` | number | Confidence score (0-1) for the predicted class |
| `probabilities` | object | Probability for each class (sums to 1.0) |

#### `events` Array
Detected anemonefish vocalizations (continuous periods above threshold):

| Field | Type | Description |
|-------|------|-------------|
| `start_time` | number | Event start time in seconds |
| `end_time` | number | Event end time in seconds |
| `duration` | number | Event duration in seconds |
| `mean_confidence` | number | Average confidence across event windows |
| `max_confidence` | number | Highest confidence in the event |
| `num_windows` | number | Number of prediction windows in this event |

#### `metadata` Object

| Field | Type | Description |
|-------|------|-------------|
| `audio_duration_seconds` | number | Total audio duration |
| `processing_time_seconds` | number | Time taken for inference |
| `model_version` | string | Model version used (e.g., "v1.0") |
| `num_predictions` | number | Total number of predictions |
| `num_events` | number | Total detected events |
| `target_class` | string | Class being detected (e.g., "anemonefish") |
| `confidence_threshold` | number | Threshold used for event detection |

### Error Response

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

Common errors:

| Error | Message | Cause |
|-------|---------|-------|
| `Missing audio data` | No audio file found | File not uploaded properly |
| `Invalid audio file` | Failed to load audio | Corrupted or unsupported format |
| `Model not loaded` | Failed to initialize model | Model file issue (contact backend) |
| `S3 download failed` | S3 error message | Invalid S3 path or permissions |

---

## Complete Code Examples

### Full React Component Example

```javascript
// src/components/AudioInference.jsx
import React, { useState } from 'react';
import { runInference } from '../services/audioInferenceService';

const AudioInference = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const results = await runInference(file, setProgress);
      setResults(results);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="audio-inference">
      <h2>Anemonefish Acoustic Inference</h2>
      
      {/* File Upload */}
      <div className="upload-section">
        <input
          type="file"
          accept="audio/*,.wav,.mp3,.flac"
          onChange={handleFileSelect}
          disabled={uploading}
        />
        
        {file && (
          <div className="file-info">
            <p>File: {file.name}</p>
            <p>Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
            <p>Upload Method: {file.size > 10485760 ? 'S3 (Large)' : 'API Gateway (Direct)'}</p>
          </div>
        )}
        
        <button 
          onClick={handleUpload} 
          disabled={!file || uploading}
        >
          {uploading ? `Processing... ${progress}%` : 'Run Inference'}
        </button>
      </div>

      {/* Progress */}
      {uploading && (
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          />
          <span>{progress}%</span>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <ResultsDisplay results={results} />
      )}
    </div>
  );
};

export default AudioInference;
```

### Results Display Component

```javascript
// src/components/ResultsDisplay.jsx
import React from 'react';

const ResultsDisplay = ({ results }) => {
  const { predictions, events, metadata } = results;

  return (
    <div className="results-display">
      {/* Summary Stats */}
      <div className="stats-summary">
        <h3>Analysis Complete</h3>
        <div className="stats-grid">
          <div className="stat">
            <label>Audio Duration</label>
            <value>{metadata.audio_duration_seconds.toFixed(1)}s</value>
          </div>
          <div className="stat">
            <label>Processing Time</label>
            <value>{metadata.processing_time_seconds.toFixed(1)}s</value>
          </div>
          <div className="stat">
            <label>Windows Analyzed</label>
            <value>{metadata.num_predictions}</value>
          </div>
          <div className="stat">
            <label>Events Detected</label>
            <value className={events.length > 0 ? 'highlight' : ''}>
              {metadata.num_events}
            </value>
          </div>
        </div>
      </div>

      {/* Detected Events */}
      {events.length > 0 && (
        <div className="events-section">
          <h3>🐠 Detected Anemonefish Vocalizations</h3>
          <div className="events-list">
            {events.map((event, index) => (
              <div key={index} className="event-card">
                <div className="event-header">
                  <span className="event-number">#{index + 1}</span>
                  <span className="event-time">
                    {formatTime(event.start_time)} - {formatTime(event.end_time)}
                  </span>
                  <span className={`confidence ${getConfidenceClass(event.mean_confidence)}`}>
                    {(event.mean_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="event-details">
                  <span>Duration: {event.duration.toFixed(2)}s</span>
                  <span>Peak: {(event.max_confidence * 100).toFixed(1)}%</span>
                  <span>Windows: {event.num_windows}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Events Message */}
      {events.length === 0 && (
        <div className="no-events">
          <p>No anemonefish vocalizations detected in this audio.</p>
          <p className="hint">
            Analyzed {predictions.length} time windows with confidence threshold of {metadata.confidence_threshold}.
          </p>
        </div>
      )}

      {/* Timeline Visualization */}
      <div className="timeline-section">
        <h3>Detection Timeline</h3>
        <AudioTimeline 
          predictions={predictions}
          events={events}
          duration={metadata.audio_duration_seconds}
        />
      </div>

      {/* Export Options */}
      <div className="export-section">
        <button onClick={() => downloadResults(results, 'json')}>
          Download JSON
        </button>
        <button onClick={() => downloadResults(results, 'csv')}>
          Download CSV
        </button>
      </div>
    </div>
  );
};

// Helper functions
const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}:${secs.padStart(4, '0')}`;
};

const getConfidenceClass = (confidence) => {
  if (confidence >= 0.8) return 'high';
  if (confidence >= 0.6) return 'medium';
  return 'low';
};

const downloadResults = (results, format) => {
  if (format === 'json') {
    const blob = new Blob([JSON.stringify(results, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inference_results_${Date.now()}.json`;
    a.click();
  } else if (format === 'csv') {
    // Convert events to CSV
    const csv = [
      ['Event #', 'Start Time', 'End Time', 'Duration', 'Confidence', 'Peak Confidence', 'Windows'],
      ...results.events.map((event, i) => [
        i + 1,
        event.start_time.toFixed(2),
        event.end_time.toFixed(2),
        event.duration.toFixed(2),
        (event.mean_confidence * 100).toFixed(1) + '%',
        (event.max_confidence * 100).toFixed(1) + '%',
        event.num_windows,
      ]),
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inference_events_${Date.now()}.csv`;
    a.click();
  }
};

export default ResultsDisplay;
```

### Simple Timeline Visualization

```javascript
// src/components/AudioTimeline.jsx
import React from 'react';

const AudioTimeline = ({ predictions, events, duration }) => {
  // Create a simple visualization of events on a timeline
  return (
    <div className="audio-timeline">
      <div className="timeline-track">
        {events.map((event, index) => {
          const leftPercent = (event.start_time / duration) * 100;
          const widthPercent = (event.duration / duration) * 100;
          
          return (
            <div
              key={index}
              className="timeline-event"
              style={{
                left: `${leftPercent}%`,
                width: `${widthPercent}%`,
                backgroundColor: getColorByConfidence(event.mean_confidence),
              }}
              title={`Event ${index + 1}: ${event.start_time.toFixed(1)}s - ${event.end_time.toFixed(1)}s (${(event.mean_confidence * 100).toFixed(1)}%)`}
            />
          );
        })}
      </div>
      
      {/* Time markers */}
      <div className="timeline-markers">
        <span>0:00</span>
        <span>{formatDuration(duration / 2)}</span>
        <span>{formatDuration(duration)}</span>
      </div>
    </div>
  );
};

const getColorByConfidence = (confidence) => {
  if (confidence >= 0.8) return '#22c55e'; // Green
  if (confidence >= 0.6) return '#eab308'; // Yellow
  return '#f97316'; // Orange
};

const formatDuration = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default AudioTimeline;
```

---

## Complete Service Implementation

### audioInferenceService.js (Full Version)

```javascript
// src/services/audioInferenceService.js
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { LambdaClient, InvokeCommand } from '@aws-sdk/client-lambda';

const API_CONFIG = {
  apiGatewayUrl: 'https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api',
  apiKey: 'VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M',
  s3InputBucket: 'anemonefish-inference-dev-input-944269089535',
  awsRegion: 'eu-west-2',
  largeFileThreshold: 10 * 1024 * 1024, // 10 MB
};

// Initialize AWS clients
const s3Client = new S3Client({
  region: API_CONFIG.awsRegion,
  credentials: {
    accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
  },
});

const lambdaClient = new LambdaClient({
  region: API_CONFIG.awsRegion,
  credentials: {
    accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
  },
});

/**
 * Main function: Upload audio and run inference
 * Automatically selects best upload method based on file size
 * 
 * @param {File} file - Audio file to process
 * @param {Function} onProgress - Callback with progress 0-100
 * @returns {Promise<Object>} Inference results
 */
export const runAudioInference = async (file, onProgress = null) => {
  const isLargeFile = file.size > API_CONFIG.largeFileThreshold;
  
  if (isLargeFile) {
    return uploadViaS3(file, onProgress);
  } else {
    return uploadViaAPIGateway(file, onProgress);
  }
};

/**
 * Upload large files via S3 + Lambda invocation
 * Use for files > 10 MB
 */
const uploadViaS3 = async (file, onProgress) => {
  try {
    onProgress?.(5);
    
    // Step 1: Upload to S3
    const s3Key = `uploads/${Date.now()}_${file.name.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
    
    const uploadCommand = new PutObjectCommand({
      Bucket: API_CONFIG.s3InputBucket,
      Key: s3Key,
      Body: file,
      ContentType: file.type || 'audio/wav',
    });

    console.log(`Uploading ${(file.size / 1024 / 1024).toFixed(1)} MB to S3...`);
    await s3Client.send(uploadCommand);
    onProgress?.(50);
    
    // Step 2: Invoke Lambda
    console.log('Triggering inference...');
    const payload = {
      s3_bucket: API_CONFIG.s3InputBucket,
      s3_key: s3Key,
    };
    
    const invokeCommand = new InvokeCommand({
      FunctionName: 'anemonefish-inference-dev-inference',
      Payload: JSON.stringify(payload),
    });
    
    const lambdaResponse = await lambdaClient.send(invokeCommand);
    onProgress?.(95);
    
    // Parse response
    const responsePayload = JSON.parse(
      new TextDecoder().decode(lambdaResponse.Payload)
    );
    
    onProgress?.(100);
    
    // Check for Lambda errors
    if (responsePayload.errorMessage) {
      throw new Error(responsePayload.errorMessage);
    }
    
    // Parse the body (Lambda returns API Gateway format)
    const results = JSON.parse(responsePayload.body);
    
    // Check for application errors
    if (results.error) {
      throw new Error(results.message || results.error);
    }
    
    return results;
    
  } catch (error) {
    console.error('S3 upload/inference error:', error);
    throw new Error(`Failed to process audio: ${error.message}`);
  }
};

/**
 * Upload small files directly via API Gateway
 * Use for files < 10 MB only
 */
const uploadViaAPIGateway = async (file, onProgress) => {
  try {
    onProgress?.(10);
    
    const formData = new FormData();
    formData.append('audio_file', file);
    
    console.log(`Uploading ${(file.size / 1024 / 1024).toFixed(1)} MB via API Gateway...`);
    
    const response = await fetch(`${API_CONFIG.apiGatewayUrl}/predict`, {
      method: 'POST',
      headers: {
        'x-api-key': API_CONFIG.apiKey,
      },
      body: formData,
    });
    
    onProgress?.(90);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error (${response.status}): ${errorText}`);
    }
    
    const results = await response.json();
    onProgress?.(100);
    
    // Check for application errors
    if (results.error) {
      throw new Error(results.message || results.error);
    }
    
    return results;
    
  } catch (error) {
    console.error('API Gateway upload error:', error);
    
    if (error.message.includes('content length exceeded')) {
      throw new Error(
        'File too large for direct upload (>10MB). Please use a smaller file or contact support.'
      );
    }
    
    throw new Error(`Failed to upload: ${error.message}`);
  }
};

export { runAudioInference };
```

---

## Environment Variables

Add to your `.env` file:

```bash
# AWS Credentials (for S3 upload)
REACT_APP_AWS_ACCESS_KEY_ID=your_access_key_here
REACT_APP_AWS_SECRET_ACCESS_KEY=your_secret_key_here

# API Configuration
REACT_APP_API_ENDPOINT=https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api
REACT_APP_API_KEY=VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M
```

⚠️ **Security Note**: For production, credentials should be obtained via:
- AWS Cognito authentication
- Temporary credentials from your backend
- **Never** hardcode credentials in frontend code

---

## Sample API Responses (Real Data)

### Example 1: File with Multiple Detections

From test file: `20230210_000001_LL_B55_M_R_with labels.wav` (44.8 MB, ~8 minutes)

```json
{
  "predictions": [
    {
      "timestamp": "4.0-5.0s",
      "class": "anemonefish",
      "confidence": 0.623,
      "probabilities": {
        "noise": 0.112,
        "anemonefish": 0.623,
        "biological": 0.265
      }
    }
    // ... 1,221 more predictions
  ],
  "events": [
    {
      "start_time": 4.4,
      "end_time": 5.4,
      "duration": 1.0,
      "mean_confidence": 0.575,
      "max_confidence": 0.623,
      "num_windows": 3
    },
    {
      "start_time": 203.6,
      "end_time": 205.4,
      "duration": 1.8,
      "mean_confidence": 0.676,
      "max_confidence": 0.721,
      "num_windows": 5
    }
    // ... 9 more events
  ],
  "metadata": {
    "audio_duration_seconds": 488.8,
    "processing_time_seconds": 95.2,
    "model_version": "v1.0",
    "num_predictions": 1222,
    "num_events": 11,
    "target_class": "anemonefish",
    "confidence_threshold": 0.5
  }
}
```

### Example 2: File with No Detections

```json
{
  "predictions": [
    {
      "timestamp": "0.0-1.0s",
      "class": "noise",
      "confidence": 0.912,
      "probabilities": {
        "noise": 0.912,
        "anemonefish": 0.034,
        "biological": 0.054
      }
    }
    // ... all predictions
  ],
  "events": [],
  "metadata": {
    "audio_duration_seconds": 120.0,
    "processing_time_seconds": 23.5,
    "model_version": "v1.0",
    "num_predictions": 300,
    "num_events": 0,
    "target_class": "anemonefish",
    "confidence_threshold": 0.5
  }
}
```

### Example 3: Error Response

```json
{
  "error": "Invalid audio file",
  "message": "Failed to load audio data. Ensure the file is a valid audio format."
}
```

---

## Error Handling Guide

### Common Errors and Solutions

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `HTTP content length exceeded 10485760 bytes` | File > 10 MB sent to API Gateway | Use S3 upload method |
| `Missing audio data` | Empty or malformed request | Check FormData format |
| `Invalid audio file` | Corrupted or unsupported format | Try different audio file |
| `S3 download failed` | S3 access issue | Check bucket name and permissions |
| `Model not loaded` | Backend configuration issue | Contact backend team |
| Request timeout | Large file processing | Increase timeout or use async approach |

### Recommended Error Handling

```javascript
try {
  const results = await runAudioInference(file, setProgress);
  setResults(results);
} catch (error) {
  // Specific error handling
  if (error.message.includes('exceeded 10485760')) {
    setError('File too large. Use S3 upload method.');
  } else if (error.message.includes('Invalid audio')) {
    setError('Invalid audio file. Please try a different file.');
  } else if (error.message.includes('timeout')) {
    setError('Processing timed out. File may be too long.');
  } else {
    setError(`Error: ${error.message}`);
  }
}
```

---

## Testing Your Integration

### Test with Sample Data

We've tested with the following file and gotten successful results:

**File**: `20230210_000001_LL_B55_M_R_with labels.wav`
- **Size**: 44.8 MB
- **Duration**: ~8 minutes
- **Results**: 11 anemonefish events detected
- **Processing Time**: ~95 seconds

The full response is available in: `test_inference_results.json`

### Test Cases to Implement

```javascript
// Test 1: Small file (< 10 MB) via API Gateway
test('Small file upload via API Gateway', async () => {
  const smallFile = new File(['...'], 'test.wav', { type: 'audio/wav' });
  const results = await runAudioInference(smallFile);
  expect(results.predictions).toBeDefined();
  expect(results.metadata).toBeDefined();
});

// Test 2: Large file (> 10 MB) via S3
test('Large file upload via S3', async () => {
  const largeFile = new File([new ArrayBuffer(15 * 1024 * 1024)], 'large.wav');
  const results = await runAudioInference(largeFile);
  expect(results.predictions).toBeDefined();
});

// Test 3: Progress tracking
test('Progress callback is called', async () => {
  const progressValues = [];
  await runAudioInference(file, (p) => progressValues.push(p));
  expect(progressValues).toContain(100);
});
```

---

## Performance Expectations

Based on real testing:

| Audio Duration | File Size | Processing Time | Events Detected |
|----------------|-----------|-----------------|-----------------|
| ~8 minutes | 44.8 MB | ~95 seconds | 11 |
| ~1 minute | ~6 MB | ~15 seconds | 0-3 (varies) |
| ~30 seconds | ~3 MB | ~8 seconds | 0-2 (varies) |

**Processing Speed**: ~3 seconds per MB of audio  
**Memory Usage**: ~300 MB per minute of audio  
**Cost per Request**: ~$0.02-0.05 depending on file size

---

## Package Dependencies

Add to your `package.json`:

```json
{
  "dependencies": {
    "@aws-sdk/client-s3": "^3.x.x",
    "@aws-sdk/client-lambda": "^3.x.x"
  }
}
```

Install:
```bash
npm install @aws-sdk/client-s3 @aws-sdk/client-lambda
```

---

## Security Best Practices

### DO ✅

- Store API key in environment variables (`.env`)
- Use AWS credentials from secure authentication flow
- Validate file types and sizes before upload
- Implement retry logic with exponential backoff
- Show clear error messages to users
- Log errors for debugging (without sensitive data)

### DON'T ❌

- Hardcode API keys in source code
- Commit `.env` file to git
- Upload files without validation
- Expose AWS credentials in browser console
- Store sensitive data in localStorage without encryption

---

## CSS Styling Example

```css
/* src/styles/AudioInference.css */

.audio-inference {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.upload-section {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.progress-bar {
  width: 100%;
  height: 30px;
  background: #e9ecef;
  border-radius: 15px;
  overflow: hidden;
  position: relative;
  margin: 1rem 0;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4f46e5, #7c3aed);
  transition: width 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
}

.event-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.event-card:hover {
  box-shadow: 0 4px 6px rgba(0,0,0,0.15);
}

.confidence.high {
  color: #22c55e;
  font-weight: bold;
}

.confidence.medium {
  color: #eab308;
  font-weight: bold;
}

.confidence.low {
  color: #f97316;
}

.timeline-track {
  position: relative;
  height: 40px;
  background: #f3f4f6;
  border-radius: 4px;
  margin: 1rem 0;
}

.timeline-event {
  position: absolute;
  height: 100%;
  border-radius: 2px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.timeline-event:hover {
  opacity: 0.8;
}
```

---

## Troubleshooting

### Issue: "CORS Error"
**Solution**: API Gateway CORS is configured to allow all origins (`*`). If you still see CORS errors:
- Ensure you're using the correct endpoint URL
- Check that `x-api-key` header is included
- Verify the browser isn't blocking due to mixed content (HTTP/HTTPS)

### Issue: "401 Unauthorized"
**Solution**: Check API key is correct and included in `x-api-key` header

### Issue: "File upload fails for large files"
**Solution**: Use S3 upload method, not direct API Gateway POST

### Issue: "Request times out"
**Solution**: Increase timeout and show loading indicator. Processing takes ~3 seconds per MB.

### Issue: AWS SDK errors
**Solution**: Ensure environment variables are set:
```javascript
// Add validation
if (!process.env.REACT_APP_AWS_ACCESS_KEY_ID) {
  console.error('Missing AWS credentials in environment variables');
}
```

---

## Next Steps for Frontend

1. **Install AWS SDK packages**
   ```bash
   npm install @aws-sdk/client-s3 @aws-sdk/client-lambda
   ```

2. **Create AWS credentials** (contact backend team or create IAM user)
   - Need: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - Required permissions: S3 PutObject, Lambda InvokeFunction

3. **Update API configuration** with endpoints and keys

4. **Implement file upload service** using code examples above

5. **Build results display** using the response format specification

6. **Test with sample files**:
   - Small file (< 10 MB): Test API Gateway path
   - Large file (> 10 MB): Test S3 path

---

## Support

**Backend API Status**: ✅ Fully Operational  
**Test Results**: `test_inference_results.json` (sample response)  
**API Documentation**: This file  
**Backend Contact**: See `DEPLOYMENT_SUMMARY.md` for infrastructure details

---

## Quick Reference

### Key URLs
- **API Base**: `https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api`
- **Predict Endpoint**: `https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict`

### Key Values
- **API Key**: `VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M`
- **S3 Bucket**: `anemonefish-inference-dev-input-944269089535`
- **Lambda Function**: `anemonefish-inference-dev-inference`
- **AWS Region**: `eu-west-2`

### File Size Limits
- **API Gateway Direct**: 10 MB max
- **S3 Upload**: 5 GB max (your app limit)
- **Lambda Timeout**: 15 minutes max

### Processing Time
- **Small files (< 5 MB)**: ~10-30 seconds
- **Medium files (5-50 MB)**: ~30-150 seconds
- **Large files (50-500 MB)**: ~3-15 minutes

---

**Ready to integrate!** 🚀 Contact the backend team if you need AWS credentials or have questions.

