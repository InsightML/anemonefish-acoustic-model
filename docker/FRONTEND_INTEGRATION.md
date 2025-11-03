# Frontend Integration Guide - Anemonefish Audio Inference API

## Overview

This document describes the API contract for the Anemonefish Audio Recognition inference endpoint. The frontend should upload audio files and receive predictions for anemonefish, biological sounds, and noise.

## API Endpoint

### Production (AWS)
```
POST https://api.your-domain.com/predict
```

### Local Testing
```
POST http://localhost:8000/predict
```

## Request Format

### Method: POST
**Content-Type**: `multipart/form-data`

### Form Field
- **Field name**: `audio` (not `audio_file`)
- **File types**: .wav, .mp3, .flac, .m4a, .ogg, .webm
- **Max size**: 5GB (24-hour audio files)

### Example Request (JavaScript/React)

```javascript
// Current implementation - KEEP THIS, just change field name
const uploadAudioFile = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('audio', file);  // Changed from 'audio_file' to 'audio'
  
  const response = await fetch(API_CONFIG.endpoint, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `API request failed: ${response.status} ${response.statusText}. ${errorText}`
    );
  }
  
  const data = await response.json();
  return data;
};
```

## Response Format

### Success Response (200 OK)

```json
{
  "predictions": [
    {
      "timestamp": "0.0-1.0s",
      "start_time": 0.0,
      "end_time": 1.0,
      "class": "anemonefish",
      "confidence": 0.92
    },
    {
      "timestamp": "0.4-1.4s",
      "start_time": 0.4,
      "end_time": 1.4,
      "class": "biological",
      "confidence": 0.75
    }
  ],
  "events": [
    {
      "start_time": 45.2,
      "end_time": 47.8,
      "class": "anemonefish",
      "duration": 2.6,
      "max_confidence": 0.94,
      "mean_confidence": 0.87
    }
  ],
  "metadata": {
    "audio_duration": 300.18,
    "processing_time": 136.83,
    "model_version": "v1.0",
    "total_windows": 748,
    "total_events": 15
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | Array | All predictions (1-second windows, 0.4s stride) |
| `predictions[].timestamp` | String | Human-readable time range |
| `predictions[].start_time` | Number | Start time in seconds |
| `predictions[].end_time` | Number | End time in seconds |
| `predictions[].class` | String | Class name: "anemonefish", "biological", or "noise" |
| `predictions[].confidence` | Number | Confidence score (0.0 to 1.0) |
| `events` | Array | Detected continuous events (filtered by confidence threshold) |
| `events[].start_time` | Number | Event start time (seconds) |
| `events[].end_time` | Number | Event end time (seconds) |
| `events[].class` | String | Detected class |
| `events[].duration` | Number | Event duration (seconds) |
| `events[].max_confidence` | Number | Highest confidence in event |
| `events[].mean_confidence` | Number | Average confidence across event |
| `metadata.audio_duration` | Number | Total audio duration (seconds) |
| `metadata.processing_time` | Number | Server-side processing time (seconds) |
| `metadata.model_version` | String | Model version used |
| `metadata.total_windows` | Number | Number of analysis windows |
| `metadata.total_events` | Number | Number of detected events |

### Error Response (4xx/5xx)

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

## Common Error Codes

| Code | Error | Cause |
|------|-------|-------|
| 400 | Missing audio file | No file provided or empty filename |
| 413 | Payload too large | File exceeds 5GB |
| 500 | Processing failed | Model loading or inference error |
| 504 | Gateway timeout | Processing took too long (>15 minutes) |

## Frontend Changes Required

### 1. Update API Config

**File**: `src/config.js` or similar

**Change**:
```javascript
// BEFORE (development)
export const API_CONFIG = {
  endpoint: 'https://your-aws-api-endpoint.com/predict',
  timeout: 600000, // 10 minutes
};

// AFTER
export const API_CONFIG = {
  // For local development
  endpoint: process.env.REACT_APP_API_ENDPOINT || 'http://localhost:8000/predict',
  // For production - will be set via environment variable
  // endpoint: 'https://api.anemonefish.your-domain.com/predict',
  timeout: 900000, // 15 minutes for large files (24-hour audio)
};
```

### 2. Update Upload Service

**File**: `src/services/apiService.js`

**Change**: Update the FormData field name from `'audio_file'` to `'audio'`

```javascript
// BEFORE
formData.append('audio_file', file);

// AFTER
formData.append('audio', file);
```

### 3. Optional: Add Progress Indicator

Since processing can take several minutes, consider adding:

```javascript
const [uploadProgress, setUploadProgress] = useState(0);
const [processingStatus, setProcessingStatus] = useState('');

// Show estimated processing time based on file size
const estimatedTime = Math.ceil(fileSizeMB * 3); // ~3 seconds per MB
setProcessingStatus(`Estimated processing time: ${estimatedTime} seconds`);
```

## Response Data Structure for Display

### Displaying Predictions

The `predictions` array contains **all** analysis windows (every 0.4 seconds). For a 5-minute audio file, you'll get ~750 predictions.

**Recommendation**: Display the `events` array instead, which contains only the detected continuous events:

```javascript
// events array example
[
  {
    "start_time": 45.2,
    "end_time": 47.8,
    "class": "anemonefish",
    "duration": 2.6,
    "max_confidence": 0.94,
    "mean_confidence": 0.87
  }
]
```

### Suggested UI Components

1. **Summary Stats** (from `metadata`):
   - Audio duration
   - Processing time
   - Number of anemonefish events detected
   - Number of biological events detected

2. **Timeline Visualization**:
   - Show events as colored bars on timeline
   - Color coding: Anemonefish (blue), Biological (green), Noise (gray)

3. **Event List/Table**:
   - Timestamp, Class, Duration, Confidence
   - Sortable and filterable

4. **Export Options**:
   - Download as CSV
   - Download as Audacity labels format (for audio editing)

## Testing Locally

### 1. Start the Backend Services

```bash
cd docker
docker compose up -d
```

This starts:
- **API Gateway proxy** on `http://localhost:8000`
- **Lambda inference** service
- **LocalStack** (mock S3)

### 2. Update Frontend Config

**For local testing**, use:
```javascript
endpoint: 'http://localhost:8000/predict'
```

### 3. Test Upload

Your existing frontend code should work with only the field name change (`'audio'` instead of `'audio_file'`).

### 4. View Logs

```bash
# API Gateway logs
docker compose logs -f api-gateway

# Lambda logs
docker compose logs -f inference-lambda
```

### 5. Stop Services

```bash
docker compose down
```

## Performance Expectations

Based on testing with real audio files:

| Audio Duration | File Size | Processing Time | Predictions |
|----------------|-----------|-----------------|-------------|
| 5 minutes | 44 MB | ~2.3 minutes | ~750 windows |
| 1 hour | ~500 MB | ~28 minutes | ~9,000 windows |
| 24 hours | ~12 GB | ~11 hours | ~216,000 windows |

**Note**: Processing is ~2.2x realtime currently. This can be optimized in Phase 3.

## Production Deployment Differences

When deployed to AWS, the only change needed is the endpoint URL:

```javascript
// Production
endpoint: 'https://api.anemonefish.your-domain.com/predict'
```

Everything else (request format, response structure) remains identical.

## Environment Variables for Frontend

Create a `.env` file in your frontend project:

```env
# Local development
REACT_APP_API_ENDPOINT=http://localhost:8000/predict

# Production (set via CI/CD)
# REACT_APP_API_ENDPOINT=https://api.anemonefish.your-domain.com/predict
```

## CORS Configuration

The local API Gateway proxy has CORS enabled for all origins (`*`). In production, this will be configured via Terraform to only allow your frontend domain.

## Summary of Changes

**Required Changes**:
1. ✅ Change FormData field name: `'audio_file'` → `'audio'`
2. ✅ Update timeout: 600000ms (10 min) → 900000ms (15 min)
3. ✅ Add environment variable for endpoint configuration

**Optional but Recommended**:
1. Add progress indicator for long processing times
2. Display `events` array instead of raw `predictions`
3. Add timeline visualization
4. Add export functionality (CSV, Audacity labels)

## Questions?

For any integration issues, check:
1. Browser console for errors
2. Network tab for request/response details
3. Backend logs: `docker compose logs api-gateway`

---

**Ready for frontend testing!** 🚀
