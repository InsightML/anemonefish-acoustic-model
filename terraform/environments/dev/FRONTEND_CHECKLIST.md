# Frontend Integration Checklist

## 📦 Files Provided for Frontend Dev

1. **`FRONTEND_QUICKSTART.md`** - 5-minute integration guide
2. **`FRONTEND_INTEGRATION_GUIDE.md`** - Complete documentation with code examples
3. **`audioInferenceService.js`** - Ready-to-use service (copy & paste)
4. **`API_RESPONSE_EXAMPLE.json`** - Real API response with annotations
5. **`test_inference_results.json`** - Full response from real test (1222 predictions)

---

## ✅ Implementation Checklist

### Phase 1: Basic Integration (30 mins) - Small Files Only

- [ ] Copy `audioInferenceService.js` to your project
- [ ] Update API endpoint in config
- [ ] Add API key to config
- [ ] Add `x-api-key` header to existing fetch call
- [ ] Test with small audio file (< 10 MB)
- [ ] Verify response format matches expected structure
- [ ] Update results display to show:
  - [ ] Number of events detected
  - [ ] List of events with timestamps
  - [ ] Confidence scores

**Result**: Working integration for files < 10 MB ✅

---

### Phase 2: Complete Integration (2-3 hours) - All File Sizes

- [ ] Install AWS SDK packages:
  ```bash
  npm install @aws-sdk/client-s3 @aws-sdk/client-lambda
  ```

- [ ] Get AWS credentials from backend team
  - [ ] Request IAM user or access keys
  - [ ] Permissions needed: S3 PutObject, Lambda InvokeFunction

- [ ] Add environment variables to `.env`:
  ```bash
  REACT_APP_AWS_ACCESS_KEY_ID=...
  REACT_APP_AWS_SECRET_ACCESS_KEY=...
  ```

- [ ] Replace upload function with smart version from `audioInferenceService.js`

- [ ] Test both upload paths:
  - [ ] Small file (< 10 MB) → API Gateway
  - [ ] Large file (> 10 MB) → S3 + Lambda

- [ ] Update progress tracking for both methods

- [ ] Handle all error cases:
  - [ ] File too large
  - [ ] Invalid audio format
  - [ ] Network errors
  - [ ] API errors
  - [ ] Missing credentials

**Result**: Working integration for ALL file sizes (up to 5 GB) ✅

---

### Phase 3: Enhanced UX (1-2 hours)

- [ ] Build results visualization:
  - [ ] Summary cards (events, duration, confidence)
  - [ ] Event list with timestamps
  - [ ] Timeline visualization
  - [ ] Confidence color coding

- [ ] Add export features:
  - [ ] Download results as JSON
  - [ ] Download events as CSV
  - [ ] Copy results to clipboard

- [ ] Improve user feedback:
  - [ ] File size validation before upload
  - [ ] Upload method indicator (API vs S3)
  - [ ] Estimated processing time
  - [ ] Real-time progress bar
  - [ ] Success/error notifications

**Result**: Production-ready user experience ✅

---

## 🎯 Quick Decision: Which Phase?

| Your Situation | Start With |
|----------------|------------|
| **Just testing/POC** | Phase 1 (30 mins) |
| **Production app with large files** | Phase 2 (2-3 hours) |
| **Need polished UX** | All phases (4-6 hours total) |

---

## 📊 What You'll Get

### API Response Contains:

1. **`predictions`** (array)
   - One per time window (~1,200 for 8-min audio)
   - Each has: timestamp, class, confidence, probabilities

2. **`events`** (array)
   - Continuous detections above threshold
   - Each has: start_time, end_time, duration, confidence
   - **This is what you display to users** ⭐

3. **`metadata`** (object)
   - Summary stats: duration, processing time, counts
   - Model version and configuration

### Display Priority

**Most Important** (show first):
- ✅ Number of events: `metadata.num_events`
- ✅ Event list: `events` array with times and confidence

**Secondary** (details view):
- Individual predictions in `predictions` array
- Timeline visualization
- Metadata and processing stats

**Optional** (advanced view):
- Full probability distributions
- Prediction heatmap over time
- Export/download options

---

## 🧪 Test Data Available

Use these to test your frontend:

1. **Sample Response**: `API_RESPONSE_EXAMPLE.json`
   - Annotated with display tips
   - Shows structure clearly

2. **Real Response**: `test_inference_results.json`
   - Full 8,644 lines
   - 1,222 predictions
   - 11 events
   - Real data from working API

3. **Test Command** (for backend testing):
   ```bash
   # Upload test file
   aws s3 cp test_audio.wav s3://anemonefish-inference-dev-input-944269089535/
   
   # Run inference
   aws lambda invoke \
     --function-name anemonefish-inference-dev-inference \
     --cli-binary-format raw-in-base64-out \
     --payload '{"s3_bucket": "anemonefish-inference-dev-input-944269089535", "s3_key": "test_audio.wav"}' \
     --region eu-west-2 \
     results.json
   ```

---

## 🔑 Critical Information

### API Endpoint
```
https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict
```

### API Key
```
VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M
```

### S3 Bucket
```
anemonefish-inference-dev-input-944269089535
```

### Region
```
eu-west-2
```

---

## 📝 Code You Need to Change

### File 1: Update `src/config.js` or `src/config/api.js`

```javascript
// OLD
export const API_CONFIG = {
  endpoint: process.env.REACT_APP_API_ENDPOINT || 'http://localhost:8000/predict',
  timeout: 900000,
};

// NEW
export const API_CONFIG = {
  endpoint: 'https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict',
  apiKey: 'VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M',
  timeout: 900000,
};
```

### File 2: Update `uploadAudioFile` function

```javascript
// ADD THIS HEADER
headers: {
  'x-api-key': API_CONFIG.apiKey,  // ← ADD THIS
},
```

### File 3: Handle response format

```javascript
// After successful upload
const results = await response.json();

// Display the results
console.log(`Found ${results.events.length} anemonefish vocalizations`);
results.events.forEach(event => {
  console.log(`  ${event.start_time}s - ${event.end_time}s (${event.mean_confidence * 100}%)`);
});
```

---

## 🚨 Important Notes for Frontend Dev

### File Size Limitation

**API Gateway has a 10 MB limit.** Your current code will fail for files > 10 MB.

**Solutions:**
1. **Quick fix**: Add file size check and show error
2. **Complete fix**: Implement S3 upload (see `audioInferenceService.js`)

### Response Format Changed

The API returns:
```javascript
{
  predictions: [...],  // All time windows
  events: [...],       // Detected vocalizations ← Show this!
  metadata: {...}      // Summary stats
}
```

NOT the old format (if there was one). Update your display logic accordingly.

### Processing Time

- **Small files (< 10 MB)**: ~10-30 seconds
- **Large files (> 10 MB)**: ~3 seconds per MB
- **Your 8-min test file**: ~95 seconds

Ensure your timeout and loading states handle this!

---

## ✅ Testing Steps

1. **Install dependencies**
   ```bash
   npm install @aws-sdk/client-s3 @aws-sdk/client-lambda
   ```

2. **Add environment variables**
   ```bash
   # .env
   REACT_APP_AWS_ACCESS_KEY_ID=your_key
   REACT_APP_AWS_SECRET_ACCESS_KEY=your_secret
   ```

3. **Copy service file**
   - Copy `audioInferenceService.js` to your project
   - Import: `import { runAudioInference } from './audioInferenceService'`

4. **Test with small file** (< 10 MB)
   ```javascript
   const results = await runAudioInference(smallFile, console.log);
   console.log(results);
   ```

5. **Test with large file** (> 10 MB)
   ```javascript
   const results = await runAudioInference(largeFile, setProgress);
   console.log(`Events: ${results.events.length}`);
   ```

6. **Build results UI**
   - Use `API_RESPONSE_EXAMPLE.json` for structure
   - See `ResultsDisplay` component in integration guide

---

## 📞 Questions?

**Backend Status**: ✅ Fully deployed and tested  
**API Status**: ✅ Working with real audio files  
**Documentation**: Complete with code examples  
**Sample Data**: Real test results provided  

**Ready for integration!** 🎉

---

## Summary

**What works NOW**:
- ✅ API Gateway endpoint live
- ✅ API key authentication enabled
- ✅ Lambda processing audio files
- ✅ Model making predictions
- ✅ Returning structured JSON results
- ✅ Tested with 44.8 MB real audio file
- ✅ Detected 11 events successfully

**What frontend needs to do**:
- Update API endpoint URL
- Add API key header
- Handle response format (predictions, events, metadata)
- For large files: Implement S3 upload (code provided)

**Estimated integration time**: 30 mins (basic) to 3 hours (complete with large file support)

