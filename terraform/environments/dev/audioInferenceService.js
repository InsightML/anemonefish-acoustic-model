/**
 * Anemonefish Acoustic Inference Service
 * 
 * Ready-to-use service for uploading audio files and running inference.
 * Automatically handles both small files (< 10 MB via API Gateway) 
 * and large files (> 10 MB via S3 + Lambda).
 * 
 * Installation:
 *   npm install @aws-sdk/client-s3 @aws-sdk/client-lambda
 * 
 * Usage:
 *   import { runAudioInference } from './audioInferenceService';
 *   
 *   const results = await runAudioInference(file, (progress) => {
 *     console.log(`Progress: ${progress}%`);
 *   });
 */

import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { LambdaClient, InvokeCommand } from '@aws-sdk/client-lambda';

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
  // API Gateway
  apiGatewayUrl: 'https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api',
  apiKey: 'VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M',
  
  // S3 & Lambda
  s3InputBucket: 'anemonefish-inference-dev-input-944269089535',
  lambdaFunctionName: 'anemonefish-inference-dev-inference',
  awsRegion: 'eu-west-2',
  
  // AWS Credentials (from environment variables)
  awsAccessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
  awsSecretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY,
  
  // File size threshold (10 MB)
  largeFileThreshold: 10 * 1024 * 1024,
  
  // Timeouts
  timeout: 900000, // 15 minutes
};

// ============================================================================
// AWS CLIENT INITIALIZATION
// ============================================================================

let s3Client = null;
let lambdaClient = null;

const initializeAwsClients = () => {
  if (!CONFIG.awsAccessKeyId || !CONFIG.awsSecretAccessKey) {
    console.warn('AWS credentials not configured. S3 upload will not be available.');
    return false;
  }

  if (!s3Client) {
    s3Client = new S3Client({
      region: CONFIG.awsRegion,
      credentials: {
        accessKeyId: CONFIG.awsAccessKeyId,
        secretAccessKey: CONFIG.awsSecretAccessKey,
      },
    });
  }

  if (!lambdaClient) {
    lambdaClient = new LambdaClient({
      region: CONFIG.awsRegion,
      credentials: {
        accessKeyId: CONFIG.awsAccessKeyId,
        secretAccessKey: CONFIG.awsSecretAccessKey,
      },
    });
  }

  return true;
};

// ============================================================================
// MAIN EXPORT FUNCTION
// ============================================================================

/**
 * Upload audio file and run inference
 * Automatically selects best upload method based on file size
 * 
 * @param {File} file - Audio file to process
 * @param {Function} onProgress - Optional callback with progress (0-100)
 * @returns {Promise<Object>} Inference results
 * 
 * @example
 * const results = await runAudioInference(audioFile, (progress) => {
 *   setUploadProgress(progress);
 * });
 * 
 * console.log(`Detected ${results.events.length} anemonefish vocalizations`);
 */
export const runAudioInference = async (file, onProgress = null) => {
  // Validate file
  if (!file) {
    throw new Error('No file provided');
  }

  // Check file size and select upload method
  const isLargeFile = file.size > CONFIG.largeFileThreshold;
  
  console.log(`File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
  console.log(`Upload method: ${isLargeFile ? 'S3 + Lambda' : 'API Gateway Direct'}`);
  
  if (isLargeFile) {
    return uploadViaS3(file, onProgress);
  } else {
    return uploadViaAPIGateway(file, onProgress);
  }
};

// ============================================================================
// UPLOAD METHOD 1: API GATEWAY (Small Files < 10 MB)
// ============================================================================

/**
 * Upload file directly via API Gateway
 * Use for files < 10 MB only
 */
const uploadViaAPIGateway = async (file, onProgress) => {
  try {
    onProgress?.(10);
    
    // Create form data
    const formData = new FormData();
    formData.append('audio_file', file);
    
    // Make request
    const response = await fetch(`${CONFIG.apiGatewayUrl}/predict`, {
      method: 'POST',
      headers: {
        'x-api-key': CONFIG.apiKey,
      },
      body: formData,
    });
    
    onProgress?.(90);
    
    // Handle errors
    if (!response.ok) {
      const errorText = await response.text();
      
      // Check for size limit error
      if (errorText.includes('content length exceeded') || errorText.includes('10485760')) {
        throw new Error(
          'File too large for direct upload (>10MB). The system will automatically use S3 upload for large files.'
        );
      }
      
      throw new Error(`API error (${response.status}): ${errorText}`);
    }
    
    // Parse results
    const results = await response.json();
    onProgress?.(100);
    
    // Check for application-level errors
    if (results.error) {
      throw new Error(results.message || results.error);
    }
    
    return results;
    
  } catch (error) {
    console.error('API Gateway upload error:', error);
    throw error;
  }
};

// ============================================================================
// UPLOAD METHOD 2: S3 + LAMBDA (Large Files > 10 MB)
// ============================================================================

/**
 * Upload file to S3 and invoke Lambda for inference
 * Use for files > 10 MB (or all files for consistency)
 */
const uploadViaS3 = async (file, onProgress) => {
  // Initialize AWS clients
  if (!initializeAwsClients()) {
    throw new Error(
      'AWS credentials not configured. Please add REACT_APP_AWS_ACCESS_KEY_ID and ' +
      'REACT_APP_AWS_SECRET_ACCESS_KEY to your environment variables.'
    );
  }

  try {
    onProgress?.(5);
    
    // Step 1: Upload to S3
    const s3Key = `uploads/${Date.now()}_${file.name.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
    
    console.log(`Uploading to S3: s3://${CONFIG.s3InputBucket}/${s3Key}`);
    
    const uploadCommand = new PutObjectCommand({
      Bucket: CONFIG.s3InputBucket,
      Key: s3Key,
      Body: file,
      ContentType: file.type || 'audio/wav',
    });

    await s3Client.send(uploadCommand);
    console.log('S3 upload complete');
    onProgress?.(50);
    
    // Step 2: Invoke Lambda for inference
    console.log('Invoking Lambda for inference...');
    
    const payload = {
      s3_bucket: CONFIG.s3InputBucket,
      s3_key: s3Key,
    };
    
    const invokeCommand = new InvokeCommand({
      FunctionName: CONFIG.lambdaFunctionName,
      Payload: JSON.stringify(payload),
    });
    
    const lambdaResponse = await lambdaClient.send(invokeCommand);
    console.log('Lambda invocation complete');
    onProgress?.(95);
    
    // Parse Lambda response
    const responsePayload = JSON.parse(
      new TextDecoder().decode(lambdaResponse.Payload)
    );
    
    // Check for Lambda-level errors
    if (responsePayload.errorMessage) {
      throw new Error(`Lambda error: ${responsePayload.errorMessage}`);
    }
    
    // Parse the body (Lambda returns API Gateway response format)
    const results = JSON.parse(responsePayload.body);
    
    // Check for application-level errors
    if (results.error) {
      throw new Error(results.message || results.error);
    }
    
    onProgress?.(100);
    console.log(`Inference complete: ${results.events?.length || 0} events detected`);
    
    return results;
    
  } catch (error) {
    console.error('S3/Lambda error:', error);
    throw new Error(`Failed to process audio via S3: ${error.message}`);
  }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Format file size for display
 */
export const formatFileSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

/**
 * Estimate processing time based on file size
 */
export const estimateProcessingTime = (fileSizeBytes) => {
  // Based on testing: ~3 seconds per MB
  const fileSizeMB = fileSizeBytes / (1024 * 1024);
  const estimatedSeconds = Math.ceil(fileSizeMB * 3);
  
  if (estimatedSeconds < 60) return `~${estimatedSeconds} seconds`;
  if (estimatedSeconds < 3600) return `~${Math.ceil(estimatedSeconds / 60)} minutes`;
  
  const hours = Math.floor(estimatedSeconds / 3600);
  const minutes = Math.ceil((estimatedSeconds % 3600) / 60);
  return `~${hours}h ${minutes}m`;
};

/**
 * Validate audio file
 */
export const validateAudioFile = (file) => {
  const validTypes = [
    'audio/wav',
    'audio/wave',
    'audio/x-wav',
    'audio/mpeg',
    'audio/mp3',
    'audio/flac',
    'audio/x-flac',
    'audio/mp4',
    'audio/ogg',
    'audio/webm',
  ];
  
  const validExtensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm'];
  const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
  
  const isValidType = validTypes.includes(file.type) || 
                     validExtensions.includes(fileExtension);
  
  if (!isValidType) {
    return {
      valid: false,
      error: 'Invalid file type. Please select an audio file (.wav, .mp3, .flac, etc.)',
    };
  }
  
  const maxSize = 5 * 1024 * 1024 * 1024; // 5 GB
  if (file.size > maxSize) {
    return {
      valid: false,
      error: 'File size exceeds maximum limit of 5 GB',
    };
  }
  
  return { valid: true };
};

/**
 * Format timestamp for display
 */
export const formatTimestamp = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}:${secs.padStart(4, '0')}`;
};

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  runAudioInference,
  formatFileSize,
  estimateProcessingTime,
  validateAudioFile,
  formatTimestamp,
  CONFIG,
};

