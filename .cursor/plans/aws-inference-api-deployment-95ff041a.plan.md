<!-- 95ff041a-4e67-4363-b6ee-7a3075fefcb4 44d7f4c5-57f6-4454-8e57-4ee6adfb4e0d -->
# AWS Inference API Deployment Plan

## Architecture Overview

Deploy a serverless inference pipeline on AWS using API Gateway, Lambda/ECS, S3, and ECR for processing audio files and running model inference.

## Phase 1: Code Refactoring for Cloud Deployment

### 1.1 Create Preprocessing Module (`src/inference/`)

- Extract spectrogram generation logic from notebooks into reusable Python module
- Create `preprocessor.py` with:
  - `AudioPreprocessor` class with configuration matching training parameters
  - Methods: `load_audio()`, `segment_audio()`, `create_spectrogram()`
  - Use same parameters: FMAX=2000Hz, N_FFT=1024, HOP_LENGTH=256, 256x256 pixels

### 1.2 Create Inference Module (`src/inference/`)

- Create `model_inference.py` with:
  - `ModelInference` class for loading and running predictions
  - Support for batch processing of spectrograms
  - Return confidence scores and timestamps

### 1.3 Create Lambda Handler (`src/lambda/`)

- `inference_handler.py` for API Lambda function
- Handle multipart form uploads from frontend
- Process audio through preprocessing → inference pipeline
- Return JSON response with predictions

### 1.4 Configuration Management

- Create `config/inference_config.yaml` with preprocessing parameters
- Ensure consistency with training configuration
- Support environment variable overrides for AWS deployment

## Phase 2: Docker Container Setup

### 2.1 Create Inference Docker Image

- Base image: `public.ecr.aws/lambda/python:3.10` for Lambda compatibility
- Install dependencies: librosa, tensorflow, numpy, matplotlib
- Package model file and preprocessing code
- Optimize for size (multi-stage build if needed)

### 2.2 Local Testing Setup

- Docker compose for local API testing
- Mock S3 using LocalStack
- Test script for end-to-end validation

## Phase 3: Terraform Infrastructure

### 3.1 Core Infrastructure (`terraform/modules/inference/`)

```hcl
# S3 Buckets
- inference-input-bucket (temporary audio storage)
- inference-output-bucket (results storage)
- model-artifacts-bucket (model files)

# Lambda/ECS Service (choose based on processing time)
- Lambda for < 15 min processing
- ECS Fargate for longer audio files

# API Gateway
- REST API with /predict endpoint
- Binary media type support for audio uploads
- CORS configuration for frontend

# ECR Repository
- Container registry for inference image
```

### 3.2 IAM Roles and Policies

- Lambda execution role with S3 access
- API Gateway invoke permissions
- CloudWatch logging permissions

### 3.3 Networking (if using ECS)

- VPC with public/private subnets
- NAT Gateway for outbound access
- Security groups for ECS tasks

## Phase 4: Model Deployment Pipeline

### 4.1 Model Artifact Management

- Upload trained model to S3
- Version control for model files
- Model metadata (training config, performance metrics)

### 4.2 CI/CD Pipeline (GitHub Actions)

```yaml
- Build and push Docker image to ECR
- Update Lambda function or ECS task definition
- Run integration tests
- Deploy to staging → production
```

## Phase 5: API Implementation Details

### 5.1 API Contract

```
POST /inference/predict
Content-Type: multipart/form-data

Request:
- audio_file: binary (supports .wav, .mp3, .flac, etc.)
- config: optional JSON (override default parameters)

Response:
{
  "predictions": [
    {
      "timestamp": "0.0-1.0s",
      "class": "anemonefish",
      "confidence": 0.92
    }
  ],
  "metadata": {
    "audio_duration": 24.5,
    "processing_time": 3.2,
    "model_version": "v1.0"
  }
}
```

### 5.2 Processing Flow

1. Upload audio to S3 input bucket
2. Trigger Lambda/ECS task
3. Load and segment audio (0.4s windows)
4. Generate spectrograms for each segment
5. Run batch inference on spectrograms
6. Aggregate predictions
7. Store results in S3 output bucket
8. Return JSON response

## Phase 6: Monitoring and Logging

### 6.1 CloudWatch Setup

- Lambda/ECS metrics and logs
- API Gateway access logs
- Custom metrics (inference latency, model accuracy)
- Alarms for errors and high latency

### 6.2 X-Ray Tracing

- End-to-end request tracing
- Performance bottleneck identification

## Phase 7: Cost Optimization

### 7.1 Compute Optimization

- Use Spot instances for ECS if applicable
- Lambda provisioned concurrency for consistent performance
- Auto-scaling based on request volume

### 7.2 Storage Optimization

- S3 lifecycle policies for temporary files
- Glacier for long-term result storage

## Implementation Todos

### Prerequisites

- Ensure model file is available and tested
- Verify preprocessing parameters match training
- Set up AWS account and credentials

### Development Tasks

- **Week 1**: Code refactoring and module creation
- **Week 2**: Docker setup and local testing
- **Week 3**: Terraform infrastructure development
- **Week 4**: API implementation and integration
- **Week 5**: CI/CD pipeline and testing
- **Week 6**: Monitoring setup and optimization

### File Structure

```
├── src/
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── preprocessor.py
│   │   ├── model_inference.py
│   │   └── utils.py
│   └── lambda/
│       └── inference_handler.py
├── terraform/
│   ├── modules/
│   │   ├── inference/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   └── monitoring/
│   ├── environments/
│   │   ├── dev/
│   │   └── prod/
│   └── main.tf
├── docker/
│   ├── Dockerfile.inference
│   └── docker-compose.yml
├── config/
│   └── inference_config.yaml
└── tests/
    ├── unit/
    └── integration/
```

## Key Considerations

1. **Audio File Size**: Frontend allows up to 5GB files. Consider:

   - Streaming processing for large files
   - ECS Fargate for long-running tasks
   - SQS for async processing

2. **Preprocessing Consistency**: Must match training exactly:

   - Same spectrogram parameters
   - Same normalization
   - Same segmentation approach

3. **Scalability**: Design for concurrent requests:

   - Lambda concurrency limits
   - ECS auto-scaling
   - API Gateway throttling

4. **Security**: 

   - API key authentication
   - S3 presigned URLs for large files
   - VPC endpoints for private communication

### To-dos

- [ ] Extract preprocessing logic from notebooks into reusable Python modules in src/inference/
- [ ] Create model inference module with batch processing support
- [ ] Create Docker image for inference with Lambda runtime compatibility
- [ ] Develop Terraform modules for S3, Lambda/ECS, API Gateway, and ECR
- [ ] Implement Lambda handler for processing audio uploads
- [ ] Integrate API Gateway with Lambda/ECS and test end-to-end flow
- [ ] Set up GitHub Actions for automated deployment
- [ ] Configure CloudWatch monitoring and alerts
- [ ] Create comprehensive unit and integration tests
- [ ] Write API documentation and deployment guide