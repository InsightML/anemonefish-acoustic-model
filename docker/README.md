# Docker Setup for Anemonefish Inference

This directory contains Docker configurations for running the inference API locally and deploying to AWS Lambda.

## Overview

The Docker setup includes:

1. **Dockerfile.inference**: Multi-stage build for AWS Lambda-compatible inference container
2. **docker-compose.yml**: Local testing environment with LocalStack and Lambda Runtime Interface Emulator
3. **Dockerfile.test**: Test runner container for end-to-end validation
4. **test-local.sh**: Automated testing script

## Prerequisites

- Docker Desktop (with Docker Compose)
- At least 8GB RAM allocated to Docker
- Trained model file in `models/target_to_noise_classifier/`

## Quick Start

### 1. Build and Test Locally

```bash
# Build images and run all tests
cd docker
chmod +x test-local.sh
./test-local.sh --build

# Run tests without rebuilding
./test-local.sh

# Run with automatic cleanup
./test-local.sh --build --cleanup
```

### 2. Start Services Manually

```bash
# Start LocalStack and Lambda
docker-compose up -d

# View logs
docker-compose logs -f inference-lambda

# Stop services
docker-compose down
```

### 3. Test Inference Endpoint

The Lambda function is available at:
```
http://localhost:9000/2015-03-31/functions/function/invocations
```

Example request:
```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "httpMethod": "POST",
    "path": "/inference/predict",
    "body": "{\"s3_bucket\": \"anemonefish-inference-input\", \"s3_key\": \"test.wav\"}"
  }'
```

## Architecture

### Services

1. **localstack**: Mock AWS services (S3, Lambda, API Gateway)
   - Port 4566: LocalStack gateway
   - Automatically creates required S3 buckets on startup

2. **inference-lambda**: Lambda function container
   - Port 9000: Lambda Runtime Interface Emulator
   - Mounts model and config for easy development
   - Uses environment variables for configuration

3. **test-runner**: Automated test execution
   - Runs pytest integration tests
   - Generates JSON test reports
   - Only starts when using `--profile test`

### File Structure

```
docker/
├── Dockerfile.inference       # Lambda-compatible inference image
├── Dockerfile.test           # Test runner image
├── docker-compose.yml        # Local development environment
├── requirements-lambda.txt   # Python dependencies for Lambda
├── init-localstack.sh       # LocalStack initialization script
├── test-local.sh            # Automated testing script
└── README.md                # This file
```

## Development Workflow

### Testing Code Changes

1. **Without Rebuild**: Code is mounted as volume, changes are immediate
   ```bash
   docker-compose restart inference-lambda
   ```

2. **With Rebuild**: Test the actual container that will be deployed
   ```bash
   docker-compose up --build inference-lambda
   ```

### Testing with Real Audio

1. Upload audio to LocalStack S3:
   ```bash
   aws --endpoint-url=http://localhost:4566 s3 cp \
     /path/to/audio.wav \
     s3://anemonefish-inference-input/
   ```

2. Invoke Lambda with S3 reference:
   ```bash
   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
     -H "Content-Type: application/json" \
     -d '{
       "httpMethod": "POST",
       "path": "/inference/predict",
       "body": "{\"s3_bucket\": \"anemonefish-inference-input\", \"s3_key\": \"audio.wav\"}"
     }'
   ```

### Updating the Model

1. Train new model and save to `models/target_to_noise_classifier/`
2. Update docker-compose.yml volume mount to point to new model
3. Restart container:
   ```bash
   docker-compose restart inference-lambda
   ```

## Configuration

### Environment Variables

Configure via `docker-compose.yml` or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_LOCAL_PATH` | Path to model file in container | `/var/task/model/best_model.keras` |
| `CONFIG_PATH` | Path to inference config | `/var/task/config/inference_config.yaml` |
| `S3_INPUT_BUCKET` | Input audio bucket | `anemonefish-inference-input` |
| `S3_OUTPUT_BUCKET` | Results bucket | `anemonefish-inference-output` |
| `S3_MODEL_BUCKET` | Model artifacts bucket | `anemonefish-model-artifacts` |
| `LOG_LEVEL` | Logging level | `DEBUG` |

### Model Configuration

Edit `config/inference_config.yaml` to adjust:
- Spectrogram parameters (must match training)
- Sliding window settings
- Prediction thresholds
- Batch size

## Deployment to AWS

### Build Production Image

```bash
# Build without development volumes
docker build -f Dockerfile.inference -t anemonefish-inference:latest ..

# Test production image
docker run -p 9000:8080 \
  -e MODEL_LOCAL_PATH=/var/task/model/best_model.keras \
  anemonefish-inference:latest
```

### Push to ECR

```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag anemonefish-inference:latest \
  <account>.dkr.ecr.us-east-1.amazonaws.com/anemonefish-inference:latest

docker push <account>.dkr.ecr.us-east-1.amazonaws.com/anemonefish-inference:latest
```

## Troubleshooting

### Container fails to start

1. Check logs:
   ```bash
   docker-compose logs inference-lambda
   ```

2. Verify model file exists:
   ```bash
   docker-compose exec inference-lambda ls -la /var/task/model/
   ```

3. Check memory allocation (Lambda needs ~2GB minimum):
   ```bash
   docker stats
   ```

### LocalStack issues

1. Verify LocalStack is healthy:
   ```bash
   curl http://localhost:4566/_localstack/health
   ```

2. Check S3 buckets:
   ```bash
   aws --endpoint-url=http://localhost:4566 s3 ls
   ```

3. View LocalStack logs:
   ```bash
   docker-compose logs localstack
   ```

### Tests fail

1. Run tests with verbose output:
   ```bash
   docker-compose --profile test run --rm test-runner pytest -v -s
   ```

2. Check test logs:
   ```bash
   cat test-results/report.json | jq
   ```

### Audio processing errors

1. Verify librosa and soundfile are installed:
   ```bash
   docker-compose exec inference-lambda pip list | grep -E "librosa|soundfile"
   ```

2. Check ffmpeg is available:
   ```bash
   docker-compose exec inference-lambda which ffmpeg
   ```

3. Test with minimal audio file (see test script examples)

## Performance Optimization

### Image Size

Current image size: ~2-3GB (includes TensorFlow and audio libraries)

Optimization strategies:
- Multi-stage build (already implemented)
- Use slim Python base image where possible
- Remove unnecessary dependencies
- Consider TensorFlow Lite for production

### Cold Start Time

Expected cold start: 3-5 seconds

Improvements:
- Use Lambda provisioned concurrency
- Implement lazy loading for model
- Optimize model file size
- Consider caching in EFS

### Memory Usage

Recommended Lambda memory: 2048-4096 MB

Monitor with:
```bash
docker stats inference-lambda
```

## Additional Resources

- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [LocalStack Documentation](https://docs.localstack.cloud/)
- [Lambda Runtime Interface Emulator](https://github.com/aws/aws-lambda-runtime-interface-emulator)

