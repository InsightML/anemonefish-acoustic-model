# Terraform Infrastructure for Anemonefish Inference API

This directory contains Terraform configurations for deploying the anemonefish audio inference API on AWS.

## Architecture Overview

The infrastructure includes:

- **S3 Buckets**: 
  - Input bucket for temporary audio file storage
  - Output bucket for inference results
  - Model artifacts bucket (must be created separately or already exist)

- **ECR Repository**: Container registry for the Lambda Docker image

- **Lambda Function**: Serverless inference function with container image deployment
  - Memory: 3008 MB (maximum for large audio processing)
  - Timeout: 900 seconds (15 minutes)
  - Supports audio files up to 5GB

- **API Gateway**: REST API with `/predict` endpoint
  - Binary media type support for `multipart/form-data`
  - CORS configuration
  - Optional API key authentication

- **IAM Roles**: Execution role for Lambda with S3 and CloudWatch permissions

- **CloudWatch**: Log groups for Lambda and API Gateway

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with credentials
3. **Terraform** >= 1.0 installed
4. **Docker** (for building and pushing container images)
5. **Model file** uploaded to S3

## Project Structure

```
terraform/
??? modules/
?   ??? inference/          # Core inference infrastructure module
?   ?   ??? main.tf
?   ?   ??? variables.tf
?   ?   ??? outputs.tf
?   ??? ecr/                # ECR repository module
?       ??? main.tf
?       ??? variables.tf
?       ??? outputs.tf
??? environments/
?   ??? dev/               # Development environment
?   ?   ??? main.tf
?   ?   ??? variables.tf
?   ?   ??? outputs.tf
?   ?   ??? terraform.tfvars.example
?   ??? prod/              # Production environment
?       ??? main.tf
?       ??? variables.tf
?       ??? outputs.tf
?       ??? terraform.tfvars.example
??? README.md
```

## Quick Start

### 1. Prepare Model File

Upload your trained model to S3:

```bash
aws s3 cp models/target_to_noise_classifier/best_model.keras \
  s3://your-model-artifacts-bucket/models/v1.0/best_model.keras
```

### 2. Configure Backend (Optional)

For remote state storage, configure an S3 backend. Edit `environments/{dev|prod}/main.tf` and set:

```hcl
backend "s3" {
  bucket         = "your-terraform-state-bucket"
  key            = "anemonefish-inference/dev/terraform.tfstate"
  region         = "us-east-1"
  encrypt        = true
  dynamodb_table = "your-state-lock-table"
}
```

Or use local backend by removing the `backend` block.

### 3. Configure Environment Variables

Copy the example tfvars file and customize:

```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
project_name = "anemonefish-inference"
model_s3_bucket = "your-model-artifacts-bucket"
model_s3_key = "models/v1.0/best_model.keras"
```

### 4. Build and Push Docker Image

Before deploying, build and push the Lambda container image to ECR:

```bash
# Get AWS account ID and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="us-east-1"

# Create ECR repository (or use Terraform to create it first)
aws ecr create-repository \
  --repository-name anemonefish-inference-dev \
  --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image
docker build -f docker/Dockerfile.inference -t anemonefish-inference:latest .

# Tag image
docker tag anemonefish-inference:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anemonefish-inference-dev:latest

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anemonefish-inference-dev:latest
```

Alternatively, let Terraform create the ECR repository first, then push the image:

```bash
# 1. Deploy ECR only (or create manually)
# 2. Get repository URL from Terraform output
terraform output -raw ecr_repository_url

# 3. Build and push using the repository URL
```

### 5. Initialize Terraform

```bash
cd terraform/environments/dev
terraform init
```

### 6. Plan Deployment

Review the infrastructure plan:

```bash
terraform plan
```

### 7. Deploy Infrastructure

```bash
terraform apply
```

You'll be prompted to confirm. Type `yes` to proceed.

### 8. Get API Endpoint

After deployment, get the API endpoint URL:

```bash
terraform output api_endpoint
```

Example output:
```
https://abc123xyz.execute-api.us-east-1.amazonaws.com/api/predict
```

### 9. Test the API

```bash
export API_ENDPOINT=$(terraform output -raw api_endpoint)

# Test with a small audio file
curl -X POST \
  -F "audio=@test_audio.wav" \
  "$API_ENDPOINT"
```

## Deployment Workflow

### Development Environment

1. Build and push Docker image with `:latest` tag
2. Deploy infrastructure (or update Lambda image URI)
3. Test API endpoint
4. Iterate on code and rebuild/push image

### Production Environment

1. Build Docker image with specific version tag (e.g., `v1.0.0`)
2. Push image to ECR
3. Update `terraform.tfvars` with the exact image URI
4. Run `terraform plan` to review changes
5. Run `terraform apply` to deploy
6. Update frontend with new API endpoint

## Updating the Lambda Function

To update the Lambda function with a new Docker image:

1. **Build new image**:
   ```bash
   docker build -f docker/Dockerfile.inference -t anemonefish-inference:v1.1.0 .
   ```

2. **Push to ECR**:
   ```bash
   docker tag anemonefish-inference:v1.1.0 \
     $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anemonefish-inference-prod:v1.1.0
   docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anemonefish-inference-prod:v1.1.0
   ```

3. **Update Terraform**:
   - Edit `terraform/environments/prod/terraform.tfvars`
   - Set `lambda_image_uri` to the new image URI

4. **Apply changes**:
   ```bash
   cd terraform/environments/prod
   terraform plan
   terraform apply
   ```

## Variables Reference

### Common Variables

- `project_name`: Project name prefix (default: `anemonefish-inference`)
- `environment`: Environment name (`dev` or `prod`)
- `model_s3_bucket`: S3 bucket containing the model file (required)
- `model_s3_key`: S3 key path to model file (default: `models/v1.0/best_model.keras`)
- `model_version`: Model version identifier
- `lambda_image_uri`: Full ECR image URI for Lambda container
- `tags`: Additional resource tags

### Lambda Configuration

- `lambda_timeout`: Function timeout in seconds (max 900, default: 900)
- `lambda_memory_size`: Memory in MB (default: 3008, max for Lambda)
- `lambda_reserved_concurrent_executions`: Reserved concurrency (null = unlimited)

### API Gateway Configuration

- `api_gateway_stage_name`: Stage name (default: `api`)
- `cors_allowed_origins`: List of allowed CORS origins
- `enable_api_key`: Enable API key authentication

### Inference Configuration

The `inference_config` object allows customization of preprocessing and prediction parameters:

- `fmax_hz`: Maximum frequency (default: 2000)
- `n_fft`: FFT window size (default: 1024)
- `hop_length`: Hop length for STFT (default: 256)
- `width_pixels`: Spectrogram width (default: 256)
- `height_pixels`: Spectrogram height (default: 256)
- `window_duration`: Window duration in seconds (default: 1.0)
- `stride_duration`: Stride duration in seconds (default: 0.4)
- `batch_size`: Prediction batch size (default: 32)
- `confidence_threshold`: Confidence threshold (default: 0.5)
- `target_class`: Target class for event detection (default: `anemonefish`)
- `min_event_duration`: Minimum event duration (default: 0.2)
- `min_gap_duration`: Minimum gap between events (default: 0.1)
- `smoothing_window`: Smoothing window size (default: 5)
- `model_classes`: Comma-separated class names (default: `noise,anemonefish,biological`)

## Outputs

After deployment, use `terraform output` to view:

- `api_endpoint`: Full HTTPS URL to the `/predict` endpoint
- `lambda_function_name`: Lambda function name
- `s3_input_bucket`: Input bucket name
- `s3_output_bucket`: Output bucket name
- `ecr_repository_url`: ECR repository URL
- `api_key_value`: API key value (if enabled, marked as sensitive)

## Security Considerations

### Production Checklist

- [ ] Set `cors_allowed_origins` to your frontend domain only
- [ ] Enable API key authentication (`enable_api_key = true`)
- [ ] Store API key securely (e.g., AWS Secrets Manager)
- [ ] Use immutable ECR image tags in production
- [ ] Enable S3 bucket versioning and encryption
- [ ] Configure CloudWatch alarms for errors and latency
- [ ] Set appropriate IAM permissions (least privilege)
- [ ] Enable AWS WAF for API Gateway if needed
- [ ] Use VPC endpoints for S3 access (if Lambda is in VPC)

### API Key Usage

If API key authentication is enabled, include the key in requests:

```bash
curl -X POST \
  -H "x-api-key: YOUR_API_KEY" \
  -F "audio=@test_audio.wav" \
  "$API_ENDPOINT"
```

Get the API key from Terraform output:

```bash
terraform output -raw api_key_value
```

## Cost Optimization

- **S3 Lifecycle Policies**: Automatically delete old files (configured via `s3_input_expiration_days` and `s3_output_expiration_days`)
- **ECR Lifecycle Policy**: Automatically delete old images (configured via `max_image_count` in ECR module)
- **CloudWatch Log Retention**: Set `log_retention_days` to control log retention
- **Lambda Reserved Concurrency**: Limit concurrent executions to control costs
- **API Gateway Throttling**: Configure throttling limits to prevent abuse

## Troubleshooting

### Lambda Function Not Found

- Ensure Docker image is pushed to ECR before deploying
- Verify `lambda_image_uri` is correct in `terraform.tfvars`
- Check Lambda function logs: `aws logs tail /aws/lambda/anemonefish-inference-dev-inference --follow`

### Model Loading Errors

- Verify model file exists in S3: `aws s3 ls s3://your-model-bucket/models/`
- Check Lambda execution role has S3 read permissions
- Verify `S3_MODEL_BUCKET` and `S3_MODEL_KEY` environment variables are set correctly

### CORS Errors

- Verify `cors_allowed_origins` includes your frontend domain
- Check API Gateway CORS configuration is deployed
- Test with `curl` to see actual error messages

### Timeout Errors

- Large audio files may exceed 15-minute Lambda timeout
- Consider using ECS Fargate for longer-running tasks (not implemented in this module)
- Or pre-process audio files to split into smaller chunks

### API Gateway 502 Bad Gateway

- Check Lambda function logs for errors
- Verify Lambda function is deployed and healthy
- Check API Gateway integration configuration

## Cleanup

To destroy all resources:

```bash
cd terraform/environments/dev
terraform destroy
```

**Warning**: This will delete all resources including S3 buckets. Make sure to back up any important data first.

## Next Steps

- [ ] Set up CI/CD pipeline for automated deployments (Phase 4)
- [ ] Configure CloudWatch alarms and monitoring (Phase 6)
- [ ] Set up X-Ray tracing for performance analysis
- [ ] Implement API Gateway rate limiting and throttling
- [ ] Add integration tests for the deployed infrastructure

## Support

For issues or questions:
1. Check CloudWatch logs for Lambda and API Gateway
2. Review Terraform plan output for configuration issues
3. Verify AWS service quotas and limits
4. Consult AWS documentation for service-specific issues
