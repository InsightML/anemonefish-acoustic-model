# Terraform Infrastructure for Anemonefish Inference API

This directory contains Terraform configurations for deploying the anemonefish audio inference API on AWS.

## Architecture Overview

The infrastructure includes:

- **S3 Buckets**: 
  - Input bucket for temporary audio file storage
  - Output bucket for inference results
  - Model artifacts bucket (optional - can be created by Terraform or separately)

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
5. **Model file** uploaded to S3 (see "Model Bucket Setup" below)

## Project Structure

```
terraform/
??? modules/
?   ??? inference/          # Core inference infrastructure module
?   ??? ecr/                # ECR repository module
?   ??? model-bucket/       # Optional model artifacts bucket module
??? environments/
?   ??? dev/               # Development environment
?   ??? prod/              # Production environment
??? scripts/
?   ??? build-and-push.sh      # Build and push Docker image
?   ??? create-model-bucket.sh # Create model artifacts bucket
??? README.md
```

## Quick Start

### 1. Create Model Bucket

You have two options:

**Option A: Use the helper script (Recommended)**
```bash
cd terraform/scripts
export AWS_REGION=eu-west-2
export ENVIRONMENT=dev
./create-model-bucket.sh
```

**Option B: Use Terraform module**
- Uncomment the `model_bucket` module in `environments/dev/main.tf`
- Set `model_s3_bucket = null` in `terraform.tfvars`
- Terraform will create the bucket automatically

**Option C: Create manually**
```bash
export AWS_REGION=eu-west-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="anemonefish-inference-dev-models-${AWS_ACCOUNT_ID}"

aws s3api create-bucket \
  --bucket "$BUCKET_NAME" \
  --region "$AWS_REGION" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION"
```

### 2. Upload Model File

```bash
aws s3 cp models/your_model.keras \
  s3://your-model-bucket/models/v1.0/best_model.keras \
  --region eu-west-2
```

### 3. Configure Environment Variables

```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
```hcl
project_name = "anemonefish-inference"
aws_region = "eu-west-2"
model_s3_bucket = "your-model-artifacts-bucket"
model_s3_key = "models/v1.0/best_model.keras"
```

### 4. Build and Push Docker Image

```bash
cd terraform/scripts
export AWS_REGION=eu-west-2
export ENVIRONMENT=dev
./build-and-push.sh
```

This will:
- Create ECR repository if needed
- Build Docker image
- Push to ECR

Copy the image URI from the output.

### 5. Configure Terraform Provider

Make sure your AWS provider is configured. You can either:

- Use AWS CLI credentials: `aws configure`
- Set environment variables: `export AWS_REGION=eu-west-2`
- Or configure in `environments/dev/main.tf`:
```hcl
provider "aws" {
  region = "eu-west-2"
}
```

### 6. Initialize Terraform

```bash
cd terraform/environments/dev
terraform init
```

### 7. Plan Deployment

```bash
terraform plan
```

### 8. Deploy Infrastructure

```bash
terraform apply
```

You'll be prompted to confirm. Type `yes` to proceed.

### 9. Get API Endpoint

```bash
terraform output api_endpoint
```

Example output:
```
https://abc123xyz.execute-api.eu-west-2.amazonaws.com/api/predict
```

### 10. Test the API

```bash
export API_ENDPOINT=$(terraform output -raw api_endpoint)
curl -X POST -F "audio=@test_audio.wav" "$API_ENDPOINT"
```

## Model Bucket Setup

The model artifacts bucket is **not created automatically** by the main inference module. You have three options:

### Option 1: Use Helper Script (Easiest)

```bash
cd terraform/scripts
export AWS_REGION=eu-west-2
export ENVIRONMENT=dev
./create-model-bucket.sh
```

Then set the bucket name in `terraform.tfvars`:
```hcl
model_s3_bucket = "anemonefish-inference-dev-models-<your-account-id>"
```

### Option 2: Use Terraform Module

1. Edit `terraform/environments/dev/main.tf`
2. Uncomment the `model_bucket` module block (lines 29-41)
3. Set `model_s3_bucket = null` in `terraform.tfvars`
4. Terraform will create the bucket automatically

### Option 3: Create Manually

Create the bucket using AWS CLI or Console, then reference it in `terraform.tfvars`.

**Why separate?** Model buckets are often shared across environments or managed separately from infrastructure, so we provide flexibility.

## Deployment Workflow

### Development Environment

1. Create model bucket (see above)
2. Upload model to S3
3. Build and push Docker image with `:latest` tag
4. Configure `terraform.tfvars`
5. Deploy: `terraform apply`
6. Test API endpoint

### Production Environment

1. Create model bucket
2. Build Docker image with specific version tag (e.g., `v1.0.0`)
3. Push image to ECR
4. Update `terraform.tfvars` with exact image URI
5. Set `cors_allowed_origins` to your frontend domain
6. Enable API key authentication
7. Run `terraform plan` to review
8. Run `terraform apply` to deploy
9. Update frontend with new API endpoint

## Updating the Lambda Function

To update the Lambda function with a new Docker image:

1. **Build new image**:
   ```bash
   cd terraform/scripts
   export IMAGE_TAG=v1.1.0
   export AWS_REGION=eu-west-2
   ./build-and-push.sh
   ```

2. **Update Terraform**:
   - Edit `terraform/environments/dev/terraform.tfvars`
   - Set `lambda_image_uri` to the new image URI

3. **Apply changes**:
   ```bash
   terraform plan
   terraform apply
   ```

## Variables Reference

### Common Variables

- `project_name`: Project name prefix (default: `anemonefish-inference`)
- `environment`: Environment name (`dev` or `prod`)
- `aws_region`: AWS region (defaults to provider region, can override)
- `model_s3_bucket`: S3 bucket containing the model file (required, or use model_bucket module)
- `model_s3_key`: S3 key path to model file (default: `models/v1.0/best_model.keras`)
- `model_version`: Model version identifier
- `lambda_image_uri`: Full ECR image URI for Lambda container

### Lambda Configuration

- `lambda_timeout`: Function timeout in seconds (max 900, default: 900)
- `lambda_memory_size`: Memory in MB (default: 3008, max for Lambda)
- `lambda_reserved_concurrent_executions`: Reserved concurrency (null = unlimited)

### API Gateway Configuration

- `api_gateway_stage_name`: Stage name (default: `api`)
- `cors_allowed_origins`: List of allowed CORS origins
- `enable_api_key`: Enable API key authentication

### Inference Configuration

The `inference_config` object allows customization of preprocessing and prediction parameters (see `terraform.tfvars.example`).

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

### API Key Usage

If API key authentication is enabled:

```bash
curl -X POST \
  -H "x-api-key: YOUR_API_KEY" \
  -F "audio=@test_audio.wav" \
  "$API_ENDPOINT"
```

Get the API key:
```bash
terraform output -raw api_key_value
```

## Troubleshooting

### Build Script Path Error

If you see `lstat ../../docker: no such file or directory`:
- The script now uses absolute paths and should work from any directory
- Make sure you're running from `terraform/scripts/`

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

### Region Configuration

If deploying to `eu-west-2`:
- Set `AWS_REGION=eu-west-2` environment variable
- Or configure in `terraform.tfvars`: `aws_region = "eu-west-2"`
- Make sure all resources are in the same region

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
