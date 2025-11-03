# Terraform Quick Start Guide

Quick reference for deploying the anemonefish inference infrastructure.

## Prerequisites

```bash
# Verify AWS CLI is configured
aws sts get-caller-identity

# Verify Terraform is installed
terraform version
```

## Step 1: Create Model Bucket

**Option A: Use helper script (Recommended)**
```bash
cd terraform/scripts
export AWS_REGION=eu-west-2
export ENVIRONMENT=dev
./create-model-bucket.sh
```

**Option B: Let Terraform create it**
- Uncomment `model_bucket` module in `environments/dev/main.tf`
- Set `model_s3_bucket = null` in `terraform.tfvars`

## Step 2: Upload Model to S3

```bash
aws s3 cp models/your_model.keras \
  s3://your-model-bucket/models/v1.0/best_model.keras \
  --region eu-west-2
```

## Step 3: Configure Environment

```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values:
# - aws_region = "eu-west-2"
# - model_s3_bucket = "your-bucket-name"
```

## Step 4: Build and Push Docker Image

```bash
cd terraform/scripts
export AWS_REGION=eu-west-2
export ENVIRONMENT=dev
./build-and-push.sh
```

## Step 5: Deploy Infrastructure

```bash
cd terraform/environments/dev
terraform init
terraform plan
terraform apply
```

## Step 6: Get API Endpoint

```bash
terraform output api_endpoint
```

## Step 7: Test API

```bash
export API_ENDPOINT=$(terraform output -raw api_endpoint)
curl -X POST -F "audio=@test_audio.wav" "$API_ENDPOINT"
```

## Updating Lambda Function

1. **Rebuild and push image:**
   ```bash
   cd terraform/scripts
   export IMAGE_TAG=v1.1.0
   ./build-and-push.sh
   ```

2. **Update Terraform:**
   ```bash
   # Edit terraform.tfvars: lambda_image_uri = "...:v1.1.0"
   terraform plan
   terraform apply
   ```

## Common Commands

```bash
# View outputs
terraform output

# View Lambda logs
aws logs tail /aws/lambda/anemonefish-inference-dev-inference --follow

# Destroy infrastructure
terraform destroy
```

## Troubleshooting

- **Lambda not found**: Ensure Docker image is pushed to ECR first
- **Model loading error**: Check S3 bucket and key are correct
- **CORS errors**: Verify `cors_allowed_origins` in terraform.tfvars
- **Timeout**: Large files may exceed 15-min limit; consider chunking

For detailed documentation, see [README.md](README.md).
