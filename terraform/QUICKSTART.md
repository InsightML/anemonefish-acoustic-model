# Terraform Quick Start Guide

Quick reference for deploying the anemonefish inference infrastructure.

## Prerequisites

```bash
# Verify AWS CLI is configured
aws sts get-caller-identity

# Verify Terraform is installed
terraform version
```

## Step 1: Upload Model to S3

```bash
aws s3 cp models/your_model.keras \
  s3://your-model-bucket/models/v1.0/best_model.keras
```

## Step 2: Configure Environment

```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

## Step 3: Build and Push Docker Image

```bash
# Option A: Use the helper script
cd terraform/scripts
export AWS_REGION=us-east-1
export ENVIRONMENT=dev
./build-and-push.sh

# Option B: Manual steps
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

cd /workspace
docker build -f docker/Dockerfile.inference -t anemonefish-inference:latest .
docker tag anemonefish-inference:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anemonefish-inference-dev:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anemonefish-inference-dev:latest
```

## Step 4: Deploy Infrastructure

```bash
cd terraform/environments/dev
terraform init
terraform plan
terraform apply
```

## Step 5: Get API Endpoint

```bash
terraform output api_endpoint
```

## Step 6: Test API

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
