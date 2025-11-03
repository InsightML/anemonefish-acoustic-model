#!/bin/bash
# Build and push Docker image to ECR for Lambda deployment

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-anemonefish-inference}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-eu-west-2}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="${DOCKERFILE:-$PROJECT_ROOT/docker/Dockerfile.inference}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Could not get AWS account ID. Is AWS CLI configured?"
    exit 1
fi

ECR_REPO_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "=========================================="
echo "Building and Pushing Docker Image"
echo "=========================================="
echo "Project:       $PROJECT_NAME"
echo "Environment:  $ENVIRONMENT"
echo "Region:       $AWS_REGION"
echo "Image Tag:    $IMAGE_TAG"
echo "ECR Repo:     $ECR_REPOSITORY_URI"
echo "Dockerfile:   $DOCKERFILE"
echo "Project Root: $PROJECT_ROOT"
echo "=========================================="
echo ""

# Verify Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE"
    exit 1
fi

# Check if ECR repository exists
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" > /dev/null 2>&1; then
    echo "ECR repository '$ECR_REPO_NAME' does not exist."
    echo "Creating ECR repository..."
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "Repository created."
else
    echo "ECR repository '$ECR_REPO_NAME' already exists."
fi

# Login to ECR
echo ""
echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REPOSITORY_URI"

# Build Docker image (from project root)
echo ""
echo "Building Docker image..."
cd "$PROJECT_ROOT"
docker build \
    -f "$DOCKERFILE" \
    -t "${PROJECT_NAME}:${IMAGE_TAG}" \
    -t "${ECR_REPOSITORY_URI}:${IMAGE_TAG}" \
    -t "${ECR_REPOSITORY_URI}:latest" \
    .

# Push image
echo ""
echo "Pushing image to ECR..."
docker push "${ECR_REPOSITORY_URI}:${IMAGE_TAG}"
docker push "${ECR_REPOSITORY_URI}:latest"

echo ""
echo "=========================================="
echo "Success!"
echo "=========================================="
echo "Image URI: ${ECR_REPOSITORY_URI}:${IMAGE_TAG}"
echo ""
echo "Update your terraform.tfvars with:"
echo "  lambda_image_uri = \"${ECR_REPOSITORY_URI}:${IMAGE_TAG}\""
echo "=========================================="
