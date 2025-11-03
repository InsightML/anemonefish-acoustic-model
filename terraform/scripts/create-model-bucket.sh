#!/bin/bash
# Quick script to create the model artifacts S3 bucket

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-anemonefish-inference}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-eu-west-2}"
BUCKET_NAME="${BUCKET_NAME:-}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Could not get AWS account ID. Is AWS CLI configured?"
    exit 1
fi

# Generate bucket name if not provided
if [ -z "$BUCKET_NAME" ]; then
    BUCKET_NAME="${PROJECT_NAME}-${ENVIRONMENT}-models-${AWS_ACCOUNT_ID}"
fi

echo "=========================================="
echo "Creating Model Artifacts S3 Bucket"
echo "=========================================="
echo "Bucket Name:  $BUCKET_NAME"
echo "Region:       $AWS_REGION"
echo "=========================================="
echo ""

# Check if bucket already exists
if aws s3api head-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" 2>/dev/null; then
    echo "Bucket '$BUCKET_NAME' already exists."
    exit 0
fi

# Create bucket
echo "Creating S3 bucket..."
if [ "$AWS_REGION" = "us-east-1" ]; then
    # us-east-1 doesn't need LocationConstraint
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$AWS_REGION"
else
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$AWS_REGION" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION"
fi

# Enable versioning
echo "Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# Enable encryption
echo "Enabling encryption..."
aws s3api put-bucket-encryption \
    --bucket "$BUCKET_NAME" \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'

# Block public access
echo "Blocking public access..."
aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo ""
echo "=========================================="
echo "Success!"
echo "=========================================="
echo "Bucket created: $BUCKET_NAME"
echo ""
echo "Upload your model with:"
echo "  aws s3 cp models/your_model.keras s3://$BUCKET_NAME/models/v1.0/best_model.keras"
echo ""
echo "Then set in terraform.tfvars:"
echo "  model_s3_bucket = \"$BUCKET_NAME\""
echo "=========================================="
