#!/bin/bash

# LocalStack initialization script
# This script runs when LocalStack is ready

echo "Initializing LocalStack services for Anemonefish Inference..."

# Set AWS CLI to use LocalStack
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
AWS_ENDPOINT="http://localhost:4566"

# Function to wait for LocalStack to be ready
wait_for_localstack() {
    echo "Waiting for LocalStack to be ready..."
    until curl -s $AWS_ENDPOINT/_localstack/health | grep -q '"s3": "available"'; do
        echo "Waiting for S3 service..."
        sleep 2
    done
    echo "LocalStack is ready!"
}

wait_for_localstack

# Create S3 buckets
echo "Creating S3 buckets..."
aws --endpoint-url=$AWS_ENDPOINT s3 mb s3://anemonefish-inference-input || true
aws --endpoint-url=$AWS_ENDPOINT s3 mb s3://anemonefish-inference-output || true
aws --endpoint-url=$AWS_ENDPOINT s3 mb s3://anemonefish-model-artifacts || true

# List buckets to verify
echo "Verifying buckets..."
aws --endpoint-url=$AWS_ENDPOINT s3 ls

# Set bucket policies (public read for testing)
echo "Setting bucket policies..."
cat > /tmp/bucket-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::anemonefish-inference-input/*"
        }
    ]
}
EOF

aws --endpoint-url=$AWS_ENDPOINT s3api put-bucket-policy \
    --bucket anemonefish-inference-input \
    --policy file:///tmp/bucket-policy.json || true

echo "LocalStack initialization complete!"

