#!/bin/bash
# Manual testing script for quick inference tests

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Clean up macOS metadata files that cause xattr errors on NAS volumes
find "$WORKSPACE_DIR" -name "._*" -type f -delete 2>/dev/null || true

LAMBDA_ENDPOINT="${LAMBDA_ENDPOINT:-http://localhost:9000/2015-03-31/functions/function/invocations}"
LOCALSTACK_ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Manual Inference Test${NC}"
echo -e "${BLUE}========================================${NC}"

# Parse arguments
AUDIO_FILE=""
USE_S3=false
OUTPUT_FILE="inference_result.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio)
            AUDIO_FILE="$2"
            shift 2
            ;;
        --use-s3)
            USE_S3=true
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 --audio <file> [--use-s3] [--output <file>]"
            exit 1
            ;;
    esac
done

if [ -z "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: --audio parameter is required${NC}"
    echo "Usage: $0 --audio <file> [--use-s3] [--output <file>]"
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    exit 1
fi

# Check file size and recommend S3 for large files
FILE_SIZE=$(stat -f%z "$AUDIO_FILE" 2>/dev/null || stat -c%s "$AUDIO_FILE" 2>/dev/null)
FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))

if [ "$USE_S3" = false ] && [ "$FILE_SIZE" -gt 10485760 ]; then
    echo -e "${YELLOW}⚠️  Warning: File size is ${FILE_SIZE_MB}MB${NC}"
    echo -e "${YELLOW}Base64 encoding large files may exceed command-line limits.${NC}"
    echo -e "${YELLOW}For files >10MB, use --use-s3 flag instead.${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Try: $0 --audio \"$AUDIO_FILE\" --use-s3${NC}"
        exit 0
    fi
fi

echo -e "${BLUE}File size: ${FILE_SIZE_MB}MB${NC}"
echo ""

# Test S3 method
if [ "$USE_S3" = true ]; then
    echo -e "${GREEN}Testing with S3 upload method...${NC}"
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI is not installed${NC}"
        echo "Install with: brew install awscli"
        exit 1
    fi
    
    # Upload to LocalStack S3
    S3_KEY="test_$(basename "$AUDIO_FILE")"
    echo -e "${BLUE}Uploading to s3://anemonefish-inference-input/$S3_KEY${NC}"
    
    aws --endpoint-url=$LOCALSTACK_ENDPOINT s3 cp \
        "$AUDIO_FILE" \
        "s3://anemonefish-inference-input/$S3_KEY"
    
    # Create Lambda event
    EVENT=$(cat <<EOF
{
  "httpMethod": "POST",
  "path": "/inference/predict",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"s3_bucket\": \"anemonefish-inference-input\", \"s3_key\": \"$S3_KEY\"}",
  "isBase64Encoded": false
}
EOF
)

else
    # Test base64 method
    echo -e "${GREEN}Testing with base64 upload method...${NC}"
    
    # Convert audio to base64
    echo -e "${BLUE}Converting audio to base64...${NC}"
    AUDIO_B64=$(base64 -i "$AUDIO_FILE")
    
    # Create Lambda event
    EVENT=$(cat <<EOF
{
  "httpMethod": "POST",
  "path": "/inference/predict",
  "headers": {
    "Content-Type": "multipart/form-data"
  },
  "body": "{\"audio_file\": \"$AUDIO_B64\", \"config\": {\"confidence_threshold\": 0.5}}",
  "isBase64Encoded": false
}
EOF
)
fi

# Invoke Lambda
echo -e "${BLUE}Invoking Lambda function...${NC}"
echo -e "${YELLOW}This may take a minute...${NC}"

RESPONSE=$(curl -s -X POST "$LAMBDA_ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$EVENT")

# Save response
echo "$RESPONSE" > "$OUTPUT_FILE"

# Parse and display results
STATUS_CODE=$(echo "$RESPONSE" | jq -r '.statusCode // 500')

if [ "$STATUS_CODE" = "200" ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Success!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Display summary
    echo "$RESPONSE" | jq -r '.body' | jq . | tee "${OUTPUT_FILE/.json/_formatted.json}"
    
    # Extract key metrics
    NUM_PREDICTIONS=$(echo "$RESPONSE" | jq -r '.body' | jq '.predictions | length')
    DURATION=$(echo "$RESPONSE" | jq -r '.body' | jq -r '.metadata.audio_duration')
    PROCESSING_TIME=$(echo "$RESPONSE" | jq -r '.body' | jq -r '.metadata.processing_time')
    
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo -e "  Audio duration: ${DURATION}s"
    echo -e "  Processing time: ${PROCESSING_TIME}s"
    echo -e "  Number of predictions: ${NUM_PREDICTIONS}"
    echo ""
    echo -e "${GREEN}Full results saved to: ${OUTPUT_FILE}${NC}"
    
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Error: Status Code $STATUS_CODE${NC}"
    echo -e "${RED}========================================${NC}"
    
    echo "$RESPONSE" | jq .
    echo -e "${RED}Results saved to: ${OUTPUT_FILE}${NC}"
    exit 1
fi

