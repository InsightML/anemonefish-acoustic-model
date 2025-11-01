#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TERRAFORM_DIR="$WORKSPACE_DIR/terraform"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Download Training Results from S3${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if terraform directory exists
if [ ! -d "$TERRAFORM_DIR" ]; then
    echo -e "${RED}ERROR: Terraform directory not found: $TERRAFORM_DIR${NC}"
    exit 1
fi

cd "$TERRAFORM_DIR"

# Check if Terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo -e "${RED}ERROR: No Terraform state found${NC}"
    echo "Have you deployed the infrastructure? Run: ./scripts/aws_deployment/deploy.sh"
    exit 1
fi

# Get S3 bucket name from Terraform
BUCKET_NAME=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

if [ -z "$BUCKET_NAME" ]; then
    echo -e "${RED}ERROR: Could not get S3 bucket name from Terraform${NC}"
    exit 1
fi

echo "S3 Bucket: $BUCKET_NAME"
echo "AWS Region: $AWS_REGION"
echo ""

# Parse command line arguments
OUTPUT_DIR="$WORKSPACE_DIR"
DOWNLOAD_MODELS=true
DOWNLOAD_LOGS=true
DOWNLOAD_TRAINING_LOG=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models-only)
            DOWNLOAD_LOGS=false
            DOWNLOAD_TRAINING_LOG=false
            shift
            ;;
        --logs-only)
            DOWNLOAD_MODELS=false
            DOWNLOAD_TRAINING_LOG=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR    Output directory (default: workspace root)"
            echo "  --models-only       Download only model files"
            echo "  --logs-only         Download only log files"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/logs"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}ERROR: AWS CLI is not installed${NC}"
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if S3 bucket has results
echo "Checking S3 bucket for results..."
RESULTS_EXIST=$(aws s3 ls "s3://$BUCKET_NAME/results/" --region "$AWS_REGION" 2>/dev/null || echo "")

if [ -z "$RESULTS_EXIST" ]; then
    echo -e "${YELLOW}WARNING: No results found in S3 bucket${NC}"
    echo "Training may still be in progress or hasn't started yet."
    echo ""
    read -p "Do you want to check the instance status? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        "$SCRIPT_DIR/connect.sh" --status
    fi
    exit 0
fi

echo -e "${GREEN}? Results found in S3${NC}"
echo ""

# Download models
if [ "$DOWNLOAD_MODELS" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading Models${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    aws s3 sync "s3://$BUCKET_NAME/results/models/" "$OUTPUT_DIR/models/" --region "$AWS_REGION"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}? Models downloaded to: $OUTPUT_DIR/models/${NC}"
    else
        echo -e "${RED}? Error downloading models${NC}"
    fi
fi

# Download logs
if [ "$DOWNLOAD_LOGS" = true ]; then
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading TensorBoard Logs${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    aws s3 sync "s3://$BUCKET_NAME/results/logs/" "$OUTPUT_DIR/logs/" --region "$AWS_REGION"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}? Logs downloaded to: $OUTPUT_DIR/logs/${NC}"
    else
        echo -e "${RED}? Error downloading logs${NC}"
    fi
fi

# Download training log
if [ "$DOWNLOAD_TRAINING_LOG" = true ]; then
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading Training Log${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    aws s3 cp "s3://$BUCKET_NAME/results/training.log" "$OUTPUT_DIR/training.log" --region "$AWS_REGION"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}? Training log downloaded to: $OUTPUT_DIR/training.log${NC}"
    else
        echo -e "${YELLOW}? Training log not available${NC}"
    fi
    
    # Download init log if available
    aws s3 cp "s3://$BUCKET_NAME/results/init.log" "$OUTPUT_DIR/init.log" --region "$AWS_REGION" 2>/dev/null || true
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${BLUE}Results saved to: $OUTPUT_DIR${NC}"
echo ""
echo "Next steps:"
echo "  ? View models: ls -lh $OUTPUT_DIR/models/"
echo "  ? View logs: ls -lh $OUTPUT_DIR/logs/"
echo "  ? View training log: cat $OUTPUT_DIR/training.log"
echo "  ? Launch TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs/"
