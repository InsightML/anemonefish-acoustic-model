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
echo -e "${BLUE}AWS Training Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}ERROR: Terraform is not installed${NC}"
    echo "Please install Terraform: https://www.terraform.io/downloads"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}ERROR: AWS CLI is not installed${NC}"
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if terraform.tfvars exists
if [ ! -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
    echo -e "${RED}ERROR: terraform.tfvars not found${NC}"
    echo "Please create terraform.tfvars from terraform.tfvars.example"
    echo ""
    echo "Steps:"
    echo "  1. cd $TERRAFORM_DIR"
    echo "  2. cp terraform.tfvars.example terraform.tfvars"
    echo "  3. Edit terraform.tfvars with your AWS configuration"
    exit 1
fi

# Parse command line arguments
SKIP_UPLOAD=false
SKIP_TERRAFORM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --skip-terraform)
            SKIP_TERRAFORM=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-upload      Skip uploading data to S3"
            echo "  --skip-terraform   Skip Terraform apply"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get bucket name from Terraform vars
cd "$TERRAFORM_DIR"
BUCKET_NAME=$(terraform output -raw s3_bucket_name 2>/dev/null || grep 's3_bucket_name' terraform.tfvars | cut -d'=' -f2 | tr -d ' "' || echo "")

# Step 1: Upload data to S3
if [ "$SKIP_UPLOAD" = false ]; then
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 1: Uploading Data to S3${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Check if bucket name is set
    if [ -z "$BUCKET_NAME" ]; then
        echo -e "${YELLOW}WARNING: S3 bucket name not set in terraform.tfvars${NC}"
        echo "Terraform will create a bucket with auto-generated name"
        echo "Skipping data upload. You can upload later after Terraform creates the bucket."
    else
        # Check if Python script exists
        if [ ! -f "$SCRIPT_DIR/upload_data_to_s3.py" ]; then
            echo -e "${RED}ERROR: upload_data_to_s3.py not found${NC}"
            exit 1
        fi
        
        # Run upload script
        python3 "$SCRIPT_DIR/upload_data_to_s3.py" \
            --bucket "$BUCKET_NAME" \
            --workspace-dir "$WORKSPACE_DIR" \
            --region "$(grep 'aws_region' terraform.tfvars | cut -d'=' -f2 | tr -d ' "' || echo 'us-east-1')"
    fi
else
    echo -e "\n${YELLOW}Skipping data upload (--skip-upload flag)${NC}"
fi

# Step 2: Initialize Terraform
if [ "$SKIP_TERRAFORM" = false ]; then
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 2: Initializing Terraform${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    cd "$TERRAFORM_DIR"
    terraform init
    
    # Step 3: Validate Terraform configuration
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 3: Validating Terraform Configuration${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    terraform validate
    
    # Step 4: Plan Terraform deployment
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 4: Planning Terraform Deployment${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    terraform plan
    
    # Step 5: Apply Terraform configuration
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 5: Applying Terraform Configuration${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    echo -e "${YELLOW}This will create AWS resources and incur costs.${NC}"
    read -p "Do you want to proceed? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo -e "${RED}Deployment cancelled${NC}"
        exit 1
    fi
    
    terraform apply
    
    # Step 6: Wait for instance to be ready
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 6: Waiting for Instance to Initialize${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    INSTANCE_ID=$(terraform output -raw instance_id)
    echo "Instance ID: $INSTANCE_ID"
    echo "Waiting for instance to be running..."
    
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    echo -e "${GREEN}? Instance is running${NC}"
    
    echo "Waiting for status checks to pass (this may take a few minutes)..."
    aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
    echo -e "${GREEN}? Instance status checks passed${NC}"
    
    # Step 7: Display connection information
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\n${BLUE}Connection Information:${NC}"
    terraform output connection_instructions
    
    echo -e "\n${YELLOW}Training has started automatically on the instance.${NC}"
    echo -e "${YELLOW}Monitor progress with:${NC}"
    echo -e "  ${BLUE}./scripts/aws_deployment/connect.sh${NC}"
    echo -e "  ${BLUE}tail -f /home/ubuntu/training.log${NC}"
    
    echo -e "\n${YELLOW}To download results after training completes:${NC}"
    echo -e "  ${BLUE}./scripts/aws_deployment/download_results.sh${NC}"
    
    echo -e "\n${YELLOW}To destroy all resources when done:${NC}"
    echo -e "  ${BLUE}./scripts/aws_deployment/cleanup.sh${NC}"
    
else
    echo -e "\n${YELLOW}Skipping Terraform apply (--skip-terraform flag)${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Script Complete${NC}"
echo -e "${GREEN}========================================${NC}"
