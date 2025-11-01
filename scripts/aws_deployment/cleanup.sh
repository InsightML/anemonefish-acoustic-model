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

echo -e "${RED}========================================${NC}"
echo -e "${RED}AWS Training Infrastructure Cleanup${NC}"
echo -e "${RED}========================================${NC}"

# Check if terraform directory exists
if [ ! -d "$TERRAFORM_DIR" ]; then
    echo -e "${RED}ERROR: Terraform directory not found: $TERRAFORM_DIR${NC}"
    exit 1
fi

cd "$TERRAFORM_DIR"

# Check if Terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo -e "${YELLOW}No Terraform state found. Nothing to destroy.${NC}"
    exit 0
fi

# Parse command line arguments
SKIP_BACKUP=false
DELETE_S3_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --delete-s3-data)
            DELETE_S3_DATA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-backup       Skip downloading results before cleanup"
            echo "  --delete-s3-data    Delete all S3 data (WARNING: This is irreversible!)"
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

# Get resource information
BUCKET_NAME=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null || echo "")

echo "Resources to be destroyed:"
echo "  ? EC2 Instance: ${INSTANCE_ID:-N/A}"
echo "  ? S3 Bucket: ${BUCKET_NAME:-N/A}"
echo "  ? Security Groups, IAM Roles, and other resources"
echo ""

# Offer to backup results
if [ "$SKIP_BACKUP" = false ] && [ -n "$BUCKET_NAME" ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Backup Results Before Cleanup?${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    read -p "Do you want to download results from S3 before cleanup? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        "$SCRIPT_DIR/download_results.sh"
        echo ""
    fi
fi

# Warning about S3 data
if [ "$DELETE_S3_DATA" = true ]; then
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}WARNING: S3 Data Deletion Enabled${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${RED}This will permanently delete ALL data in the S3 bucket:${NC}"
    echo -e "${RED}  ? Training data${NC}"
    echo -e "${RED}  ? Model artifacts${NC}"
    echo -e "${RED}  ? Training logs${NC}"
    echo -e "${RED}  ? All other files${NC}"
    echo ""
    echo -e "${YELLOW}This action CANNOT be undone!${NC}"
    echo ""
    read -p "Are you absolutely sure? Type 'DELETE' to confirm: " -r
    if [[ $REPLY != "DELETE" ]]; then
        echo -e "${GREEN}Deletion cancelled${NC}"
        DELETE_S3_DATA=false
    else
        # Empty the S3 bucket before Terraform destroy
        if [ -n "$BUCKET_NAME" ]; then
            echo -e "${YELLOW}Emptying S3 bucket...${NC}"
            aws s3 rm "s3://$BUCKET_NAME" --recursive
            echo -e "${GREEN}? S3 bucket emptied${NC}"
        fi
    fi
else
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Note: S3 Bucket Data Will Be Preserved${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "The S3 bucket will be destroyed, but if it contains data,"
    echo "Terraform will fail to delete it (S3 buckets must be empty to be deleted)."
    echo ""
    echo "To also delete all S3 data, run with --delete-s3-data flag"
    echo ""
fi

# Confirm destruction
echo -e "${RED}========================================${NC}"
echo -e "${RED}Final Confirmation${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo -e "${YELLOW}This will destroy ALL AWS resources created by Terraform.${NC}"
echo -e "${YELLOW}You will be charged for resources until they are destroyed.${NC}"
echo ""
read -p "Do you want to proceed with destruction? (yes/no): " -r
echo ""
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${GREEN}Cleanup cancelled${NC}"
    exit 0
fi

# Run terraform destroy
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Terraform Destroy${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

terraform destroy

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

if [ "$DELETE_S3_DATA" = false ] && [ -n "$BUCKET_NAME" ]; then
    echo -e "\n${YELLOW}Note: If Terraform destroy failed due to non-empty S3 bucket,${NC}"
    echo -e "${YELLOW}you can manually delete it with:${NC}"
    echo -e "  aws s3 rb s3://$BUCKET_NAME --force"
fi

echo -e "\n${BLUE}All AWS resources have been destroyed.${NC}"
echo -e "${BLUE}No further charges will be incurred.${NC}"
