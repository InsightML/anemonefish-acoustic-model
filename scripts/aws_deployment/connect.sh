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
echo -e "${BLUE}Connect to Training Instance${NC}"
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

# Get instance information from Terraform
INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "")
SSH_KEY_NAME=$(grep 'ssh_key_name' terraform.tfvars | cut -d'=' -f2 | tr -d ' "' || echo "")

if [ -z "$INSTANCE_IP" ]; then
    echo -e "${RED}ERROR: Could not get instance IP from Terraform${NC}"
    exit 1
fi

if [ -z "$SSH_KEY_NAME" ]; then
    echo -e "${RED}ERROR: Could not get SSH key name from terraform.tfvars${NC}"
    exit 1
fi

# Check if key file exists
KEY_FILE="$HOME/.ssh/${SSH_KEY_NAME}.pem"
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}WARNING: Key file not found at $KEY_FILE${NC}"
    echo "Trying alternative location: $HOME/.ssh/${SSH_KEY_NAME}"
    KEY_FILE="$HOME/.ssh/${SSH_KEY_NAME}"
    if [ ! -f "$KEY_FILE" ]; then
        echo -e "${RED}ERROR: SSH key file not found${NC}"
        echo "Please specify the full path to your SSH key:"
        read -p "Key file path: " KEY_FILE
        if [ ! -f "$KEY_FILE" ]; then
            echo -e "${RED}ERROR: File not found: $KEY_FILE${NC}"
            exit 1
        fi
    fi
fi

# Ensure key has correct permissions
chmod 400 "$KEY_FILE"

# Parse command line arguments
MODE="shell"
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tensorboard)
            MODE="tensorboard"
            shift
            ;;
        --logs)
            MODE="logs"
            shift
            ;;
        --status)
            MODE="status"
            shift
            ;;
        --command)
            MODE="command"
            COMMAND="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tensorboard    Connect with TensorBoard port forwarding"
            echo "  --logs           Show training logs"
            echo "  --status         Check training status"
            echo "  --command CMD    Execute a command on the instance"
            echo "  --help           Show this help message"
            echo ""
            echo "Default: Open interactive SSH shell"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Connect based on mode
case $MODE in
    tensorboard)
        echo -e "${GREEN}Connecting with TensorBoard port forwarding...${NC}"
        echo -e "${BLUE}TensorBoard will be available at: http://localhost:6006${NC}"
        echo -e "${YELLOW}Press Ctrl+C to disconnect${NC}"
        echo ""
        ssh -i "$KEY_FILE" -L 6006:localhost:6006 ubuntu@$INSTANCE_IP
        ;;
    
    logs)
        echo -e "${GREEN}Showing training logs...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
        echo ""
        ssh -i "$KEY_FILE" ubuntu@$INSTANCE_IP "tail -f /home/ubuntu/training.log"
        ;;
    
    status)
        echo -e "${GREEN}Checking training status...${NC}"
        echo ""
        ssh -i "$KEY_FILE" ubuntu@$INSTANCE_IP << 'EOF'
echo "=========================================="
echo "Training Service Status"
echo "=========================================="
systemctl status training.service --no-pager || echo "Training service not found"

echo ""
echo "=========================================="
echo "TensorBoard Service Status"
echo "=========================================="
systemctl status tensorboard.service --no-pager || echo "TensorBoard service not found"

echo ""
echo "=========================================="
echo "GPU Status"
echo "=========================================="
nvidia-smi || echo "nvidia-smi not available"

echo ""
echo "=========================================="
echo "Disk Usage"
echo "=========================================="
df -h /

echo ""
echo "=========================================="
echo "Last 10 Lines of Training Log"
echo "=========================================="
tail -n 10 /home/ubuntu/training.log 2>/dev/null || echo "Training log not found"
EOF
        ;;
    
    command)
        echo -e "${GREEN}Executing command: $COMMAND${NC}"
        echo ""
        ssh -i "$KEY_FILE" ubuntu@$INSTANCE_IP "$COMMAND"
        ;;
    
    shell)
        echo -e "${GREEN}Opening SSH shell...${NC}"
        echo -e "${YELLOW}Type 'exit' to close the connection${NC}"
        echo ""
        ssh -i "$KEY_FILE" ubuntu@$INSTANCE_IP
        ;;
esac
