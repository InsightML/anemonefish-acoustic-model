#!/bin/bash
# Local testing script for inference Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Clean up macOS metadata files that cause xattr errors on NAS volumes
echo -e "${BLUE}Cleaning up macOS metadata files...${NC}"
find "$WORKSPACE_DIR" -name "._*" -type f -delete 2>/dev/null || true
find "$WORKSPACE_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Anemonefish Inference - Local Testing${NC}"
echo -e "${BLUE}========================================${NC}"

# Parse arguments
BUILD_FLAG=""
TEST_ONLY=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Stopping containers...${NC}"
    cd "$SCRIPT_DIR"
    docker-compose down
}

# Set trap for cleanup on exit
if [ "$CLEANUP" = true ]; then
    trap cleanup EXIT
fi

cd "$SCRIPT_DIR"

# Build and start services
if [ "$TEST_ONLY" = false ]; then
    echo -e "${GREEN}Starting services...${NC}"
    docker-compose up $BUILD_FLAG -d localstack inference-lambda
    
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10
    
    # Check if LocalStack is ready
    echo -e "${BLUE}Checking LocalStack...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:4566/_localstack/health | grep -q '"s3": "available"'; then
            echo -e "${GREEN}LocalStack is ready!${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}LocalStack failed to start${NC}"
            docker-compose logs localstack
            exit 1
        fi
        sleep 2
    done
    
    # Check if Lambda is ready
    echo -e "${BLUE}Checking Lambda container...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:9000/ | grep -q "Lambda"; then
            echo -e "${GREEN}Lambda container is ready!${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}Lambda container failed to start${NC}"
            docker-compose logs inference-lambda
            exit 1
        fi
        sleep 2
    done
fi

# Run tests
echo -e "${GREEN}Running integration tests...${NC}"
docker-compose --profile test run --rm test-runner

# Check test results
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Tests failed${NC}"
    echo -e "${RED}========================================${NC}"
    
    # Show logs
    echo -e "${YELLOW}Lambda logs:${NC}"
    docker-compose logs --tail=50 inference-lambda
    
    exit 1
fi

# Manual testing instructions
echo ""
echo -e "${BLUE}Manual Testing:${NC}"
echo "Lambda endpoint: http://localhost:9000/2015-03-31/functions/function/invocations"
echo "LocalStack S3: http://localhost:4566"
echo ""
echo "To test manually, use:"
echo "  curl -XPOST \"http://localhost:9000/2015-03-31/functions/function/invocations\" -d '{...}'"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f inference-lambda"
echo ""
echo "To stop services:"
echo "  docker-compose down"

