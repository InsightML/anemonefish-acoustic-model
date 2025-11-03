#!/bin/bash
# Verification script for Phase 2 Docker setup

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 2 Setup Verification${NC}"
echo -e "${BLUE}========================================${NC}"

ERRORS=0

# Check function
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found directory: $1"
    else
        echo -e "${RED}✗${NC} Missing directory: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} Executable: $1"
    else
        echo -e "${RED}✗${NC} Not executable: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

echo ""
echo -e "${BLUE}Checking Docker files...${NC}"
check_file "$SCRIPT_DIR/Dockerfile.inference"
check_file "$SCRIPT_DIR/Dockerfile.test"
check_file "$SCRIPT_DIR/docker-compose.yml"
check_file "$SCRIPT_DIR/requirements-lambda.txt"

echo ""
echo -e "${BLUE}Checking scripts...${NC}"
check_file "$SCRIPT_DIR/init-localstack.sh"
check_executable "$SCRIPT_DIR/init-localstack.sh"
check_file "$SCRIPT_DIR/test-local.sh"
check_executable "$SCRIPT_DIR/test-local.sh"
check_file "$SCRIPT_DIR/test-manual.sh"
check_executable "$SCRIPT_DIR/test-manual.sh"

echo ""
echo -e "${BLUE}Checking documentation...${NC}"
check_file "$SCRIPT_DIR/README.md"
check_file "$SCRIPT_DIR/.env.example"

echo ""
echo -e "${BLUE}Checking test files...${NC}"
check_dir "$WORKSPACE_DIR/tests/integration"
check_file "$WORKSPACE_DIR/tests/integration/__init__.py"
check_file "$WORKSPACE_DIR/tests/integration/test_inference_e2e.py"

echo ""
echo -e "${BLUE}Checking Phase 1 prerequisites...${NC}"
check_dir "$WORKSPACE_DIR/src/anemonefish_acoustics/inference"
check_file "$WORKSPACE_DIR/src/anemonefish_acoustics/inference/preprocessor.py"
check_file "$WORKSPACE_DIR/src/anemonefish_acoustics/inference/model_inference.py"
check_file "$WORKSPACE_DIR/src/anemonefish_acoustics/inference/utils.py"
check_file "$WORKSPACE_DIR/src/anemonefish_acoustics/lambda/inference_handler.py"
check_file "$WORKSPACE_DIR/config/inference_config.yaml"

echo ""
echo -e "${BLUE}Checking model files...${NC}"
if ls "$WORKSPACE_DIR/models/target_to_noise_classifier/"*/best_model.keras 1> /dev/null 2>&1; then
    MODEL_PATH=$(ls -t "$WORKSPACE_DIR/models/target_to_noise_classifier/"*/best_model.keras | head -1)
    echo -e "${GREEN}✓${NC} Found model: $MODEL_PATH"
else
    echo -e "${RED}✗${NC} No trained model found in models/target_to_noise_classifier/"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo -e "${BLUE}Checking Docker installation...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"
    DOCKER_VERSION=$(docker --version)
    echo -e "  $DOCKER_VERSION"
    
    if docker compose version &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker Compose is available"
        COMPOSE_VERSION=$(docker compose version)
        echo -e "  $COMPOSE_VERSION"
    else
        echo -e "${RED}✗${NC} Docker Compose not available"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗${NC} Docker is not installed"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo -e "${BLUE}========================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 2 setup is complete!${NC}"
    echo -e "${GREEN}All required files are in place.${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Build and test: cd docker && ./test-local.sh --build"
    echo "2. Manual testing: ./test-manual.sh --audio /path/to/audio.wav"
    echo "3. View README: cat docker/README.md"
    exit 0
else
    echo -e "${RED}✗ Setup incomplete: $ERRORS error(s) found${NC}"
    echo ""
    echo "Please ensure all required files are created."
    exit 1
fi

