#!/bin/bash
# Deploy H-JEPA Model Server
# Usage: ./scripts/deploy.sh [OPTIONS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DEPLOYMENT_TYPE="docker"
DEVICE="cpu"
MODEL_PATH=""
PORT=8000
WORKERS=4
IMAGE_NAME="h-jepa:inference-cpu"

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE          Deployment type (docker|kubernetes|local) [default: docker]"
    echo "  -d, --device DEVICE      Device (cpu|cuda) [default: cpu]"
    echo "  -m, --model PATH         Path to model checkpoint [required]"
    echo "  -p, --port PORT          Port to expose [default: 8000]"
    echo "  -w, --workers NUM        Number of workers [default: 4]"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --type docker --device cpu --model checkpoints/model_best.pt"
    echo "  $0 --type kubernetes --device cuda --model /path/to/model.pt"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate model path
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path is required${NC}"
    usage
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

# Set image name based on device
if [ "$DEVICE" = "cuda" ]; then
    IMAGE_NAME="h-jepa:inference-gpu"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}H-JEPA Model Server Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Deployment type: ${YELLOW}$DEPLOYMENT_TYPE${NC}"
echo -e "Device: ${YELLOW}$DEVICE${NC}"
echo -e "Model path: ${YELLOW}$MODEL_PATH${NC}"
echo -e "Port: ${YELLOW}$PORT${NC}"
echo -e "Workers: ${YELLOW}$WORKERS${NC}"
echo ""

# Deploy based on type
case $DEPLOYMENT_TYPE in
    docker)
        echo -e "${GREEN}Deploying with Docker...${NC}"

        # Build Docker image
        echo -e "${YELLOW}Building Docker image...${NC}"
        docker build \
            -f Dockerfile.inference \
            --build-arg TORCH_DEVICE=$DEVICE \
            -t $IMAGE_NAME \
            .

        # Stop existing container if running
        if docker ps -a | grep -q h-jepa-server; then
            echo -e "${YELLOW}Stopping existing container...${NC}"
            docker stop h-jepa-server || true
            docker rm h-jepa-server || true
        fi

        # Run container
        echo -e "${YELLOW}Starting container...${NC}"
        if [ "$DEVICE" = "cuda" ]; then
            docker run -d \
                --name h-jepa-server \
                --runtime=nvidia \
                -e NVIDIA_VISIBLE_DEVICES=0 \
                -e MODEL_PATH=/app/models/checkpoint.pt \
                -e DEVICE=cuda \
                -v "$(dirname $MODEL_PATH):/app/models:ro" \
                -p $PORT:8000 \
                $IMAGE_NAME
        else
            docker run -d \
                --name h-jepa-server \
                -e MODEL_PATH=/app/models/checkpoint.pt \
                -e DEVICE=cpu \
                -v "$(dirname $MODEL_PATH):/app/models:ro" \
                -p $PORT:8000 \
                $IMAGE_NAME
        fi

        echo -e "${GREEN}Container started successfully!${NC}"
        echo -e "API available at: ${YELLOW}http://localhost:$PORT${NC}"
        echo -e "Documentation: ${YELLOW}http://localhost:$PORT/docs${NC}"
        ;;

    kubernetes)
        echo -e "${GREEN}Deploying to Kubernetes...${NC}"

        # Check if kubectl is available
        if ! command -v kubectl &> /dev/null; then
            echo -e "${RED}Error: kubectl not found${NC}"
            exit 1
        fi

        # Create ConfigMap for model
        echo -e "${YELLOW}Creating ConfigMap...${NC}"
        kubectl create configmap hjepa-config \
            --from-literal=device=$DEVICE \
            --from-literal=port=$PORT \
            --dry-run=client -o yaml | kubectl apply -f -

        # Apply Kubernetes manifests
        echo -e "${YELLOW}Applying Kubernetes manifests...${NC}"
        kubectl apply -f kubernetes/deployment.yaml
        kubectl apply -f kubernetes/service.yaml

        echo -e "${GREEN}Deployed to Kubernetes!${NC}"
        echo -e "Check status: ${YELLOW}kubectl get pods -l app=h-jepa${NC}"
        ;;

    local)
        echo -e "${GREEN}Deploying locally...${NC}"

        # Check if Python is available
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}Error: python3 not found${NC}"
            exit 1
        fi

        # Set environment variables
        export MODEL_PATH=$MODEL_PATH
        export DEVICE=$DEVICE

        # Run server
        echo -e "${YELLOW}Starting server...${NC}"
        python3 -m uvicorn src.serving.model_server:app \
            --host 0.0.0.0 \
            --port $PORT \
            --workers $WORKERS &

        SERVER_PID=$!
        echo $SERVER_PID > /tmp/hjepa-server.pid

        echo -e "${GREEN}Server started (PID: $SERVER_PID)${NC}"
        echo -e "API available at: ${YELLOW}http://localhost:$PORT${NC}"
        echo -e "Stop server: ${YELLOW}kill $SERVER_PID${NC}"
        ;;

    *)
        echo -e "${RED}Unknown deployment type: $DEPLOYMENT_TYPE${NC}"
        usage
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
