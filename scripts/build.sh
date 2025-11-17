#!/bin/bash
# Build script: Initialize submodules and build Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Initialize submodules
if [ ! -d "3rd_party/cutlass" ]; then
    echo "Initializing CUTLASS submodule..."
    git submodule add https://github.com/NVIDIA/cutlass.git 3rd_party/cutlass 2>/dev/null || true
fi
git submodule update --init --recursive

# Build Docker image
echo "Building Docker image..."
docker build -t cutlass_study:latest .

# Run container
CONTAINER_NAME="cutlass_study_dev"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    cutlass_study:latest \
    tail -f /dev/null

echo "Done! Enter container with: docker exec -it $CONTAINER_NAME bash"
