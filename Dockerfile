FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.py CMakeLists.txt ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

CMD ["/bin/bash"]

