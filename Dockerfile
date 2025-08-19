# ===== Dockerfile (FINAL) =====
FROM python:3.11-slim-bookworm
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the Docker (CPU) environment
COPY requirements.docker.txt .
RUN pip install --upgrade pip
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DLLAMA_METAL=OFF -DLLAMA_OPENBLAS=ON" pip install --no-cache-dir -r requirements.docker.txt

CMD ["bash"]