# Video Retrieval System - Docker Image
# Base: NVIDIA CUDA for GPU acceleration
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
 python3.10 \
 python3-pip \
 python3-dev \
 ffmpeg \
 libsm6 \
 libxext6 \
 libgl1-mesa-glx \
 libglib2.0-0 \
 curl \
 && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
 ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/videos /app/data/keyframes /app/data/cache

# Expose API port
EXPOSE 8000

# Set environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV GPU_ENABLED=true
ENV USE_FP16=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
 CMD curl -f http://localhost:8000/api/health || exit 1

# Run the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
