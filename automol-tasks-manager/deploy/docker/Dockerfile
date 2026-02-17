# AutoMol Server Dockerfile
# Container for molecular property prediction API using original AutoMol
#
# Build from AutoMol root directory:
#   docker build -t automol-server -f automol-claude-plugin/deploy/docker/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 -v /path/to/models:/models automol-server

# Use PyTorch base image for GPU support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy AutoMol package and requirements
COPY automol/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional server dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    python-multipart>=0.0.6 \
    requests>=2.31.0

# Copy AutoMol source code
COPY automol /app/automol

# Copy deployment code
COPY automol-claude-plugin/deploy /app/deploy

# Copy automol_resources (pre-trained models)
COPY automol_resources /app/automol_resources

# Install automol package
RUN pip install --no-cache-dir -e /app/automol

# Create models directory
RUN mkdir -p /models

# Environment variables
ENV MODEL_PATH=/models/model.pt
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV LOG_LEVEL=info
ENV USE_GPU=false
# Path to automol_resources for pre-trained encoders
ENV AUTOMOL_RESOURCES=/app/automol_resources

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the server
CMD ["sh", "-c", "python -m deploy.server.app --model $MODEL_PATH --host $HOST --port $PORT --workers $WORKERS"]
