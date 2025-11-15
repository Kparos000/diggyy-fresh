# Dockerfile for training RL agent
# Usage: docker build -f docker/Dockerfile -t diggyy-fresh-train .
#        docker run -v $(pwd)/models:/app/models diggyy-fresh-train

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create output directories
RUN mkdir -p models logs data/processed

# Generate synthetic data
RUN python src/synthetic_data_generator.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: train agent
CMD ["python", "src/agent.py"]
