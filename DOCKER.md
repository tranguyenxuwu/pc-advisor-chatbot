# Docker Setup Guide

This guide will help you run the RAG Chatbot using Docker and Docker Compose.

## Prerequisites

- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### 1. Configure Environment Variables (Optional)

If you need to set OpenAI API keys:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API credentials
# Use your preferred text editor (nano, vim, code, etc.)
nano .env
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The application will be available at: **http://localhost:8501**

### 3. Alternative: Build and Run with Docker (without Compose)

```bash
# Build the image
docker build -t rag-chatbot .

# Run the container
docker run -d \
  --name rag-chatbot \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  rag-chatbot

# View logs
docker logs -f rag-chatbot

# Stop and remove the container
docker stop rag-chatbot
docker rm rag-chatbot
```

## GPU Support (NVIDIA)

If you have an NVIDIA GPU and want to use it for faster inference:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Update the `Dockerfile` to use CUDA-enabled PyTorch:

```dockerfile
# Replace the CPU PyTorch installation line with:
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Rebuild and run:

```bash
docker-compose up -d --build
```

## Useful Docker Commands

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs
docker-compose logs -f rag-chatbot

# Restart the service
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build

# Remove all containers and volumes
docker-compose down -v

# Execute commands inside the container
docker-compose exec rag-chatbot bash

# Check resource usage
docker stats rag-chatbot
```

## Troubleshooting

### Port Already in Use

If port 8501 is already in use, change it in `docker-compose.yml`:

```yaml
ports:
  - "8502:8501" # Use 8502 instead
```

### Data Persistence

The ChromaDB data is persisted in the `./data` directory which is mounted as a volume. This ensures your embeddings and data are not lost when the container is stopped.

### Model Cache

HuggingFace models are cached in a Docker volume named `huggingface-cache` to avoid re-downloading models every time the container is rebuilt.

### Memory Issues

If the container runs out of memory, you can increase Docker's memory limit in Docker Desktop settings or add memory limits to `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

## Production Deployment

For production deployment, consider:

1. Using environment variables for sensitive data
2. Setting up proper logging
3. Adding a reverse proxy (nginx)
4. Enabling HTTPS
5. Setting resource limits
6. Using Docker Swarm or Kubernetes for orchestration

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit in Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
