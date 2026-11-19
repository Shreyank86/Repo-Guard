# RepoGuard Deployment Guide

RepoGuard is architected to progress through three distinct deployment stages.

---

## Stage 1: Local Dockerized Deployment (Current)

The local deployment builds and orchestrates the backend API and frontend static server with persistent storage.

### Prerequisites:
- Docker Engine 20+ and Docker Compose v2 installed.

### Commands:
```bash
# Build containers
docker compose -f deployment/docker-compose.yml build

# Start services in background
docker compose -f deployment/docker-compose.yml up -d

# Check status
docker compose -f deployment/docker-compose.yml ps

# View logs
docker compose -f deployment/docker-compose.yml logs -f

# Stop services
docker compose -f deployment/docker-compose.yml down
```

### Accessing Endpoints:
- Frontend Web UI: `http://localhost:3000` (or `http://localhost:5173` in local dev)
- Backend API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## Stage 2: Integrated ML Deployment (Local GPU)

For running local model inference with NVIDIA GPU acceleration (e.g. RTX 3050 Laptop GPU):
1. Install NVIDIA Container Toolkit.
2. In `.env`:
   ```env
   MOCK_MODE=false
   LLM_DEVICE=cuda
   LLM_QUANTIZATION_BITS=4
   EMBEDDING_DEVICE=cuda
   ```
3. Update `docker-compose.yml` to mount GPU resources:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

---

## Stage 3: Cloud Production Deployment

To deploy RepoGuard to the cloud cost-effectively:

1. **Frontend**:
   - Deploy as static assets to Cloudflare Pages, AWS S3 + CloudFront, or Vercel.
   - Zero compute cost for static UI hosting.

2. **Backend API**:
   - Deploy containerized FastAPI to AWS ECS Fargate, Google Cloud Run, or a lightweight VPS (Hetzner / DigitalOcean, ~\$5-\$10/month).
   - Handles AST parsing, verification, and API routing on CPU.

3. **Decoupled Model Inference**:
   - Rather than paying for an expensive 24/7 cloud GPU instance ($200+/month), use a serverless GPU inference endpoint (RunPod Serverless or Modal Labs, ~$0.20/hour active, auto-scales to 0 when idle).
   - Alternatively, use an external LLM endpoint compatible with OpenAI/vLLM format.

4. **Persistent Artifact Storage**:
   - Store generated FAISS indexes and symbol databases in an S3/GCS bucket mounted via FUSE or downloaded on startup.

