# Development Setup Guide

Follow this guide to set up your local development environment for **RepoGuard**.

---

## 1. Prerequisites
- **Python**: Version 3.10 or 3.11 installed.
- **Node.js**: Version 18+ or 20+ installed.
- **Git**: Installed and configured.
- **Hardware**: Compatible with CPU-only machines as well as GPU machines (e.g. NVIDIA RTX 3050 Laptop GPU with 6GB VRAM).

---

## 2. Quickstart for Team Members

### Step 1: Clone and Environment Setup
```bash
git clone https://github.com/your-org/RepoGuard.git
cd RepoGuard

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Copy default environment configuration
cp .env.example .env
```

### Step 2: Install Dependencies
```bash
# Core dependencies (FastAPI, Pydantic, Uvicorn, AST, Pytest, Ruff)
pip install -e .

# To install dev and test tools:
pip install -e ".[dev]"

# Optional: To install heavy ML libraries (Torch, Transformers, FAISS, Sentence-Transformers)
# (Only needed when running live neural models instead of mock mode)
pip install -e ".[ml]"
```

### Step 3: Run Tests
RepoGuard is designed to pass all tests immediately out of the box using mock mode without requiring GPU downloads:
```bash
pytest tests/ -v
```

### Step 4: Run Backend API
```bash
uvicorn repoguard.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser to:
- Interactive API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Step 5: Run Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 3. Working With Mock Mode vs Live ML Mode

In `.env`:
- `MOCK_MODE=true` (Default): Uses fast, deterministic mock implementations for embeddings, vector search, LLM generation, and agent transitions. Perfect for CPU laptops, rapid unit testing, and UI integration.
- `MOCK_MODE=false`: Loads real Sentence Transformers (`all-MiniLM-L6-v2`) and HuggingFace models (`Qwen2.5-Coder`). Requires PyTorch and compatible hardware.

