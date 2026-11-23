.PHONY: help install install-all test lint format index validate-index retrieval-eval run-backend run-frontend docker-build docker-up docker-down clean

help:
	@echo "RepoGuard Development Commands:"
	@echo "  make install         Install core backend dependencies"
	@echo "  make install-all     Install backend with ML and Dev dependencies"
	@echo "  make test            Run pytest test suite"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Run ruff code formatter"
	@echo "  make run-backend     Start FastAPI backend in development mode"
	@echo "  make run-frontend    Start Vite frontend in development mode"
	@echo "  make docker-build    Build Docker containers"
	@echo "  make docker-up       Start Docker Compose services"
	@echo "  make docker-down     Stop Docker Compose services"
	@echo "  make index           Fetch pinned repos (configs/repos.yaml) and build their indexes"
	@echo "  make validate-index  Cross-check every built index against its source"
	@echo "  make retrieval-eval  Build the retrieval benchmark and evaluate retrieval + RL free signals"
	@echo "  make clean           Clean cache and temporary files"

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

test:
	pytest tests/ -v --cov=repoguard --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

index:
	python scripts/fetch_repos.py --index

validate-index:
	@for d in data/indexes/*/; do n=$$(basename $$d); \
		python scripts/validate_index.py --repo data/repos/$$n --index $$d --report reports/index_validation/$$n.json || exit 1; \
	done

retrieval-eval:
	python scripts/build_retrieval_benchmark.py
	python scripts/evaluate_retrieval.py

run-backend:
	uvicorn repoguard.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	cd frontend && npm run dev

docker-build:
	docker compose -f deployment/docker-compose.yml build

docker-up:
	docker compose -f deployment/docker-compose.yml up -d

docker-down:
	docker compose -f deployment/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

