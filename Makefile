# Advanced Multi-Agent RAG System Makefile

.PHONY: help install install-dev test test-coverage lint format clean run-api run-ui run-demo docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-coverage - Run tests with coverage"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean temporary files"
	@echo "  run-api      - Run API server"
	@echo "  run-ui       - Run Streamlit UI"
	@echo "  run-demo     - Run simple demo"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ *.py
	isort src/ tests/ *.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

run-api:
	python start_server.py

run-ui:
	streamlit run streamlit_app.py

run-demo:
	python simple_demo.py

run-multi-agent:
	python run_multi_agent_demo.py

docker-build:
	docker build -t advanced-rag-system .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f