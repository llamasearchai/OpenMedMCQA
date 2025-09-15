.PHONY: setup test lint type fmt run datasette api docker-build docker-run precommit test-cov test-all prefetch-models prefetch-datasets

setup:
	uv pip install -e ".[dev,cli,data]"

fmt:
	ruff format src tests

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest -q

test-cov:
	pytest -q --disable-warnings --maxfail=1 --cov=medagent --cov-report=term-missing

test-all:
	tox

run:
	medagent menu

datasette:
	medagent serve-datasette

api:
	uvicorn medagent.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t medagent:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env -v $$(pwd):/app medagent:latest

precommit:
	pre-commit run -a

prefetch-models:
	python scripts/prefetch_models.py allenai/scibert_scivocab_uncased

prefetch-datasets:
	python scripts/prefetch_datasets.py medmcqa || true

