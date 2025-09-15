# syntax=docker/dockerfile:1

# Build stage (installs with dev extras to compile anything needed)
FROM python:3.11-slim as build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl jq && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml hatch.toml tox.ini Makefile /app/
COPY src /app/src
COPY scripts /app/scripts
RUN pip install --upgrade pip && pip install uv && uv pip install -e ".[dev,cli,data]"

# Runtime stage (minimal)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 appuser
WORKDIR /app

# Copy installed site-packages from build image for speed
COPY --from=build /usr/local /usr/local
COPY pyproject.toml hatch.toml tox.ini Makefile /app/
COPY src /app/src
COPY scripts /app/scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENV LIGHT_TESTS=1 \
    MEDAGENT_TEMPERATURE=0.0

USER appuser

CMD ["uvicorn", "medagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
