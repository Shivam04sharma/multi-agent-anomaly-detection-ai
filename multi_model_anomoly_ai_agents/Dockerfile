FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
# DOC-DEVOPS-001: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
# Copy workspace root for uv sync
COPY pyproject.toml uv.lock ./
COPY requirements.txt ./requirements.txt
# Install deps from the monorepo root pyproject.toml
RUN uv sync --group anomaly --no-dev
COPY src/ ./src/
# Copy pre-downloaded embedding model from repo
COPY models/all-MiniLM-L6-v2/ ./models/all-MiniLM-L6-v2/
# Put the virtualenv on PATH so we don't need `uv run` at runtime
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 9005
ENV PORT=9005
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
# HF_HUB_CACHE has highest priority for cache lookups (per HF docs) 
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9005"]