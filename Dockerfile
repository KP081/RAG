FROM python:3.12-slim

WORKDIR /app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /app

# install uv (fast python package manager)
RUN pip install uv

# Copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# Install dependencies (production only)
RUN uv sync --frozen --no-dev

# copy application code
COPY --chown=appuser:appuser app/ app/

# Expose port
EXPOSE 8000

# health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Run with Uvicorn
CMD [ "uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]