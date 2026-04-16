# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Install Cython first (required by hdbscan) and CPU-only PyTorch before
# the full requirements.txt so Docker layer caching works efficiently.
RUN pip install --no-cache-dir Cython
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Port ──────────────────────────────────────────────────────────────────────
# Railway injects $PORT at runtime; we expose 8000 as the fallback default.
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# Use shell form so the $PORT variable is expanded at runtime by the shell.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
