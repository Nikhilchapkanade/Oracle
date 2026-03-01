FROM python:3.11-slim AS base

LABEL maintainer="ORACLE Team"
LABEL description="ORACLE — Predictive Viral Evolution Engine"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ─── Dependencies ───
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ─── Application ───
COPY . .

# ─── Expose ports ───
EXPOSE 8501 8100 8101 8102 8103

# ─── Health check ───
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ─── Default: run dashboard ───
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
