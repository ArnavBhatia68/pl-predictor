FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[api,xgboost]"

# Build production artifacts from the versioned match data and selected model
# configurations. Serialized models stay inside Render's image rather than
# being transferred through GitHub.
RUN python -m pl_predictor features \
    && python -m pl_predictor v4 \
        --first-fold 2018 \
        --last-fold 2023 \
        --test-season 2025 \
        --production-season 2026 \
    && python -m pl_predictor stats \
        --first-fold 2022 \
        --last-fold 2024 \
        --test-season 2025 \
        --production-season 2026

RUN mkdir -p /var/data

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn pl_predictor.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
