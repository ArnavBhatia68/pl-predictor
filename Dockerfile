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

# Build final artifacts from the versioned data and the already-validated
# winning configurations. Research-time walk-forward reports remain checked in.
RUN python -m pl_predictor features \
    && python -m pl_predictor.production

RUN mkdir -p /var/data

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn pl_predictor.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
