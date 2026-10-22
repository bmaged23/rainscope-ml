FROM python:3.13-slim
LABEL org.opencontainers.image.source="https://github.com/bmaged23/rainscope-ml" org.opencontainers.image.licenses="GPL-3.0" org.opencontainers.image.title="RainScope ML CPU prediction API"
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY pyproject.toml LICENSE config.json ./
COPY src ./src
COPY scripts/download_data.py scripts/train_serving_model.py ./scripts/
COPY docs/upstream.json docs/ATTRIBUTION.md ./docs/
COPY deployment/example_request.json ./deployment/
RUN pip install --no-cache-dir --no-deps . \
    && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/download_data.py \
    && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/train_serving_model.py \
    && rm -rf data/raw \
    && useradd --uid 10001 --create-home appuser
USER appuser
ENV RAINSCOPE_MODEL=/app/artifacts/serving_model.joblib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"
CMD ["uvicorn", "rainscope.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
