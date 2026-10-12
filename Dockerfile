FROM python:3.12-slim AS models

WORKDIR /build
COPY download_models.py ./
RUN python download_models.py --output /models

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models \
    ORT_INTRA_THREADS=4 \
    ORT_INTER_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=models /models /models
RUN find /models -type d -exec chmod 0555 {} + \
    && find /models -type f -exec chmod 0444 {} + \
    && groupadd --gid 10001 joytag \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin joytag
COPY app.py LICENSE NOTICE ./

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); assert json.load(response)['loaded'] is True"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
