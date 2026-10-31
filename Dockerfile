# Sketch2Shape: CPU image by default; GPU via docker-compose (the `gpu` profile).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf U2NET_HOME=/models/u2net OUTPUT_DIR=/app/outputs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 fonts-dejavu-core libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
ARG GPU=0
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$GPU" = "1" ]; then pip install --no-cache-dir -r requirements-gpu.txt; fi
COPY . .
# bake model weights into the image layer cache (skip with --build-arg PREFETCH=0)
ARG PREFETCH=1
RUN if [ "$PREFETCH" = "1" ]; then python scripts/download_models.py $( [ "$GPU" = "1" ] && echo --gpu ); fi
EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
