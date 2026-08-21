# Python audio / Live pipeline — Cloud Run / Docker Hub / local compose.
#
# Build (from this directory):
#   docker build -t python-service .

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ ./proto/
COPY src/ ./src/

RUN python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/feedback_ingestion.proto

ENV PORT=8080
ENV GRPC_PORT=50051
ENV METRICS_PORT=9100
ENV DESKTOP_WS_ENABLED=true

# Cloud Run ingress (WSS + /health)
EXPOSE 8080
# Internal gRPC leftover port (no inbound StreamAudio)
EXPOSE 50051
# Prometheus metrics (internal)
EXPOSE 9100

CMD ["python", "src/main.py"]
