FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libffi-dev \
    libssl-dev \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY proto/ ./proto/
COPY src/ ./src/

# Gerar código gRPC a partir do .proto
RUN python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/audio_pipeline.proto \
    && python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/feedback_ingestion.proto

# Expor porta gRPC
EXPOSE 50051

# Expor porta de métricas Prometheus
EXPOSE 9100

# Comando para iniciar o servidor
CMD ["python", "src/main.py"]
