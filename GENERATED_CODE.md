# Código Python Gerado a partir do .proto

Quando você executa o comando `grpc_tools.protoc`, ele gera **2 arquivos Python** a partir do arquivo `.proto`:

## Arquivos Gerados

### 1. `audio_pipeline_pb2.py`
Contém as **classes das mensagens** definidas no `.proto`:
- `AudioChunk` - Classe para representar um chunk de áudio
- `StreamAudioResponse` - Classe para a resposta do servidor

**Campos disponíveis:**
- `AudioChunk.meeting_id` (string)
- `AudioChunk.participant_id` (string)
- `AudioChunk.track` (string)
- `AudioChunk.wav_data` (bytes)
- `AudioChunk.sample_rate` (int32)
- `AudioChunk.channels` (int32)
- `AudioChunk.timestamp_ms` (int64)
- `AudioChunk.sequence` (int32)

### 2. `audio_pipeline_pb2_grpc.py`
Contém as **classes do serviço gRPC**:
- `AudioPipelineServiceStub` - Cliente gRPC (para fazer chamadas)
- `AudioPipelineServiceServicer` - Base para implementar o servidor

**Métodos disponíveis:**
- `StreamAudio(request_iterator, context)` - Método que você implementa no servidor

## Comando de Geração

O código é gerado automaticamente durante o build do Docker com este comando:

```bash
python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/audio_pipeline.proto
```

**Parâmetros:**
- `--proto_path`: Diretório onde estão os arquivos .proto
- `--python_out`: Onde gerar os arquivos `*_pb2.py` (mensagens)
- `--grpc_python_out`: Onde gerar os arquivos `*_pb2_grpc.py` (serviços)
- Último argumento: arquivo .proto a processar

## Onde os Arquivos São Gerados

No Dockerfile, os arquivos são gerados em `./proto/`:
- `proto/audio_pipeline_pb2.py`
- `proto/audio_pipeline_pb2_grpc.py`

## Como São Usados no Código

No `server.py`, você importa assim:

```python
import audio_pipeline_pb2          # Classes das mensagens
import audio_pipeline_pb2_grpc     # Classes do serviço
```

E usa assim:

```python
# Herdar da classe base do servidor
class AudioPipelineServicer(audio_pipeline_pb2_grpc.AudioPipelineServiceServicer):
    def StreamAudio(self, request_iterator, context):
        for chunk in request_iterator:
            # chunk é uma instância de audio_pipeline_pb2.AudioChunk
            meeting_id = chunk.meeting_id
            wav_data = chunk.wav_data
            # ...
        
        # Retornar resposta
        return audio_pipeline_pb2.StreamAudioResponse(
            success=True,
            message="OK",
            chunks_received=count
        )
```

## Verificação

Os arquivos são gerados automaticamente durante o build do Docker. Se precisar gerar manualmente (para desenvolvimento local):

```bash
cd python-service
docker run --rm -v $(pwd):/workspace python:3.11-slim bash -c "
    pip install grpcio-tools &&
    python -m grpc_tools.protoc \
        --proto_path=/workspace/proto \
        --python_out=/workspace/proto \
        --grpc_python_out=/workspace/proto \
        /workspace/proto/audio_pipeline.proto
"
```
