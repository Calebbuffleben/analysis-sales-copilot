# Audio Pipeline Service (Python gRPC)

Serviço Python que recebe streams de áudio via gRPC do backend NestJS.

## Estrutura

```
python-service/
├── Dockerfile              # Imagem Docker do serviço
├── docker-compose.yml      # Orquestração Docker
├── requirements.txt        # Dependências Python
├── proto/
│   └── audio_pipeline.proto # Definição do protocolo gRPC
└── src/
    ├── main.py             # Entry point da aplicação
    ├── config/             # Configurações
    │   ├── settings.py      # Configurações centralizadas
    │   └── logging_config.py # Configuração de logging
    ├── handlers/            # Handlers gRPC
    │   └── audio_handler.py # Handler para streams de áudio
    ├── services/            # Lógica de negócio
    │   ├── audio_service.py # Serviço de processamento de áudio
    │   └── stream_service.py # Gerenciamento de streams
    ├── utils/               # Utilitários
    │   └── proto_utils.py   # Utilitários para código proto
    └── grpc_server/         # Servidor gRPC
        └── server.py        # Setup e inicialização do servidor
```

## Como executar

### Via Docker Compose

```bash
cd python-service

# Construir e iniciar o serviço COM LOGS NO TERMINAL (recomendado)
docker-compose up --build

# Ou em background (sem logs no terminal)
docker-compose up -d --build

# Ver logs do serviço em background
docker-compose logs -f

# Parar o serviço
docker-compose down
```

### Via Docker diretamente

```bash
cd python-service

# Construir a imagem
docker build -t audio-pipeline-service .

# Executar o container
docker run -p 50051:50051 audio-pipeline-service
```

## Variáveis de Ambiente

- `GRPC_PORT`: Porta do servidor gRPC (padrão: 50051)
- `GRPC_WORKERS`: Número de workers do ThreadPoolExecutor (padrão: 10)
- `STORAGE_DIR`: Diretório para armazenar arquivos (padrão: /app/storage)
- `LOG_LEVEL`: Nível de logging - DEBUG, INFO, WARNING, ERROR, CRITICAL (padrão: INFO)
- `GRPC_FEEDBACK_URL`: Host:porta do ingress gRPC de feedback no backend (ex.: `localhost:50052` ou `*.railway.internal:50052`)
- `AUDIO_BUFFER_WINDOW_SECONDS`, `AUDIO_BUFFER_MIN_WINDOW_SECONDS`, `AUDIO_BUFFER_MIN_INTERVAL_MS`: política de janela deslizante
- `WHISPER_VAD_FILTER`: `true`/`false` — usa filtro VAD do faster-whisper (padrão: `true`). Desligar para testar hipótese de fala removida pelo VAD.
- `WHISPER_EMPTY_DIAGNOSTIC_NO_VAD`: `true`/`false` — se `true`, quando o STT vier vazio com VAD ligado, agenda uma segunda passagem **só para log** sem VAD em thread separada (não bloqueia a próxima janela); o texto publicado não muda. Em produção, prefira `false`.
- `WINDOW_QUEUE_MAX_SIZE`: tamanho máximo da fila de janelas prontas antes do STT (padrão: `8`)
- `WINDOW_WORKER_THREADS`: threads que consomem a fila e rodam `TranscriptionPipelineService.process_window` (padrão: `2`)
- `WINDOW_MAX_AGE_MS`: descarta janelas cuja idade (`now - window_end_ms`) exceda este valor ao enfileirar ou ao processar (padrão: `25000`)
- `WINDOW_LOW_PRIORITY_SPEECH_RATIO_BELOW`: abaixo deste `speech_ratio` a janela é considerada baixa prioridade para eviction quando a fila está cheia (padrão: `0.02`)
- `PRELOAD_ML_MODELS`: `true`/`false` — se `true` (padrão), o processo carrega o Whisper e o modelo de embeddings (`sentence-transformers`) **antes** de subir o gRPC, evitando atraso de vários minutos na primeira janela (download HF + init). Em ambientes efêmeros sem cache de modelo, deixe `true`.
- `WHISPER_LOW_ENERGY_DBFS`: limiar (dBFS); janelas com RMS médio abaixo disso **não** chamam o Whisper (atalho `low_energy_precheck`), só logs — reduz fila e CPU em silêncio absoluto.
- `WHISPER_DEFAULT_LANGUAGE`: idioma opcional do Whisper (ex.: `pt`). Se definido, o serviço fixa esse idioma em todas as janelas; sem essa variável o serviço usa autodetecção e só reaproveita o último idioma bem-sucedido do stream como fallback de recuperação.

## Endpoints

- **gRPC**: `0.0.0.0:50051` (dentro do container)
- **gRPC**: `localhost:50051` (do host)

## Logs

O serviço loga:
- Início de cada stream de áudio (inclui `sample_rate`, `channels`, bytes/s do contrato s16le)
- `🔊 Window ready` — janela pronta para STT: duração, RMS, proporção de amostras “com fala”, pico (metadados incluem `speech_ratio` / `mean_rms_dbfs` para priorização)
- `📥 Window dequeue` — worker retirou a janela da fila; inclui `queue_wait_ms`
- `⏱️ Pipeline latency` — tempos aproximados de STT, análise e publish por janela
- `📝 Transcription completed` / `📝 STT empty` — resultado do Whisper com motivo aproximado quando vazio (`reason=`), idioma detectado e fallback usado
- `📝 STT skip | reason=low_energy_precheck` — RMS abaixo do limiar; Whisper não foi chamado
- `📝 STT recovered with language fallback` — janela que estava vazia na autodetecção mas recuperou texto ao repetir com idioma conhecido
- `⏭️ Pipeline skip (empty transcript)` — pipeline interrompido antes da análise quando não há texto
- `📨 Feedback published` — feedback enviado ao backend (inclui `transcript_chars` e janela em ms)
- Estatísticas a cada 100 chunks
- Finalização de streams
- Erros

### Smoke test (latência ponta a ponta)

Com backend e Python no ar, valide em logs que:

- `queue_wait_ms` em `📥 Window dequeue` permanece baixo (segundos, não minutos) sob carga moderada
- `⏱️ Pipeline latency` mostra STT como maior fatia; `total_ms` não explode sem backlog
- Janelas antigas geram `dropped` / não chegam a `📨 Feedback published` com `window_end_ms` muito anterior ao `ts` observado
- Sob fila cheia, eviction favorece manter janelas com `speech_ratio` mais alto (ver metadados em `🔊 Window ready`)

## Arquitetura

O serviço está organizado em módulos com responsabilidades bem definidas:

- **config/**: Centraliza todas as configurações da aplicação
- **handlers/**: Implementa os handlers gRPC que recebem requisições
- **services/**: Contém a lógica de negócio (processamento de áudio, gerenciamento de streams)
- **utils/**: Utilitários auxiliares (geração de código proto)
- **grpc_server/**: Setup e inicialização do servidor gRPC
- **main.py**: Entry point que orquestra a inicialização

## Próximos Passos

O serviço atualmente apenas recebe e loga os chunks de áudio. Para implementar:

1. Salvar arquivos WAV em disco
2. Enviar para serviço de transcrição
3. Processar com modelos de IA
4. Gerar feedback e eventos
