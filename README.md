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

## Endpoints

- **gRPC**: `0.0.0.0:50051` (dentro do container)
- **gRPC**: `localhost:50051` (do host)

## Logs

O serviço loga:
- Início de cada stream de áudio
- Cada chunk recebido (com detalhes)
- Estatísticas a cada 100 chunks
- Finalização de streams
- Erros

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
