# 🚀 Self-Hosted Deployment Guide

Complete setup guide for running the Meet Sales Co-pilot with **100% FREE LLM** using Ollama.

## 📋 Prerequisites

- **Server**: Any machine with 8GB+ RAM (Hetzner, DigitalOcean, local server, etc.)
- **Docker**: Installed and running
- **Disk Space**: ~10GB (4GB for Ollama models + storage)
- **Network**: Ports 50051, 9100, 11434 accessible (for Docker services)

**Minimum Server Specs:**
- CPU: 2 cores (4 cores recommended)
- RAM: 8GB (16GB recommended)
- Disk: 50GB SSD
- OS: Ubuntu 20.04+ / Debian 11+

**Recommended Server (Hetzner CX31):**
- 2 vCPU, 8GB RAM, 160GB SSD
- Cost: ~€8/month
- Location: EU (GDPR compliant)

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd meet/python-service
```

### Step 2: Run Deployment Script

```bash
# First-time setup (downloads everything)
./deploy.sh
```

The script will:
1. ✅ Check Docker installation
2. ✅ Create `.env` from template
3. ✅ Start Ollama service
4. ✅ Download AI model (~4.1GB, takes 2-5 minutes)
5. ✅ Start python-service

### Step 3: Verify Installation

```bash
# Check all services are running
./deploy.sh --status

# Test Ollama is working
curl http://localhost:11434/api/tags

# Test python-service metrics
curl http://localhost:9100/metrics
```

Expected output: Services showing as "Up" and healthy.

---

## 📦 Manual Setup (If Preferred)

If you prefer manual steps:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Start Ollama
docker-compose up -d ollama

# 3. Wait for Ollama to be ready (check with)
curl http://localhost:11434/
# Should return: "Ollama is running"

# 4. Download model (Portuguese-optimized)
docker exec ollama ollama pull qwen2.5:7b

# 5. Start python-service
docker-compose up -d

# 6. Check logs
docker-compose logs -f audio-pipeline-service
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# LLM Model Selection
OLLAMA_MODEL=qwen2.5:7b  # Best for Portuguese

# Gemini cloud mode (recommended for production)
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
# Single-key mode:
GEMINI_API_KEY=your_api_key_here
# Multi-key pool (sticky by tenant_id; capacity ~= keys * GEMINI_RPM_LIMIT):
# GEMINI_API_KEYS=key1,key2,key3
GEMINI_RPM_LIMIT=12
GEMINI_RPM_WINDOW_SEC=60
GEMINI_KEY_ROUTING=tenant

# Backend Connection
GRPC_FEEDBACK_URL=localhost:50052  # Your backend server (host:port for gRPC)

# Auth for PublishFeedback (escolha A ou B)
#
# A) JWT estático (role=SERVICE), mint manual via POST /auth/service-token:
# BACKEND_SERVICE_TOKEN=eyJ...
#
# B) Auto-renovação (recomendado): mesmo SERVICE_BOOTSTRAP_KEY do backend +
#    base URL HTTP do Nest (para POST /auth/service-token). O SERVICE JWT é
#    global; o tenant por chamada vem de x-tenant-id (derivado do áudio).
# BACKEND_HTTP_BASE_URL=http://localhost:3001
# SERVICE_BOOTSTRAP_KEY=<mesmo valor do backend>
# SERVICE_TOKEN_MINT_TTL_SECONDS=3600

# Logging
LOG_LEVEL=INFO  # Use DEBUG for troubleshooting
```

### Gemini multi-key pool

Para tráfego de produção com vários tenants/reuniões simultâneas, use o pool de chaves Gemini. Guia completo: **[docs/gemini-multi-key-pool.md](../docs/gemini-multi-key-pool.md)**.

**Motivação:** a API Gemini impõe RPM por key. Com uma única key, todos os tenants disputavam o mesmo teto (~12 RPM global antes do pool), gerando filas, defer e fallback rule-based. Com N keys, a capacidade nominal sobe para **N × `GEMINI_RPM_LIMIT`** RPM.

**Comportamento:**

- `GEMINI_API_KEYS=key1,key2,key3` — um slot por key; `GEMINI_API_KEY` é ignorada quando a lista está definida.
- Roteamento **sticky** por `tenant_id` (`sha256(tenant_id) % N`) — o mesmo tenant sempre usa a mesma key.
- Cada slot tem `GeminiAnalyzer` próprio (`google-genai`), janela RPM independente e backoff 429 isolado.
- Se só `GEMINI_API_KEY` estiver definida, o modo single-key continua válido (retrocompat).

**Configuração recomendada:**

```bash
GEMINI_API_KEYS=AIzaSy...,AIzaSy...   # CSV, sem espaços; keys AI Studio (prefixo AIza)
GEMINI_RPM_LIMIT=12                   # ~80% do tier Google (free ≈ 15 RPM)
GEMINI_KEY_ROUTING=tenant
```

**Não** coloque várias keys em `GEMINI_API_KEY` separadas por vírgula — use `GEMINI_API_KEYS`. O serviço tenta split automático com warning, mas a forma correta é a variável dedicada.

**Implicações:** keys de projetos Google diferentes somam quota separada; keys do mesmo projeto compartilham limites diários. Monitore `gemini_pool_slots`, `gemini_key_calls_total{slot}` e `gemini_key_rpm_limited_total{slot}` em `:9100/metrics`.

### Multi-tenant ingress contract

- `BackendFeedbackClient` envia em cada chamada gRPC:
  - `authorization: Bearer <SERVICE JWT>` (estático em `BACKEND_SERVICE_TOKEN` ou obtido via `ServiceJwtProvider` + `/auth/service-token`)
  - `x-tenant-id: <tenant efetivo>` (obrigatório para tokens `role=SERVICE`;
    o backend valida tenant ativo e usa esse valor como tenant da chamada).
- `tenant_id` acompanha o áudio desde o `AudioChunk` de entrada até o
  `PublishFeedback` final, passando por `AudioService.process_chunk` →
  `AudioBufferService` → meta da janela → `TranscriptionChunk` →
  `BackendFeedbackEvent`. Publicar com `tenant_id` vazio é erro e gera
  `ValueError`.
- Detalhes completos do contrato em
  [`../docs/auth-architecture.md`](../docs/auth-architecture.md) e
  [`../docs/tenancy.md`](../docs/tenancy.md).

### Available Models

| Model | Size | PT-BR Quality | Speed | RAM | Command |
|-------|------|--------------|-------|-----|---------|
| **qwen2.5:7b** | 4.1GB | ⭐⭐⭐⭐⭐ | ~80ms/token | 6GB | `ollama pull qwen2.5:7b` |
| llama3.1:8b | 4.7GB | ⭐⭐⭐⭐ | ~100ms/token | 7GB | `ollama pull llama3.1:8b` |
| qwen2.5:3b | 2.3GB | ⭐⭐⭐⭐ | ~50ms/token | 4GB | `ollama pull qwen2.5:3b` |
| mistral:7b | 4.1GB | ⭐⭐⭐ | ~90ms/token | 6GB | `ollama pull mistral:7b` |

**Recommendation**: Use `qwen2.5:7b` for Portuguese sales calls.

### Switching Models

```bash
# Download new model
docker exec ollama ollama pull llama3.1:8b

# Update .env
sed -i 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=llama3.1:8b/' .env

# Restart service
docker-compose restart audio-pipeline-service
```

---

## 🔍 Monitoring & Maintenance

### Check Service Status

```bash
./deploy.sh --status
```

### View Logs

```bash
# Real-time logs
./deploy.sh --logs

# Or manually
docker-compose logs -f audio-pipeline-service
docker-compose logs -f ollama
```

### Prometheus Metrics

```bash
# View metrics
curl http://localhost:9100/metrics

# Key metrics to monitor
curl http://localhost:9100/metrics | grep -E "llm_|window_"
```

### Resource Usage

```bash
# Docker stats
docker stats --no-stream

# Disk usage (models take ~4GB each)
docker exec ollama du -sh /root/.ollama

# List downloaded models
docker exec ollama ollama list
```

### Update Models

```bash
# Pull latest version
docker exec ollama ollama pull qwen2.5:7b

# Remove old models to free space
docker exec ollama ollama rm llama3.1:8b
```

---

## 🛠️ Troubleshooting

### Service Won't Start

```bash
# Check Docker
docker ps

# Check logs
docker-compose logs ollama
docker-compose logs audio-pipeline-service

# Common issue: Out of memory
free -h  # Check available RAM
```

### Ollama Not Responding

```bash
# Restart Ollama
docker-compose restart ollama

# Test connection
curl http://localhost:11434/api/tags

# If still failing, recreate container
docker-compose down ollama
docker-compose up -d ollama
```

### Model Not Found

```bash
# Check available models
docker exec ollama ollama list

# Re-download if missing
docker exec ollama ollama pull qwen2.5:7b
```

### High Latency

```bash
# Check if using too small model for RAM
docker stats

# If RAM-limited, use smaller model
docker exec ollama ollama pull qwen2.5:3b
sed -i 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=qwen2.5:3b/' .env
docker-compose restart audio-pipeline-service
```

### Connection to Backend Failing

```bash
# Test backend connectivity
curl -v http://localhost:50052  # Should show gRPC

# Update .env with correct URL
echo "GRPC_FEEDBACK_URL=your-backend-ip:50052" >> .env
docker-compose restart audio-pipeline-service
```

---

## 🔄 Updating the Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Verify
./deploy.sh --status
```

---

## 📊 Performance Tuning

### For CPU-Only Servers (No GPU)

Use smaller models and increase timeouts:

```bash
# .env
OLLAMA_MODEL=qwen2.5:3b  # Smaller, faster
OLLAMA_TIMEOUT=60        # More time for CPU inference
```

### For Servers with GPU

Ollama auto-detects NVIDIA GPUs. Just ensure you have:
- NVIDIA drivers installed
- `nvidia-docker` or NVIDIA Container Toolkit

```bash
# Verify GPU is being used
docker exec ollama nvidia-smi
```

### Memory Optimization

If running low on RAM:

```bash
# docker-compose.yml - reduce Ollama memory
deploy:
  resources:
    limits:
      memory: 6G  # Instead of 8G

# Use smaller model
OLLAMA_MODEL=qwen2.5:3b
```

---

## 🚨 Production Checklist

Before going live:

- [ ] Server has 8GB+ RAM
- [ ] Model downloaded and verified
- [ ] `.env` configured with correct backend URL
- [ ] Services showing as healthy
- [ ] Prometheus metrics accessible
- [ ] Logs being collected
- [ ] Backup strategy for `.env` and storage
- [ ] Firewall rules configured (only expose needed ports)

---

## 📞 Common Commands

```bash
# Deploy (first time)
./deploy.sh

# Check status
./deploy.sh --status

# View logs
./deploy.sh --logs

# Restart services
./deploy.sh --restart

# Stop everything
./deploy.sh --stop

# Update to latest code
git pull && ./deploy.sh --restart
```

---

## 💡 Tips

1. **Use `qwen2.5:7b`** for Portuguese - it's the best open-source model for PT-BR
2. **Monitor RAM usage** - if consistently above 80%, switch to `qwen2.5:3b`
3. **Keep models updated** - Ollama regularly improves model quality
4. **Backup `.env`** - contains your configuration
5. **Use DEBUG logging** temporarily when troubleshooting issues

---

## 🎯 Next Steps

After deployment:

1. ✅ Verify service is working with test call
2. ✅ Set up Prometheus + Grafana (see `../prometheus-grafana-dashboard-setup.md`)
3. ✅ Configure firewall rules (only expose necessary ports)
4. ✅ Set up log rotation
5. ✅ Monitor metrics for first few days to ensure stability

---

## 📚 Additional Resources

- [Gemini multi-key pool](../docs/gemini-multi-key-pool.md)
- [Free LLM Setup Guide](free-llm-setup-ollama.md)
- [Migration Guide](migration-gemini-to-ollama.md)
- [LLM Improvements Summary](../docs/llm-improvements-summary.md)
- [Ollama Documentation](https://ollama.ai/docs)

---

## 🆘 Need Help?

1. Check logs: `./deploy.sh --logs`
2. Check status: `./deploy.sh --status`
3. Review troubleshooting section above
4. Check Ollama docs: https://ollama.ai/docs
