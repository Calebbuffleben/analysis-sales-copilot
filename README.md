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

# Backend Connection
GRPC_FEEDBACK_URL=localhost:50052  # Your backend server

# Logging
LOG_LEVEL=INFO  # Use DEBUG for troubleshooting
```

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
