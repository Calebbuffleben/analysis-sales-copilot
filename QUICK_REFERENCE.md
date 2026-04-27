# ⚡ Quick Reference Card - Self-Hosted Deployment

## 🚀 First Time Setup (3 commands)

```bash
git clone <repo> && cd meet/python-service
./deploy.sh
```

That's it! Everything else is automatic.

---

## 📋 Daily Operations

```bash
# Check if everything is running
./deploy.sh --status

# Watch live logs
./deploy.sh --logs

# Restart services
./deploy.sh --restart
```

---

## 🔧 Common Tasks

### Change Model
```bash
docker exec ollama ollama pull qwen2.5:3b
sed -i 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=qwen2.5:3b/' .env
./deploy.sh --restart
```

### Check Resource Usage
```bash
docker stats --no-stream
docker exec ollama du -sh /root/.ollama
```

### View Metrics
```bash
curl http://localhost:9100/metrics | grep llm_
```

---

## 🆘 Emergency Fixes

### Service Down
```bash
./deploy.sh --restart
```

### Out of Memory
```bash
# Switch to smaller model
sed -i 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=qwen2.5:3b/' .env
./deploy.sh --restart
```

### Ollama Stuck
```bash
docker-compose down ollama
docker-compose up -d ollama
```

---

## 📊 Model Quick Reference

| Model | Quality | Speed | RAM | Use Case |
|-------|---------|-------|-----|----------|
| **qwen2.5:7b** | ⭐⭐⭐⭐⭐ | Medium | 6GB | **Default (PT-BR)** |
| qwen2.5:3b | ⭐⭐⭐⭐ | Fast | 4GB | Low RAM servers |
| llama3.1:8b | ⭐⭐⭐⭐ | Medium | 7GB | English calls |

---

## 🔍 Log Patterns to Watch

```bash
# Good - LLM working
grep "Ollama analysis" logs

# Warning - Fallback activated (check why)
grep "fallback" logs

# Error - Service issues
grep "ERROR" logs
```

---

## 💡 Pro Tips

1. ✅ **Keep default model** unless you have RAM issues
2. ✅ **Monitor metrics** weekly at minimum
3. ✅ **Update monthly**: `git pull && ./deploy.sh --restart`
4. ✅ **Backup `.env`** to secure location
5. ✅ **Use DEBUG** temporarily for troubleshooting

---

## 📞 Need Help?

1. `./deploy.sh --status`
2. `./deploy.sh --logs`
3. Check [README.md](README.md) troubleshooting section
