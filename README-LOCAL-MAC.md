# 🍎 Local Development on MacBook Pro

Run the entire Python service + Ollama locally on your Mac for **free development and testing**.

**Benefits:**
- ✅ 100% FREE (no server costs)
- ✅ Fast iteration (edit code, restart, test)
- ✅ Apple Silicon (M1/M2/M3) has excellent GPU acceleration
- ✅ Test everything before deploying to production

---

## 🚀 Quick Setup (5 Minutes)

### Option A: Native Ollama + Docker Python Service (Recommended)

This runs Ollama natively on macOS (best performance) and python-service in Docker.

#### Step 1: Install Ollama

```bash
# Install via Homebrew
brew install --cask ollama

# Or download from https://ollama.ai
```

#### Step 2: Start Ollama

```bash
# Start Ollama in background
ollama serve &

# Or use the menu bar app (if installed via .dmg)
```

#### Step 3: Download Model

```bash
# For Portuguese (recommended)
ollama pull qwen2.5:7b

# Takes ~2-5 minutes (4.1GB download)
```

#### Step 4: Run Python Service in Docker

```bash
cd python-service

# Build and run with local Ollama
docker-compose -f docker-compose.yml \
  -f docker-compose.local.yml \
  up --build
```

---

### Option B: Everything in Docker (Simpler)

Run both Ollama and python-service in Docker containers.

```bash
cd python-service

# Start everything
docker-compose up -d ollama
docker exec -it ollama ollama pull qwen2.5:7b
docker-compose up --build
```

**Note**: Slightly slower on Mac due to Docker VM overhead, but simpler setup.

---

## 📋 Detailed Setup (Option A - Recommended)

### 1. Install Ollama

```bash
# Check if already installed
ollama --version

# If not installed, use Homebrew
brew install --cask ollama

# Or download from https://ollama.ai/download
```

### 2. Start Ollama

```bash
# Start Ollama (runs in background)
ollama serve &

# Verify it's running
curl http://localhost:11434/
# Should return: "Ollama is running"
```

### 3. Download Model

```bash
# List available models
ollama list

# Download recommended model for PT-BR
ollama pull qwen2.5:7b

# Wait for download (~4.1GB)
# Progress: [====>                    ] 20% (~1 min remaining)
```

### 4. Configure Environment

```bash
cd python-service

# Copy env template
cp .env.example .env

# Edit .env
cat > .env << 'EOF'
# Use local Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT=30

# Backend connection (if backend is also local)
GRPC_FEEDBACK_URL=host.docker.internal:50052

# Logging (use DEBUG for development)
LOG_LEVEL=DEBUG
EOF
```

### 5. Build and Run

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up

# Or run in background
docker-compose up -d
```

### 6. Verify Everything Works

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Test inference
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "Hello from Mac!",
  "stream": false
}'

# Check python-service logs
docker-compose logs -f audio-pipeline-service

# Expected log:
# Using Ollama LLM provider (model: qwen2.5:7b)
# Ollama connected at http://host.docker.internal:11434
```

---

## 🐛 Troubleshooting

### Docker Can't Reach Ollama

```bash
# Test from Mac
curl http://localhost:11434/

# Test from Docker container
docker run --rm alpine wget -qO- http://host.docker.internal:11434/

# If fails, use IP address instead
# Find your Mac's IP: ipconfig getifaddr en0
# Update .env:
OLLAMA_BASE_URL=http://192.168.1.X:11434
```

### Ollama Not Using GPU (Apple Silicon)

```bash
# Check if Metal GPU is active
ollama serve --help

# Should automatically use Metal on M1/M2/M3
# You'll see: "Using Metal GPU" in logs

# Verify in activity monitor
# Ollama should show "Apple GPU" usage
```

### Model Download Stuck

```bash
# Cancel and retry
ollama pull qwen2.5:7b

# Or try smaller model first
ollama pull qwen2.5:3b  # 2.3GB, faster download
```

### Python Service Can't Import Modules

```bash
# Rebuild with no cache
docker-compose build --no-cache
docker-compose up
```

---

## ⚡ Development Workflow

### Edit Code and Test

```bash
# 1. Edit code
code src/modules/text_analysis/ollama_analyzer.py

# 2. Restart service (fast)
docker-compose restart

# 3. Watch logs
docker-compose logs -f audio-pipeline-service
```

### Run Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run unit tests
cd python-service
pytest tests/ -v
```

### Check Performance

```bash
# Monitor Ollama resource usage
ollama ps

# Check Mac's GPU usage
sudo powermetrics --samplers gpu_power -i 1000 -n 10
```

---

## 📊 Performance on Mac

**Apple Silicon (M1/M2/M3):**
- qwen2.5:7b: ~50-80ms/token (Metal GPU)
- llama3.1:8b: ~60-100ms/token
- RAM usage: ~6GB

**Intel Mac:**
- qwen2.5:7b: ~150-300ms/token (CPU only)
- llama3.1:8b: ~200-400ms/token
- RAM usage: ~7GB

---

## 🎯 Testing with Real Calls

### Connect to Local Backend

If you're also running the backend locally:

```bash
# Start backend (in another terminal)
cd ../backend
npm run start:dev

# Backend should be on port 50052
# Python service connects via host.docker.internal:50052
```

### Test with Chrome Extension

```bash
# Update extension config (chrome-extension/config.json)
{
  "backendUrl": "ws://localhost:3001"
}

# Load extension in Chrome
# 1. chrome://extensions
# 2. Enable Developer Mode
# 3. Load unpacked (select chrome-extension folder)
# 4. Start a Google Meet call
# 5. Watch python-service logs for analysis
```

---

## 🔄 Switching Models

```bash
# Download different model
ollama pull llama3.1:8b

# Update .env
sed -i '' 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=llama3.1:8b/' .env

# Restart
docker-compose restart
```

---

## 🧹 Cleanup

```bash
# Stop services
docker-compose down

# Remove Ollama models (frees ~4GB)
ollama rm qwen2.5:7b

# Remove Docker images
docker-compose down --rmi all

# Remove containers
docker-compose rm -f
```

---

## 💡 Pro Tips

1. **Use DEBUG logging** during development:
   ```bash
   LOG_LEVEL=DEBUG docker-compose up
   ```

2. **Test with qwen2.5:3b first** (faster download, less RAM):
   ```bash
   ollama pull qwen2.5:3b
   ```

3. **Watch Ollama logs**:
   ```bash
   # Ollama logs (if running in foreground)
   ollama serve
   ```

4. **Use Docker volumes for persistence**:
   ```bash
   # Models persist across restarts
   docker volume ls | grep ollama
   ```

5. **Apple Silicon is 3-5x faster** than Intel for inference

---

## 📝 Example Development Session

```bash
# 1. Start Ollama (in terminal 1)
ollama serve

# 2. Start python-service (in terminal 2)
cd python-service
docker-compose up --build

# 3. Watch logs for successful connection
# Should see:
# "Ollama connected at http://host.docker.internal:11434"
# "Model 'qwen2.5:7b' is available in Ollama"

# 4. Start a test Google Meet call
# 5. Watch analysis happen in real-time
# 6. Edit code → restart → test again
```

---

## ✅ Checklist

- [ ] Ollama installed (`ollama --version`)
- [ ] Model downloaded (`ollama list`)
- [ ] `.env` configured with `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- [ ] Docker service running (`docker-compose up`)
- [ ] Logs show successful connection
- [ ] Test inference works (`curl http://localhost:11434/api/generate`)

You're now developing with **100% free LLM** on your Mac! 🎉
