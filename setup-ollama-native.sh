#!/bin/bash
# ==========================================
# Setup Ollama Nativo no Mac
# ==========================================
# Instala Ollama nativamente para 3-5x mais performance
# ==========================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================"
echo "🚀 Ollama Native Mac Setup"
echo -e "========================================${NC}"

# Step 1: Check Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}❌ Homebrew não encontrado${NC}"
    echo "Instale com: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi
echo -e "${GREEN}✅ Homebrew encontrado${NC}"

# Step 2: Install Ollama
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama já instalado ($(ollama --version))${NC}"
else
    echo -e "${YELLOW}📦 Instalando Ollama via Homebrew...${NC}"
    brew install --cask ollama
    echo -e "${GREEN}✅ Ollama instalado${NC}"
fi

# Step 3: Start Ollama
echo -e "${YELLOW}🔧 Iniciando Ollama service...${NC}"
# Kill any existing ollama process
pkill ollama 2>/dev/null || true
sleep 2

# Start Ollama in background
ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to be ready
echo -e "${YELLOW}⏳ Aguardando Ollama inicializar...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:11434/ &> /dev/null; then
        echo -e "${GREEN}✅ Ollama está rodando${NC}"
        break
    fi
    sleep 2
done

# Verify
if ! curl -s http://localhost:11434/ &> /dev/null; then
    echo -e "${RED}❌ Ollama não inicializou. Verifique /tmp/ollama.log${NC}"
    exit 1
fi

# Step 4: Download Model
MODEL="qwen2.5:3b"
echo -e "${YELLOW}📥 Verificando modelo ${MODEL}...${NC}"

if ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo -e "${GREEN}✅ Modelo ${MODEL} já está disponível${NC}"
else
    echo -e "${YELLOW}📦 Baixando modelo ${MODEL} (~2GB, pode levar 3-10 min)...${NC}"
    ollama pull "$MODEL"
    echo -e "${GREEN}✅ Modelo ${MODEL} baixado${NC}"
fi

# Step 5: Stop Docker Ollama
echo -e "${YELLOW}🐳 Parando Ollama do Docker...${NC}"
cd "$(dirname "$0")"
docker compose -f docker-compose.local.yml stop ollama 2>/dev/null || true
echo -e "${GREEN}✅ Ollama Docker parado${NC}"

# Step 6: Restart Python Service
echo -e "${YELLOW}🔄 Reiniciando Python Service para conectar ao Ollama nativo...${NC}"
docker compose -f docker-compose.local.yml up -d --force-recreate python-service
echo -e "${GREEN}✅ Python Service reiniciado${NC}"

# Step 7: Verify
sleep 10
echo -e "${GREEN}========================================"
echo "🔍 Verificação Final"
echo -e "========================================${NC}"

echo -e "\n📊 Ollama nativo:"
curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null | head -10

echo -e "\n📊 Python Service:"
docker compose -f docker-compose.local.yml logs python-service 2>&1 | grep -E "OLLAMA|Ollama connected|LLM Provider" | tail -3

echo -e "\n${GREEN}========================================"
echo "✅ Setup Completo!"
echo -e "========================================${NC}"

echo -e "\n🎯 Ollama agora roda nativamente no Mac (mais rápido!)"
echo -e "🐳 O container Docker do Ollama está parado"
echo -e "📡 Python Service conecta via host.docker.internal:11434"
echo -e "\n🧪 Para testar, inicie uma chamada no Google Meet!"
echo -e "\n⚡ Para parar Ollama nativo quando não usar:"
echo -e "   pkill ollama"
echo -e "\n▶️  Para iniciar Ollama nativo:"
echo -e "   ollama serve &"
