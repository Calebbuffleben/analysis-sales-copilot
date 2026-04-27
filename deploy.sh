#!/bin/bash
# ==========================================
# Self-Hosted Deployment Script
# ==========================================
# Automates setup of python-service with Ollama
# 100% FREE LLM inference
#
# Usage:
#   ./deploy.sh          # Full deployment
#   ./deploy.sh --setup  # First-time setup only
#   ./deploy.sh --start  # Start services
#   ./deploy.sh --status # Check status
# ==========================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================"
echo "🚀 Meet Sales Co-pilot - Self-Hosted Deploy"
echo -e "========================================${NC}"
echo ""

# ==========================================
# Functions
# ==========================================

check_docker() {
    if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker and docker-compose are required${NC}"
        echo "Install Docker: https://docs.docker.com/engine/install/"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker found${NC}"
}

setup_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}⚙️  Creating .env from template...${NC}"
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file${NC}"
        echo -e "${YELLOW}📝 Please review and edit .env if needed${NC}"
    else
        echo -e "${GREEN}✅ .env file already exists${NC}"
    fi
}

start_ollama() {
    echo -e "${YELLOW}🔧 Starting Ollama service...${NC}"
    docker-compose up -d ollama
    
    echo -e "${YELLOW}⏳ Waiting for Ollama to be ready...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:11434/ &> /dev/null; then
            echo -e "${GREEN}✅ Ollama is running${NC}"
            return 0
        fi
        sleep 2
    done
    
    echo -e "${RED}❌ Ollama failed to start${NC}"
    docker-compose logs ollama
    exit 1
}

check_model() {
    MODEL=$(grep OLLAMA_MODEL .env | cut -d'=' -f2)
    echo -e "${YELLOW}📦 Checking for model: ${MODEL}${NC}"
    
    if docker exec ollama ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo -e "${GREEN}✅ Model ${MODEL} already downloaded${NC}"
    else
        echo -e "${YELLOW}📥 Downloading model ${MODEL} (this may take a few minutes)...${NC}"
        echo -e "${YELLOW}   Model size: ~4.1GB for qwen2.5:7b${NC}"
        docker exec ollama ollama pull "$MODEL"
        echo -e "${GREEN}✅ Model downloaded successfully${NC}"
    fi
}

start_services() {
    echo -e "${YELLOW}🚀 Starting all services...${NC}"
    docker-compose up -d
    
    echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
    sleep 10
    
    # Check service status
    echo ""
    echo -e "${GREEN}========================================"
    echo "📊 Service Status"
    echo -e "========================================${NC}"
    docker-compose ps
    echo ""
    
    # Show logs
    echo -e "${GREEN}📝 Recent logs from python-service:${NC}"
    docker-compose logs --tail=20 audio-pipeline-service
    echo ""
    
    echo -e "${GREEN}========================================"
    echo "✅ Deployment Complete!"
    echo -e "========================================${NC}"
    echo ""
    echo -e "🔍 Check service health:"
    echo "   docker-compose ps"
    echo ""
    echo -e "📊 View Prometheus metrics:"
    echo "   curl http://localhost:9100/metrics"
    echo ""
    echo -e "📝 Watch logs:"
    echo "   docker-compose logs -f audio-pipeline-service"
    echo ""
    echo -e "🛑 Stop services:"
    echo "   docker-compose down"
    echo ""
}

show_status() {
    echo -e "${GREEN}========================================"
    echo "📊 Service Status"
    echo -e "========================================${NC}"
    docker-compose ps
    echo ""
    
    echo -e "${GREEN}========================================"
    echo "💾 Resource Usage"
    echo -e "========================================${NC}"
    docker stats --no-stream
    echo ""
    
    echo -e "${GREEN}========================================"
    echo "🧠 Ollama Models"
    echo -e "========================================${NC}"
    docker exec ollama ollama list 2>/dev/null || echo -e "${RED}Ollama not running${NC}"
    echo ""
    
    echo -e "${GREEN}========================================"
    echo "📝 Recent Logs"
    echo -e "========================================${NC}"
    docker-compose logs --tail=30 audio-pipeline-service
}

# ==========================================
# Main Script
# ==========================================

case "${1:-deploy}" in
    --setup|setup)
        echo -e "${YELLOW}Running first-time setup...${NC}"
        check_docker
        setup_env
        start_ollama
        check_model
        echo -e "${GREEN}✅ Setup complete! Run './deploy.sh --start' to start services${NC}"
        ;;
    
    --start|start)
        echo -e "${YELLOW}Starting services...${NC}"
        check_docker
        start_services
        ;;
    
    --status|status)
        show_status
        ;;
    
    --stop|stop)
        echo -e "${YELLOW}Stopping services...${NC}"
        docker-compose down
        echo -e "${GREEN}✅ Services stopped${NC}"
        ;;
    
    --restart|restart)
        echo -e "${YELLOW}Restarting services...${NC}"
        docker-compose down
        docker-compose up -d
        echo -e "${GREEN}✅ Services restarted${NC}"
        ;;
    
    --logs|logs)
        docker-compose logs -f audio-pipeline-service
        ;;
    
    --deploy|deploy|"")
        echo -e "${YELLOW}Running full deployment...${NC}"
        check_docker
        setup_env
        start_ollama
        check_model
        start_services
        ;;
    
    --help|help|-h)
        echo -e "${GREEN}Usage:${NC}"
        echo "  ./deploy.sh              # Full deployment (setup + start)"
        echo "  ./deploy.sh --setup      # First-time setup only"
        echo "  ./deploy.sh --start      # Start services"
        echo "  ./deploy.sh --status     # Check service status"
        echo "  ./deploy.sh --stop       # Stop services"
        echo "  ./deploy.sh --restart    # Restart services"
        echo "  ./deploy.sh --logs       # Watch logs"
        echo ""
        echo -e "${GREEN}First time? Run:${NC} ./deploy.sh"
        ;;
    
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Run './deploy.sh --help' for usage"
        exit 1
        ;;
esac
