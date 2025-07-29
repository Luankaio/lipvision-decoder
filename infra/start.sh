#!/bin/bash

# LipVision Decoder - Script de inicialização Docker
# Este script facilita a execução do projeto usando Docker

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "🔥 ===================================== 🔥"
echo "     LipVision Decoder - Docker Setup    "
echo "🔥 ===================================== 🔥"
echo -e "${NC}"

# Função para mostrar ajuda
show_help() {
    echo "Uso: $0 [COMANDO] [OPÇÕES]"
    echo ""
    echo "Comandos:"
    echo "  build                 - Constrói a imagem Docker"
    echo "  run [mediapipe|simple] - Executa o detector (padrão: mediapipe)"
    echo "  stop                  - Para os containers"
    echo "  clean                 - Remove containers e imagens"
    echo "  logs                  - Mostra logs do container"
    echo "  shell                 - Abre shell no container"
    echo "  check                 - Verifica pré-requisitos"
    echo "  help                  - Mostra esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 build"
    echo "  $0 run mediapipe"
    echo "  $0 run simple"
    echo "  $0 check"
}

# Função para verificar pré-requisitos
check_prerequisites() {
    echo -e "${YELLOW}🔍 Verificando pré-requisitos...${NC}"
    
    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker não encontrado!${NC}"
        echo "Instale o Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker encontrado${NC}"
    
    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose não encontrado!${NC}"
        echo "Instale o Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker Compose encontrado${NC}"
    
    # Verificar se está no diretório correto
    if [ ! -f "../main.py" ]; then
        echo -e "${RED}❌ Execute este script do diretório infra/!${NC}"
        echo "cd infra && ./start.sh"
        exit 1
    fi
    echo -e "${GREEN}✅ Diretório correto${NC}"
    
    # Verificar câmera
    if [ ! -e "/dev/video0" ]; then
        echo -e "${YELLOW}⚠️  Câmera não encontrada em /dev/video0${NC}"
        echo "Verifique se a câmera está conectada"
    else
        echo -e "${GREEN}✅ Câmera encontrada${NC}"
    fi
    
    # Verificar X11 (para GUI)
    if [ -z "$DISPLAY" ]; then
        echo -e "${YELLOW}⚠️  DISPLAY não configurado${NC}"
        echo "Execute: export DISPLAY=:0"
    else
        echo -e "${GREEN}✅ DISPLAY configurado: $DISPLAY${NC}"
    fi
    
    echo -e "${GREEN}🎉 Pré-requisitos verificados!${NC}"
}

# Função para construir a imagem
build_image() {
    echo -e "${YELLOW}🔨 Construindo imagem Docker...${NC}"
    docker-compose build
    echo -e "${GREEN}✅ Imagem construída com sucesso!${NC}"
}

# Função para executar o projeto
run_project() {
    local method=${1:-mediapipe}
    
    echo -e "${YELLOW}🚀 Iniciando LipVision Decoder com método: $method${NC}"
    
    # Configurar X11 para GUI
    echo -e "${YELLOW}🔑 Configurando acesso X11...${NC}"
    
    # Permitir conexões X11 do Docker
    if command -v xhost &> /dev/null; then
        xhost +local:docker || true
        xhost +local:root || true
    fi
    
    # Verificar se DISPLAY está configurado
    if [ -z "$DISPLAY" ]; then
        echo -e "${YELLOW}⚠️  Configurando DISPLAY...${NC}"
        export DISPLAY=:0
    fi
    
    echo -e "${GREEN}✅ X11 configurado: DISPLAY=$DISPLAY${NC}"
    
    # Parar containers existentes
    docker-compose down --remove-orphans || true
    
    if [ "$method" = "simple" ]; then
        docker-compose --profile simple up lipvision-simple
    else
        docker-compose up lipvision
    fi
}

# Função para parar containers
stop_containers() {
    echo -e "${YELLOW}🛑 Parando containers...${NC}"
    docker-compose down --remove-orphans
    echo -e "${GREEN}✅ Containers parados${NC}"
}

# Função para limpar tudo
clean_all() {
    echo -e "${YELLOW}🧹 Limpando containers e imagens...${NC}"
    docker-compose down --remove-orphans --volumes
    docker rmi lipvision-decoder:latest || true
    docker system prune -f
    echo -e "${GREEN}✅ Limpeza concluída${NC}"
}

# Função para mostrar logs
show_logs() {
    docker-compose logs -f
}

# Função para abrir shell
open_shell() {
    echo -e "${YELLOW}🐚 Abrindo shell no container...${NC}"
    docker-compose run --rm lipvision bash
}

# Verificar argumentos
case "${1:-help}" in
    "check")
        check_prerequisites
        ;;
    "build")
        check_prerequisites
        build_image
        ;;
    "run")
        check_prerequisites
        run_project "$2"
        ;;
    "stop")
        stop_containers
        ;;
    "clean")
        clean_all
        ;;
    "logs")
        show_logs
        ;;
    "shell")
        open_shell
        ;;
    "help"|*)
        show_help
        ;;
esac
