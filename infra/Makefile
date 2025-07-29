# LipVision Decoder - Makefile
.PHONY: help check build run-mediapipe run-simple stop clean logs shell dev-up dev-down jupyter

# Configurações
COMPOSE_FILE := docker-compose.yml
DEV_COMPOSE_FILE := docker-compose.dev.yml
IMAGE_NAME := lipvision-decoder:latest

# Comando padrão
help: ## Mostra esta ajuda
	@echo "🔥 LipVision Decoder - Comandos Make 🔥"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

check: ## Verifica pré-requisitos
	@./start.sh check

build: ## Constrói a imagem Docker
	@./start.sh build

run-mediapipe: ## Executa com método MediaPipe
	@./start.sh run mediapipe

run-simple: ## Executa com método simples
	@./start.sh run simple

stop: ## Para todos os containers
	@./start.sh stop

clean: ## Remove containers e imagens
	@./start.sh clean

logs: ## Mostra logs dos containers
	@./start.sh logs

shell: ## Abre shell no container
	@./start.sh shell

# Comandos de desenvolvimento
dev-up: ## Inicia ambiente de desenvolvimento
	@docker-compose -f $(DEV_COMPOSE_FILE) up -d lipvision-dev
	@echo "🚀 Ambiente de desenvolvimento iniciado!"
	@echo "Execute: make dev-shell para acessar"

dev-down: ## Para ambiente de desenvolvimento
	@docker-compose -f $(DEV_COMPOSE_FILE) down
	@echo "🛑 Ambiente de desenvolvimento parado"

dev-shell: ## Abre shell no container de desenvolvimento
	@docker exec -it lipvision-decoder-dev bash

jupyter: ## Inicia Jupyter Notebook
	@docker-compose -f $(DEV_COMPOSE_FILE) --profile jupyter up -d lipvision-jupyter
	@echo "📓 Jupyter iniciado em: http://localhost:8888"

# Comandos de manutenção
rebuild: clean build ## Reconstrói completamente a imagem

restart: stop run-mediapipe ## Reinicia com MediaPipe

# Comandos avançados
inspect: ## Inspeciona a imagem Docker
	@docker image inspect $(IMAGE_NAME)

size: ## Mostra o tamanho da imagem
	@docker image ls $(IMAGE_NAME)

prune: ## Remove recursos não utilizados do Docker
	@docker system prune -f
	@echo "🧹 Limpeza do sistema Docker concluída"
