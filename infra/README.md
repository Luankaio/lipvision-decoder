# 🐳 LipVision Decoder - Infraestrutura Docker

Este diretório contém toda a infraestrutura Docker para executar o LipVision Decoder de forma containerizada.

## 🚀 Início Rápido

### 1. Verificar Pré-requisitos
```bash
cd infra
./start.sh check
```

### 2. Construir a Imagem
```bash
./start.sh build
```

### 3. Executar o Projeto
```bash
# Método MediaPipe (padrão)
./start.sh run mediapipe

# Método simples (Haar Cascades)
./start.sh run simple
```

## 📋 Comandos Disponíveis

| Comando | Descrição |
|---------|-----------|
| `./start.sh check` | Verifica pré-requisitos do sistema |
| `./start.sh build` | Constrói a imagem Docker |
| `./start.sh run [mediapipe\|simple]` | Executa o detector |
| `./start.sh stop` | Para os containers |
| `./start.sh clean` | Remove containers e imagens |
| `./start.sh logs` | Mostra logs do container |
| `./start.sh shell` | Abre shell no container |
| `./start.sh help` | Mostra ajuda |

## 🛠️ Desenvolvimento

### Modo Desenvolvimento (Hot Reload)
```bash
# Iniciar container de desenvolvimento
docker-compose -f docker-compose.dev.yml up -d lipvision-dev

# Executar comandos no container
docker exec -it lipvision-decoder-dev python main.py --method mediapipe

# Abrir shell para desenvolvimento
docker exec -it lipvision-decoder-dev bash
```

### Jupyter Notebook
```bash
# Iniciar Jupyter para experimentação
docker-compose -f docker-compose.dev.yml --profile jupyter up lipvision-jupyter

# Acesse: http://localhost:8888
```

## 🔧 Configuração

### Câmera
O container mapeia automaticamente `/dev/video0`. Se sua câmera estiver em outro dispositivo, edite o `docker-compose.yml`:

```yaml
devices:
  - "/dev/video1:/dev/video0"  # Mapear video1 para video0 no container
```

### Display (GUI)
Para funcionar corretamente com GUI:

1. **Linux**: Execute `xhost +local:docker` antes de iniciar
2. **macOS**: Instale XQuartz e configure `DISPLAY=host.docker.internal:0`
3. **Windows**: Use VcXsrv ou similar

### Volumes Persistentes
Os dados capturados são salvos em:
- `../lipvision/data_collection/data/` (mapeado para o host)

## 📁 Estrutura dos Arquivos

```
infra/
├── Dockerfile              # Imagem principal
├── docker-compose.yml      # Configuração de produção
├── docker-compose.dev.yml  # Configuração de desenvolvimento
├── start.sh                # Script de inicialização
├── .dockerignore           # Arquivos ignorados no build
└── README.md               # Esta documentação
```

## 🐛 Solução de Problemas

### Câmera não encontrada
```bash
# Verificar dispositivos de vídeo
ls /dev/video*

# Verificar permissões
sudo usermod -a -G video $USER
```

### Problemas com GUI
```bash
# Permitir acesso ao X11
xhost +local:docker

# Verificar DISPLAY
echo $DISPLAY
```

### Container não inicia
```bash
# Verificar logs
./start.sh logs

# Limpar e reconstruir
./start.sh clean
./start.sh build
```

### Problemas de permissão
```bash
# Ajustar permissões dos dados
sudo chown -R $USER:$USER ../lipvision/data_collection/data/
```

## 🔒 Segurança

- O container roda com usuário não-root (`lipvision`)
- Acesso à câmera é restrito ao dispositivo mapeado
- Volumes são mapeados com permissões específicas

## 📈 Performance

### Otimizações incluídas:
- Multi-stage build para imagem menor
- Cache de dependências Python
- Apenas dependências necessárias do sistema
- Usuário não-root para segurança

### Monitoramento:
```bash
# Ver uso de recursos
docker stats lipvision-decoder

# Ver logs em tempo real
./start.sh logs
```

## 🤝 Contribuindo

Para contribuir com melhorias na infraestrutura:

1. Teste suas mudanças localmente
2. Documente novas configurações
3. Mantenha compatibilidade com diferentes sistemas
4. Atualize este README se necessário

## 📄 Licença

A infraestrutura Docker segue a mesma licença do projeto principal.
