# 🎥 LipVision Decoder

Multimodal pipeline for lip reading: from visual input to decoded textual output using LLMs.

Este projeto implementa um sistema de detecção e recorte de lábios em tempo real usando visão computacional com OpenCV. O sistema serve como base para desenvolvimento de aplicações de leitura labial.

## 🚀 Funcionalidades

- **Detecção de lábios em tempo real** usando câmera
- **Dois métodos de detecção**:
  - **MediaPipe**: Detecção precisa usando landmarks faciais (recomendado)
  - **Haar Cascades**: Método mais simples e rápido
- **Captura e salvamento** de recortes dos lábios
- **Interface visual** com feedback em tempo real
- **Visualização** do recorte detectado

## 📋 Requisitos

- Python 3.8+
- Câmera (webcam)
- OpenCV
- MediaPipe (para método avançado)
- NumPy

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone <repository-url>
cd lipvision-decoder
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🐳 Execução com Docker (Recomendado)

O projeto inclui suporte completo ao Docker para facilitar a execução sem configuração manual do ambiente.

### Pré-requisitos Docker
- Docker instalado
- Docker Compose instalado
- Câmera conectada
- Sistema com interface gráfica (X11 no Linux)

### Passos para Executar

1. **Navegue para o diretório Docker:**
```bash
cd infra
```

2. **Verifique os pré-requisitos:**
```bash
./start.sh check
```

3. **Construa a imagem Docker:**
```bash
./start.sh build
```

4. **Execute o projeto:**

**Método MediaPipe (Recomendado):**
```bash
./start.sh run mediapipe
```

**Método Simples (Mais Rápido):**
```bash
./start.sh run simple
```

### Comandos Úteis Docker

- **Parar containers:**
```bash
./start.sh stop
```

- **Ver logs em tempo real:**
```bash
./start.sh logs
```

- **Abrir shell no container:**
```bash
./start.sh shell
```

- **Limpar tudo (containers e imagens):**
```bash
./start.sh clean
```

- **Ajuda:**
```bash
./start.sh help
```

### Vantagens do Docker
- ✅ **Ambiente isolado** - Não interfere no sistema host
- ✅ **Dependências gerenciadas** - Todas as bibliotecas incluídas
- ✅ **Configuração automática** - X11 e câmera configurados automaticamente
- ✅ **Portabilidade** - Funciona em qualquer sistema com Docker
- ✅ **Fácil limpeza** - Remove tudo com um comando

## 🎯 Uso (Execução Nativa)

### Método Principal (Recomendado)
```bash
python main.py --method mediapipe
```

### Método Simples (Mais Rápido)
```bash
python main.py --method simple
```

### Execução Direta dos Detectores

**Detector MediaPipe (Avançado):**
```bash
python lip_detector.py
```

**Detector Simples:**
```bash
python simple_lip_detector.py
```

## ⌨️ Controles

Durante a execução:
- **`c`**: Capturar e salvar recorte dos lábios
- **`q`**: Sair do programa

## 📁 Estrutura do Projeto

```
lipvision-decoder/
├── main.py                    # Script principal
├── lip_detector.py           # Detector MediaPipe (avançado)
├── simple_lip_detector.py    # Detector Haar Cascades (simples)
├── requirements.txt          # Dependências
├── infra/                    # Configurações Docker
│   ├── start.sh             # Script de controle Docker
│   ├── Dockerfile           # Imagem Docker
│   └── docker-compose.yml   # Orquestração de containers
├── lipvision/               # Módulos do projeto
│   └── data_collection/     # Coleta e processamento de dados
│       ├── data/           # Dados salvos
│       │   ├── lip_crops/        # Recortes MediaPipe
│       │   └── lip_crops_simple/ # Recortes método simples
│       ├── lip_detector.py       # Detector MediaPipe
│       └── simple_lip_detector.py # Detector simples
└── README.md               # Documentação
```

## 🔧 Métodos de Detecção

### 1. MediaPipe (Recomendado)
- **Precisão**: Alta
- **Performance**: Moderada
- **Características**:
  - Usa 468 landmarks faciais
  - Detecção precisa dos contornos dos lábios
  - Melhor para aplicações que requerem alta precisão

### 2. Haar Cascades (Simples)
- **Precisão**: Moderada
- **Performance**: Alta
- **Características**:
  - Detecção de face + estimativa da região da boca
  - Mais rápido e com menor uso de recursos
  - Adequado para prototipagem rápida

## 🖼️ Saída

Os recortes dos lábios são salvos automaticamente quando você pressiona `c`:
- **MediaPipe**: `lip_crops/lip_crop_YYYYMMDD_HHMMSS_mmm.jpg`
- **Simples**: `lip_crops_simple/mouth_crop_YYYYMMDD_HHMMSS_mmm.jpg`

## 🚀 Próximos Passos

Este projeto serve como base para:
- Sistemas de leitura labial
- Análise de movimento dos lábios
- Aplicações de acessibilidade
- Interfaces multimodais
- Integração com modelos de linguagem (LLMs)

## 🐛 Solução de Problemas

### Problemas com Docker

**Erro de permissão X11:**
```
qt.qpa.xcb: could not connect to display :0
```
**Solução:**
```bash
export DISPLAY=:0
xhost +local:docker
./start.sh run mediapipe
```

**Câmera não acessível no Docker:**
```
❌ Erro: Câmera não encontrada ou não acessível
```
**Soluções:**
- Verifique se a câmera está em `/dev/video0`
- Execute com privilégios: o script já configura automaticamente
- Use `./start.sh check` para diagnosticar

### Problemas de Execução Nativa

### Câmera não encontrada
```
❌ Erro: Câmera não encontrada ou não acessível
```
**Soluções:**
- Verifique se a câmera está conectada
- Feche outros programas que possam estar usando a câmera
- Verifique as permissões de acesso à câmera

### Erro de importação
```
❌ Erro de importação: No module named 'cv2'
```
**Solução:**
```bash
pip install -r requirements.txt
```

### Performance lenta
- Use o método `simple` para melhor performance:
```bash
python main.py --method simple
```

## 📄 Licença

Este projeto está licenciado sob a licença especificada no arquivo LICENSE.
