FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgtk-3-dev \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    # Dependências para câmera e vídeo
    v4l-utils \
    # Dependências para áudio (sounddevice)
    libasound2-dev \
    portaudio19-dev \
    # Dependências adicionais
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Configurar ambiente
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Criar usuário não-root para segurança
RUN useradd -m -s /bin/bash lipvision && \
    usermod -a -G audio,video lipvision

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Criar diretórios necessários
RUN mkdir -p lipvision/data_collection/data/lip_crops && \
    mkdir -p lipvision/data_collection/data/lip_crops_simple && \
    chown -R lipvision:lipvision /app

# Copiar script de health check
COPY infra/healthcheck.py /app/healthcheck.py

# Mudar para usuário não-root
USER lipvision

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Comando padrão
CMD ["python", "main.py", "--method", "mediapipe"]
