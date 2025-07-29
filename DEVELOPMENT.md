# 🔧 Guia de Desenvolvimento - LipVision Decoder

Este documento fornece informações para desenvolvedores que desejam estender ou modificar o projeto.

## 🏗️ Arquitetura do Sistema

```
LipVision Decoder
├── Frontend (Interface de usuário)
│   ├── main.py           # Script principal
│   └── examples.py       # Exemplos de uso
├── Core (Lógica principal)
│   ├── lip_detector.py   # Detector MediaPipe
│   └── simple_lip_detector.py  # Detector Haar Cascades
├── Configuration
│   └── config.py         # Configurações centralizadas
└── Output
    ├── lip_crops/        # Recortes MediaPipe
    └── lip_crops_simple/ # Recortes método simples
```

## 🔌 Extensões Possíveis

### 1. Integração com Modelos de IA

```python
class LipReadingPipeline:
    def __init__(self):
        self.lip_detector = LipDetector()
        self.lip_reader_model = self.load_model()
    
    def load_model(self):
        # Carregar modelo de leitura labial
        # Exemplo: modelo Transformer, CNN+RNN, etc.
        pass
    
    def predict_text(self, lip_sequence):
        # Converter sequência de lábios em texto
        pass
```

### 2. Processamento de Vídeo

```python
class VideoProcessor:
    def __init__(self, detector_type='mediapipe'):
        if detector_type == 'mediapipe':
            self.detector = LipDetector()
        else:
            self.detector = SimpleLipDetector()
    
    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, lip_crop, bbox = self.detector.process_frame(frame)
            
            if lip_crop is not None:
                cv2.imwrite(f"{output_dir}/frame_{frame_count:06d}.jpg", lip_crop)
            
            frame_count += 1
        
        cap.release()
        return frame_count
```

### 3. API REST

```python
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)
detector = LipDetector()

@app.route('/detect_lips', methods=['POST'])
def detect_lips():
    # Receber imagem em base64
    image_data = request.json['image']
    
    # Processar imagem
    # ... código de processamento ...
    
    return jsonify({
        'lip_detected': True,
        'bbox': bbox,
        'lip_crop_base64': base64.b64encode(lip_crop).decode()
    })
```

### 4. Interface Gráfica com Tkinter

```python
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class LipDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LipVision Decoder")
        self.setup_ui()
        
    def setup_ui(self):
        # Criar interface gráfica
        pass
    
    def run(self):
        self.root.mainloop()
```

## 🧪 Testes

### Estrutura de Testes

```
tests/
├── test_lip_detector.py
├── test_simple_detector.py
├── test_config.py
└── fixtures/
    ├── test_images/
    └── expected_outputs/
```

### Exemplo de Teste

```python
import unittest
import cv2
import numpy as np
from lip_detector import LipDetector

class TestLipDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LipDetector()
        
    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.face_mesh)
        self.assertEqual(len(self.detector.LIPS_POINTS), 32)
    
    def test_process_frame_with_face(self):
        # Carregar imagem de teste
        test_image = cv2.imread('tests/fixtures/test_images/face.jpg')
        
        # Processar
        processed_frame, lip_crop, bbox = self.detector.process_frame(test_image)
        
        # Verificar resultados
        self.assertIsNotNone(processed_frame)
        # Adicionar mais assertivas conforme necessário
    
    def test_process_frame_without_face(self):
        # Criar imagem vazia
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Processar
        processed_frame, lip_crop, bbox = self.detector.process_frame(empty_image)
        
        # Verificar que não detectou lábios
        self.assertIsNone(lip_crop)
        self.assertIsNone(bbox)

if __name__ == '__main__':
    unittest.main()
```

## 📊 Métricas e Avaliação

### Métricas de Performance

```python
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
        
    def start_monitoring(self):
        self.start_time = time.time()
        self.frame_count = 0
    
    def log_frame(self):
        self.frame_count += 1
        
    def get_fps(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            return self.frame_count / elapsed
        return 0
    
    def get_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
```

### Avaliação de Qualidade

```python
class QualityAssessment:
    def __init__(self):
        pass
    
    def assess_lip_crop(self, lip_crop):
        """Avalia a qualidade do recorte dos lábios"""
        if lip_crop is None or lip_crop.size == 0:
            return {'quality': 0, 'reasons': ['No crop available']}
        
        metrics = {}
        
        # Verificar tamanho mínimo
        h, w = lip_crop.shape[:2]
        if h < 20 or w < 40:
            metrics['size_adequate'] = False
        else:
            metrics['size_adequate'] = True
        
        # Verificar nitidez (usando variância do Laplaciano)
        gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = laplacian_var
        metrics['is_sharp'] = laplacian_var > 100  # Threshold empírico
        
        # Verificar iluminação
        mean_brightness = np.mean(gray)
        metrics['brightness'] = mean_brightness
        metrics['well_lit'] = 30 < mean_brightness < 200
        
        # Calcular score geral
        score = 0
        if metrics['size_adequate']:
            score += 30
        if metrics['is_sharp']:
            score += 40
        if metrics['well_lit']:
            score += 30
        
        return {
            'quality_score': score,
            'metrics': metrics,
            'is_good_quality': score >= 70
        }
```

## 🔧 Otimizações

### 1. Otimização de Performance

```python
# Cache de modelos
_model_cache = {}

def get_cached_detector(detector_type):
    if detector_type not in _model_cache:
        if detector_type == 'mediapipe':
            _model_cache[detector_type] = LipDetector()
        else:
            _model_cache[detector_type] = SimpleLipDetector()
    return _model_cache[detector_type]

# Processamento multi-thread
import threading
from queue import Queue

class ThreadedProcessor:
    def __init__(self, detector):
        self.detector = detector
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_loop(self):
        while True:
            frame = self.input_queue.get()
            if frame is None:
                break
            result = self.detector.process_frame(frame)
            self.output_queue.put(result)
```

### 2. Configuração Adaptativa

```python
class AdaptiveDetector:
    def __init__(self):
        self.detector = LipDetector()
        self.performance_monitor = PerformanceMonitor()
        self.quality_assessor = QualityAssessment()
        
    def auto_adjust_settings(self, current_fps, target_fps=30):
        """Ajusta configurações automaticamente baseado na performance"""
        if current_fps < target_fps * 0.8:  # Performance baixa
            # Reduzir qualidade para melhorar performance
            self.detector.face_mesh = self.detector.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,  # Reduzir precisão
                min_detection_confidence=0.3,  # Reduzir threshold
                min_tracking_confidence=0.3
            )
        elif current_fps > target_fps * 1.2:  # Performance alta
            # Aumentar qualidade
            self.detector.face_mesh = self.detector.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Aumentar precisão
                min_detection_confidence=0.7,  # Aumentar threshold
                min_tracking_confidence=0.7
            )
```

## 📚 Recursos Adicionais

### Datasets para Treinamento
- [LRW (Lip Reading in the Wild)](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
- [LRS2 (Lip Reading Sentences 2)](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
- [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)

### Modelos Pré-treinados
- MediaPipe Face Mesh
- OpenCV Haar Cascades
- Dlib facial landmarks
- FaceAlignment (PyTorch)

### Ferramentas Úteis
- OpenCV para processamento de imagem
- MediaPipe para detecção facial
- PyTorch/TensorFlow para modelos de IA
- scikit-learn para machine learning clássico
- Jupyter Notebooks para experimentação

## 🚀 Deploy

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Serviço Systemd

```ini
[Unit]
Description=LipVision Decoder Service
After=network.target

[Service]
Type=simple
User=lipvision
WorkingDirectory=/opt/lipvision-decoder
ExecStart=/opt/lipvision-decoder/.venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📞 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Padrões de Código

- Use PEP 8 para formatação Python
- Documente funções com docstrings
- Adicione testes para novas funcionalidades
- Mantenha a compatibilidade com Python 3.8+
