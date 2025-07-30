#!/usr/bin/env python3
"""
Teste de detecção de boca aberta usando 3D Saliency em tempo real.

Este teste usa 3D Saliency para detectar movimentos/mudanças na região da boca
e determinar quando ela está aberta. A saliência 3D considera mudanças temporais
entre frames consecutivos para identificar regiões de interesse.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import numpy as np
import time
from lipvision.data_collection.lip_detector import LipDetector

# Parâmetros ajustáveis para 3D Saliency
SALIENCY_THRESHOLD = 30  # Threshold para considerar região saliente
MOUTH_REGION_RATIO = 0.4  # Proporção da face que corresponde à região da boca
FRAME_BUFFER_SIZE = 3  # Número de frames para análise temporal
MOUTH_OPEN_SALIENCY_MIN = 0.15  # Saliência mínima na região da boca para considerar aberta

class SaliencyMouthDetector:
    def __init__(self):
        """Inicializa o detector de boca usando 3D Saliency"""
        self.lip_detector = LipDetector()
        self.frame_buffer = []
        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        
    def compute_3d_saliency(self, current_frame):
        """
        Computa 3D Saliency considerando mudanças temporais entre frames.
        Retorna mapa de saliência que destaca regiões com movimento/mudança.
        """
        if len(self.frame_buffer) < FRAME_BUFFER_SIZE:
            # Ainda não temos frames suficientes
            return None
            
        # Converter frames para escala de cinza
        gray_frames = []
        for frame in self.frame_buffer:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular diferenças temporais
        temporal_diff = np.zeros_like(current_gray, dtype=np.float32)
        
        for i, prev_frame in enumerate(gray_frames):
            # Peso maior para frames mais recentes
            weight = (i + 1) / len(gray_frames)
            diff = cv2.absdiff(current_gray, prev_frame).astype(np.float32)
            temporal_diff += weight * diff
        
        # Normalizar
        temporal_diff = temporal_diff / len(gray_frames)
        
        # Aplicar filtro Gaussiano para suavizar
        temporal_diff = cv2.GaussianBlur(temporal_diff, (5, 5), 0)
        
        # Combinar com saliência estática
        success, static_saliency = self.saliency.computeSaliency(current_gray)
        if success:
            static_saliency = (static_saliency * 255).astype(np.uint8)
            # Combinar saliência temporal e estática
            combined_saliency = 0.7 * temporal_diff + 0.3 * static_saliency.astype(np.float32)
        else:
            combined_saliency = temporal_diff
        
        # Normalizar para 0-255
        combined_saliency = cv2.normalize(combined_saliency, None, 0, 255, cv2.NORM_MINMAX)
        return combined_saliency.astype(np.uint8)
    
    def compute_mouth_crop_saliency(self, mouth_crop):
        """
        Computa 3D saliency especificamente para o recorte da boca.
        
        Args:
            mouth_crop: Recorte da região da boca (numpy array)
            
        Returns:
            Mapa de saliência do recorte da boca ou None se insuficientes frames
        """
        if len(self.frame_buffer) < 2:
            return None
            
        # Usar o método existente de compute_3d_saliency mas apenas no recorte
        return self.compute_3d_saliency(mouth_crop)
    
    def is_mouth_open_saliency(self, frame, debug=False):
        """
        Detecta se a boca está aberta usando 3D Saliency aplicado apenas na região do recorte da boca.
        
        Estratégia:
        1. Detecta a face e região da boca usando MediaPipe
        2. Extrai o recorte da boca
        3. Mantém buffer de recortes da boca ao longo do tempo
        4. Computa 3D Saliency apenas no recorte da boca
        5. Boca aberta = alta saliência no recorte da boca
        """
        # Detectar região da boca usando MediaPipe
        processed_frame, lip_crop, bbox = self.lip_detector.process_frame(frame.copy())
        
        if bbox is None or lip_crop is None or lip_crop.size == 0:
            return False, frame if debug else frame
        
        # Redimensionar recorte da boca para tamanho consistente
        mouth_crop_resized = cv2.resize(lip_crop, (64, 32))
        
        # Atualizar buffer de recortes da boca
        if len(self.frame_buffer) >= FRAME_BUFFER_SIZE:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(mouth_crop_resized.copy())
        
        # Computar 3D Saliency apenas no recorte da boca
        mouth_saliency_map = self.compute_mouth_crop_saliency(mouth_crop_resized)
        
        if mouth_saliency_map is None:
            return False, frame if debug else frame
        
        # Calcular métricas de saliência no recorte da boca
        mean_saliency = np.mean(mouth_saliency_map)
        max_saliency = np.max(mouth_saliency_map)
        saliency_ratio = mean_saliency / 255.0
        
        # Critério: boca aberta se saliência média no recorte for alta
        mouth_open = saliency_ratio > MOUTH_OPEN_SALIENCY_MIN
        
        if debug:
            debug_frame = frame.copy()
            
            # Desenhar bounding box da boca
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if mouth_open else (0, 0, 255)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            
            # Mostrar métricas na região da boca
            cv2.putText(debug_frame, f"Saliencia: {saliency_ratio:.3f}", 
                       (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(debug_frame, f"Max: {max_saliency}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Criar visualização do mapa de saliência do recorte da boca
            mouth_saliency_colored = cv2.applyColorMap(mouth_saliency_map, cv2.COLORMAP_HOT)
            
            # Preparar dados para janela separada da boca
            mouth_data = {
                'lip_crop': lip_crop,
                'mouth_saliency_map': mouth_saliency_map,
                'mouth_saliency_colored': mouth_saliency_colored,
                'mouth_open': mouth_open,
                'saliency_ratio': saliency_ratio,
                'max_saliency': max_saliency
            }
            
            return mouth_open, debug_frame, mouth_data
        
        return mouth_open, frame

def create_mouth_visualization(mouth_data):
    """
    Cria uma visualização detalhada da região da boca com saliência.
    
    Args:
        mouth_data: Dicionário com dados da boca (lip_crop, saliency_map, etc.)
        
    Returns:
        Imagem combinada para exibição na janela separada
    """
    # Extrair dados
    lip_crop = mouth_data['lip_crop']
    mouth_saliency_map = mouth_data['mouth_saliency_map']
    mouth_saliency_colored = mouth_data['mouth_saliency_colored']
    mouth_open = mouth_data['mouth_open']
    saliency_ratio = mouth_data['saliency_ratio']
    max_saliency = mouth_data['max_saliency']
    
    # Tamanho da visualização (maior para melhor qualidade)
    display_width = 400
    display_height = 300
    
    # Redimensionar recorte da boca
    lip_crop_resized = cv2.resize(lip_crop, (display_width//2, display_height//2))
    
    # Redimensionar mapa de saliência
    saliency_resized = cv2.resize(mouth_saliency_colored, (display_width//2, display_height//2))
    
    # Criar canvas para a visualização
    visualization = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # Posicionar recorte da boca (lado esquerdo)
    h_crop, w_crop = lip_crop_resized.shape[:2]
    y_offset = 20
    visualization[y_offset:y_offset+h_crop, 0:w_crop] = lip_crop_resized
    
    # Posicionar mapa de saliência (lado direito)
    h_sal, w_sal = saliency_resized.shape[:2]
    visualization[y_offset:y_offset+h_sal, display_width//2:display_width//2+w_sal] = saliency_resized
    
    # Adicionar labels
    cv2.putText(visualization, "Original", (10, 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(visualization, "3D Saliency", (display_width//2 + 10, 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Status da boca
    status_text = "BOCA ABERTA" if mouth_open else "BOCA FECHADA"
    status_color = (0, 255, 0) if mouth_open else (0, 0, 255)
    
    # Informações na parte inferior
    info_y_start = y_offset + max(h_crop, h_sal) + 30
    cv2.putText(visualization, status_text, (10, info_y_start), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    cv2.putText(visualization, f"Saliencia Media: {saliency_ratio:.3f}", 
               (10, info_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(visualization, f"Saliencia Maxima: {max_saliency}", 
               (10, info_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(visualization, f"Threshold: {MOUTH_OPEN_SALIENCY_MIN:.3f}", 
               (10, info_y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Linha divisória
    cv2.line(visualization, (display_width//2, 0), (display_width//2, display_height), (128, 128, 128), 1)
    
    return visualization

def test_3d_saliency_mouth_detection():
    """Testa detecção de boca aberta usando 3D Saliency"""
    print("=== TESTE: Detecção de Boca Aberta com 3D Saliency ===")
    print("🎥 Janela principal: Visualização geral com detecção")
    print("👄 Janela separada: Análise detalhada da região da boca")
    print()
    print("Controles:")
    print("Pressione 'q' para sair")
    print("Pressione 'r' para resetar buffer de frames")
    print("Pressione '+' para aumentar threshold")
    print("Pressione '-' para diminuir threshold")
    print()
    
    detector = SaliencyMouthDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera")
        return
    
    print("✅ Câmera aberta com sucesso")
    
    global MOUTH_OPEN_SALIENCY_MIN
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro: Não foi possível ler o frame da câmera")
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Espelhar horizontalmente
        
        # Detectar se a boca está aberta usando 3D Saliency
        result = detector.is_mouth_open_saliency(frame, debug=True)
        
        if len(result) == 3:  # Debug mode retorna 3 valores
            mouth_open, debug_frame, mouth_data = result
            
            # Criar janela separada para visualização detalhada da boca
            if mouth_data:
                mouth_display = create_mouth_visualization(mouth_data)
                cv2.imshow('Mouth Region Analysis', mouth_display)
        else:  # Modo normal retorna 2 valores
            mouth_open, debug_frame = result
        
        # Mostrar status e parâmetros
        status_text = "BOCA ABERTA" if mouth_open else "BOCA FECHADA"
        status_color = (0, 255, 0) if mouth_open else (0, 0, 255)
        
        cv2.putText(debug_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(debug_frame, f"Threshold: {MOUTH_OPEN_SALIENCY_MIN:.3f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Frame: {frame_count}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar instruções
        cv2.putText(debug_frame, "q:sair r:reset +:++ -:-- | Janela da boca eh redimensionavel", 
                   (10, debug_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('3D Saliency Mouth Detection', debug_frame)
        
        # Tornar a janela da boca redimensionável
        cv2.namedWindow('Mouth Region Analysis', cv2.WINDOW_NORMAL)
        
        # Processar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.frame_buffer = []
            print("🔄 Buffer de frames resetado")
        elif key == ord('+') or key == ord('='):
            MOUTH_OPEN_SALIENCY_MIN = min(1.0, MOUTH_OPEN_SALIENCY_MIN + 0.01)
            print(f"📈 Threshold aumentado para: {MOUTH_OPEN_SALIENCY_MIN:.3f}")
        elif key == ord('-') or key == ord('_'):
            MOUTH_OPEN_SALIENCY_MIN = max(0.0, MOUTH_OPEN_SALIENCY_MIN - 0.01)
            print(f"📉 Threshold diminuído para: {MOUTH_OPEN_SALIENCY_MIN:.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== TESTE FINALIZADO ===")

if __name__ == "__main__":
    test_3d_saliency_mouth_detection()
