import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import cv2
import time
import numpy as np
from lipvision.data_collection.lip_detector import LipDetector



EXTRACTION_DIR = os.path.join(os.path.dirname(__file__), 'extraction')
os.makedirs(EXTRACTION_DIR, exist_ok=True)


# Parâmetros ajustáveis:
# Distância mínima (em pixels) entre os landmarks centrais dos lábios para considerar a boca aberta
MOUTH_OPEN_PIXEL_DISTANCE = 3
# Janela de silêncio (em segundos) após fechar a boca para encerrar a gravação
POST_SILENCE_WINDOW = 0.3
# Tempo de pre-gravação (em segundos) antes da detecção da abertura da boca
PRE_DETECTION_BUFFER = 0.15


# Nova versão: captura da câmera, salva recortes de "fala" em extraction

from lipvision.data_collection.simple_lip_detector import SimpleLipDetector

class SpeakingExtractor:
    def __init__(self, method='mediapipe', post_silence_window=POST_SILENCE_WINDOW, 
                 pre_detection_buffer=PRE_DETECTION_BUFFER, camera_index=0, fps=30):
        if method == 'mediapipe':
            self.detector = LipDetector()
        elif method == 'simple':
            self.detector = SimpleLipDetector()
        else:
            raise ValueError(f"Método desconhecido: {method}")
        self.post_silence_window = post_silence_window
        self.camera_index = camera_index
        self.fps = fps
        
        # Buffer circular para gravar antes da detecção
        self.pre_detection_seconds = pre_detection_buffer
        self.pre_buffer_size = int(self.fps * self.pre_detection_seconds)
        self.frame_buffer = []  # Buffer circular de frames
        self.lip_crop_buffer = []  # Buffer circular de recortes da boca

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return

        print("Iniciando extração de segmentos de fala (pressione 'q' para sair)")
        print(f"Pre-buffer: {self.pre_detection_seconds}s ({self.pre_buffer_size} frames)")
        
        speaking = False
        post_silence_counter = 0
        silence_frames = int(self.post_silence_window * self.fps)
        segment_id = 1
        writer = None
        current_output_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame da câmera")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detectar se a boca está aberta e obter recorte
            mouth_open, debug_frame, lip_crop = self._is_mouth_open(frame, debug=True)
            
            # Sempre adicionar frame e recorte ao buffer circular
            self.add_frame_to_buffer(frame, lip_crop)

            if mouth_open and not speaking:
                # Começou a falar - iniciar gravação COM pre-buffer
                speaking = True
                post_silence_counter = 0
                writer, current_output_path = self.start_recording_with_prebuffer(segment_id)

            if speaking:
                if writer is not None and lip_crop is not None and lip_crop.size > 0:
                    writer.write(lip_crop)  # Gravar apenas o recorte da boca

                if not mouth_open:
                    post_silence_counter += 1
                    if post_silence_counter >= silence_frames:
                        # Parou de falar - finalizar gravação
                        speaking = False
                        post_silence_counter = 0
                        if writer is not None:
                            writer.release()
                            writer = None
                        print(f"💾 Segmento de boca limpo salvo: {current_output_path}")
                        segment_id += 1
                else:
                    post_silence_counter = 0

            # Mostrar visualização
            status = "FALANDO" if speaking else "SILENCIO"
            color = (0, 255, 0) if mouth_open else (0, 0, 255)
            cv2.putText(debug_frame, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(debug_frame, f"Buffer: {len(self.frame_buffer)}/{self.pre_buffer_size} frames", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"Pre-buffer: {self.pre_detection_seconds}s", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"Gravando: recortes limpos (sem pontos)", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if speaking and post_silence_counter > 0:
                remaining_frames = silence_frames - post_silence_counter
                cv2.putText(debug_frame, f"Encerrando em: {remaining_frames} frames", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Speaking Extraction - Face', debug_frame)
            
            # Mostrar recorte da boca em janela separada (LIMPO, sem pontos MediaPipe)
            if lip_crop is not None and lip_crop.size > 0:
                # Redimensionar para visualização melhor
                if lip_crop.shape[0] > 0 and lip_crop.shape[1] > 0:
                    lip_resized = cv2.resize(lip_crop, (300, 150))  # Tamanho maior para melhor visualização
                    
                    # Adicionar informações no recorte
                    lip_display = lip_resized.copy()
                    cv2.putText(lip_display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    if speaking:
                        cv2.putText(lip_display, "GRAVANDO (LIMPO)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(lip_display, "Sem pontos MediaPipe", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imshow('Speaking Extraction - Lip Crop (Clean)', lip_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Cleanup
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Extração finalizada. {segment_id-1} segmentos salvos.")

    def _is_mouth_open(self, frame, debug=False):
        """
        Detecta se a boca está aberta usando o LipDetector.
        Retorna mouth_open, debug_frame, lip_crop_clean (sem pontos do MediaPipe)
        """
        if isinstance(self.detector, LipDetector):
            # Processar frame para detectar landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.face_mesh.process(rgb_frame)
            
            debug_frame = frame.copy() if debug else frame
            mouth_open = False
            lip_crop_clean = None
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    
                    # Calcular se a boca está aberta usando os landmarks centrais
                    upper_idx = 13
                    lower_idx = 14
                    upper = face_landmarks.landmark[upper_idx]
                    lower = face_landmarks.landmark[lower_idx]
                    upper_y = int(upper.y * h)
                    lower_y = int(lower.y * h)
                    mouth_distance = abs(lower_y - upper_y)
                    mouth_open = mouth_distance > MOUTH_OPEN_PIXEL_DISTANCE
                    
                    # Extrair recorte da boca do FRAME ORIGINAL (sem pontos)
                    lip_landmarks = self.detector.get_lip_landmarks(face_landmarks.landmark, h, w)
                    lip_crop_clean, bbox = self.detector.crop_lip_region(frame.copy(), lip_landmarks)
                    
                    # Para debug, desenhar landmarks apenas no debug_frame
                    if debug:
                        debug_frame = self.detector.draw_lip_landmarks(debug_frame, lip_landmarks, face_landmarks.landmark, h, w)
                        if bbox:
                            cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                            cv2.putText(debug_frame, "Lips", (bbox[0], bbox[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    break
                
                return mouth_open, debug_frame, lip_crop_clean
            else:
                return False, debug_frame, None
                
        elif isinstance(self.detector, SimpleLipDetector):
            processed_frame, mouth_crop, mouth_bbox = self.detector.process_frame(frame.copy())
            if mouth_bbox is not None:
                x1, y1, x2, y2 = mouth_bbox
                height = y2 - y1
                width = x2 - x1
                mouth_open = height > 0.20 * width
                
                # Extrair recorte limpo do frame original (sem anotações)
                lip_crop_clean = frame[y1:y2, x1:x2].copy()
                
                if debug:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0,255,0) if mouth_open else (0,0,255), 2)
                    return mouth_open, processed_frame, lip_crop_clean
                return mouth_open, frame, lip_crop_clean
            return False, processed_frame if debug else frame, None
        else:
            raise RuntimeError("Detector desconhecido para detecção de boca aberta")

    def add_frame_to_buffer(self, frame, lip_crop):
        """Adiciona frame e recorte da boca ao buffer circular"""
        self.frame_buffer.append(frame.copy())
        if lip_crop is not None and lip_crop.size > 0:
            self.lip_crop_buffer.append(lip_crop.copy())
        else:
            # Se não há recorte, adicionar um frame vazio do tamanho padrão
            self.lip_crop_buffer.append(np.zeros((100, 200, 3), dtype=np.uint8))
        
        if len(self.frame_buffer) > self.pre_buffer_size:
            self.frame_buffer.pop(0)  # Remove o frame mais antigo
        if len(self.lip_crop_buffer) > self.pre_buffer_size:
            self.lip_crop_buffer.pop(0)  # Remove o recorte mais antigo

    def start_recording_with_prebuffer(self, segment_id):
        """Inicia gravação incluindo recortes da boca do buffer"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(EXTRACTION_DIR, f'lip_segment_{segment_id}_{timestamp}.mp4')
        
        # Obter dimensões do recorte da boca
        if self.lip_crop_buffer:
            h, w = self.lip_crop_buffer[0].shape[:2]
        else:
            h, w = 100, 200  # Fallback para recorte da boca
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        
        # Escrever recortes da boca do buffer (0.15s antes da detecção)
        for lip_crop in self.lip_crop_buffer:
            if lip_crop is not None and lip_crop.size > 0:
                writer.write(lip_crop)
        
        print(f"🟢 Iniciando gravação de recortes limpos da boca com {len(self.lip_crop_buffer)} frames de pre-buffer ({self.pre_detection_seconds}s)")
        return writer, output_path


if __name__ == '__main__':
    extractor = SpeakingExtractor()
    extractor.run()
