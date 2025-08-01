import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import cv2
import time
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
            
            # Sempre adicionar frame ao buffer circular
            self.add_frame_to_buffer(frame)
            
            # Detectar se a boca está aberta
            mouth_open, debug_frame = self._is_mouth_open(frame, debug=True)

            if mouth_open and not speaking:
                # Começou a falar - iniciar gravação COM pre-buffer
                speaking = True
                post_silence_counter = 0
                writer, current_output_path = self.start_recording_with_prebuffer(segment_id)

            if speaking:
                if writer is not None:
                    writer.write(frame)

                if not mouth_open:
                    post_silence_counter += 1
                    if post_silence_counter >= silence_frames:
                        # Parou de falar - finalizar gravação
                        speaking = False
                        post_silence_counter = 0
                        if writer is not None:
                            writer.release()
                            writer = None
                        print(f"💾 Segmento salvo: {current_output_path}")
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
            
            if speaking and post_silence_counter > 0:
                remaining_frames = silence_frames - post_silence_counter
                cv2.putText(debug_frame, f"Encerrando em: {remaining_frames} frames", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Speaking Extraction', debug_frame)

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
        Detecta se a boca está aberta usando o detector selecionado.
        Para MediaPipe: usa a distância vertical entre os landmarks centrais dos lábios (linha amarela).
        Para Simple: verifica altura da região da boca detectada.
        Se debug=True, retorna também o frame com landmarks/desenhos.
        """
        if isinstance(self.detector, LipDetector):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.face_mesh.process(rgb_frame)
            debug_frame = frame.copy()
            mouth_open = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    # Pontos centrais dos lábios (MediaPipe: 13 = centro inferior, 14 = centro superior)
                    upper_idx = 13
                    lower_idx = 14
                    upper = face_landmarks.landmark[upper_idx]
                    lower = face_landmarks.landmark[lower_idx]
                    upper_y = int(upper.y * h)
                    lower_y = int(lower.y * h)
                    # Distância vertical entre os dois pontos
                    mouth_distance = abs(lower_y - upper_y)
                    # Critério: boca aberta se distância > MOUTH_OPEN_PIXEL_DISTANCE
                    mouth_open = mouth_distance > MOUTH_OPEN_PIXEL_DISTANCE
                    if debug:
                        cv2.line(debug_frame, (int(upper.x * w), upper_y), (int(lower.x * w), lower_y), (0,255,255), 2)
                        color = (0,255,0) if mouth_open else (0,0,255)
                        cv2.circle(debug_frame, (int(upper.x * w), upper_y), 4, color, -1)
                        cv2.circle(debug_frame, (int(lower.x * w), lower_y), 4, color, -1)
                        return mouth_open, debug_frame
                    return mouth_open, frame
            return False, frame
        elif isinstance(self.detector, SimpleLipDetector):
            processed_frame, mouth_crop, mouth_bbox = self.detector.process_frame(frame.copy())
            if mouth_bbox is not None:
                x1, y1, x2, y2 = mouth_bbox
                height = y2 - y1
                width = x2 - x1
                mouth_open = height > 0.20 * width
                if debug:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0,255,0) if mouth_open else (0,0,255), 2)
                    return mouth_open, processed_frame
                return mouth_open, frame
            return False, frame
        else:
            raise RuntimeError("Detector desconhecido para detecção de boca aberta")

    def add_frame_to_buffer(self, frame):
        """Adiciona frame ao buffer circular"""
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.pre_buffer_size:
            self.frame_buffer.pop(0)  # Remove o frame mais antigo

    def start_recording_with_prebuffer(self, segment_id):
        """Inicia gravação incluindo frames do buffer"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(EXTRACTION_DIR, f'segment_{segment_id}_{timestamp}.mp4')
        
        # Obter dimensões do frame
        if self.frame_buffer:
            h, w = self.frame_buffer[0].shape[:2]
        else:
            h, w = 480, 640  # Fallback
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        
        # Escrever frames do buffer (0.15s antes da detecção)
        for buffered_frame in self.frame_buffer:
            writer.write(buffered_frame)
        
        print(f"🟢 Iniciando gravação com {len(self.frame_buffer)} frames de pre-buffer ({self.pre_detection_seconds}s)")
        return writer, output_path


if __name__ == '__main__':
    extractor = SpeakingExtractor()
    extractor.run()
