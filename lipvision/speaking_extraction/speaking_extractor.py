import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import time
from lipvision.data_collection.lip_detector import LipDetector


EXTRACTION_DIR = os.path.join(os.path.dirname(__file__), 'extraction')
os.makedirs(EXTRACTION_DIR, exist_ok=True)


# Parâmetros ajustáveis:
# Distância mínima (em pixels) entre os landmarks centrais dos lábios para considerar a boca aberta
MOUTH_OPEN_PIXEL_DISTANCE = 3
# Janela de silêncio (em segundos) após fechar a boca para encerrar a gravação
POST_SILENCE_WINDOW = 0.7


# Nova versão: captura da câmera, salva recortes de "fala" em extraction

from lipvision.data_collection.simple_lip_detector import SimpleLipDetector

class SpeakingExtractor:
    def __init__(self, method='mediapipe', post_silence_window=POST_SILENCE_WINDOW, camera_index=0, fps=30):
        if method == 'mediapipe':
            self.detector = LipDetector()
        elif method == 'simple':
            self.detector = SimpleLipDetector()
        else:
            raise ValueError(f"Método desconhecido: {method}")
        self.post_silence_window = post_silence_window
        self.camera_index = camera_index
        self.fps = fps

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return

        print("Iniciando extração de segmentos de fala (pressione 'q' para sair)")
        speaking = False
        post_silence_counter = 0
        segment_frames = []
        segment_count = 0
        last_save_time = None


        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame da câmera")
                break
            frame = cv2.flip(frame, 1)

            # Detectar se a boca está aberta
            mouth_open, debug_frame = self._is_mouth_open(frame, debug=True)

            # Mostrar feedback visual
            cv2.putText(debug_frame, f"Falando: {'SIM' if speaking else 'NAO'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if speaking else (0,0,255), 2)
            cv2.imshow('Speaking Extraction', debug_frame)

            if mouth_open:
                if not speaking:
                    segment_frames = []
                    speaking = True
                    post_silence_counter = 0
                    print("🟢 Falando (boca aberta)")
                segment_frames.append(frame.copy())
                post_silence_counter = 0
            else:
                if speaking:
                    post_silence_counter += 1
                    segment_frames.append(frame.copy())
                    if post_silence_counter >= int(self.post_silence_window * self.fps):
                        # Salvar segmento
                        if len(segment_frames) > 0:
                            timestamp = time.strftime('%Y%m%d_%H%M%S')
                            out_path = os.path.join(EXTRACTION_DIR, f'segment_{segment_count}_{timestamp}.mp4')
                            h, w = segment_frames[0].shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
                            for f in segment_frames:
                                out.write(f)
                            out.release()
                            print(f"💾 Segmento salvo: {out_path}")
                            segment_count += 1
                        speaking = False
                        segment_frames = []
                        post_silence_counter = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

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


if __name__ == '__main__':
    extractor = SpeakingExtractor()
    extractor.run()

if __name__ == '__main__':
    extractor = SpeakingExtractor()
    extractor.run()
