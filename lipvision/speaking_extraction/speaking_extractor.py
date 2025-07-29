import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import time
from lipvision.data_collection.lip_detector import LipDetector

EXTRACTION_DIR = os.path.join(os.path.dirname(__file__), 'extraction')
os.makedirs(EXTRACTION_DIR, exist_ok=True)


# Nova versão: captura da câmera, salva recortes de "fala" em extraction

from lipvision.data_collection.simple_lip_detector import SimpleLipDetector

class SpeakingExtractor:
    def __init__(self, method='mediapipe', post_silence_window=0.5, camera_index=0, fps=30):
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
            mouth_open = self._is_mouth_open(frame)

            # Mostrar feedback visual
            cv2.putText(frame, f"Falando: {'SIM' if speaking else 'NAO'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if speaking else (0,0,255), 2)
            cv2.imshow('Speaking Extraction', frame)

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

    def _is_mouth_open(self, frame):
        """
        Detecta se a boca está aberta usando o detector selecionado.
        Para MediaPipe: verifica distância vertical entre landmarks dos lábios.
        Para Simple: verifica altura da região da boca detectada.
        """
        if isinstance(self.detector, LipDetector):
            processed_frame, lip_crop, bbox = self.detector.process_frame(frame)
            # Heurística: se o recorte dos lábios for "alto" o suficiente, considera boca aberta
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                height = y2 - y1
                width = x2 - x1
                # Critério simples: altura > 20% da largura
                return height > 0.20 * width
            return False
        elif isinstance(self.detector, SimpleLipDetector):
            processed_frame, mouth_crop, mouth_bbox = self.detector.process_frame(frame)
            if mouth_bbox is not None:
                x1, y1, x2, y2 = mouth_bbox
                height = y2 - y1
                width = x2 - x1
                return height > 0.20 * width
            return False
        else:
            raise RuntimeError("Detector desconhecido para detecção de boca aberta")


if __name__ == '__main__':
    extractor = SpeakingExtractor()
    extractor.run()

if __name__ == '__main__':
    extractor = SpeakingExtractor()
    extractor.run()
