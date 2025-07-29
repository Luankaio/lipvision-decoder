import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
import numpy as np
from lipvision.data_collection.lip_detector import LipDetector
from lipvision.data_collection.simple_lip_detector import SimpleLipDetector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return

    mediapipe_detector = LipDetector()
    simple_detector = SimpleLipDetector()

    print("Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame da câmera")
            break
        frame = cv2.flip(frame, 1)

        # MediaPipe
        mp_frame = frame.copy()
        mp_processed, mp_crop, _ = mediapipe_detector.process_frame(mp_frame)
        cv2.imshow('Lip Reading (MediaPipe)', mp_processed)

        # Simple
        simple_frame = frame.copy()
        simple_processed, simple_crop, _ = simple_detector.process_frame(simple_frame)
        cv2.imshow('Simple Detector (Haar)', simple_processed)

        # Sair
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
