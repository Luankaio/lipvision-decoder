#!/usr/bin/env python3
"""
Script de teste para verificar se o LipDetector está detectando
tanto o lábio superior quanto o inferior corretamente.
"""

import cv2
import numpy as np
from lipvision.data_collection.lip_detector import LipDetector

def test_lip_detection():
    """Testa a detecção de lábios com uma imagem de exemplo ou câmera"""
    detector = LipDetector()
    
    print("=== TESTE DO LIP DETECTOR ===")
    print("Verificando se detecta lábio superior E inferior...")
    print()
    
    # Testar com câmera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera")
        return
    
    print("✅ Câmera aberta com sucesso")
    print("Pressione 'q' para sair, 'space' para analisar frame atual")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro: Não foi possível ler o frame da câmera")
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Espelhar horizontalmente
        
        # Processar frame
        processed_frame, lip_crop, bbox = detector.process_frame(frame)
        
        # Análise detalhada a cada 30 frames
        if frame_count % 30 == 0:
            print(f"\n--- Frame {frame_count} ---")
            
            # Obter dados dos lábios
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    upper_lip, lower_lip = detector.get_lip_regions_separately(face_landmarks.landmark, h, w)
                    validation = detector.validate_lip_detection(upper_lip, lower_lip)
                    
                    print(f"🔍 Lábio Superior: {validation['upper_lip_points']} pontos")
                    print(f"🔍 Lábio Inferior: {validation['lower_lip_points']} pontos")
                    
                    if validation['both_lips_detected']:
                        print("✅ AMBOS OS LÁBIOS DETECTADOS CORRETAMENTE!")
                    else:
                        print("❌ Detecção incompleta dos lábios")
                        if not validation['upper_lip_detected']:
                            print("   - Lábio superior não detectado adequadamente")
                        if not validation['lower_lip_detected']:
                            print("   - Lábio inferior não detectado adequadamente")
            else:
                print("❌ Nenhuma face detectada")
        
        # Mostrar frame
        cv2.imshow('Teste Lip Detection', processed_frame)
        
        # Mostrar crop se disponível
        if lip_crop is not None and lip_crop.size > 0:
            if lip_crop.shape[0] > 0 and lip_crop.shape[1] > 0:
                lip_resized = cv2.resize(lip_crop, (200, 100))
                cv2.imshow('Lip Crop', lip_resized)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Análise detalhada do frame atual
            print("\n=== ANÁLISE DETALHADA ===")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    print(f"Face {i+1}:")
                    h, w, _ = frame.shape
                    
                    # Pontos completos dos lábios
                    lip_landmarks = detector.get_lip_landmarks(face_landmarks.landmark, h, w)
                    print(f"  Total de pontos dos lábios: {len(lip_landmarks)}")
                    
                    # Lábios separados
                    upper_lip, lower_lip = detector.get_lip_regions_separately(face_landmarks.landmark, h, w)
                    print(f"  Pontos lábio superior: {len(upper_lip)}")
                    print(f"  Pontos lábio inferior: {len(lower_lip)}")
                    
                    # Validação
                    validation = detector.validate_lip_detection(upper_lip, lower_lip)
                    print(f"  Status: {'✅ OK' if validation['both_lips_detected'] else '❌ PROBLEMA'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== TESTE FINALIZADO ===")

if __name__ == "__main__":
    test_lip_detection()
