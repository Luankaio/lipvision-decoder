#!/usr/bin/env python3
"""
Exemplo de uso programático dos detectores de lábios

Este arquivo demonstra como usar as classes de detecção em seus próprios projetos.
"""

import cv2
import numpy as np
from lip_detector import LipDetector
from simple_lip_detector import SimpleLipDetector

def example_mediapipe_usage():
    """Exemplo de uso do detector MediaPipe"""
    print("=== Exemplo: Detector MediaPipe ===")
    
    # Inicializar detector
    detector = LipDetector()
    
    # Simular processamento de uma imagem
    # (Em um caso real, você carregaria uma imagem com cv2.imread)
    
    print("Detector MediaPipe inicializado com sucesso!")
    print("- Detecta até 1 face por frame")
    print("- Usa 468 landmarks faciais")
    print("- Salva recortes em 'lip_crops/'")
    print()

def example_simple_usage():
    """Exemplo de uso do detector simples"""
    print("=== Exemplo: Detector Simples ===")
    
    # Inicializar detector
    detector = SimpleLipDetector()
    
    print("Detector simples inicializado com sucesso!")
    print("- Usa Haar Cascades para detecção de face")
    print("- Estima região da boca na parte inferior da face")
    print("- Salva recortes em 'lip_crops_simple/'")
    print()

def process_image_example(image_path=None):
    """Exemplo de processamento de uma imagem estática"""
    print("=== Exemplo: Processamento de Imagem ===")
    
    if image_path is None:
        print("Para processar uma imagem específica:")
        print("1. Coloque a imagem no diretório do projeto")
        print("2. Modifique o código para usar o caminho correto")
        print()
        return
    
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            return
        
        # Usar detector MediaPipe
        detector = LipDetector()
        processed_frame, lip_crop, bbox = detector.process_frame(image)
        
        if lip_crop is not None:
            print("✅ Lábios detectados com sucesso!")
            print(f"   Tamanho do recorte: {lip_crop.shape}")
            print(f"   Bounding box: {bbox}")
            
            # Salvar resultado
            cv2.imwrite("resultado_processamento.jpg", processed_frame)
            cv2.imwrite("recorte_labios.jpg", lip_crop)
            print("   Arquivos salvos: resultado_processamento.jpg, recorte_labios.jpg")
        else:
            print("❌ Nenhum lábio detectado na imagem")
    
    except Exception as e:
        print(f"Erro: {e}")

def batch_processing_example():
    """Exemplo de processamento em lote"""
    print("=== Exemplo: Processamento em Lote ===")
    print("Para processar múltiplas imagens:")
    print("""
import os
import glob

detector = LipDetector()
image_files = glob.glob("imagens/*.jpg")

for img_path in image_files:
    image = cv2.imread(img_path)
    processed_frame, lip_crop, bbox = detector.process_frame(image)
    
    if lip_crop is not None:
        filename = os.path.basename(img_path)
        cv2.imwrite(f"resultados/{filename}", lip_crop)
        print(f"Processado: {filename}")
""")
    print()

def custom_callback_example():
    """Exemplo de uso com callback personalizado"""
    print("=== Exemplo: Callback Personalizado ===")
    print("Para integrar com outros sistemas:")
    print("""
def meu_callback(lip_crop, bbox):
    # Sua lógica personalizada aqui
    if lip_crop is not None:
        # Exemplo: análise adicional
        altura, largura = lip_crop.shape[:2]
        print(f"Lábios detectados: {largura}x{altura}")
        
        # Exemplo: enviar para API
        # enviar_para_api(lip_crop)
        
        # Exemplo: salvar com metadados
        # salvar_com_metadados(lip_crop, bbox)

# Usar durante processamento:
detector = LipDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    processed_frame, lip_crop, bbox = detector.process_frame(frame)
    
    # Chamar seu callback
    meu_callback(lip_crop, bbox)
    
    cv2.imshow('Camera', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
""")
    print()

def performance_tips():
    """Dicas de performance"""
    print("=== Dicas de Performance ===")
    print("1. Para melhor performance:")
    print("   - Use o detector simples para aplicações em tempo real")
    print("   - Reduza a resolução da câmera se necessário")
    print("   - Processe apenas a cada N frames em aplicações não críticas")
    print()
    print("2. Para melhor precisão:")
    print("   - Use o detector MediaPipe")
    print("   - Garanta boa iluminação")
    print("   - Mantenha o rosto centralizado na imagem")
    print()
    print("3. Configurações da câmera:")
    print("""
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduzir resolução
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)            # Definir FPS
""")
    print()

def main():
    """Função principal com exemplos"""
    print("🎥 LipVision Decoder - Exemplos de Uso")
    print("=" * 50)
    print()
    
    # Executar exemplos
    example_mediapipe_usage()
    example_simple_usage()
    process_image_example()
    batch_processing_example()
    custom_callback_example()
    performance_tips()
    
    print("💡 Para executar os detectores:")
    print("   python main.py --method mediapipe")
    print("   python main.py --method simple")
    print()
    print("📚 Para mais informações, consulte o README.md")

if __name__ == "__main__":
    main()
