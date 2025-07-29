#!/usr/bin/env python3
"""
LipVision Decoder - Detector de Lábios/Boca

Este programa oferece duas abordagens para detectar e recortar a região dos lábios:
1. MediaPipe (Avançado) - Usa landmarks faciais precisos
2. Haar Cascades (Simples) - Usa detecção de face e estimativa da região da boca

Uso:
    python main.py [--method mediapipe|simple]
"""

import argparse
import sys
import cv2

def check_camera():
    """Verifica se a câmera está disponível"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: Câmera não encontrada ou não acessível")
        print("Verifique se:")
        print("  - A câmera está conectada")
        print("  - Não há outros programas usando a câmera")
        print("  - Você tem permissões para acessar a câmera")
        return False
    cap.release()
    return True

def print_banner():
    """Imprime o banner do programa"""
    print("=" * 60)
    print("🎥 LipVision Decoder - Detector de Lábios/Boca")
    print("=" * 60)
    print("Multimodal pipeline para leitura labial")
    print("Detecta e recorta a região dos lábios em tempo real")
    print("=" * 60)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Detector de lábios usando visão computacional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Métodos disponíveis:
  mediapipe  - Usa MediaPipe para detecção precisa de landmarks (recomendado)
  simple     - Usa Haar Cascades para detecção básica (mais rápido)

Exemplos:
  python main.py --method mediapipe
  python main.py --method simple
  python main.py  # usa MediaPipe por padrão
        """
    )
    
    parser.add_argument(
        "--method",
        choices=["mediapipe", "simple"],
        default="mediapipe",
        help="Método de detecção a usar (padrão: mediapipe)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Verificar se a câmera está disponível
    if not check_camera():
        sys.exit(1)
    
    print(f"🔧 Método selecionado: {args.method}")
    print()
    
    try:
        if args.method == "mediapipe":
            print("📡 Carregando detector MediaPipe...")
            from lipvision.data_collection.lip_detector import LipDetector
            detector = LipDetector()
            detector.run_camera()
        elif args.method == "simple":
            print("📡 Carregando detector simples...")
            from lipvision.data_collection.simple_lip_detector import SimpleLipDetector
            detector = SimpleLipDetector()
            detector.run_camera()
    
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 Programa interrompido pelo usuário")
    
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        sys.exit(1)
    
    print("✅ Programa finalizado com sucesso")

if __name__ == "__main__":
    main()
