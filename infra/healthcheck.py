#!/usr/bin/env python3
"""
Health Check Script para LipVision Decoder
Verifica se todos os componentes estão funcionando corretamente
"""

import sys
import os
import subprocess
import importlib

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    print("🐍 Verificando versão do Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Versão muito antiga!")
        return False

def check_imports():
    """Verifica se todos os módulos necessários estão instalados"""
    print("\n📦 Verificando dependências...")
    
    required_modules = [
        'cv2',
        'numpy',
        'mediapipe',
        'matplotlib',
        'sounddevice'
    ]
    
    all_good = True
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - OK")
        except ImportError:
            print(f"❌ {module} - NÃO ENCONTRADO!")
            all_good = False
    
    return all_good

def check_camera():
    """Verifica se a câmera está acessível"""
    print("\n📷 Verificando acesso à câmera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Câmera acessível - Resolução: {width}x{height}")
                cap.release()
                return True
            else:
                print("❌ Câmera acessível mas não consegue capturar frames")
                cap.release()
                return False
        else:
            print("❌ Não consegue acessar a câmera")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar câmera: {e}")
        return False

def check_display():
    """Verifica se o display está configurado"""
    print("\n🖥️  Verificando configuração do display...")
    display = os.environ.get('DISPLAY')
    if display:
        print(f"✅ DISPLAY configurado: {display}")
        return True
    else:
        print("❌ DISPLAY não configurado - GUI pode não funcionar")
        return False

def check_directories():
    """Verifica se os diretórios necessários existem"""
    print("\n📁 Verificando diretórios...")
    
    required_dirs = [
        '/app/lipvision/data_collection/data/lip_crops',
        '/app/lipvision/data_collection/data/lip_crops_simple'
    ]
    
    all_good = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory} - OK")
        else:
            print(f"❌ {directory} - NÃO ENCONTRADO!")
            all_good = False
    
    return all_good

def check_permissions():
    """Verifica permissões de escrita"""
    print("\n🔒 Verificando permissões...")
    
    test_dirs = [
        '/app/lipvision/data_collection/data/lip_crops',
        '/app/lipvision/data_collection/data/lip_crops_simple'
    ]
    
    all_good = True
    for directory in test_dirs:
        try:
            test_file = os.path.join(directory, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✅ {directory} - Escrita OK")
        except Exception as e:
            print(f"❌ {directory} - Sem permissão de escrita: {e}")
            all_good = False
    
    return all_good

def run_quick_test():
    """Executa um teste rápido do detector"""
    print("\n🧪 Executando teste rápido...")
    try:
        # Teste básico do MediaPipe
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe inicializado com sucesso")
        
        # Teste básico do OpenCV
        import cv2
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        print("✅ OpenCV funcionando")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def main():
    """Função principal do health check"""
    print("🏥 ===== LipVision Decoder - Health Check =====\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_imports),
        ("Camera Access", check_camera),
        ("Display Config", check_display),
        ("Directories", check_directories),
        ("Permissions", check_permissions),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erro em {name}: {e}")
            results.append((name, False))
    
    # Resumo
    print("\n" + "="*50)
    print("📊 RESUMO DOS TESTES:")
    print("="*50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! Sistema pronto para uso.")
        return 0
    else:
        print("⚠️  Alguns testes falharam. Verifique a configuração.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
