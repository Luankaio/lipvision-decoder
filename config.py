"""
Configurações para os detectores de lábios

Este arquivo contém configurações que podem ser ajustadas para otimizar
a detecção de acordo com suas necessidades específicas.
"""

# Configurações do MediaPipe
MEDIAPIPE_CONFIG = {
    # Número máximo de faces a detectar
    'max_num_faces': 1,
    
    # Refinar landmarks (mais preciso, mas mais lento)
    'refine_landmarks': True,
    
    # Confiança mínima para detecção
    'min_detection_confidence': 0.5,
    
    # Confiança mínima para rastreamento
    'min_tracking_confidence': 0.5,
    
    # Margem ao redor dos lábios (em pixels)
    'lip_margin': 20,
}

# Configurações do detector simples
SIMPLE_CONFIG = {
    # Fator de escala para detecção de face
    'scale_factor': 1.3,
    
    # Número mínimo de vizinhos para validar detecção
    'min_neighbors': 5,
    
    # Região da boca (proporção da face)
    'mouth_region': {
        'y_start_ratio': 0.6,  # Começar a 60% da altura da face
        'y_end_ratio': 1.0,    # Até o final da face
        'x_start_ratio': 0.2,  # Começar a 20% da largura
        'x_end_ratio': 0.8,    # Até 80% da largura
    },
    
    # Margem ao redor da boca
    'mouth_margin': 10,
}

# Configurações da câmera
CAMERA_CONFIG = {
    # Índice da câmera (0 para câmera padrão)
    'camera_index': 0,
    
    # Resolução da câmera
    'width': 640,
    'height': 480,
    
    # FPS da câmera
    'fps': 30,
    
    # Espelhar imagem horizontalmente
    'flip_horizontal': True,
}

# Configurações de visualização
DISPLAY_CONFIG = {
    # Cores (BGR format)
    'face_color': (255, 0, 0),      # Azul para face
    'lip_color': (0, 255, 0),       # Verde para lábios
    'bbox_color': (0, 255, 255),    # Amarelo para bounding box
    'text_color': (255, 255, 255),  # Branco para texto
    
    # Espessura das linhas
    'line_thickness': 2,
    'text_thickness': 2,
    
    # Tamanho da fonte
    'font_scale': 0.7,
    'font_small': 0.5,
    
    # Tamanho da janela de visualização do recorte
    'crop_preview_size': (200, 100),
}

# Configurações de salvamento
SAVE_CONFIG = {
    # Diretórios de saída
    'mediapipe_output_dir': 'lip_crops',
    'simple_output_dir': 'lip_crops_simple',
    
    # Formato do timestamp
    'timestamp_format': '%Y%m%d_%H%M%S_%f',
    
    # Qualidade da imagem JPEG (0-100)
    'jpeg_quality': 95,
    
    # Redimensionamento automático dos recortes
    'auto_resize': {
        'enabled': False,
        'target_size': (128, 64),
    },
}

# Configurações de performance
PERFORMANCE_CONFIG = {
    # Processar apenas a cada N frames (para economizar recursos)
    'frame_skip': 1,  # 1 = processar todos os frames
    
    # Redimensionar frame antes do processamento
    'resize_for_processing': {
        'enabled': False,
        'scale': 0.5,  # 50% do tamanho original
    },
    
    # Aplicar filtros para melhorar detecção
    'preprocessing': {
        'gaussian_blur': False,
        'histogram_equalization': False,
        'bilateral_filter': True,
    },
}

# Configurações avançadas do MediaPipe
ADVANCED_MEDIAPIPE = {
    # Índices específicos dos landmarks dos lábios
    'lip_landmarks': {
        # Contorno externo dos lábios
        'outer_lip': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
        
        # Contorno interno dos lábios
        'inner_lip': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80],
        
        # Pontos específicos
        'lip_corners': [61, 291],  # Cantos da boca
        'lip_center_top': [13],    # Centro do lábio superior
        'lip_center_bottom': [14], # Centro do lábio inferior
    },
    
    # Desenhar landmarks específicos
    'draw_landmarks': {
        'outer_lip': True,
        'inner_lip': False,
        'lip_corners': True,
        'center_points': True,
    },
}

def get_config(config_name):
    """
    Retorna uma configuração específica
    
    Args:
        config_name (str): Nome da configuração
        
    Returns:
        dict: Configuração solicitada
    """
    configs = {
        'mediapipe': MEDIAPIPE_CONFIG,
        'simple': SIMPLE_CONFIG,
        'camera': CAMERA_CONFIG,
        'display': DISPLAY_CONFIG,
        'save': SAVE_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'advanced_mediapipe': ADVANCED_MEDIAPIPE,
    }
    
    return configs.get(config_name, {})

def print_current_config():
    """Imprime todas as configurações atuais"""
    print("📋 Configurações Atuais:")
    print("=" * 40)
    
    configs = [
        ('MediaPipe', MEDIAPIPE_CONFIG),
        ('Detector Simples', SIMPLE_CONFIG),
        ('Câmera', CAMERA_CONFIG),
        ('Visualização', DISPLAY_CONFIG),
        ('Salvamento', SAVE_CONFIG),
        ('Performance', PERFORMANCE_CONFIG),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print_current_config()
