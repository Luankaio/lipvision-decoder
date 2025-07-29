import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

class LipDetector:
    def __init__(self):
        """Inicializa o detector de lábios usando MediaPipe"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configuração do Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Índices dos pontos dos lábios no MediaPipe Face Mesh
        self.LIPS_POINTS = [
            # Lábio superior externo
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Lábio inferior externo
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            # Lábio superior interno
            80, 81, 82, 13, 312, 311, 310, 415,
            # Lábio inferior interno
            78, 95, 88, 178, 87, 14, 317, 402
        ]
        
        # Criar diretório para salvar recortes
        self.output_dir = "lipvision/data_collection/data/lip_crops"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_lip_landmarks(self, landmarks, img_height, img_width):
        """Extrai as coordenadas dos pontos dos lábios"""
        lip_coords = []
        
        # Pontos específicos para contorno dos lábios
        lip_indices = [
            # Contorno externo dos lábios
            61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308,
            324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        
        for idx in lip_indices:
            x = int(landmarks[idx].x * img_width)
            y = int(landmarks[idx].y * img_height)
            lip_coords.append([x, y])
        
        return np.array(lip_coords, dtype=np.int32)
    
    def crop_lip_region(self, image, lip_landmarks):
        """Recorta a região dos lábios da imagem"""
        # Encontrar bounding box dos lábios
        x_min = np.min(lip_landmarks[:, 0])
        x_max = np.max(lip_landmarks[:, 0])
        y_min = np.min(lip_landmarks[:, 1])
        y_max = np.max(lip_landmarks[:, 1])
        
        # Adicionar margem ao redor dos lábios
        margin = 20
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        
        # Recortar a região
        lip_crop = image[y_min:y_max, x_min:x_max]
        
        return lip_crop, (x_min, y_min, x_max, y_max)
    
    def draw_lip_landmarks(self, image, lip_landmarks):
        """Desenha os pontos dos lábios na imagem"""
        for point in lip_landmarks:
            cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)
        
        # Desenhar contorno dos lábios
        cv2.polylines(image, [lip_landmarks], True, (255, 0, 0), 2)
        
        return image
    
    def process_frame(self, frame):
        """Processa um frame para detectar e recortar lábios"""
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pontos faciais
        results = self.face_mesh.process(rgb_frame)
        
        lip_crop = None
        bbox = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obter coordenadas dos lábios
                h, w, _ = frame.shape
                lip_landmarks = self.get_lip_landmarks(face_landmarks.landmark, h, w)
                
                # Desenhar landmarks dos lábios
                frame = self.draw_lip_landmarks(frame, lip_landmarks)
                
                # Recortar região dos lábios
                lip_crop, bbox = self.crop_lip_region(frame, lip_landmarks)
                
                # Desenhar bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                cv2.putText(frame, "Lips", (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, lip_crop, bbox
    
    def save_lip_crop(self, lip_crop):
        """Salva o recorte dos lábios"""
        if lip_crop is not None and lip_crop.size > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.output_dir}/lipvision/data/lip_crop_{timestamp}.jpg"
            cv2.imwrite(filename, lip_crop)
            return filename
        return None
    
    def run_camera(self):
        """Executa a detecção em tempo real usando a câmera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return
        
        print("Iniciando detecção de lábios...")
        print("Pressione 'c' para capturar um recorte dos lábios")
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera")
                break
            
            # Espelhar a imagem horizontalmente para parecer natural
            frame = cv2.flip(frame, 1)
            
            # Processar frame
            processed_frame, lip_crop, bbox = self.process_frame(frame)
            
            # Mostrar instruções na tela
            cv2.putText(processed_frame, "Pressione 'c' para capturar, 'q' para sair", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame principal
            cv2.imshow('Lip Detection', processed_frame)
            
            # Mostrar recorte dos lábios se disponível
            if lip_crop is not None and lip_crop.size > 0:
                # Redimensionar para visualização melhor
                if lip_crop.shape[0] > 0 and lip_crop.shape[1] > 0:
                    lip_resized = cv2.resize(lip_crop, (200, 100))
                    cv2.imshow('Lip Crop', lip_resized)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if lip_crop is not None:
                    filename = self.save_lip_crop(lip_crop)
                    if filename:
                        print(f"Recorte salvo: {filename}")
                    else:
                        print("Erro ao salvar o recorte")
                else:
                    print("Nenhum lábio detectado para capturar")
        
        # Limpar recursos
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Função principal"""
    detector = LipDetector()
    detector.run_camera()

if __name__ == "__main__":
    main()
