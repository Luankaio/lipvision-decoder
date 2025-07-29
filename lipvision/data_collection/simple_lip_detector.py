import cv2
import numpy as np
import os
from datetime import datetime
from .config import get_config

class SimpleLipDetector:
    def __init__(self):
        """Inicializa o detector simples usando Haar Cascades"""
        # Carregar configurações
        self.simple_config = get_config('simple')
        self.save_config = get_config('save')
        
        # Carregar classificadores Haar para face e boca
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Criar diretório para salvar recortes
        self.output_dir = os.path.join("lipvision", "data_collection", "data", self.save_config['simple_output_dir'])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def detect_mouth_region(self, face_roi):
        """Detecta a região da boca dentro da face"""
        # A boca geralmente está na metade inferior da face
        h, w = face_roi.shape[:2]
        
        # Usar configurações do arquivo config.py
        mouth_region = self.simple_config['mouth_region']
        
        # Definir região de interesse para a boca
        mouth_y_start = int(h * mouth_region['y_start_ratio'])
        mouth_y_end = int(h * mouth_region['y_end_ratio'])
        mouth_x_start = int(w * mouth_region['x_start_ratio'])
        mouth_x_end = int(w * mouth_region['x_end_ratio'])
        
        mouth_roi = face_roi[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        
        return mouth_roi, (mouth_x_start, mouth_y_start, mouth_x_end, mouth_y_end)
    
    def enhance_mouth_detection(self, mouth_roi):
        """Aplica filtros para melhorar a detecção da boca"""
        if mouth_roi.size == 0:
            return mouth_roi
        
        # Converter para escala de cinza se necessário
        if len(mouth_roi.shape) == 3:
            gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_mouth = mouth_roi
        
        # Aplicar equalização de histograma
        enhanced = cv2.equalizeHist(gray_mouth)
        
        # Aplicar filtro bilateral para suavizar mantendo bordas
        if len(mouth_roi.shape) == 3:
            enhanced_color = cv2.bilateralFilter(mouth_roi, 9, 75, 75)
            return enhanced_color
        else:
            return enhanced
    
    def process_frame(self, frame):
        """Processa um frame para detectar e extrair a região da boca"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces
        faces = self.face_cascade.detectMultiScale(gray, 
                                                    self.simple_config['scale_factor'], 
                                                    self.simple_config['min_neighbors'])
        
        mouth_crop = None
        mouth_bbox = None
        
        for (x, y, w, h) in faces:
            # Desenhar retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Extrair ROI da face
            face_roi = frame[y:y+h, x:x+w]
            
            # Detectar região da boca
            mouth_roi, mouth_coords = self.detect_mouth_region(face_roi)
            
            if mouth_roi.size > 0:
                # Coordenadas absolutas da boca na imagem original
                abs_mouth_x1 = x + mouth_coords[0]
                abs_mouth_y1 = y + mouth_coords[1]
                abs_mouth_x2 = x + mouth_coords[2]
                abs_mouth_y2 = y + mouth_coords[3]
                
                # Desenhar retângulo ao redor da boca
                cv2.rectangle(frame, (abs_mouth_x1, abs_mouth_y1), 
                             (abs_mouth_x2, abs_mouth_y2), (0, 255, 0), 2)
                cv2.putText(frame, "Mouth", (abs_mouth_x1, abs_mouth_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Melhorar a qualidade do recorte da boca
                mouth_crop = self.enhance_mouth_detection(mouth_roi)
                mouth_bbox = (abs_mouth_x1, abs_mouth_y1, abs_mouth_x2, abs_mouth_y2)
                
                # Usar apenas a primeira face detectada
                break
        
        return frame, mouth_crop, mouth_bbox
    
    def save_mouth_crop(self, mouth_crop):
        """Salva o recorte da boca"""
        if mouth_crop is not None and mouth_crop.size > 0:
            timestamp = datetime.now().strftime(self.save_config['timestamp_format'])[:-3]
            filename = f"{self.output_dir}/mouth_crop_{timestamp}.jpg"
            
            # Salvar com qualidade configurada
            cv2.imwrite(filename, mouth_crop, [cv2.IMWRITE_JPEG_QUALITY, self.save_config['jpeg_quality']])
            return filename
        return None
    
    def run_camera(self):
        """Executa a detecção em tempo real usando a câmera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return
        
        print("Iniciando detecção simples de boca...")
        print("Pressione 'c' para capturar um recorte da boca")
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera")
                break
            
            # Espelhar a imagem horizontalmente
            frame = cv2.flip(frame, 1)
            
            # Processar frame
            processed_frame, mouth_crop, mouth_bbox = self.process_frame(frame)
            
            # Mostrar instruções na tela
            cv2.putText(processed_frame, "Pressione 'c' para capturar, 'q' para sair", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame principal
            cv2.imshow('Simple Mouth Detection', processed_frame)
            
            # Mostrar recorte da boca se disponível
            if mouth_crop is not None and mouth_crop.size > 0:
                # Redimensionar para visualização melhor
                if mouth_crop.shape[0] > 0 and mouth_crop.shape[1] > 0:
                    mouth_resized = cv2.resize(mouth_crop, (150, 80))
                    cv2.imshow('Mouth Crop', mouth_resized)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if mouth_crop is not None:
                    filename = self.save_mouth_crop(mouth_crop)
                    if filename:
                        print(f"Recorte da boca salvo: {filename}")
                    else:
                        print("Erro ao salvar o recorte")
                else:
                    print("Nenhuma boca detectada para capturar")
        
        # Limpar recursos
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Função principal"""
    detector = SimpleLipDetector()
    detector.run_camera()

if __name__ == "__main__":
    main()
