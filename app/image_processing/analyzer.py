import cv2
import numpy as np
from uuid import uuid4

class SprayAnalyzer:
    @staticmethod
    def analyze_image(image_bytes: bytes) -> tuple[float, int, int]:
        """
        Analiza una imagen usando técnicas similares a ImageJ para detección de fluorescencia.
        
        Args:
            image_bytes: Imagen en formato bytes
            
        Returns:
            tuple: (porcentaje de cobertura, área total, área rociada)
        """
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convertir a escala de grises (similar a ImageJ)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold automático (similar al Auto-Threshold de ImageJ)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Eliminar ruido y mejorar la segmentación
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos (similar a Analyze Particles en ImageJ)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear máscara para el área total (sin fondo)
        total_mask = np.zeros_like(gray)
        cv2.drawContours(total_mask, contours, -1, (255), -1)
        
        # Calcular áreas usando los contornos
        total_area = cv2.countNonZero(total_mask)
        
        # Aplicar threshold para la fluorescencia
        _, fluorescence_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        sprayed_area = cv2.countNonZero(cv2.bitwise_and(fluorescence_mask, total_mask))
        
        # Calcular porcentaje de cobertura
        coverage = (sprayed_area / total_area * 100) if total_area > 0 else 0
        
        return coverage, total_area, sprayed_area

    @staticmethod
    def generate_image_id() -> str:
        return str(uuid4())

    @staticmethod
    def save_debug_image(image_bytes: bytes, output_path: str):
        """
        Guarda imágenes de debug para verificar el proceso de análisis.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Crear visualización del análisis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Guardar imagen de debug
        cv2.imwrite(output_path, debug_image)