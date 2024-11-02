import cv2
import numpy as np
from uuid import uuid4

class SprayAnalyzer:
    @staticmethod
    def analyze_image(image_bytes: bytes) -> tuple[float, int, int]:
        """
        Analiza una imagen para detectar el área rociada.
        
        Args:
            image_bytes: Imagen en formato bytes
            
        Returns:
            tuple: (porcentaje de cobertura, área total, área rociada)
        """
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Calcular área total en píxeles
        total_pixels = image.shape[0] * image.shape[1]
        
        # Convertir a HSV para mejor detección de gotas de agua
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detectar áreas rociadas (estos valores se pueden ajustar)
        lower = np.array([0, 0, 200])  # Detectar áreas brillantes/blancas
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Calcular área rociada y porcentaje
        sprayed_pixels = cv2.countNonZero(mask)
        coverage = (sprayed_pixels / total_pixels) * 100
        
        return coverage, total_pixels, sprayed_pixels

    @staticmethod
    def generate_image_id() -> str:
        """
        Genera un identificador único para cada imagen analizada.
        
        Returns:
            str: Identificador único
        """
        return str(uuid4())