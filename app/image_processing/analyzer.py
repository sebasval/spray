import cv2
import numpy as np
from uuid import uuid4
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import base64
from app.models.image import ImageAnalysisResponse

class SprayAnalyzer:
    # Parámetros ajustados para fluorescencia azul
    MIN_SPRAY_AREA = 5
    MORPH_KERNEL_SIZE = 3
    
    @staticmethod
    def _detect_leaf_mask(image: np.ndarray) -> np.ndarray:
        """
        Detecta la máscara de la hoja usando múltiples canales
        """
        # Convertir a diferentes espacios de color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Crear máscara base usando el canal gris
        _, base_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Limpiar máscara
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar el contorno más grande (la hoja)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cleaned_mask
            
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        return final_mask

    @staticmethod
    def _detect_fluorescence(image: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detecta la fluorescencia azul específicamente
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Rango para detectar azul brillante
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Crear máscara para azul
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Aplicar máscara de la hoja
        fluorescence_mask = cv2.bitwise_and(blue_mask, leaf_mask)
        
        # Suavizar y limpiar la máscara
        kernel = np.ones((3,3), np.uint8)
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_OPEN, kernel)
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calcular umbral usado (para debug)
        mean_intensity = np.mean(hsv[:,:,2][fluorescence_mask > 0]) if cv2.countNonZero(fluorescence_mask) > 0 else 0
        
        return fluorescence_mask, mean_intensity

    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Analiza una imagen para detectar cobertura de spray usando fluorescencia UV
        
        Returns:
            tuple: (coverage_percentage, leaf_area, sprayed_area, processed_image_base64)
        """
        # Convertir bytes a imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Reducir ruido
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Detectar máscara de la hoja
        leaf_mask = SprayAnalyzer._detect_leaf_mask(denoised)
        
        # Detectar fluorescencia
        fluorescence_mask, threshold_used = SprayAnalyzer._detect_fluorescence(denoised, leaf_mask)
        
        # Calcular áreas y cobertura
        leaf_area = cv2.countNonZero(leaf_mask)
        sprayed_area = cv2.countNonZero(fluorescence_mask)
        coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0
        
        # Crear imagen procesada con áreas de rocío en amarillo
        processed_image = image.copy()
        processed_image[fluorescence_mask > 0] = [0, 255, 255]  # Amarillo para áreas con spray
        
        # Codificar imagen procesada a base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Guardar imágenes de debug
        if save_debug:
            # Guardar imágenes
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_fluorescence_mask.jpg", fluorescence_mask)
            cv2.imwrite("debug_result.jpg", processed_image)
            
            # Histograma de azul
            plt.figure(figsize=(10, 6))
            blue_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0]
            plt.hist(blue_channel.ravel(), 256, [0, 256])
            plt.title("Distribución del Canal Azul (Matiz)")
            plt.savefig("debug_histogram.jpg")
            plt.close()

        return round(coverage, 2), leaf_area, sprayed_area, processed_image_base64

    @staticmethod
    def generate_image_id() -> str:
        return str(uuid4())

    @staticmethod
    def calculate_batch_summary(analyses: List[ImageAnalysisResponse]) -> dict:
        if not analyses:
            return {
                "total_images": 0,
                "average_coverage": 0,
                "min_coverage": 0,
                "max_coverage": 0,
                "total_area_analyzed": 0,
                "total_area_sprayed": 0,
                "global_coverage": 0,
            }

        coverages = [analysis.coverage_percentage for analysis in analyses]
        total_areas = [analysis.total_area for analysis in analyses]
        sprayed_areas = [analysis.sprayed_area for analysis in analyses]

        return {
            "total_images": len(analyses),
            "average_coverage": round(sum(coverages) / len(coverages), 2),
            "min_coverage": round(min(coverages), 2),
            "max_coverage": round(max(coverages), 2),
            "total_area_analyzed": sum(total_areas),
            "total_area_sprayed": sum(sprayed_areas),
            "global_coverage": round(
                sum(sprayed_areas) / sum(total_areas) * 100, 2
            ) if sum(total_areas) > 0 else 0,
        }