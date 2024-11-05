import cv2
import numpy as np
from uuid import uuid4
from typing import List
from app.models.image import ImageAnalysisResponse

class SprayAnalyzer:
    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int]:
        """
        Analiza una imagen usando técnicas similares a ImageJ para detección de fluorescencia.
        Args:
            image_bytes: Imagen en formato bytes
            save_debug: Si es True, guarda imágenes de debug
        Returns:
            tuple: (porcentaje de cobertura, área total, área rociada)
        """
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Obtener dimensiones originales
        height, width = image.shape[:2]
        print(f"Dimensiones de la imagen: {width}x{height}")
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold automático
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Eliminar ruido y mejorar la segmentación
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos externos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear máscara para el área total
        total_mask = np.zeros_like(gray)
        cv2.drawContours(total_mask, contours, -1, (255), -1)
        
        # Ajustar el threshold para la fluorescencia
        _, fluorescence_mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        
        # Aplicar la máscara total al área rociada
        sprayed_mask = cv2.bitwise_and(fluorescence_mask, total_mask)
        
        # Calcular áreas
        total_area = cv2.countNonZero(total_mask)
        sprayed_area = cv2.countNonZero(sprayed_mask)
        
        # Aplicar factor de corrección
        correction_factor = 0.80
        sprayed_area = int(sprayed_area * correction_factor)
        
        # Calcular porcentaje de cobertura
        coverage = (sprayed_area / total_area * 100) if total_area > 0 else 0
        
        # Guardar imágenes de debug
        if save_debug:
            # Imagen original con contornos
            debug_original = image.copy()
            cv2.drawContours(debug_original, contours, -1, (0, 255, 0), 2)
            cv2.imwrite('debug_contours.jpg', debug_original)
            
            # Máscara del área total
            cv2.imwrite('debug_total_mask.jpg', total_mask)
            
            # Máscara del área rociada
            cv2.imwrite('debug_sprayed_mask.jpg', sprayed_mask)
            
            # Visualización combinada
            debug_combined = np.zeros((height, width, 3), dtype=np.uint8)
            debug_combined[total_mask > 0] = [255, 0, 0]    # Área total en rojo
            debug_combined[sprayed_mask > 0] = [0, 255, 0]  # Área rociada en verde
            cv2.imwrite('debug_combined.jpg', debug_combined)
        
        return coverage, total_area, sprayed_area

    @staticmethod
    def generate_image_id() -> str:
        return str(uuid4())

    @staticmethod
    def calculate_batch_summary(analyses: List[ImageAnalysisResponse]) -> dict:
        """
        Calcula estadísticas para un lote de análisis
        """
        coverages = [analysis.coverage_percentage for analysis in analyses]
        total_areas = [analysis.total_area for analysis in analyses]
        sprayed_areas = [analysis.sprayed_area for analysis in analyses]
        
        return {
            "total_images": len(analyses),
            "average_coverage": round(sum(coverages) / len(coverages), 2) if coverages else 0,
            "min_coverage": round(min(coverages), 2) if coverages else 0,
            "max_coverage": round(max(coverages), 2) if coverages else 0,
            "total_area_analyzed": sum(total_areas),
            "total_area_sprayed": sum(sprayed_areas)
        }