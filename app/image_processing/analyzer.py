import cv2
import numpy as np
from uuid import uuid4
from typing import List
import matplotlib.pyplot as plt
from app.models.image import ImageAnalysisResponse

class SprayAnalyzer:
    # Parámetros ajustados para mejor detección de fluorescencia
    LOWER_THRESHOLD = 180  # Umbral para detectar áreas brillantes
    UPPER_THRESHOLD = 255  # Máximo valor de brillo
    MIN_SPRAY_AREA = 5     # Área mínima para considerar una gota
    MORPH_KERNEL_SIZE = 2  # Tamaño del kernel para operaciones morfológicas

    @staticmethod
    def analyze_image(
        image_bytes: bytes, save_debug: bool = True
    ) -> tuple[float, int, int]:
        """
        Analiza una imagen para detectar cobertura de spray usando fluorescencia UV
        """
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        # Convertir a HSV para mejor detección de fluorescencia
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extraer canal V (brillo) para detectar fluorescencia
        v_channel = hsv[:,:,2]
        
        # Aplicar desenfoque Gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(v_channel, (3,3), 0)
        
        # Crear máscara para fluorescencia
        fluorescence_mask = cv2.inRange(
            blurred, SprayAnalyzer.LOWER_THRESHOLD, SprayAnalyzer.UPPER_THRESHOLD
        )

        # Operaciones morfológicas para limpiar la máscara
        kernel = np.ones(
            (SprayAnalyzer.MORPH_KERNEL_SIZE, SprayAnalyzer.MORPH_KERNEL_SIZE), 
            np.uint8
        )
        cleaned_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar y filtrar contornos
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = [
            cnt
            for cnt in contours
            if cv2.contourArea(cnt) > SprayAnalyzer.MIN_SPRAY_AREA
        ]

        # Crear máscara final con contornos válidos
        final_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(final_mask, valid_contours, -1, 255, -1)

        # Detectar área de la hoja (usando umbral bajo en canal V)
        leaf_mask = cv2.threshold(v_channel, 30, 255, cv2.THRESH_BINARY)[1]
        leaf_area = cv2.countNonZero(leaf_mask)
        sprayed_area = cv2.countNonZero(final_mask)

        # Calcular porcentaje de cobertura
        coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0

        # Guardar imágenes de debug si está habilitado
        if save_debug:
            # Histograma de luminosidad
            plt.figure(figsize=(10, 6))
            plt.hist(v_channel.ravel(), 256, [0, 256])
            plt.title("Distribución de Luminosidad")
            plt.axvline(x=SprayAnalyzer.LOWER_THRESHOLD, color="r", linestyle="--")
            plt.savefig("debug_histogram.jpg")
            plt.close()

            # Máscaras intermedias
            cv2.imwrite("debug_v_channel.jpg", v_channel)
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_fluorescence_mask.jpg", fluorescence_mask)
            cv2.imwrite("debug_cleaned_mask.jpg", cleaned_mask)
            cv2.imwrite("debug_final_mask.jpg", final_mask)
            
            # Visualización de resultados
            overlay = image.copy()
            overlay[final_mask > 0] = [0, 255, 0]  # Áreas de spray en verde
            cv2.imwrite("debug_overlay.jpg", overlay)

        return round(coverage, 2), leaf_area, sprayed_area

    @staticmethod
    def generate_image_id() -> str:
        """
        Genera un ID único para cada imagen
        """
        return str(uuid4())

    @staticmethod
    def calculate_batch_summary(analyses: List[ImageAnalysisResponse]) -> dict:
        """
        Calcula estadísticas resumen para un lote de imágenes analizadas
        """
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
