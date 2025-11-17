import cv2
import numpy as np
from uuid import uuid4
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import base64
from app.models.image import ImageAnalysisResponse


class SprayAnalyzer:
    # Parámetros ajustados para fluorescencia azul
    MIN_SPRAY_AREA = 5  # Mantenemos un mínimo absoluto
    MORPH_KERNEL_SIZE = 3
    
    @staticmethod
    def _detect_leaf_mask(image: np.ndarray) -> np.ndarray:
        """
        Detecta la máscara de la hoja usando múltiples canales
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, base_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        _, otsu_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        sv_mask = ((sat > 10) & (val > 15)).astype(np.uint8) * 255

        combined = cv2.bitwise_or(base_mask, otsu_mask)
        combined = cv2.bitwise_or(combined, sv_mask)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cleaned_mask  # Devuelve máscara vacía si no hay contornos
            
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        return final_mask

    @staticmethod
    def _detect_fluorescence(image: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detecta la fluorescencia CÍAN/AZUL y rechaza PÚRPURA
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1) Detección por rango HSV (Cian/Azul)
        # Matiz (H) en OpenCV: Cían (85-100), Azul (100-130). Púrpura/Magenta es 140+.
        # Este rango excluye el púrpura.
        lower_cyan_blue = np.array([85, 40, 50])
        upper_cyan_blue = np.array([135, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_cyan_blue, upper_cyan_blue)

        # 2) Filtro BGR para Cían (Verde > Rojo)
        # Esto es crucial para eliminar el Púrpura (Rojo > Verde)
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        
        # El canal verde debe ser más brillante que el rojo
        # El '+ 5' da un pequeño margen para evitar ruido
        bgr_cyan_mask = (g > r + 5).astype(np.uint8) * 255

        # 3) Requisitos mínimos de valor/saturación para evitar sombras
        sat_ok = (hsv[:, :, 1] > 20).astype(np.uint8) * 255
        val_ok = (hsv[:, :, 2] > 25).astype(np.uint8) * 255

        # 4) Combinar máscaras: DEBE ser cían/azul (HSV) Y (Verde > Rojo) (BGR)
        combined = cv2.bitwise_and(hsv_mask, bgr_cyan_mask)
        combined = cv2.bitwise_and(combined, sat_ok)
        combined = cv2.bitwise_and(combined, val_ok)
        fluorescence_mask = cv2.bitwise_and(combined, leaf_mask)

        # 5) Limpieza morfológica:
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_OPEN, kernel3)
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_CLOSE, kernel5)
        fluorescence_mask = cv2.dilate(fluorescence_mask, kernel3, iterations=1)

        mean_intensity = (
            float(np.mean(hsv[:, :, 2][fluorescence_mask > 0]))
            if cv2.countNonZero(fluorescence_mask) > 0
            else 0.0
        )

        return fluorescence_mask, mean_intensity

    @staticmethod
    def _filter_valid_droplets(image: np.ndarray, fluorescence_mask: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filtra la máscara de fluorescencia por tamaño para eliminar ruido.
        La lógica de color principal ya se aplicó en _detect_fluorescence.
        """
        total_leaf_area = cv2.countNonZero(leaf_mask)
        total_fluorescence_area = cv2.countNonZero(fluorescence_mask)

        if total_leaf_area == 0 or total_fluorescence_area == 0:
            return fluorescence_mask, False

        # Establecer tamaño mínimo de gota (valor original)
        MIN_DROPLET_SIZE = SprayAnalyzer.MIN_SPRAY_AREA  # Usamos el valor de clase, 5

        # Etiquetar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fluorescence_mask, connectivity=8
        )
        
        filtered_mask = np.zeros_like(fluorescence_mask)
        valid_droplets_count = 0
        
        # Empezamos desde 1 para evitar el fondo (etiqueta 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_DROPLET_SIZE:
                # El filtro de color ya se hizo. Solo filtramos por tamaño.
                filtered_mask[labels == i] = 255
                valid_droplets_count += 1
        
        # Considerar válido si se encuentra algún componente que pase el filtro
        has_valid_droplets = valid_droplets_count > 0
                        
        return filtered_mask, has_valid_droplets

    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Analiza una imagen para detectar cobertura de spray usando fluorescencia UV
        """
        # 1. Decodificar y Pre-procesar
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0, 0, 0, None
            
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # 2. Detección de Hoja
        leaf_mask = SprayAnalyzer._detect_leaf_mask(denoised)
        leaf_area = cv2.countNonZero(leaf_mask)
        
        if leaf_area == 0:
            _, buffer = cv2.imencode('.jpg', image)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            return 0.0, 0, 0, processed_image_base64
            
        # 3. Detección de Fluorescencia (filtra por color CÍAN vs PÚRPURA)
        fluorescence_mask, _ = SprayAnalyzer._detect_fluorescence(denoised, leaf_mask)
        
        # 4. Filtrado de Gotas Válidas (filtra por TAMAÑO/ruido)
        filtered_mask, has_valid_droplets = SprayAnalyzer._filter_valid_droplets(denoised, fluorescence_mask, leaf_mask)
        
        # 5. Calcular Resultados
        if has_valid_droplets:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage = (sprayed_area / leaf_area * 100)
        else:
            sprayed_area = 0
            coverage = 0.0
        
        # 6. Crear Imagen de Salida
        processed_image = image.copy()
        processed_image[filtered_mask > 0] = [0, 255, 255]  # Amarillo para áreas con spray
        
        # 7. Codificar a Base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 8. Guardar Debug
        if save_debug:
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_fluorescence_mask.jpg", fluorescence_mask)  # Sospechosos (post-color)
            cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)  # Confirmados (post-tamaño)
            cv2.imwrite("debug_result.jpg", processed_image)
            
            plt.figure(figsize=(10, 6))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_in_leaf = hsv[:, :, 0][leaf_mask > 0]
            if hue_in_leaf.size > 0:
                plt.hist(hue_in_leaf.ravel(), 180, [0, 180])
                plt.title("Distribución de Matiz (Hue) dentro de la Hoja")
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
            "total_area_sprayed": sum(sprayed_areas),  # ERROR CORREGIDO: Debería ser sum(sprayed_areas)
            "global_coverage": round(
                sum(sprayed_areas) / sum(total_areas) * 100, 2
            ) if sum(total_areas) > 0 else 0,
        }
