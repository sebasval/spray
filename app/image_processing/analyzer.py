import cv2
import numpy as np
from uuid import uuid4
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import base64
from app.models.image import ImageAnalysisResponse


class SprayAnalyzer:
    # Parámetros ajustados para fluorescencia
    MIN_SPRAY_AREA = 5  # Mínimo absoluto de área para considerar una gota
    MORPH_KERNEL_SIZE = 3
    
    @staticmethod
    def _detect_leaf_mask(image: np.ndarray) -> np.ndarray:
        """
        Detecta la máscara de la hoja usando múltiples canales.
        Ajustado para mejorar la robustez en los bordes y evitar el fondo.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Umbralización simple para eliminar el fondo negro obvio
        _, base_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Umbral Otsu en el canal de Saturación (ayuda con los bordes de la hoja)
        sat = hsv[:, :, 1]
        _, sat_otsu_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combinar máscaras
        combined_mask = cv2.bitwise_or(base_mask, sat_otsu_mask)
        
        # Operaciones morfológicas para limpiar y conectar la máscara
        kernel = np.ones((7, 7), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Rellenar cualquier agujero dentro de la hoja
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(cleaned_mask)
            
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)

        return final_mask

    @staticmethod
    def _detect_fluorescence(image: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detecta la fluorescencia del spray usando OpenCV.
        Combina dos estrategias:
          1) Detección por BRILLO: áreas significativamente más brillantes que el fondo
             de la hoja (captura spray blanco/brillante con baja saturación).
          2) Detección por COLOR: áreas con tono cian/azul y saturación moderada
             (captura spray con fluorescencia azulada).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        val_channel = hsv[:, :, 2]

        # ================================================================
        # MÉTODO 1: Detección por BRILLO RELATIVO (spray blanco/brillante)
        # ================================================================
        # El spray fluorescente aparece como áreas mucho más brillantes que
        # el color base de la hoja. Calculamos un umbral adaptativo.
        leaf_pixels_val = val_channel[leaf_mask > 0]

        if len(leaf_pixels_val) > 0:
            median_val = float(np.median(leaf_pixels_val))
            std_val = float(np.std(leaf_pixels_val))
            # Umbral: mediana + 1.0*std, pero mínimo 120 para evitar falsos positivos
            brightness_threshold = max(median_val + 1.0 * std_val, 120)
            brightness_mask = (val_channel > brightness_threshold).astype(np.uint8) * 255
        else:
            brightness_mask = np.zeros_like(leaf_mask)

        # ================================================================
        # MÉTODO 2: Detección por COLOR (cian/azul con saturación moderada)
        # ================================================================
        # Rango HSV para tonos cian/azul (con brillo mínimo de 80 para evitar falsos en zonas oscuras)
        lower_cyan_blue = np.array([85, 30, 80])
        upper_cyan_blue = np.array([130, 255, 255])
        hsv_color_mask = cv2.inRange(hsv, lower_cyan_blue, upper_cyan_blue)

        # Filtro BGR: canales azul/verde dominan sobre rojo
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        cyan_check = (g > r + 5) & (b > r)
        blue_check = (b > r + 15) & (b > g + 5)
        bgr_mask = (cyan_check | blue_check).astype(np.uint8) * 255

        color_mask = cv2.bitwise_and(hsv_color_mask, bgr_mask)

        # ================================================================
        # COMBINAR ambos métodos
        # ================================================================
        combined = cv2.bitwise_or(brightness_mask, color_mask)

        # La máscara final DEBE estar dentro de la máscara de la hoja
        fluorescence_mask = cv2.bitwise_and(combined, leaf_mask)

        # Limpieza morfológica
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        fluorescence_mask = cv2.dilate(fluorescence_mask, kernel_small, iterations=1)

        mean_intensity = (
            float(np.mean(hsv[:, :, 2][fluorescence_mask > 0]))
            if cv2.countNonZero(fluorescence_mask) > 0
            else 0.0
        )

        return fluorescence_mask, mean_intensity

    @staticmethod
    def _filter_valid_droplets(image: np.ndarray, fluorescence_mask: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filtra la máscara de fluorescencia por tamaño para eliminar ruido.
        """
        total_fluorescence_area = cv2.countNonZero(fluorescence_mask)

        if total_fluorescence_area == 0:
            return np.zeros_like(fluorescence_mask), False

        MIN_DROPLET_SIZE = SprayAnalyzer.MIN_SPRAY_AREA

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fluorescence_mask, connectivity=8
        )
        
        filtered_mask = np.zeros_like(fluorescence_mask)
        valid_droplets_count = 0
        
        for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_DROPLET_SIZE:
                filtered_mask[labels == i] = 255
                valid_droplets_count += 1
        
        has_valid_droplets = valid_droplets_count > 0
                        
        return filtered_mask, has_valid_droplets

    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Analiza una imagen para detectar cobertura de spray usando fluorescencia UV.
        
        Combina detección por brillo relativo (áreas blancas/brillantes) y por color
        (cian/azul) para máxima cobertura de detección.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0, 0, 0, None

        # Guardar imagen original para la salida visual
        original_image = image.copy()

        # ============================================================
        # Pre-procesamiento: Contraste alto B&W
        # Replica el filtro "Contraste alto de B&W" de Microsoft Designer
        # Usa CLAHE + curva S sigmoid (NO binario — conserva tonos de gris)
        # ============================================================
        gray_pre = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CLAHE para realzar contraste local
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        high_contrast = clahe.apply(gray_pre)
        # Curva S sigmoid — empuja oscuros a negro y claros a blanco
        # pero conserva gradiente (no es binario puro)
        normalized = high_contrast.astype(np.float32) / 255.0
        contrast_strength = 12  # pendiente de la curva (más alto = más contraste)
        midpoint = 0.5
        curved = 1.0 / (1.0 + np.exp(-contrast_strength * (normalized - midpoint)))
        bw_image = (curved * 255).astype(np.uint8)

        # ============================================================
        # Detección de hoja usando la imagen original (antes del B&W)
        # ============================================================
        denoised_orig = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        leaf_mask = SprayAnalyzer._detect_leaf_mask(denoised_orig)
        leaf_area = cv2.countNonZero(leaf_mask)
        
        if leaf_area == 0:
            _, buffer = cv2.imencode('.jpg', original_image)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            return 0.0, 0, 0, processed_image_base64

        # ============================================================
        # Detección de spray en la imagen B&W de alto contraste
        # Las gotas de spray aparecen como píxeles brillantes (blancos)
        # sobre la hoja que es gris/oscura
        # ============================================================
        # Calcular umbral adaptativo dentro de la hoja
        leaf_pixels = bw_image[leaf_mask > 0]
        if len(leaf_pixels) > 0:
            median_val = float(np.median(leaf_pixels))
            std_val = float(np.std(leaf_pixels))
            # Spray = píxeles más brillantes que la mediana + 1 desviación estándar
            spray_threshold = median_val + 1.0 * std_val
            spray_threshold = min(spray_threshold, 220)  # máximo 220 para no perder spray
            spray_threshold = max(spray_threshold, 100)   # mínimo 100 para evitar falsos en fondo
            import logging
            logging.getLogger(__name__).info(
                f"B&W stats: median={median_val:.1f}, std={std_val:.1f}, threshold={spray_threshold:.1f}"
            )
        else:
            spray_threshold = 150
        
        fluorescence_mask = cv2.inRange(bw_image, int(spray_threshold), 255)
        fluorescence_mask = cv2.bitwise_and(fluorescence_mask, leaf_mask)

        # Filtrado de gotas válidas (eliminar ruido por tamaño)
        filtered_mask, has_valid_droplets = SprayAnalyzer._filter_valid_droplets(original_image, fluorescence_mask)
        
        # 5. Calcular Resultados
        if has_valid_droplets:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage = (sprayed_area / leaf_area * 100)
            coverage = min(coverage, 100.0) 
        else:
            sprayed_area = 0
            coverage = 0.0
        
        # 6. Crear Imagen de Salida (usar imagen original, no la pre-procesada)
        processed_image = original_image.copy()
        processed_image[filtered_mask > 0] = [0, 255, 255]  # Amarillo para áreas con spray
        
        # 7. Codificar a Base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 8. Guardar Debug
        if save_debug:
            cv2.imwrite("debug_bw_highcontrast.jpg", bw_image)
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_fluorescence_mask.jpg", fluorescence_mask) 
            cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)       
            cv2.imwrite("debug_result.jpg", processed_image)
            
            plt.figure(figsize=(10, 6))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_in_leaf = hsv[:, :, 0][leaf_mask > 0]
            if hue_in_leaf.size > 0:
                plt.hist(hue_in_leaf.ravel(), 180, [0, 180], color='gray', alpha=0.7, label='Hoja Completa')
            
            hue_in_fluorescence = hsv[:, :, 0][filtered_mask > 0]
            if hue_in_fluorescence.size > 0:
                plt.hist(hue_in_fluorescence.ravel(), 180, [0, 180], color='cyan', alpha=0.7, label='Fluorescencia Detectada')
            
            plt.title("Distribución de Matiz (Hue) dentro de la Hoja y Fluorescencia")
            plt.xlabel("Matiz (Hue)")
            plt.ylabel("Píxeles")
            plt.legend()
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
