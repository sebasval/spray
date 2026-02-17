import cv2
import numpy as np
from uuid import uuid4
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import base64
import logging
from app.models.image import ImageAnalysisResponse

logger = logging.getLogger(__name__)


class SprayAnalyzer:
    MIN_SPRAY_AREA = 5
    MORPH_KERNEL_SIZE = 3
    
    @staticmethod
    def _detect_leaf_mask(bw_image: np.ndarray) -> np.ndarray:
        """
        Detecta la máscara de la hoja en la imagen B&W de alto contraste.
        Trabaja sobre la imagen ya filtrada (CLAHE + sigmoid).
        
        En B&W de alto contraste:
          - Fondo Photoroom (gris/negro) → comprimido a ~0-30 (muy oscuro)
          - Hoja → valores medios/altos (~50-220)
          - Spray → valores muy altos (~240-255)
        Otsu separa fondo oscuro de hoja+spray de forma limpia.
        """
        if len(bw_image.shape) == 3:
            gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bw_image
        
        # Otsu en la imagen B&W: el alto contraste genera distribución bimodal
        # clara entre fondo oscuro (~0-30) y hoja (~50-255)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Umbral fijo bajo para reforzar eliminación de fondo negro
        _, fixed_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # AND: debe pasar ambos filtros
        combined_mask = cv2.bitwise_and(otsu_mask, fixed_mask)
        
        # Operaciones morfológicas para limpiar y conectar la máscara
        kernel = np.ones((7, 7), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Rellenar cualquier agujero dentro de la hoja
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray)
        
        # Tomar solo el contorno más grande (la hoja)
        image_area = gray.shape[0] * gray.shape[1]
        min_contour_area = image_area * 0.01
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        
        if not valid_contours:
            return np.zeros_like(gray)
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        leaf_area = cv2.countNonZero(final_mask)
        leaf_pct = (leaf_area / image_area) * 100
        logger.info(f"Leaf detection: area={leaf_area}, {leaf_pct:.1f}% of image")
        
        # Si la hoja ocupa >90% de la imagen, probablemente falló la detección
        if leaf_pct > 90:
            logger.warning(f"Leaf mask too large ({leaf_pct:.1f}%), retrying with stricter threshold")
            _, stricter_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            stricter_mask = cv2.morphologyEx(stricter_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            stricter_mask = cv2.morphologyEx(stricter_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            contours2, _ = cv2.findContours(stricter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid2 = [c for c in contours2 if cv2.contourArea(c) >= min_contour_area]
            if valid2:
                largest2 = max(valid2, key=cv2.contourArea)
                final_mask = np.zeros_like(gray)
                cv2.drawContours(final_mask, [largest2], -1, 255, -1)
                leaf_area2 = cv2.countNonZero(final_mask)
                logger.info(f"Stricter leaf detection: area={leaf_area2}, {(leaf_area2/image_area)*100:.1f}% of image")

        return final_mask

    @staticmethod
    def _filter_valid_droplets(fluorescence_mask: np.ndarray) -> Tuple[np.ndarray, bool]:
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
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_DROPLET_SIZE:
                filtered_mask[labels == i] = 255
                valid_droplets_count += 1
        
        has_valid_droplets = valid_droplets_count > 0
                        
        return filtered_mask, has_valid_droplets

    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Analiza una imagen para detectar cobertura de spray.
        
        Pipeline completo:
          1. Imagen original → Filtro B&W (CLAHE + sigmoid)
          2. B&W → Detección de hoja (Otsu)
          3. B&W → Detección de spray (umbral adaptativo dentro de la hoja)
          4. B&W → Imagen de salida con overlay de spray detectado
        
        Todo el análisis trabaja sobre la imagen B&W filtrada.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0, 0, 0, None

        # ============================================================
        # PASO 1: Filtro B&W de alto contraste
        # Replica el filtro "Contraste alto de B&W" de Microsoft Designer
        # CLAHE + curva S sigmoid (conserva tonos de gris, no es binario)
        # De aquí en adelante, TODO trabaja sobre bw_image
        # ============================================================
        gray_pre = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        high_contrast = clahe.apply(gray_pre)
        normalized = high_contrast.astype(np.float32) / 255.0
        contrast_strength = 12
        midpoint = 0.5
        curved = 1.0 / (1.0 + np.exp(-contrast_strength * (normalized - midpoint)))
        bw_image = (curved * 255).astype(np.uint8)

        # ============================================================
        # PASO 2: Detección de hoja en la imagen B&W
        # ============================================================
        leaf_mask = SprayAnalyzer._detect_leaf_mask(bw_image)
        leaf_area = cv2.countNonZero(leaf_mask)
        
        if leaf_area == 0:
            bw_bgr = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
            _, buffer = cv2.imencode('.jpg', bw_bgr)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            return 0.0, 0, 0, processed_image_base64

        # ============================================================
        # PASO 3: Detección de spray en la imagen B&W
        # Las gotas de spray = píxeles brillantes (blancos) sobre la hoja
        # ============================================================
        leaf_pixels = bw_image[leaf_mask > 0]
        if len(leaf_pixels) > 0:
            median_val = float(np.median(leaf_pixels))
            std_val = float(np.std(leaf_pixels))
            spray_threshold = median_val + 1.0 * std_val
            spray_threshold = min(spray_threshold, 220)
            spray_threshold = max(spray_threshold, 100)
            logger.info(
                f"B&W spray stats: median={median_val:.1f}, std={std_val:.1f}, threshold={spray_threshold:.1f}"
            )
        else:
            spray_threshold = 150
        
        fluorescence_mask = cv2.inRange(bw_image, int(spray_threshold), 255)
        fluorescence_mask = cv2.bitwise_and(fluorescence_mask, leaf_mask)

        # Filtrado de gotas válidas (eliminar ruido por tamaño)
        filtered_mask, has_valid_droplets = SprayAnalyzer._filter_valid_droplets(fluorescence_mask)
        
        # PASO 4: Calcular resultados
        if has_valid_droplets:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage = (sprayed_area / leaf_area * 100)
            coverage = min(coverage, 100.0) 
        else:
            sprayed_area = 0
            coverage = 0.0
        
        # PASO 5: Crear imagen de salida sobre la imagen B&W
        # Convertir B&W a BGR para poder superponer el overlay en color
        processed_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
        processed_image[filtered_mask > 0] = [0, 255, 255]  # Amarillo/cyan para spray detectado
        
        # Codificar a Base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Debug: guardar imágenes intermedias
        if save_debug:
            cv2.imwrite("debug_bw_highcontrast.jpg", bw_image)
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_fluorescence_mask.jpg", fluorescence_mask) 
            cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)       
            cv2.imwrite("debug_result.jpg", processed_image)
            
            # Histograma de intensidad B&W dentro de la hoja
            plt.figure(figsize=(10, 6))
            leaf_intensities = bw_image[leaf_mask > 0]
            if leaf_intensities.size > 0:
                plt.hist(leaf_intensities.ravel(), 256, [0, 256], color='gray', alpha=0.7, label='Hoja (B&W)')
            
            spray_intensities = bw_image[filtered_mask > 0]
            if spray_intensities.size > 0:
                plt.hist(spray_intensities.ravel(), 256, [0, 256], color='cyan', alpha=0.7, label='Spray detectado')
            
            if len(leaf_pixels) > 0:
                plt.axvline(x=spray_threshold, color='red', linestyle='--', label=f'Umbral spray ({spray_threshold:.0f})')
            
            plt.title("Distribución de Intensidad B&W — Hoja vs Spray")
            plt.xlabel("Intensidad (0=negro, 255=blanco)")
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
