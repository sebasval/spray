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

    # Rangos razonables para objeto sobre fondo oscuro (flor, hoja, etc.)
    MIN_OBJECT_PCT = 2.0
    MAX_OBJECT_PCT = 85.0

    @staticmethod
    def _preprocess_for_segmentation(image: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento adaptativo para imágenes con fondo oscuro y objeto iluminado.
        Usa CLAHE suave + normalización para mejorar contraste sin sobreajustar.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # CLAHE adaptativo: clipLimit bajo para no exagerar en imágenes ya contrastadas
        mean_val = np.mean(gray)
        clip_limit = 3.0 if mean_val < 80 else 2.5  # Menos agresivo si ya hay contraste
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Curva sigmoid suave para separar fondo oscuro de objeto
        normalized = enhanced.astype(np.float32) / 255.0
        contrast_strength = 10
        midpoint = 0.45  # Ligeramente más bajo para fondos muy oscuros
        curved = 1.0 / (1.0 + np.exp(-contrast_strength * (normalized - midpoint)))
        return (curved * 255).astype(np.uint8)

    @staticmethod
    def _get_object_mask_from_threshold(
        gray: np.ndarray, threshold: int, kernel_size: int = 5
    ) -> Tuple[np.ndarray, float]:
        """Aplica un umbral y devuelve la máscara del objeto más grande con su % de área."""
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = gray.shape[0] * gray.shape[1]
        min_area = image_area * 0.005

        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            return np.zeros_like(gray), 0.0

        largest = max(valid, key=cv2.contourArea)
        result = np.zeros_like(gray)
        cv2.drawContours(result, [largest], -1, 255, -1)
        pct = (cv2.countNonZero(result) / image_area) * 100
        return result, pct

    @staticmethod
    def _detect_object_mask_robust(bw_image: np.ndarray) -> np.ndarray:
        """
        Detección robusta del objeto principal (flor, hoja, etc.) sobre fondo oscuro.
        Prueba múltiples estrategias y elige la que da un tamaño de objeto razonable.
        Funciona con distintos lotes sin necesidad de ajustes manuales.
        """
        if len(bw_image.shape) == 3:
            gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bw_image

        image_area = gray.shape[0] * gray.shape[1]
        min_pct, max_pct = SprayAnalyzer.MIN_OBJECT_PCT, SprayAnalyzer.MAX_OBJECT_PCT

        candidates: List[Tuple[np.ndarray, float, str]] = []

        # Estrategia 1: Otsu (adaptativo a la distribución de la imagen)
        try:
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask, pct = SprayAnalyzer._get_object_mask_from_threshold(gray, int(otsu_thresh))
            if mask is not None and pct > 0:
                candidates.append((mask.copy(), pct, "otsu"))
        except Exception as e:
            logger.debug(f"Otsu falló: {e}")

        # Estrategia 2: Percentiles del histograma global
        # Percentiles 65-85 suelen marcar el límite fondo/objeto en imágenes oscuras
        flat = gray.ravel()
        if len(flat) > 100:
            for p in [65, 70, 75, 80]:
                thresh = int(np.percentile(flat, p))
                if 15 <= thresh <= 220:
                    mask, pct = SprayAnalyzer._get_object_mask_from_threshold(gray, thresh)
                    if pct > 0 and min_pct <= pct <= max_pct:
                        candidates.append((mask.copy(), pct, f"p{p}"))

        # Estrategia 3: Umbrales fijos en rango típico para fondos oscuros
        for thresh in [15, 25, 35, 45]:
            mask, pct = SprayAnalyzer._get_object_mask_from_threshold(gray, thresh)
            if pct > 0 and min_pct <= pct <= max_pct:
                candidates.append((mask.copy(), pct, f"fixed_{thresh}"))

        # Estrategia 4: Si Otsu dio objeto demasiado grande, intentar umbrales más altos
        if candidates:
            otsu_candidates = [c for c in candidates if c[2] == "otsu" and c[1] > max_pct]
            if otsu_candidates:
                otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                for extra in [30, 50, 70]:
                    thresh = min(int(otsu_thresh) + extra, 200)
                    mask, pct = SprayAnalyzer._get_object_mask_from_threshold(gray, thresh)
                    if min_pct <= pct <= max_pct:
                        candidates.append((mask.copy(), pct, f"otsu+{extra}"))

        # Elegir el mejor candidato: preferir tamaño intermedio (20-60%)
        def score(c: Tuple[np.ndarray, float, str]) -> float:
            _, pct, _ = c
            if not (min_pct <= pct <= max_pct):
                return -1
            # Preferir objetos que ocupen 20-60% (típico para flor/hoja centrada)
            ideal = 40
            return -abs(pct - ideal)

        valid = [c for c in candidates if min_pct <= c[1] <= max_pct]
        if valid:
            best = max(valid, key=score)
            logger.info(f"Object detection: {best[2]}, area={best[1]:.1f}% of image")
            return best[0]

        # Fallback: usar el candidato con menor desviación del rango válido
        if candidates:
            fallback = min(candidates, key=lambda c: abs(c[1] - 40) if c[1] > 0 else 999)
            logger.warning(f"Using fallback detection: {fallback[2]}, area={fallback[1]:.1f}%")
            return fallback[0]

        # Último recurso: Otsu sin validación
        try:
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask, _ = SprayAnalyzer._get_object_mask_from_threshold(gray, int(otsu_thresh))
            return mask if mask is not None else np.zeros_like(gray)
        except Exception:
            return np.zeros_like(gray)

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
        Usa ImageJ (via PyImageJ) para threshold y análisis de partículas,
        garantizando resultados 100% compatibles con ImageJ.
        """
        try:
            from app.image_processing.imagej_analyzer import analyze_with_imagej
            logger.info("Using ImageJ engine for analysis")
            return analyze_with_imagej(image_bytes, save_debug)
        except Exception as e:
            logger.error(f"ImageJ analysis failed, falling back to OpenCV: {e}", exc_info=True)
            return SprayAnalyzer._analyze_image_opencv(image_bytes, save_debug)

    @staticmethod
    def _analyze_image_opencv(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Fallback: análisis con OpenCV puro (sin ImageJ).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0, 0, 0, None

        bw_image = SprayAnalyzer._preprocess_for_segmentation(image)
        leaf_mask = SprayAnalyzer._detect_object_mask_robust(bw_image)
        leaf_area = cv2.countNonZero(leaf_mask)
        
        if leaf_area == 0:
            bw_bgr = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
            _, buffer = cv2.imencode('.jpg', bw_bgr)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            return 0.0, 0, 0, processed_image_base64

        leaf_pixels = bw_image[leaf_mask > 0]
        if len(leaf_pixels) > 100:
            spray_threshold = float(np.percentile(leaf_pixels, 85))
            spray_threshold = np.clip(spray_threshold, 60, 245)
        else:
            spray_threshold = 150

        fluorescence_mask = cv2.inRange(bw_image, int(spray_threshold), 255)
        fluorescence_mask = cv2.bitwise_and(fluorescence_mask, leaf_mask)
        filtered_mask, has_valid_droplets = SprayAnalyzer._filter_valid_droplets(fluorescence_mask)
        
        if has_valid_droplets:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage = (sprayed_area / leaf_area * 100)
            coverage = min(coverage, 100.0) 
        else:
            sprayed_area = 0
            coverage = 0.0
        
        processed_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
        processed_image[filtered_mask > 0] = [0, 255, 255]
        
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

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
