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
        # Convertir a espacios de color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Suavizar para Otsu
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Máscara 1: umbral fijo muy permisivo (evita perder hojas oscuras)
        _, base_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # Máscara 2: Otsu en gris para separar fondo negro de hoja
        _, otsu_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Máscara 3: píxeles con saturación y valor mínimos (descarta fondo negro)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        sv_mask = ((sat > 10) & (val > 15)).astype(np.uint8) * 255

        # Combinar máscaras y limpiar
        combined = cv2.bitwise_or(base_mask, otsu_mask)
        combined = cv2.bitwise_or(combined, sv_mask)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
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

        # Calcular el valor medio del canal azul (en BGR) en la máscara de hoja
        # Esto nos ayuda a diferenciar entre imágenes con fluorescencia real y sin ella
        mean_blue = np.mean(image[:, :, 0][leaf_mask > 0]) if cv2.countNonZero(leaf_mask) > 0 else 0

        # 1) Detección por rango HSV (azules)
        if mean_blue > 70:
            # Para imágenes con fluorescencia UV real que tienen azul brillante
            lower_blue = np.array([90, 45, 45])  # ligeramente más permisivo
            upper_blue = np.array([140, 255, 255])
        else:
            # Para imágenes normales, mantener parámetros algo más estrictos
            lower_blue = np.array([95, 65, 60])
            upper_blue = np.array([125, 255, 255])
        hsv_blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 2) Índice de exceso de azul con umbral adaptativo (robusto a iluminación)
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        blue_excess = b - 0.5 * (g + r)
        # Normalizar a 0-255 para operar en uint8 de forma estable
        blue_excess_norm = blue_excess - blue_excess.min()
        max_val = blue_excess_norm.max() if blue_excess_norm.max() > 0 else 1.0
        blue_excess_norm = (blue_excess_norm / max_val * 255).astype(np.uint8)

        # Umbral adaptativo dentro de la hoja: percentil alto o media+desv
        bex_in_leaf = blue_excess_norm[leaf_mask > 0]
        if bex_in_leaf.size > 0:
            # Umbral por percentil para capturar zonas más intensas
            p88 = np.percentile(bex_in_leaf, 84)
            mu = float(np.mean(bex_in_leaf))
            sigma = float(np.std(bex_in_leaf))
            adaptive_thr = max(p88, mu + 1.0 * sigma)
        else:
            adaptive_thr = 0
        bex_mask = (blue_excess_norm >= adaptive_thr).astype(np.uint8) * 255

        # 2b) Umbral de Otsu (calculado solo con píxeles de hoja)
        if bex_in_leaf.size > 0:
            # aplicar Otsu sobre el histograma de la hoja
            # para generar un corte automático adicional
            hist_otsu_src = bex_in_leaf.astype(np.uint8)
            # Otsu requiere imagen; recreamos pequeña imagen 1D
            otsu_thr, _ = cv2.threshold(hist_otsu_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bex_otsu_mask = (blue_excess_norm >= otsu_thr).astype(np.uint8) * 255
        else:
            bex_otsu_mask = np.zeros_like(bex_mask)

        # 3) Requisitos mínimos de valor/saturación para evitar sombras
        # Más permisivos para escenas oscuras; aún evitamos sombras puras
        sat_ok = (hsv[:, :, 1] > 15).astype(np.uint8) * 255
        val_ok = (hsv[:, :, 2] > 20).astype(np.uint8) * 255

        # 4) Combinar máscaras y restringir a la hoja
        combined = cv2.bitwise_or(hsv_blue_mask, bex_mask)
        combined = cv2.bitwise_or(combined, bex_otsu_mask)
        combined = cv2.bitwise_and(combined, sat_ok)
        combined = cv2.bitwise_and(combined, val_ok)
        fluorescence_mask = cv2.bitwise_and(combined, leaf_mask)

        # 5) Limpieza morfológica: abrir para ruido y cerrar para rellenar huecos
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_OPEN, kernel3)
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_CLOSE, kernel5)
        # Dilatar un poco para recuperar gotas muy pequeñas perdidas por umbrales
        fluorescence_mask = cv2.dilate(fluorescence_mask, kernel3, iterations=1)

        # Intensidad media (para fines de debug) usando el canal V del HSV en las zonas detectadas
        mean_intensity = (
            float(np.mean(hsv[:, :, 2][fluorescence_mask > 0]))
            if cv2.countNonZero(fluorescence_mask) > 0
            else 0.0
        )

        return fluorescence_mask, mean_intensity

    @staticmethod
    def _filter_valid_droplets(image: np.ndarray, fluorescence_mask: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filtra la máscara de fluorescencia para verificar si contiene gotas reales
        
        Returns:
            Tuple[np.ndarray, bool]: Máscara filtrada y booleano indicando si contiene gotas válidas
        """
        # Características globales
        total_leaf_area = cv2.countNonZero(leaf_mask)
        total_fluorescence_area = cv2.countNonZero(fluorescence_mask)
        mean_blue_leaf = (
            float(np.mean(image[:, :, 0][leaf_mask > 0]))
            if total_leaf_area > 0
            else 0.0
        )

        # Establecer tamaño mínimo de gota adaptativo al área de la hoja
        # más permisivo para hojas pequeñas y escenas de baja señal
        adaptive_min_size = int(max(8, 0.00008 * float(total_leaf_area)))
        MIN_DROPLET_SIZE = adaptive_min_size

        # Etiquetar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fluorescence_mask, connectivity=8
        )
        
        # Filtrar componentes demasiado pequeños (posible ruido)
        filtered_mask = np.zeros_like(fluorescence_mask)
        valid_droplets_count = 0
        
        # Si la fluorescencia cubre casi toda la hoja pero no tiene componentes distinguibles, probablemente es falso positivo
        # Umbral más permisivo: 85% en lugar de 70%
        if total_fluorescence_area > 0.85 * total_leaf_area and num_labels < 5:
            return filtered_mask, False
        
        # Empezamos desde 1 para evitar el fondo (etiqueta 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_DROPLET_SIZE:
                comp_mask = labels == i
                # Pureza de azul relativa al rojo/verde dentro del componente
                b_vals = image[:, :, 0][comp_mask]
                g_vals = image[:, :, 1][comp_mask]
                r_vals = image[:, :, 2][comp_mask]
                blue_ratio = (float(np.mean(b_vals)) + 1.0) / (float(np.mean(g_vals + r_vals)) / 2.0 + 1.0)
                # exigir predominio de azul
                if blue_ratio < 1.10:
                    continue
                # Diferencia respecto al promedio de la hoja
                if float(np.mean(b_vals)) < mean_blue_leaf + 4.0:
                    continue
                # Si pasa filtros de color, mantener
                filtered_mask[comp_mask] = 255
                valid_droplets_count += 1
        
        # Verificación de gotas: menos estricta y con tolerancia cuando hay cobertura pequeña pero real
        has_valid_droplets = valid_droplets_count >= 2
        if not has_valid_droplets and valid_droplets_count == 1:
            coverage_ratio = (total_fluorescence_area / total_leaf_area) if total_leaf_area > 0 else 0.0
            if coverage_ratio >= 0.02:  # al menos 2% de la hoja con señal
                has_valid_droplets = True

        # Anti falso-positivo: si hay muy pocas regiones y el azul medio no supera a la hoja, invalidar
        if has_valid_droplets and valid_droplets_count <= 2:
            mean_blue_in_mask = (
                float(np.mean(image[:, :, 0][filtered_mask > 0]))
                if cv2.countNonZero(filtered_mask) > 0
                else 0.0
            )
            if mean_blue_in_mask < mean_blue_leaf + 6.0:
                has_valid_droplets = False

        # Anti falso-positivo adicional: gran cobertura con pocas regiones => resplandor uniforme
        filtered_area = cv2.countNonZero(filtered_mask)
        coverage_ratio = (filtered_area / total_leaf_area) if total_leaf_area > 0 else 0.0
        if has_valid_droplets and coverage_ratio > 0.35 and valid_droplets_count < 15:
            has_valid_droplets = False

        # Si hay suficientes regiones, comprobar circularidad media para evitar formas alargadas
        if has_valid_droplets and valid_droplets_count > 0:
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circularities = []
            for c in contours[:50]:  # limitar por rendimiento
                area = cv2.contourArea(c)
                if area <= 0:
                    continue
                per = cv2.arcLength(c, True)
                if per <= 0:
                    continue
                circ = 4 * np.pi * area / (per * per)
                circularities.append(circ)
            if circularities:
                median_circ = float(np.median(circularities))
                if median_circ < 0.18 and valid_droplets_count < 10:
                    has_valid_droplets = False

        # Chequeos de textura y saturación para evitar resplandor uniforme sin gotas
        sat_mask = None
        speckle_ratio = None
        too_large_component = None
        if has_valid_droplets and coverage_ratio > 0.25:
            # Saturación media en máscara vs hoja
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat_leaf = float(np.mean(hsv[:, :, 1][leaf_mask > 0])) if total_leaf_area > 0 else 0.0
            sat_mask = float(np.mean(hsv[:, :, 1][filtered_mask > 0])) if cv2.countNonZero(filtered_mask) > 0 else 0.0

            # Textura (varianza del Laplaciano) en canal azul
            blue = image[:, :, 0]
            lap = cv2.Laplacian(blue, cv2.CV_32F)
            var_lap_leaf = float(np.var(lap[leaf_mask > 0])) if total_leaf_area > 0 else 0.0
            var_lap_mask = float(np.var(lap[filtered_mask > 0])) if cv2.countNonZero(filtered_mask) > 0 else 0.0
            speckle_ratio = (var_lap_mask + 1e-6) / (var_lap_leaf + 1e-6)

            # Tamaño típico de componente (mediana)
            comp_areas = [int(cv2.contourArea(c)) for c in cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
            median_area = float(np.median(comp_areas)) if comp_areas else 0.0
            # 0.4% del área de la hoja como umbral de componente demasiado grande para ser gota típica
            too_large_component = median_area > 0.004 * float(total_leaf_area)

        # Condiciones de descarte por brillo uniforme
        if has_valid_droplets and coverage_ratio > 0.25:
            sat_mask_val = sat_mask if sat_mask is not None else 255.0
            speckle_ratio_val = speckle_ratio if speckle_ratio is not None else 10.0
            too_large_component_val = too_large_component if too_large_component is not None else False
            if (sat_mask_val < 30 and (mean_blue_in_mask - mean_blue_leaf) < 12) or speckle_ratio_val < 1.15 or too_large_component_val:
                has_valid_droplets = False

        # Aceptación de alta confianza: cobertura muy alta con azul significativamente mayor
        high_confidence = False
        if not has_valid_droplets and filtered_area > 0:
            mean_blue_in_mask = (
                float(np.mean(image[:, :, 0][filtered_mask > 0]))
                if cv2.countNonZero(filtered_mask) > 0
                else 0.0
            )
            if coverage_ratio > 0.45 and (mean_blue_in_mask - mean_blue_leaf) >= 25:
                has_valid_droplets = True
                high_confidence = True
        
        # Solo aplicar verificación de circularidad si hay muy pocas gotas
        # y la cobertura total es muy alta (posible falso positivo)
        if not high_confidence and valid_droplets_count == 1 and total_fluorescence_area > 0.4 * total_leaf_area:
            # Verificar si las formas son circulares (como gotas reales)
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # El área y perímetro nos ayudan a calcular la circularidad
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Umbral de circularidad menos estricto
                    if circularity < 0.3:  # No es suficientemente circular
                        has_valid_droplets = False
                        break
                        
        return filtered_mask, has_valid_droplets

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
        
        # Filtrar gotas válidas
        filtered_mask, has_valid_droplets = SprayAnalyzer._filter_valid_droplets(denoised, fluorescence_mask, leaf_mask)
        
        # Calcular áreas y cobertura
        leaf_area = cv2.countNonZero(leaf_mask)
        
        # Calcular cobertura inicial basada en la máscara de fluorescencia original
        # para casos con cobertura media-alta real
        initial_sprayed_area = cv2.countNonZero(fluorescence_mask)
        initial_coverage = (initial_sprayed_area / leaf_area * 100) if leaf_area > 0 else 0
        
        # Verificación especial para imágenes fluorescentes UV
        # Calcular el promedio del canal azul (BGR) en toda la hoja
        mean_blue_in_leaf = np.mean(image[:,:,0][leaf_mask > 0]) if cv2.countNonZero(leaf_mask) > 0 else 0
        is_fluorescent_image = mean_blue_in_leaf > 70  # Umbral para considerar que es una imagen bajo luz UV
        
        # Si no hay gotas válidas pero hay señal de fluorescencia, hacer verificación adicional
        if not has_valid_droplets:
            # Caso especial: Imágenes fluorescentes UV como T2C1P3TMInterno
            if is_fluorescent_image and initial_sprayed_area > 0:
                # Verificar si la distribución de fluorescencia es típica de un patrón de gotas
                contours, _ = cv2.findContours(fluorescence_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Heurística adicional: textura y tamaño típico de componentes
                blue = image[:, :, 0]
                lap = cv2.Laplacian(blue, cv2.CV_32F)
                var_lap_leaf = float(np.var(lap[leaf_mask > 0])) if leaf_area > 0 else 0.0
                var_lap_mask = float(np.var(lap[fluorescence_mask > 0])) if initial_sprayed_area > 0 else 0.0
                speckle_ratio = (var_lap_mask + 1e-6) / (var_lap_leaf + 1e-6)
                comp_areas = [int(cv2.contourArea(c)) for c in contours]
                median_area = float(np.median(comp_areas)) if comp_areas else 0.0
                small_components = median_area < 0.001 * float(leaf_area)
                if len(contours) >= 20 and speckle_ratio > 1.2 and small_components:
                    has_valid_droplets = True
                    filtered_mask = fluorescence_mask.copy()
                # Si toda la imagen es brillante (azul intenso), probablemente es fluorescente
                elif mean_blue_in_leaf > 80 and initial_coverage > 20:
                    has_valid_droplets = True
                    filtered_mask = fluorescence_mask.copy()
            # Caso normal: Verificar cobertura significativa pero intermedia
            elif 10 < initial_coverage < 90 and initial_sprayed_area > 0 and not is_fluorescent_image:
                contours, _ = cv2.findContours(fluorescence_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Requerir patrón de gotas mucho más claro para rescatar
                blue = image[:, :, 0]
                lap = cv2.Laplacian(blue, cv2.CV_32F)
                var_lap_leaf = float(np.var(lap[leaf_mask > 0])) if leaf_area > 0 else 0.0
                var_lap_mask = float(np.var(lap[fluorescence_mask > 0])) if initial_sprayed_area > 0 else 0.0
                speckle_ratio = (var_lap_mask + 1e-6) / (var_lap_leaf + 1e-6)
                comp_areas = [int(cv2.contourArea(c)) for c in contours]
                median_area = float(np.median(comp_areas)) if comp_areas else 0.0
                small_components = median_area < 0.001 * float(leaf_area)
                if len(contours) >= 25 and speckle_ratio > 1.25 and small_components:
                    has_valid_droplets = True
                    filtered_mask = fluorescence_mask.copy()
            
            # Si definitivamente no hay gotas válidas, establecer cobertura a 0
            if not has_valid_droplets:
                sprayed_area = 0
                coverage = 0
                # Fallback CONDICIONADO: solo en imágenes claramente UV y con muchas regiones pequeñas
                contours, _ = cv2.findContours(fluorescence_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                fluorescence_pixels = image[:,:,0][(leaf_mask > 0) & (fluorescence_mask > 0)]
                leaf_blue_pixels = image[:,:,0][leaf_mask > 0]
                if is_fluorescent_image and len(contours) >= 20 and leaf_blue_pixels.size > 0 and fluorescence_pixels.size > 0:
                    mu_leaf = float(np.mean(leaf_blue_pixels))
                    mu_fluo = float(np.mean(fluorescence_pixels))
                    if mu_fluo >= mu_leaf + 8:  # más estricto para evitar FP
                        sprayed_area = int(initial_sprayed_area)
                        coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0
            else:
                sprayed_area = cv2.countNonZero(filtered_mask)
                coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0
        else:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0
            
            # Verificación para casos extremos
            if coverage > 70:
                # Análisis adicional para alta cobertura
                mean_fluorescence = np.mean(image[:,:,0][filtered_mask > 0]) if cv2.countNonZero(filtered_mask) > 0 else 0
                
                if is_fluorescent_image:
                    # Para imágenes bajo luz UV: Si tiene alta componente azul, es fluorescencia válida
                    if mean_fluorescence < 60:  # Insuficiente azul para ser fluorescencia real
                        coverage = 0
                        sprayed_area = 0
                else:
                    # Para imágenes normales: Ser más estricto para evitar falsos positivos
                    if mean_fluorescence < 50 or not has_valid_droplets:
                        coverage = 0
                        sprayed_area = 0
        
        # Aplicar corrección para cobertura muy alta (>90%)
        if coverage > 90:
            # Reducir entre 10-15% dependiendo de qué tan cerca está del 100%
            correction_factor = 0.10 + (0.05 * (coverage - 90) / 10)
            sprayed_area = int(sprayed_area * (1 - correction_factor))
            coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0
        
        # Crear imagen procesada con áreas de rocío en amarillo
        processed_image = image.copy()
        processed_image[filtered_mask > 0] = [0, 255, 255]  # Amarillo para áreas con spray
        
        # Codificar imagen procesada a base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Guardar imágenes de debug
        if save_debug:
            # Guardar imágenes
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_fluorescence_mask.jpg", fluorescence_mask)
            cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)
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