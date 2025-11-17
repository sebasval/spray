import cv2
import numpy as np
from uuid import uuid4
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import base64
from app.models.image import ImageAnalysisResponse

class SprayAnalyzer:
    """
    Analizador de spray mejorado con detección precisa de gotas azul cian
    """
    
    # Parámetros optimizados
    MIN_DROPLET_AREA = 5  # Área mínima de una gota en píxeles (reducido para capturar gotas pequeñas)
    MAX_DROPLET_AREA_RATIO = 0.05  # Máximo 5% del área de hoja por gota individual
    MIN_DROPLETS_COUNT = 1  # Mínimo de gotas para considerar válido (reducido para mayor sensibilidad)
    
    @staticmethod
    def _detect_leaf_mask(image: np.ndarray) -> np.ndarray:
        """
        Detecta la máscara de la hoja de manera robusta
        """
        # Convertir a escala de grises y HSV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Suavizar para reducir ruido
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Método 1: Umbral Otsu para separar hoja de fondo negro
        _, otsu_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Método 2: Umbral fijo conservador
        _, fixed_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        
        # Método 3: Por saturación y valor en HSV (descarta fondo negro)
        sv_mask = ((hsv[:, :, 1] > 10) | (hsv[:, :, 2] > 20)).astype(np.uint8) * 255
        
        # Combinar las tres máscaras
        combined = cv2.bitwise_or(otsu_mask, fixed_mask)
        combined = cv2.bitwise_or(combined, sv_mask)
        
        # Limpieza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Encontrar el contorno más grande (la hoja principal)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return combined
        
        # Crear máscara final con el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(combined)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        return final_mask
    
    @staticmethod
    def _detect_cyan_droplets(image: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Detecta específicamente gotas de color azul cian usando múltiples espacios de color
        
        Returns:
            Tuple[np.ndarray, dict]: Máscara de gotas detectadas y diccionario con estadísticas
        """
        # Convertir a diferentes espacios de color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extraer canales
        b, g, r = cv2.split(image)
        h, s, v = cv2.split(hsv)
        l, a, b_lab = cv2.split(lab)
        
        # ========================================
        # PASO 1: Detección por HSV (azul cian)
        # ========================================
        # Azul cian está entre 85-105 en H (matiz)
        # Rangos ajustados para capturar azul cian específicamente
        lower_cyan = np.array([85, 40, 40])   # H, S, V mínimos
        upper_cyan = np.array([105, 255, 255])  # H, S, V máximos
        cyan_mask_hsv = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # ========================================
        # PASO 2: Índice de exceso de azul
        # ========================================
        # ExB = B - 0.5 * (G + R)
        # Detecta píxeles donde el azul domina
        b_float = b.astype(np.float32)
        g_float = g.astype(np.float32)
        r_float = r.astype(np.float32)
        
        blue_excess = b_float - 0.5 * (g_float + r_float)
        
        # Normalizar a rango 0-255
        blue_excess_min = blue_excess.min()
        blue_excess_max = blue_excess.max()
        if blue_excess_max > blue_excess_min:
            blue_excess_norm = ((blue_excess - blue_excess_min) / (blue_excess_max - blue_excess_min) * 255).astype(np.uint8)
        else:
            blue_excess_norm = np.zeros_like(b)
        
        # Calcular umbral adaptativo solo dentro de la hoja
        blue_excess_in_leaf = blue_excess_norm[leaf_mask > 0]
        if blue_excess_in_leaf.size > 0:
            # Usar percentil 85 como umbral (más sensible que 90)
            # Si hay muchas gotas, el percentil será más alto naturalmente
            threshold_excess = np.percentile(blue_excess_in_leaf, 85)
            # Umbral mínimo más bajo para capturar gotas menos intensas
            threshold_excess = max(threshold_excess, 80)
        else:
            threshold_excess = 80
        
        excess_mask = (blue_excess_norm > threshold_excess).astype(np.uint8) * 255
        
        # ========================================
        # PASO 3: Detección por LAB
        # ========================================
        # En LAB, el azul cian tiene:
        # - L medio-alto (luminosidad)
        # - a negativo (hacia azul, no rojo)
        # - b negativo (hacia azul, no amarillo)
        
        # Crear máscara LAB
        lab_mask = np.zeros_like(l)
        lab_mask[(l > 50) & (a < 128) & (b_lab < 128)] = 255
        
        # ========================================
        # PASO 4: Ratio B/(R+G)
        # ========================================
        # Las gotas azul cian tienen mucho más azul que rojo+verde
        blue_ratio = b_float / (r_float + g_float + 1.0)
        blue_ratio_norm = np.clip(blue_ratio * 100, 0, 255).astype(np.uint8)
        
        # Umbral para ratio (azul debe ser al menos 1.2x más que promedio de R+G)
        # Reducido de 1.3 a 1.2 para mayor sensibilidad
        ratio_mask = (blue_ratio > 1.2).astype(np.uint8) * 255
        
        # ========================================
        # PASO 5: Combinar máscaras de manera INTELIGENTE
        # ========================================
        # Sistema de votación flexible:
        # - Si HSV detecta azul cian (método más confiable), aceptar con 1 voto adicional
        # - Si no hay HSV pero hay 2+ otros métodos, también aceptar
        # Esto balancea sensibilidad y precisión
        
        # Convertir a binario para contar
        mask1 = (cyan_mask_hsv > 0).astype(np.uint8)
        mask2 = (excess_mask > 0).astype(np.uint8)
        mask3 = (lab_mask > 0).astype(np.uint8)
        mask4 = (ratio_mask > 0).astype(np.uint8)
        
        # Suma de coincidencias
        vote_sum = mask1 + mask2 + mask3 + mask4
        
        # Regla flexible: 
        # - Si HSV (mask1) está presente Y al menos otro método → aceptar
        # - O si hay 2+ métodos sin HSV → aceptar
        # - O si hay 3+ métodos → aceptar (muy confiable)
        hsv_present = mask1 > 0
        other_votes = (mask2 + mask3 + mask4) > 0
        strong_consensus = vote_sum >= 3
        
        combined_mask = (
            (hsv_present & other_votes) |  # HSV + otro método
            (vote_sum >= 2)  # 2+ métodos sin importar cuáles
        ).astype(np.uint8) * 255
        
        # ========================================
        # PASO 6: Restringir a la hoja y filtrar por brillo
        # ========================================
        # Eliminar píxeles muy oscuros (sombras)
        brightness_mask = (v > 30).astype(np.uint8) * 255
        
        # Aplicar todas las restricciones
        final_mask = cv2.bitwise_and(combined_mask, leaf_mask)
        final_mask = cv2.bitwise_and(final_mask, brightness_mask)
        
        # ========================================
        # PASO 7: Limpieza morfológica
        # ========================================
        # Eliminar ruido pequeño
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Cerrar huecos pequeños en gotas
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        # ========================================
        # PASO 8: Recolectar estadísticas
        # ========================================
        stats = {
            'mean_blue_channel': float(np.mean(b[final_mask > 0])) if cv2.countNonZero(final_mask) > 0 else 0,
            'mean_hue': float(np.mean(h[final_mask > 0])) if cv2.countNonZero(final_mask) > 0 else 0,
            'mean_saturation': float(np.mean(s[final_mask > 0])) if cv2.countNonZero(final_mask) > 0 else 0,
            'mean_value': float(np.mean(v[final_mask > 0])) if cv2.countNonZero(final_mask) > 0 else 0,
            'threshold_used': float(threshold_excess),
            'pixels_detected': int(cv2.countNonZero(final_mask))
        }
        
        return final_mask, stats
    
    @staticmethod
    def _validate_and_filter_droplets(
        image: np.ndarray,
        droplet_mask: np.ndarray,
        leaf_mask: np.ndarray,
        stats: dict
    ) -> Tuple[np.ndarray, bool, dict]:
        """
        Valida que las detecciones sean gotas reales y no falsos positivos
        
        Returns:
            Tuple[np.ndarray, bool, dict]: Máscara filtrada, si es válida, y estadísticas de validación
        """
        leaf_area = cv2.countNonZero(leaf_mask)
        if leaf_area == 0:
            return droplet_mask, False, {'reason': 'no_leaf_detected'}
        
        # Encontrar componentes conectados (gotas individuales)
        num_labels, labels, component_stats, centroids = cv2.connectedComponentsWithStats(
            droplet_mask, connectivity=8
        )
        
        # Filtrar componentes por tamaño
        max_droplet_area = int(leaf_area * SprayAnalyzer.MAX_DROPLET_AREA_RATIO)
        filtered_mask = np.zeros_like(droplet_mask)
        valid_droplets = []
        
        for i in range(1, num_labels):  # Saltar fondo (label 0)
            area = component_stats[i, cv2.CC_STAT_AREA]
            
            # Filtrar por tamaño
            if area < SprayAnalyzer.MIN_DROPLET_AREA or area > max_droplet_area:
                continue
            
            # Extraer la gota
            droplet_pixels = (labels == i)
            
            # Calcular características de color
            b_mean = float(np.mean(image[:, :, 0][droplet_pixels]))
            g_mean = float(np.mean(image[:, :, 1][droplet_pixels]))
            r_mean = float(np.mean(image[:, :, 2][droplet_pixels]))
            
            # Verificar que el azul domine (umbral más permisivo)
            blue_dominance = b_mean / (0.5 * (g_mean + r_mean) + 1.0)
            if blue_dominance < 1.15:  # Reducido de 1.2 a 1.15 para mayor sensibilidad
                continue
            
            # Calcular circularidad (gotas tienden a ser circulares)
            contours, _ = cv2.findContours(
                droplet_pixels.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
            else:
                circularity = 0
            
            # Guardar información de la gota
            valid_droplets.append({
                'area': area,
                'circularity': circularity,
                'blue_dominance': blue_dominance,
                'mean_blue': b_mean
            })
            
            # Agregar a la máscara filtrada
            filtered_mask[droplet_pixels] = 255
        
        # ========================================
        # Validación global
        # ========================================
        validation_stats = {
            'num_droplets': len(valid_droplets),
            'total_droplet_area': int(cv2.countNonZero(filtered_mask)),
            'leaf_area': leaf_area,
            'coverage_ratio': cv2.countNonZero(filtered_mask) / leaf_area if leaf_area > 0 else 0
        }
        
        # Criterios de validación
        is_valid = True
        reason = 'valid'
        
        # 1. Debe haber suficientes gotas o cobertura significativa
        if len(valid_droplets) < SprayAnalyzer.MIN_DROPLETS_COUNT:
            # Si hay cobertura significativa (>2%), aceptar aunque haya pocas gotas
            # (puede ser una gota grande o varias gotas que se fusionaron)
            if validation_stats['coverage_ratio'] < 0.02:  # Reducido de 0.05 a 0.02
                is_valid = False
                reason = 'insufficient_droplets'
        
        # 2. Evitar cobertura excesiva uniforme (probablemente iluminación, no gotas)
        if validation_stats['coverage_ratio'] > 0.85:
            # Verificar si realmente son gotas discretas
            if len(valid_droplets) < 20:
                is_valid = False
                reason = 'excessive_uniform_coverage'
        
        # 3. Verificar circularidad promedio
        if len(valid_droplets) > 0:
            avg_circularity = np.mean([d['circularity'] for d in valid_droplets])
            validation_stats['avg_circularity'] = float(avg_circularity)
            
            # Si la circularidad es muy baja y hay pocas gotas, probablemente no son gotas
            # Pero ser más permisivo: solo rechazar si es muy bajo (<0.15) y muy pocas gotas
            if avg_circularity < 0.15 and len(valid_droplets) < 5:  # Más estricto solo en casos extremos
                is_valid = False
                reason = 'low_circularity'
        
        # 4. Verificar dominancia de azul promedio
        if len(valid_droplets) > 0:
            avg_blue_dominance = np.mean([d['blue_dominance'] for d in valid_droplets])
            validation_stats['avg_blue_dominance'] = float(avg_blue_dominance)
            
            # Si el azul no domina lo suficiente, no son gotas azul cian
            # Umbral más permisivo para capturar gotas reales
            if avg_blue_dominance < 1.25:  # Reducido de 1.3 a 1.25
                is_valid = False
                reason = 'insufficient_blue_dominance'
        
        validation_stats['reason'] = reason
        validation_stats['valid_droplets_details'] = valid_droplets[:10]  # Primeras 10 para debug
        
        return filtered_mask, is_valid, validation_stats
    
    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Analiza una imagen para detectar cobertura de spray con gotas azul cian
        
        Returns:
            tuple: (coverage_percentage, leaf_area, sprayed_area, processed_image_base64)
        """
        # Decodificar imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        # Reducir ruido de la imagen
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        
        # Paso 1: Detectar la hoja
        leaf_mask = SprayAnalyzer._detect_leaf_mask(denoised)
        leaf_area = cv2.countNonZero(leaf_mask)
        
        if leaf_area == 0:
            # No se detectó hoja
            return 0.0, 0, 0, None
        
        # Paso 2: Detectar gotas azul cian
        droplet_mask, detection_stats = SprayAnalyzer._detect_cyan_droplets(denoised, leaf_mask)
        
        # Paso 3: Validar y filtrar gotas
        filtered_mask, is_valid, validation_stats = SprayAnalyzer._validate_and_filter_droplets(
            denoised, droplet_mask, leaf_mask, detection_stats
        )
        
        # Calcular cobertura
        if is_valid:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage_percentage = (sprayed_area / leaf_area * 100.0) if leaf_area > 0 else 0.0
            
            # Validación final: Si el coverage es muy bajo (<0.5%), considerar que no hay gotas
            # Esto evita falsos positivos mínimos
            if coverage_percentage < 0.5:
                sprayed_area = 0
                coverage_percentage = 0.0
        else:
            # No se detectaron gotas válidas
            sprayed_area = 0
            coverage_percentage = 0.0
        
        # Crear imagen procesada con visualización
        processed_image = image.copy()
        
        # Marcar área de hoja en verde tenue
        processed_image[leaf_mask > 0] = cv2.addWeighted(
            processed_image[leaf_mask > 0],
            0.7,
            np.full_like(processed_image[leaf_mask > 0], [0, 50, 0]),
            0.3,
            0
        )
        
        # Marcar gotas detectadas en amarillo brillante
        processed_image[filtered_mask > 0] = [0, 255, 255]
        
        # Agregar texto con información
        text = f"Coverage: {coverage_percentage:.2f}% | Gotas: {validation_stats.get('num_droplets', 0)}"
        cv2.putText(
            processed_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Codificar a base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Guardar imágenes de debug
        if save_debug:
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_droplet_mask.jpg", droplet_mask)
            cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)
            cv2.imwrite("debug_result.jpg", processed_image)
            
            # Guardar histograma de matiz (Hue)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_channel = hsv[:, :, 0]
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.hist(hue_channel[leaf_mask > 0].ravel(), 180, [0, 180], color='blue', alpha=0.7)
            plt.title("Histograma de Matiz (Hue) en Hoja")
            plt.xlabel("Matiz (0-180)")
            plt.ylabel("Frecuencia")
            plt.axvline(x=85, color='r', linestyle='--', label='Cian inicio (85)')
            plt.axvline(x=105, color='r', linestyle='--', label='Cian fin (105)')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            blue_channel = image[:, :, 0]
            plt.hist(blue_channel[leaf_mask > 0].ravel(), 256, [0, 256], color='cyan', alpha=0.7)
            plt.title("Histograma de Canal Azul en Hoja")
            plt.xlabel("Intensidad Azul (0-255)")
            plt.ylabel("Frecuencia")
            
            plt.subplot(1, 3, 3)
            if cv2.countNonZero(filtered_mask) > 0:
                plt.hist(blue_channel[filtered_mask > 0].ravel(), 256, [0, 256], color='yellow', alpha=0.7)
                plt.title("Histograma de Azul en Gotas Detectadas")
                plt.xlabel("Intensidad Azul (0-255)")
                plt.ylabel("Frecuencia")
            else:
                plt.text(0.5, 0.5, 'Sin gotas detectadas', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title("Sin gotas detectadas")
            
            plt.tight_layout()
            plt.savefig("debug_histogram.jpg", dpi=100)
            plt.close()
            
            # Guardar estadísticas en archivo de texto
            with open("debug_stats.txt", "w") as f:
                f.write("=== ESTADÍSTICAS DE DETECCIÓN ===\n\n")
                f.write(f"Área de hoja: {leaf_area} píxeles\n")
                f.write(f"Área con spray: {sprayed_area} píxeles\n")
                f.write(f"Cobertura: {coverage_percentage:.2f}%\n")
                f.write(f"Válido: {is_valid}\n\n")
                f.write("=== ESTADÍSTICAS DE DETECCIÓN ===\n")
                for key, value in detection_stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n=== ESTADÍSTICAS DE VALIDACIÓN ===\n")
                for key, value in validation_stats.items():
                    if key != 'valid_droplets_details':
                        f.write(f"{key}: {value}\n")
        
        return round(coverage_percentage, 2), leaf_area, sprayed_area, processed_image_base64
    
    @staticmethod
    def generate_image_id() -> str:
        """Genera un ID único para una imagen"""
        return str(uuid4())
    
    @staticmethod
    def calculate_batch_summary(analyses: List[ImageAnalysisResponse]) -> dict:
        """
        Calcula estadísticas resumidas de un lote de análisis
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
