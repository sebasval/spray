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
        Ajustado para mejorar la robustez en los bordes y evitar el fondo.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Umbralización simple para eliminar el fondo negro obvio
        _, base_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)  # Umbral más bajo para asegurar captura

        # Umbral Otsu en el canal de Saturación (ayuda con los bordes de la hoja)
        sat = hsv[:, :, 1]
        _, sat_otsu_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combinar máscaras
        combined_mask = cv2.bitwise_or(base_mask, sat_otsu_mask)
        
        # Operaciones morfológicas para limpiar y conectar la máscara
        kernel = np.ones((7, 7), np.uint8)  # Kernel más grande para una mejor conexión
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Rellenar cualquier agujero dentro de la hoja
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(cleaned_mask)
            
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)  # Rellenar el contorno más grande

        return final_mask

    @staticmethod
    def _detect_fluorescence(image: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detecta la fluorescencia CÍAN/AZUL, rechazando PÚRPURA.
        Ajustado para ser más estricto y solo capturar azules/cian brillantes.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1) Detección por rango HSV (Cian/Azul) - Rango de matiz más estricto
        # El rango ahora se enfoca más en el cian-azul puro, evitando los azules que tiran a púrpura
        lower_cyan_blue = np.array([85, 50, 60])  # Saturación y Valor más altos para exigir brillo
        upper_cyan_blue = np.array([125, 255, 255])  # Rango de matiz ligeramente más estrecho en el extremo superior
        hsv_mask = cv2.inRange(hsv, lower_cyan_blue, upper_cyan_blue)

        # 2) Filtro BGR para Cían/Azul dominante - Requisitos más estrictos
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        
        # Requerir que el azul (o verde para cian) sea SIGNIFICATIVAMENTE dominante
        # Esto filtra los "azules" tenues o mezclados con mucho rojo/púrpura
        cyan_check = (g > r + 10) & (b > r + 5)  # Verde debe ser notablemente mayor que rojo, y azul mayor que rojo
        blue_check = (b > r + 20) & (b > g + 10)  # Azul debe ser MUCHO mayor que rojo y verde
        
        bgr_mask = ((cyan_check | blue_check)).astype(np.uint8) * 255

        # 3) Requisitos mínimos de valor/saturación para evitar sombras y ruido oscuro
        # Estos umbrales se han elevado para asegurar que solo los píxeles más BRILANTES y SATURADOS pasen
        sat_ok = (hsv[:, :, 1] > 60).astype(np.uint8) * 255  # Exige alta saturación
        val_ok = (hsv[:, :, 2] > 70).astype(np.uint8) * 255  # Exige alto brillo

        # 4) Combinar máscaras: DEBE pasar HSV, BGR, y tener suficiente brillo/saturación
        combined = cv2.bitwise_and(hsv_mask, bgr_mask)
        combined = cv2.bitwise_and(combined, sat_ok)
        combined = cv2.bitwise_and(combined, val_ok)
        
        # La máscara de fluorescencia final DEBE estar dentro de la máscara de la hoja
        fluorescence_mask = cv2.bitwise_and(combined, leaf_mask)

        # 5) Limpieza morfológica:
        # Se ha añadido una operación de apertura más agresiva para eliminar pequeños ruidos.
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Apertura: Elimina pequeños objetos y aísla las gotas
        fluorescence_mask = cv2.morphologyEx(fluorescence_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Cierre: Conecta pequeñas brechas en las gotas restantes
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

        MIN_DROPLET_SIZE = SprayAnalyzer.MIN_SPRAY_AREA  # Usamos el valor de clase, 5

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
        Analiza una imagen para detectar cobertura de spray usando fluorescencia UV
        """
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
            
        # 3. Detección de Fluorescencia (filtra por color CÍAN vs PÚRPURA y limita a la hoja)
        fluorescence_mask, _ = SprayAnalyzer._detect_fluorescence(denoised, leaf_mask)
        
        # 4. Filtrado de Gotas Válidas (filtra por TAMAÑO/ruido)
        filtered_mask, has_valid_droplets = SprayAnalyzer._filter_valid_droplets(denoised, fluorescence_mask)
        
        # 5. Calcular Resultados
        if has_valid_droplets:
            sprayed_area = cv2.countNonZero(filtered_mask)
            coverage = (sprayed_area / leaf_area * 100)
            coverage = min(coverage, 100.0) 
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
            cv2.imwrite("debug_fluorescence_mask_pre_filter.jpg", fluorescence_mask) 
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
