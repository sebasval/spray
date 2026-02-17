"""
Spray coverage analyzer — ImageJ-compatible, pure Python.

Replicates ImageJ's analysis pipeline without requiring Java:
- IsoData threshold: ported from ImageJ source (AutoThresholder.java)
- Particle analysis: Watershed + circularity via scikit-image/scipy
  (replicates ImageJ's Analyze Particles with size & circularity filters)
"""
import cv2
import numpy as np
from uuid import uuid4
from typing import List, Optional
import base64
import logging
from skimage import measure, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from app.models.image import ImageAnalysisResponse

logger = logging.getLogger(__name__)


class SprayAnalyzer:
    MIN_PARTICLE_AREA = 5
    MIN_CIRCULARITY = 0.20
    MAX_CIRCULARITY = 1.50
    MIN_SOLIDITY = 0.50

    # ──────────────────────────────────────────────
    #  IsoData Threshold (port of ImageJ's Default)
    # ──────────────────────────────────────────────

    @staticmethod
    def _imagej_isodata_threshold(histogram) -> int:
        """
        Port of ImageJ's Default threshold (modified IsoData).
        Source: ij/process/AutoThresholder.java -> defaultIsoData() + IJIsoData()

        This is THE algorithm ImageJ uses when you do
        Image > Adjust > Threshold > Auto (Default).
        """
        data = list(histogram)
        n = len(data)

        # --- defaultIsoData: cap the mode if it dominates ---
        mode = 0
        max_count = 0
        for i in range(n):
            if data[i] > max_count:
                max_count = data[i]
                mode = i

        max_count2 = 0
        for i in range(n):
            if data[i] > max_count2 and i != mode:
                max_count2 = data[i]

        if max_count > (max_count2 * 2) and max_count2 != 0:
            data[mode] = int(max_count2 * 1.5)

        # --- IJIsoData ---
        max_value = n - 1
        count0 = data[0]
        data[0] = 0
        count_max = data[max_value]
        data[max_value] = 0

        min_idx = 0
        while data[min_idx] == 0 and min_idx < max_value:
            min_idx += 1

        max_idx = max_value
        while data[max_idx] == 0 and max_idx > 0:
            max_idx -= 1

        if min_idx >= max_idx:
            data[0] = count0
            data[max_value] = count_max
            return n // 2

        moving_index = min_idx
        result = 0.0

        while True:
            sum1 = sum2 = sum3 = sum4 = 0.0
            for i in range(min_idx, moving_index + 1):
                sum1 += i * data[i]
                sum2 += data[i]
            for i in range(moving_index + 1, max_idx + 1):
                sum3 += i * data[i]
                sum4 += data[i]

            if sum2 == 0 or sum4 == 0:
                moving_index += 1
                if moving_index >= max_idx - 1:
                    break
                continue

            result = (sum1 / sum2 + sum3 / sum4) / 2.0
            moving_index += 1

            if not ((moving_index + 1) <= result and moving_index < max_idx - 1):
                break

        data[0] = count0
        data[max_value] = count_max

        # Java's Math.round() rounds 0.5 up; Python's round() uses banker's rounding
        return int(result + 0.5)

    # ──────────────────────────────────────────────
    #  Leaf / Object Detection
    # ──────────────────────────────────────────────

    @staticmethod
    def _detect_leaf_mask(gray: np.ndarray, background_threshold: int = 30) -> np.ndarray:
        """
        Detect leaf/object mask, compatible with Photoroom backgrounds.
        Uses Otsu on the grayscale with a minimum threshold to exclude dark backgrounds.
        """
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, fixed_mask = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_and(otsu_mask, fixed_mask)

        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray)

        image_area = gray.shape[0] * gray.shape[1]
        min_area = image_area * 0.003
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not valid:
            return np.zeros_like(gray)

        mask = np.zeros_like(gray)
        for contour in valid:
            cv2.drawContours(mask, [contour], -1, 255, -1)

        leaf_pct = cv2.countNonZero(mask) / image_area * 100
        if leaf_pct > 85:
            logger.warning(f"Object mask too large ({leaf_pct:.1f}%), retrying with threshold=80")
            _, strict = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, kernel, iterations=2)
            strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, kernel, iterations=2)
            contours2, _ = cv2.findContours(strict, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid2 = [c for c in contours2 if cv2.contourArea(c) >= min_area]
            if valid2:
                mask = np.zeros_like(gray)
                for contour in valid2:
                    cv2.drawContours(mask, [contour], -1, 255, -1)

        return mask

    # ──────────────────────────────────────────────
    #  Particle Analyzer (Watershed + Circularity)
    # ──────────────────────────────────────────────

    @staticmethod
    def _filter_particles(spray_mask: np.ndarray) -> tuple:
        """
        Replicates ImageJ's Analyze Particles in pure Python (scikit-image).

        Pipeline (matching ImageJ's ParticleAnalyzer.java):
        1. Watershed to separate touching droplets
        2. For each particle, measure area, perimeter, circularity, solidity
        3. Filter by:
           - area >= MIN_PARTICLE_AREA (removes noise)
           - MIN_CIRCULARITY <= circularity (removes elongated artifacts)
           - solidity >= MIN_SOLIDITY (removes irregular/fragmented shapes)

        Note: unlike ImageJ's EXCLUDE_EDGE_PARTICLES (designed for individual
        particle measurement), we DO NOT exclude edge particles because
        coverage analysis needs total sprayed area, not individual accuracy.

        Circularity: 4*pi*area / perimeter^2  (ImageJ caps at 1.0 for display)
        Solidity: area / convex_area  (1.0 = perfectly convex, <0.5 = very irregular)

        Returns: (filtered_mask uint8, valid_droplet_count int)
        """
        if cv2.countNonZero(spray_mask) == 0:
            return np.zeros_like(spray_mask), 0

        binary = (spray_mask > 0).astype(np.uint8)

        # Watershed segmentation to split touching droplets
        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(distance, min_distance=3, labels=binary)

        if len(coords) == 0:
            # No peaks found — fall back to connected components
            labels = measure.label(binary)
        else:
            peak_mask = np.zeros(distance.shape, dtype=bool)
            peak_mask[tuple(coords.T)] = True
            markers = ndi.label(peak_mask)[0]
            labels = segmentation.watershed(-distance, markers, mask=binary)

        props = measure.regionprops(labels)
        filtered_mask = np.zeros_like(spray_mask)
        valid_count = 0
        rejected = {"area": 0, "circularity": 0, "solidity": 0}

        for p in props:
            # --- Size filter ---
            if p.area < SprayAnalyzer.MIN_PARTICLE_AREA:
                rejected["area"] += 1
                continue

            # --- Perimeter sanity ---
            if p.perimeter == 0:
                rejected["area"] += 1
                continue

            # --- Circularity filter (ImageJ formula) ---
            circularity = (4.0 * np.pi * p.area) / (p.perimeter ** 2)
            # ImageJ caps circularity at 1.0 when maxCircularity <= 1.0
            # We allow up to MAX_CIRCULARITY to handle discretization artifacts
            if circularity > 1.0:
                circularity = 1.0
            if circularity < SprayAnalyzer.MIN_CIRCULARITY:
                rejected["circularity"] += 1
                continue

            # --- Solidity filter (area / convex_area) ---
            # Real droplets are convex; irregular shapes are artifacts
            if p.convex_area > 0:
                solidity = p.area / p.convex_area
                if solidity < SprayAnalyzer.MIN_SOLIDITY:
                    rejected["solidity"] += 1
                    continue

            filtered_mask[labels == p.label] = 255
            valid_count += 1

        logger.info(
            f"Particle analysis: {len(props)} total, {valid_count} valid | "
            f"Rejected: area={rejected['area']}, circ={rejected['circularity']}, "
            f"solidity={rejected['solidity']}"
        )

        return filtered_mask, valid_count

    # ──────────────────────────────────────────────
    #  Main Analysis Pipeline
    # ──────────────────────────────────────────────

    @staticmethod
    def analyze_image(image_bytes: bytes, save_debug: bool = True) -> tuple[float, int, int, Optional[str]]:
        """
        Analyze spray coverage using an ImageJ-compatible pipeline.

        Pipeline:
        1. Convert to 8-bit grayscale + HSV
        2. Detect leaf/object mask
        3. Brightness detection via ImageJ IsoData threshold
        4. Color detection for cyan/blue fluorescence
        5. Combine both masks
        6. Filter particles (Watershed + circularity)
        7. Calculate coverage percentage

        Returns: (coverage_percentage, leaf_area, sprayed_area, processed_image_b64)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return 0.0, 0, 0, None

        original_image = image.copy()

        # Step 1: Grayscale + HSV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Step 2: Detect leaf/object mask
        leaf_mask = SprayAnalyzer._detect_leaf_mask(gray)
        leaf_area = cv2.countNonZero(leaf_mask)

        if leaf_area == 0:
            _, buffer = cv2.imencode('.jpg', original_image)
            return 0.0, 0, 0, base64.b64encode(buffer).decode('utf-8')

        logger.info(f"Leaf area: {leaf_area} px ({leaf_area / (gray.shape[0] * gray.shape[1]) * 100:.1f}% of image)")

        # Step 3: Brightness detection (ImageJ IsoData threshold)
        # CRITICAL: compute histogram ONLY on leaf pixels (not black background).
        # ImageJ computes threshold on the ROI, not the full image.
        # Using full-image histogram would skew threshold too low due to
        # dominant black background, causing massive false positives.
        histogram = cv2.calcHist([gray], [0], leaf_mask, [256], [0, 256]).flatten().astype(int)
        threshold = SprayAnalyzer._imagej_isodata_threshold(histogram)
        logger.info(f"ImageJ IsoData threshold (brightness, leaf-only): {threshold}")

        _, brightness_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        brightness_mask = cv2.bitwise_and(brightness_mask, leaf_mask)

        # Filter brightness mask: exclude purple/magenta UV reflections.
        # Under UV light, leaf surface reflects as bright PURPLE (H >120),
        # while spray fluoresces as CYAN/BLUE (H 75-120) or appears WHITE
        # (desaturated + bright). Only keep bright pixels with spray-compatible color.
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        spray_compatible = (
            ((sat < 60) & (val >= 100)) |          # desaturated AND bright (white = dense spray)
            ((hue >= 75) & (hue <= 120))           # cyan/blue hue = spray fluorescence
        ).astype(np.uint8) * 255
        brightness_before = cv2.countNonZero(brightness_mask)
        brightness_mask = cv2.bitwise_and(brightness_mask, spray_compatible)
        brightness_after = cv2.countNonZero(brightness_mask)

        if brightness_before > 0 and brightness_after < brightness_before:
            removed_pct = (1 - brightness_after / brightness_before) * 100
            logger.info(
                f"Brightness filter: removed {removed_pct:.0f}% of bright pixels "
                f"(purple/magenta UV reflections), kept {brightness_after} spray-compatible"
            )

        # Step 4: Color detection (cyan/blue fluorescence under UV)
        # H 80-120: cyan to blue, excluding blue-purple (H>120) which is
        # leaf surface color under UV, not spray fluorescence.
        lower_cyan = np.array([80, 30, 60])
        upper_cyan = np.array([120, 255, 255])
        color_mask_hsv = cv2.inRange(hsv, lower_cyan, upper_cyan)

        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        cyan_check = ((g > r + 5) & (b > r)).astype(np.uint8) * 255
        blue_check = ((b > r + 15) & (b > g + 5)).astype(np.uint8) * 255
        bgr_mask = cv2.bitwise_or(cyan_check, blue_check)

        color_mask = cv2.bitwise_and(color_mask_hsv, bgr_mask)
        color_mask = cv2.bitwise_and(color_mask, leaf_mask)

        # Step 5: Combine brightness + color
        spray_mask = cv2.bitwise_or(brightness_mask, color_mask)

        logger.info(
            f"Detection: brightness={cv2.countNonZero(brightness_mask)}px, "
            f"color={cv2.countNonZero(color_mask)}px, "
            f"combined={cv2.countNonZero(spray_mask)}px"
        )

        # Step 6: Morphological cleanup before particle analysis
        # Small opening removes 1-2px noise, small closing fills tiny gaps in droplets
        morph_kernel = np.ones((3, 3), np.uint8)
        spray_mask = cv2.morphologyEx(spray_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        spray_mask = cv2.morphologyEx(spray_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        # Step 7: Filter particles (Watershed + circularity + solidity)
        filtered_mask, droplet_count = SprayAnalyzer._filter_particles(spray_mask)

        # Step 8: Calculate coverage
        sprayed_area = cv2.countNonZero(filtered_mask)
        coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0.0
        coverage = min(coverage, 100.0)

        logger.info(
            f"Analysis result: threshold={threshold}, leaf={leaf_area}, "
            f"spray={sprayed_area}, droplets={droplet_count}, coverage={coverage:.2f}%"
        )

        # Step 9: Create output image with yellow overlay
        processed_image = original_image.copy()
        processed_image[filtered_mask > 0] = [0, 255, 255]

        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        if save_debug:
            cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
            cv2.imwrite("debug_brightness_mask.jpg", brightness_mask)
            cv2.imwrite("debug_spray_mask.jpg", spray_mask)
            cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)
            cv2.imwrite("debug_result.jpg", processed_image)

        return round(coverage, 2), leaf_area, sprayed_area, processed_image_b64

    # ──────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────

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
