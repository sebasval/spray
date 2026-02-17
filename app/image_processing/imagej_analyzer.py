"""
ImageJ-compatible spray analyzer.
Implements ImageJ's Default (IsoData) threshold algorithm in pure Python
for 100% compatible results without requiring Java.

The IsoData algorithm is ported directly from ImageJ source:
https://github.com/imagej/ImageJ/blob/master/ij/process/AutoThresholder.java
"""
import numpy as np
import cv2
import base64
import logging

logger = logging.getLogger(__name__)


def imagej_isodata_threshold(histogram):
    """
    Port of ImageJ's Default threshold (modified IsoData).
    Source: ij/process/AutoThresholder.java -> defaultIsoData() + IJIsoData()
    
    This is THE algorithm ImageJ uses when you do Image > Adjust > Threshold > Auto (Default).
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
    data[0] = 0  # exclude erased areas
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
    
    return int(round(result))


def detect_leaf_mask(gray, background_threshold=30):
    """
    Detect leaf/object mask, compatible with Photoroom backgrounds.
    Uses Otsu on the grayscale with a minimum threshold to exclude dark backgrounds.
    """
    # Otsu to separate foreground from background
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Also apply fixed threshold to handle gray Photoroom backgrounds
    _, fixed_mask = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)
    
    # Intersection: must pass both
    combined = cv2.bitwise_and(otsu_mask, fixed_mask)
    
    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours, take largest as leaf
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray)
    
    image_area = gray.shape[0] * gray.shape[1]
    min_area = image_area * 0.003  # at least 0.3% of image (catch smaller leaves)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid:
        return np.zeros_like(gray)
    
    # Draw ALL valid contours, not just the largest
    # This handles multiple leaves, petals, flowers in one image
    mask = np.zeros_like(gray)
    for contour in valid:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Safety check: if mask is >85% of image, background detection failed
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


def analyze_with_imagej(image_bytes: bytes, save_debug: bool = True):
    """
    Analyze spray coverage using ImageJ's IsoData threshold algorithm.
    
    Pipeline (replicates ImageJ manual workflow):
    1. Convert to 8-bit grayscale
    2. Apply ImageJ Default (IsoData) auto-threshold
    3. Detect leaf area
    4. Count sprayed pixels (above threshold, within leaf)
    5. Calculate coverage percentage
    
    Returns: (coverage_percentage, leaf_area, sprayed_area, processed_image_b64)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return 0.0, 0, 0, None
    
    original_image = image.copy()
    
    # Step 1: Prepare both grayscale and color representations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Step 2: Detect leaf/object mask
    leaf_mask = detect_leaf_mask(gray)
    leaf_area = cv2.countNonZero(leaf_mask)
    
    if leaf_area == 0:
        _, buffer = cv2.imencode('.jpg', original_image)
        return 0.0, 0, 0, base64.b64encode(buffer).decode('utf-8')
    
    logger.info(f"Leaf area: {leaf_area} px ({leaf_area / (gray.shape[0]*gray.shape[1]) * 100:.1f}% of image)")
    
    # ── METHOD 1: Brightness (ImageJ IsoData threshold) ──
    # Detects bright white/yellow spray spots
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(int)
    threshold = imagej_isodata_threshold(histogram)
    logger.info(f"ImageJ IsoData threshold (brightness): {threshold}")
    
    _, brightness_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    brightness_mask = cv2.bitwise_and(brightness_mask, leaf_mask)
    
    # ── METHOD 2: Color detection (cyan/blue fluorescence) ──
    # Detects spray droplets that glow cyan/blue under UV
    # HSV range for cyan/blue: Hue 85-130, Saturation >30, Value >60
    lower_cyan = np.array([85, 30, 60])
    upper_cyan = np.array([130, 255, 255])
    color_mask_hsv = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    # Additional BGR check: blue or green channel must dominate red
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    cyan_check = ((g > r + 5) & (b > r)).astype(np.uint8) * 255
    blue_check = ((b > r + 15) & (b > g + 5)).astype(np.uint8) * 255
    bgr_mask = cv2.bitwise_or(cyan_check, blue_check)
    
    color_mask = cv2.bitwise_and(color_mask_hsv, bgr_mask)
    color_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    # ── COMBINE both methods ──
    # Spray = bright spots OR cyan/blue fluorescent spots
    spray_mask = cv2.bitwise_or(brightness_mask, color_mask)
    
    logger.info(f"Detection: brightness={cv2.countNonZero(brightness_mask)}px, color={cv2.countNonZero(color_mask)}px, combined={cv2.countNonZero(spray_mask)}px")
    
    # Step 5: Filter small particles (noise) — like ImageJ Analyze Particles
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spray_mask, connectivity=8)
    filtered_mask = np.zeros_like(spray_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 5:  # Min particle size
            filtered_mask[labels == i] = 255
    
    sprayed_area = cv2.countNonZero(filtered_mask)
    coverage = (sprayed_area / leaf_area * 100) if leaf_area > 0 else 0.0
    coverage = min(coverage, 100.0)
    
    logger.info(f"ImageJ analysis: threshold={threshold}, leaf={leaf_area}, spray={sprayed_area}, coverage={coverage:.2f}%")
    
    # Step 6: Create output image with yellow overlay
    processed_image = original_image.copy()
    processed_image[filtered_mask > 0] = [0, 255, 255]  # Yellow for spray
    
    _, buffer = cv2.imencode('.jpg', processed_image)
    processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    if save_debug:
        cv2.imwrite("debug_leaf_mask.jpg", leaf_mask)
        cv2.imwrite("debug_threshold.jpg", thresholded)
        cv2.imwrite("debug_spray_mask.jpg", spray_mask)
        cv2.imwrite("debug_filtered_mask.jpg", filtered_mask)
        cv2.imwrite("debug_result.jpg", processed_image)
    
    return round(coverage, 2), leaf_area, sprayed_area, processed_image_b64
