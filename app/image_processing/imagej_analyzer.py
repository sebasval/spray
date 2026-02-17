"""
ImageJ-compatible spray analyzer.
All threshold algorithms ported directly from ImageJ source code:
https://github.com/imagej/ImageJ/blob/master/ij/process/AutoThresholder.java

Particle analysis replicates ImageJ's Analyze Particles command.
"""
import numpy as np
import cv2
import base64
import logging
import math

logger = logging.getLogger(__name__)


# =====================================================================
# ImageJ Threshold Algorithms (ported from Java)
# =====================================================================

def imagej_default_threshold(histogram):
    """ImageJ Default = modified IsoData. Used by Image > Adjust > Threshold > Auto."""
    data = list(histogram)
    n = len(data)
    
    # Cap the mode if it dominates (defaultIsoData)
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
    
    return _ij_isodata(data)


def _ij_isodata(data):
    """Core IsoData implementation from ImageJ."""
    max_value = len(data) - 1
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
        return len(data) // 2
    
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


def imagej_otsu_threshold(histogram):
    """ImageJ Otsu's threshold algorithm."""
    data = list(histogram)
    n = len(data)
    num_pixels = sum(data)
    if num_pixels == 0:
        return 0
    
    term = 1.0 / num_pixels
    histo = [term * d for d in data]
    
    # Cumulative normalized histogram
    cnh = [0.0] * n
    cnh[0] = histo[0]
    for i in range(1, n):
        cnh[i] = cnh[i - 1] + histo[i]
    
    # Mean
    mean = [0.0] * n
    for i in range(1, n):
        mean[i] = mean[i - 1] + i * histo[i]
    
    total_mean = mean[n - 1]
    
    # Find threshold that maximizes between-class variance
    threshold = 0
    max_bcv = 0.0
    for i in range(n):
        denom = cnh[i] * (1.0 - cnh[i])
        if denom == 0:
            continue
        bcv = (total_mean * cnh[i] - mean[i]) ** 2 / denom
        if bcv > max_bcv:
            max_bcv = bcv
            threshold = i
    
    return threshold


def imagej_triangle_threshold(histogram):
    """ImageJ Triangle threshold algorithm (Zack et al. 1977)."""
    data = list(histogram)
    n = len(data)
    
    # Find min and max
    min_idx = 0
    for i in range(n):
        if data[i] > 0:
            min_idx = i
            break
    if min_idx > 0:
        min_idx -= 1
    
    min2 = 0
    for i in range(n - 1, 0, -1):
        if data[i] > 0:
            min2 = i
            break
    if min2 < n - 1:
        min2 += 1
    
    dmax = 0
    max_idx = 0
    for i in range(n):
        if data[i] > dmax:
            max_idx = i
            dmax = data[i]
    
    inverted = False
    if (max_idx - min_idx) < (min2 - max_idx):
        inverted = True
        left, right = 0, n - 1
        while left < right:
            data[left], data[right] = data[right], data[left]
            left += 1
            right -= 1
        min_idx = n - 1 - min2
        max_idx = n - 1 - max_idx
    
    if min_idx == max_idx:
        return min_idx
    
    nx = data[max_idx]
    ny = min_idx - max_idx
    d = math.sqrt(nx * nx + ny * ny)
    nx /= d
    ny /= d
    d = nx * min_idx + ny * data[min_idx]
    
    split = min_idx
    split_distance = 0
    for i in range(min_idx + 1, max_idx + 1):
        new_distance = nx * i + ny * data[i] - d
        if new_distance > split_distance:
            split = i
            split_distance = new_distance
    split -= 1
    
    if inverted:
        left, right = 0, n - 1
        while left < right:
            data[left], data[right] = data[right], data[left]
            left += 1
            right -= 1
        return n - 1 - split
    return split


def imagej_yen_threshold(histogram):
    """ImageJ Yen threshold algorithm."""
    data = list(histogram)
    n = len(data)
    total = sum(data)
    if total == 0:
        return 0
    
    norm_histo = [d / total for d in data]
    
    P1 = [0.0] * n
    P1[0] = norm_histo[0]
    for i in range(1, n):
        P1[i] = P1[i - 1] + norm_histo[i]
    
    P1_sq = [0.0] * n
    P1_sq[0] = norm_histo[0] * norm_histo[0]
    for i in range(1, n):
        P1_sq[i] = P1_sq[i - 1] + norm_histo[i] * norm_histo[i]
    
    P2_sq = [0.0] * n
    P2_sq[n - 1] = 0.0
    for i in range(n - 2, -1, -1):
        P2_sq[i] = P2_sq[i + 1] + norm_histo[i + 1] * norm_histo[i + 1]
    
    threshold = -1
    max_crit = float('-inf')
    for it in range(n):
        p1p2 = P1_sq[it] * P2_sq[it]
        p1_1mp1 = P1[it] * (1.0 - P1[it])
        crit = -1.0 * (math.log(p1p2) if p1p2 > 0 else 0.0) + 2 * (math.log(p1_1mp1) if p1_1mp1 > 0 else 0.0)
        if crit > max_crit:
            max_crit = crit
            threshold = it
    
    return threshold


# All available methods
THRESHOLD_METHODS = {
    'Default': imagej_default_threshold,
    'Otsu': imagej_otsu_threshold,
    'Triangle': imagej_triangle_threshold,
    'Yen': imagej_yen_threshold,
}


# =====================================================================
# Leaf/Object Detection
# =====================================================================

def detect_object_mask(gray):
    """
    Detect the leaf/flower/object separating it from the background.
    
    Strategy: Two-pass approach
    1. Use ImageJ Default threshold on the FULL image to separate object from background
    2. Morphological cleanup + largest contour
    """
    # Use ImageJ's own threshold to separate object from background
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(int)
    bg_threshold = imagej_default_threshold(histogram)
    
    logger.info(f"Object detection - IsoData threshold for background: {bg_threshold}")
    
    # If threshold is very low (<10), the image is mostly dark = background
    # Bump it up to avoid including gray Photoroom backgrounds
    effective_threshold = max(bg_threshold, 25)
    
    _, object_mask = cv2.threshold(gray, effective_threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours, keep largest as the object
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray), 0
    
    image_area = gray.shape[0] * gray.shape[1]
    min_area = image_area * 0.005
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid:
        return np.zeros_like(gray), 0
    
    largest = max(valid, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    object_area = cv2.countNonZero(mask)
    object_pct = object_area / image_area * 100
    
    # Safety: if object is >85% of image, background detection failed
    # Try progressively higher thresholds
    if object_pct > 85:
        for higher_thresh in [50, 80, 100]:
            logger.warning(f"Object mask {object_pct:.1f}% of image, retrying with threshold={higher_thresh}")
            _, strict = cv2.threshold(gray, higher_thresh, 255, cv2.THRESH_BINARY)
            strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, kernel, iterations=2)
            strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, kernel, iterations=2)
            contours2, _ = cv2.findContours(strict, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid2 = [c for c in contours2 if cv2.contourArea(c) >= min_area]
            if valid2:
                largest2 = max(valid2, key=cv2.contourArea)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [largest2], -1, 255, -1)
                new_pct = cv2.countNonZero(mask) / image_area * 100
                if new_pct < 85:
                    object_area = cv2.countNonZero(mask)
                    break
    
    logger.info(f"Object area: {object_area} px ({object_area / image_area * 100:.1f}% of image)")
    return mask, object_area


# =====================================================================
# Analyze Particles (ImageJ-compatible)
# =====================================================================

def analyze_particles(binary_mask, min_size=5, max_size=float('inf'), 
                      min_circularity=0.0, max_circularity=1.0,
                      exclude_edges=False, image_shape=None):
    """
    Replicates ImageJ's Analyze Particles command.
    
    For each connected component:
    - Measures area
    - Measures circularity = 4π × area / perimeter²
    - Filters by size and circularity
    - Optionally excludes particles touching image edges
    
    Returns: (filtered_mask, particle_results)
    where particle_results is a list of dicts with Area, Mean, etc.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    filtered_mask = np.zeros_like(binary_mask)
    results = []
    h, w = binary_mask.shape
    
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Size filter
        if area < min_size or area > max_size:
            continue
        
        # Edge exclusion
        if exclude_edges:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h:
                continue
        
        # Circularity filter
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
        else:
            circularity = 0
        
        if circularity < min_circularity or circularity > max_circularity:
            continue
        
        # This particle passes all filters
        filtered_mask[labels == i] = 255
        results.append({
            'id': i,
            'area': area,
            'circularity': round(circularity, 4),
            'centroid_x': round(centroids[i][0], 1),
            'centroid_y': round(centroids[i][1], 1),
        })
    
    return filtered_mask, results


# =====================================================================
# Main Analysis Pipeline
# =====================================================================

def analyze_with_imagej(image_bytes: bytes, save_debug: bool = True,
                        threshold_method: str = 'Default',
                        min_particle_size: int = 5,
                        exclude_edge_particles: bool = False):
    """
    Full spray coverage analysis using ImageJ algorithms.
    
    Pipeline (replicates ImageJ manual workflow):
    
    ┌─────────────────────────────────────────────┐
    │ 1. INPUT: Imagen original (color)           │
    │    └→ Convertir a escala de grises (8-bit)  │
    │                                             │
    │ 2. DETECTAR OBJETO (hoja/flor)              │
    │    └→ IsoData threshold sobre imagen        │
    │       completa → separar fondo vs objeto    │
    │    └→ Contorno más grande = el objeto       │
    │                                             │
    │ 3. SEGUNDO THRESHOLD (dentro del objeto)    │
    │    └→ Histograma SOLO de píxeles del objeto │
    │    └→ IsoData/Otsu/Triangle sobre ese       │
    │       histograma → separar hoja vs spray    │
    │                                             │
    │ 4. ANALYZE PARTICLES                        │
    │    └→ Componentes conectados del spray      │
    │    └→ Filtrar por tamaño y circularidad     │
    │    └→ Medir área de cada partícula          │
    │                                             │
    │ 5. OUTPUT                                   │
    │    └→ Coverage = spray_area / object_area   │
    │    └→ Imagen con overlay amarillo           │
    │    └→ Tabla de partículas (como ImageJ)     │
    └─────────────────────────────────────────────┘
    
    Returns: (coverage_percentage, object_area, sprayed_area, processed_image_b64)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return 0.0, 0, 0, None
    
    original_image = image.copy()
    h, w = image.shape[:2]
    
    # ── Step 1: Convert to 8-bit grayscale ──
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.info(f"Image: {w}x{h}, {w*h} pixels")
    
    # ── Step 2: Detect object (leaf/flower) ──
    # First threshold: separate object from background using full image histogram
    object_mask, object_area = detect_object_mask(gray)
    
    if object_area == 0:
        _, buffer = cv2.imencode('.jpg', original_image)
        return 0.0, 0, 0, base64.b64encode(buffer).decode('utf-8')
    
    # ── Step 3: Second threshold WITHIN the object ──
    # Extract histogram of ONLY the object pixels (exclude background)
    object_pixels = gray[object_mask > 0]
    
    # Build histogram of just the object
    obj_histogram = np.zeros(256, dtype=int)
    for val in object_pixels:
        obj_histogram[val] += 1
    
    # Apply selected ImageJ threshold method on the object histogram
    threshold_fn = THRESHOLD_METHODS.get(threshold_method, imagej_default_threshold)
    spray_threshold = threshold_fn(obj_histogram)
    
    logger.info(f"Spray threshold ({threshold_method}) within object: {spray_threshold}")
    
    # Apply threshold: pixels ABOVE threshold within the object = spray
    _, spray_binary = cv2.threshold(gray, spray_threshold, 255, cv2.THRESH_BINARY)
    spray_within_object = cv2.bitwise_and(spray_binary, object_mask)
    
    # ── Step 4: Analyze Particles ──
    filtered_mask, particles = analyze_particles(
        spray_within_object,
        min_size=min_particle_size,
        exclude_edges=exclude_edge_particles,
        image_shape=(h, w)
    )
    
    sprayed_area = cv2.countNonZero(filtered_mask)
    coverage = (sprayed_area / object_area * 100) if object_area > 0 else 0.0
    coverage = min(coverage, 100.0)
    
    num_particles = len(particles)
    logger.info(
        f"Analysis complete: threshold={spray_threshold}, "
        f"object={object_area}px, spray={sprayed_area}px, "
        f"coverage={coverage:.2f}%, particles={num_particles}"
    )
    
    # ── Step 5: Create output image ──
    processed_image = original_image.copy()
    processed_image[filtered_mask > 0] = [0, 255, 255]  # Yellow overlay for spray
    
    _, buffer = cv2.imencode('.jpg', processed_image)
    processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # ── Debug output ──
    if save_debug:
        cv2.imwrite("debug_grayscale.jpg", gray)
        cv2.imwrite("debug_object_mask.jpg", object_mask)
        cv2.imwrite("debug_spray_binary.jpg", spray_within_object)
        cv2.imwrite("debug_filtered_particles.jpg", filtered_mask)
        cv2.imwrite("debug_result.jpg", processed_image)
        
        # Log particle table (like ImageJ Results window)
        if particles:
            logger.info("─── Particle Results (ImageJ-style) ───")
            logger.info(f"{'#':>4} {'Area':>10} {'Circ.':>8} {'X':>8} {'Y':>8}")
            for p in particles[:20]:  # Show first 20
                logger.info(f"{p['id']:>4} {p['area']:>10} {p['circularity']:>8.4f} {p['centroid_x']:>8.1f} {p['centroid_y']:>8.1f}")
            if len(particles) > 20:
                logger.info(f"  ... and {len(particles) - 20} more particles")
    
    return round(coverage, 2), object_area, sprayed_area, processed_image_b64
