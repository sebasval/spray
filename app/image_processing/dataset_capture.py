"""
Dataset capture for future YOLOv8-seg training.

Goal: accumulate labeled images (OpenCV result + Moondream result + eventual
ImageJ ground truth from client) without growing storage costs.

Strategy:
- Compress images aggressively (JPEG quality 60, max width 1024px)
- Store metadata as compact JSON (one row per image)
- Async writes to avoid blocking the API response
- Configurable storage path; defaults to local ./dataset/
"""
import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DATASET_DIR = os.getenv("DATASET_DIR", "./dataset")
DATASET_MAX_WIDTH = int(os.getenv("DATASET_MAX_WIDTH", "1024"))
DATASET_JPEG_QUALITY = int(os.getenv("DATASET_JPEG_QUALITY", "60"))
DATASET_ENABLED = os.getenv("DATASET_ENABLED", "true").lower() == "true"


class DatasetCapture:
    """Lightweight dataset writer (compressed JPEG + JSON metadata)."""

    @staticmethod
    def _ensure_dirs():
        os.makedirs(os.path.join(DATASET_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "metadata"), exist_ok=True)

    @staticmethod
    def _compress_image(image_bytes: bytes) -> bytes:
        """Resize + re-encode JPEG to minimize storage cost."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        h, w = img.shape[:2]
        if w > DATASET_MAX_WIDTH:
            scale = DATASET_MAX_WIDTH / w
            new_size = (DATASET_MAX_WIDTH, int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DATASET_JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", img, encode_params)
        if not ok:
            return image_bytes
        return buf.tobytes()

    @staticmethod
    def _save_sync(
        image_id: str,
        image_bytes: bytes,
        file_name: str,
        coverage_opencv: float,
        coverage_moondream: float,
        coverage_final: float,
        validation_flag: str,
    ):
        try:
            DatasetCapture._ensure_dirs()

            compressed = DatasetCapture._compress_image(image_bytes)
            img_path = os.path.join(DATASET_DIR, "images", f"{image_id}.jpg")
            with open(img_path, "wb") as f:
                f.write(compressed)

            metadata = {
                "image_id": image_id,
                "file_name": file_name,
                "timestamp": datetime.now().isoformat(),
                "coverage_opencv": round(coverage_opencv, 2),
                "coverage_moondream": round(coverage_moondream, 2) if coverage_moondream >= 0 else None,
                "coverage_final": round(coverage_final, 2),
                "validation_flag": validation_flag,
                "ground_truth_imagej": None,  # Filled in later when client reports
                "image_size_bytes": len(compressed),
            }
            meta_path = os.path.join(DATASET_DIR, "metadata", f"{image_id}.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Dataset entry saved: {image_id} ({len(compressed)} bytes)")
        except Exception as e:
            logger.warning(f"Dataset capture failed for {image_id}: {e}")

    @staticmethod
    def save_async(
        image_id: str,
        image_bytes: bytes,
        file_name: str,
        coverage_opencv: float,
        coverage_moondream: float,
        coverage_final: float,
        validation_flag: str,
    ):
        """Fire-and-forget save. Doesn't block the API response."""
        if not DATASET_ENABLED:
            return

        thread = threading.Thread(
            target=DatasetCapture._save_sync,
            args=(
                image_id,
                image_bytes,
                file_name,
                coverage_opencv,
                coverage_moondream,
                coverage_final,
                validation_flag,
            ),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def add_ground_truth(image_id: str, ground_truth_pct: float) -> bool:
        """
        When client reports ImageJ-measured % for a previously analyzed image,
        update its metadata file. Returns True if updated.
        """
        try:
            meta_path = os.path.join(DATASET_DIR, "metadata", f"{image_id}.json")
            if not os.path.exists(meta_path):
                return False
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            metadata["ground_truth_imagej"] = round(ground_truth_pct, 2)
            metadata["ground_truth_added_at"] = datetime.now().isoformat()
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"Failed to add ground truth for {image_id}: {e}")
            return False
