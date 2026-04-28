from pydantic import BaseModel
from typing import List, Optional

class ImageAnalysisResponse(BaseModel):
    coverage_percentage: float
    coverage_opencv: Optional[float] = None
    coverage_moondream: Optional[float] = None
    validation_flag: Optional[str] = None  # "opencv_only" | "validated" | "backup_used"
    total_area: int
    sprayed_area: int
    image_id: str
    file_name: str
    processed_image: Optional[str] = None  # Base64 de la imagen procesada

class BatchAnalysisResponse(BaseModel):
    analyses: List[ImageAnalysisResponse]
    summary: dict
    analysis_id: str
