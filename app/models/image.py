from pydantic import BaseModel
from typing import List, Optional

class ImageAnalysisResponse(BaseModel):
    coverage_percentage: float
    total_area: int
    sprayed_area: int
    image_id: str
    file_name: str
    processed_image: Optional[str] = None  # Base64 de la imagen procesada

class BatchAnalysisResponse(BaseModel):
    analyses: List[ImageAnalysisResponse]
    summary: dict
    analysis_id: str  # Nuevo campo para identificar el análisis para la descarga del Excel