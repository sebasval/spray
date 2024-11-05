from pydantic import BaseModel
from typing import Optional, List

class ImageAnalysisResponse(BaseModel):
    coverage_percentage: float
    total_area: int
    sprayed_area: int
    image_id: str
    file_name: str
    status: str = "success"
    message: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    analyses: List[ImageAnalysisResponse]
    summary: dict