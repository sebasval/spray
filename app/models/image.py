from pydantic import BaseModel
from typing import Optional

class ImageAnalysisResponse(BaseModel):
    coverage_percentage: float     # Porcentaje de área cubierta por el rociado
    total_area: int               # Área total de la imagen en píxeles
    sprayed_area: int            # Área rociada en píxeles
    image_id: str                # Identificador único de la imagen
    status: str = "success"      # Estado del análisis
    message: Optional[str] = None # Mensaje opcional (útil para errores)