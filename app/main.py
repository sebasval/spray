from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io

# Importamos nuestras clases personalizadas
from app.models.image import ImageAnalysisResponse
from app.image_processing.analyzer import SprayAnalyzer

# Configuración de la app
app = FastAPI(
    title="Spray Analyzer API",
    description="API para análisis de cobertura de rociado en hojas",
    version="1.0.0"
)

# Configurar CORS (mantener el que ya teníamos)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mantenemos los endpoints existentes
@app.get("/")
async def read_root():
    return {"message": "Bienvenido a Spray Analyzer API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

# Nuevo endpoint para analizar imágenes
@app.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile):
    # Verificar que el archivo sea una imagen
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer el contenido de la imagen
        contents = await file.read()
        
        # Crear instancia del analizador y procesar la imagen
        analyzer = SprayAnalyzer()
        coverage, total_area, sprayed_area = analyzer.analyze_image(contents)
        
        # Generar y retornar la respuesta
        return ImageAnalysisResponse(
            coverage_percentage=round(coverage, 2),
            total_area=total_area,
            sprayed_area=sprayed_area,
            image_id=analyzer.generate_image_id()
        )
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))