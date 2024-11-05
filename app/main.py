from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io

from app.models.image import ImageAnalysisResponse, BatchAnalysisResponse
from app.image_processing.analyzer import SprayAnalyzer

app = FastAPI(
    title="Spray Analyzer API",
    description="API para análisis de cobertura de rociado en hojas",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Bienvenido a Spray Analyzer API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_single_image(file: UploadFile):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="El archivo debe ser una imagen")
    
    try:
        contents = await file.read()
        analyzer = SprayAnalyzer()
        coverage, total_area, sprayed_area = analyzer.analyze_image(contents)
        
        return ImageAnalysisResponse(
            coverage_percentage=round(coverage, 2),
            total_area=total_area,
            sprayed_area=sprayed_area,
            image_id=analyzer.generate_image_id(),
            file_name=file.filename
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_multiple_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, detail="No se proporcionaron archivos")
    
    analyses = []
    errors = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            errors.append(f"{file.filename} no es una imagen válida")
            continue
        
        try:
            contents = await file.read()
            analyzer = SprayAnalyzer()
            coverage, total_area, sprayed_area = analyzer.analyze_image(contents)
            
            analysis = ImageAnalysisResponse(
                coverage_percentage=round(coverage, 2),
                total_area=total_area,
                sprayed_area=sprayed_area,
                image_id=analyzer.generate_image_id(),
                file_name=file.filename
            )
            analyses.append(analysis)
            
        except Exception as e:
            errors.append(f"Error procesando {file.filename}: {str(e)}")
    
    if not analyses:
        raise HTTPException(400, detail="No se pudo procesar ninguna imagen")
    
    summary = SprayAnalyzer.calculate_batch_summary(analyses)
    if errors:
        summary["errors"] = errors
    
    return BatchAnalysisResponse(
        analyses=analyses,
        summary=summary
    )