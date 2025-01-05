from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict
import io
from openpyxl import Workbook
from datetime import datetime
from app.models.image import ImageAnalysisResponse, BatchAnalysisResponse
from app.image_processing.analyzer import SprayAnalyzer

# Almacenamiento temporal de resultados (en producción usar Redis o similar)
analysis_results: Dict[str, dict] = {}

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
    # Validar número máximo de archivos
    MAX_FILES = 100
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB por archivo
    
    if not files:
        raise HTTPException(400, detail="No se proporcionaron archivos")
    
    if len(files) > MAX_FILES:
        raise HTTPException(
            400, 
            detail=f"Se excedió el límite de archivos. Máximo permitido: {MAX_FILES}"
        )
    
    analyses = []
    errors = []
    
    for file in files:
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            errors.append(f"{file.filename} no es una imagen válida")
            continue
        
        # Validar tamaño del archivo
        try:
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                errors.append(
                    f"{file.filename} excede el tamaño máximo permitido de {MAX_FILE_SIZE/1024/1024}MB"
                )
                continue
                
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

    analysis_id = SprayAnalyzer.generate_image_id()
    
    analysis_results[analysis_id] = {
        "analyses": [analysis.dict() for analysis in analyses],
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }
    
    return BatchAnalysisResponse(
        analyses=analyses,
        summary=summary,
        analysis_id=analysis_id
    )

@app.get("/download-excel/{analysis_id}")
async def download_excel(analysis_id: str):
    # Verificar si existe el análisis
    if analysis_id not in analysis_results:
        raise HTTPException(404, detail="Análisis no encontrado")
    
    results = analysis_results[analysis_id]
    
    # Crear el archivo Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Análisis de Rociado"
    
    # Agregar encabezados
    headers = ["Archivo", "Cobertura (%)", "Área Total", "Área Rociada", "ID de Imagen"]
    ws.append(headers)
    
    # Agregar datos de cada análisis
    for analysis in results["analyses"]:
        ws.append([
            analysis["file_name"],
            analysis["coverage_percentage"],
            analysis["total_area"],
            analysis["sprayed_area"],
            analysis["image_id"]
        ])
    
    # Agregar resumen
    ws.append([])  # Línea en blanco
    ws.append(["RESUMEN"])
    ws.append(["Total de imágenes", results["summary"]["total_images"]])
    ws.append(["Cobertura promedio", f"{results['summary']['average_coverage']}%"])
    ws.append(["Cobertura mínima", f"{results['summary']['min_coverage']}%"])
    ws.append(["Cobertura máxima", f"{results['summary']['max_coverage']}%"])
    ws.append(["Área total analizada", results["summary"]["total_area_analyzed"]])
    ws.append(["Área total rociada", results["summary"]["total_area_sprayed"]])
    
    if "errors" in results["summary"]:
        ws.append([])
        ws.append(["ERRORES"])
        for error in results["summary"]["errors"]:
            ws.append([error])
    
    # Guardar el Excel en memoria
    excel_file = io.BytesIO()
    wb.save(excel_file)
    excel_file.seek(0)
    
    # Crear nombre de archivo con fecha
    filename = f"analisis_rociado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    return StreamingResponse(
        excel_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )