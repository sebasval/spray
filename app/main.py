from fastapi import FastAPI, UploadFile, HTTPException, File, Depends, Security, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Dict, Optional
import io
from openpyxl import Workbook
from datetime import datetime

# Importaciones existentes
from app.models.image import ImageAnalysisResponse, BatchAnalysisResponse
from app.image_processing.analyzer import SprayAnalyzer

# Nuevas importaciones para autenticación
from app.auth.security import verify_token, create_access_token, security, verify_password, oauth2_scheme
from app.auth.models import Token, UserCreate, User
from app.auth.database import DatabaseManager
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.auth.security import SECRET_KEY, ALGORITHM

# WhatsApp Bot
from app.whatsapp_bot import router as whatsapp_router

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
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Añade esta línea
)

# WhatsApp Bot routes
app.include_router(whatsapp_router)

# Rutas de autenticación
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = DatabaseManager.get_user(form_data.username)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Email o contraseña incorrectos"
        )
    
    # Verificar la contraseña
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Email o contraseña incorrectos"
        )
    
    # Actualizar último login
    DatabaseManager.update_last_login(user.email)
    
    access_token = create_access_token(
        data={"sub": user.email}
    )
    return Token(access_token=access_token)

@app.post("/users", response_model=User)
async def create_user(
    user: UserCreate,
    authorization: Optional[str] = Header(None)
):
    """
    Crear un nuevo usuario. El primer usuario se puede crear sin autenticación.
    Los siguientes usuarios requieren autenticación del administrador.
    """
    # Verificar si ya hay usuarios existentes
    user_count = DatabaseManager.count_users()
    
    # Verificar si ya hay 7 usuarios (o el límite configurado)
    if user_count >= 7:
        raise HTTPException(
            status_code=400,
            detail="Se ha alcanzado el límite máximo de usuarios permitidos"
        )
    
    # Si no es el primer usuario, verificar que quien lo crea esté autenticado
    if user_count > 0:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Se requiere autenticación para crear usuarios adicionales"
            )
        
        # Extraer y verificar el token
        token = authorization.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if not email:
                raise HTTPException(
                    status_code=401,
                    detail="Token inválido"
                )
        except Exception:
            raise HTTPException(
                status_code=401,
                detail="Se requiere autenticación válida para crear usuarios adicionales"
            )
    
    return DatabaseManager.create_user(user)

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: str = Depends(verify_token)):
    """
    Obtener información del usuario actual
    """
    user = DatabaseManager.get_user(current_user)
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

# Rutas existentes modificadas para requerir autenticación
@app.get("/")
async def read_root():
    return {"message": "Bienvenido a Spray Analyzer API"}

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    return """<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Política de Privacidad - Spray Analyzer</title>
<style>body{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;color:#333;line-height:1.6}h1{color:#2e7d32}h2{color:#388e3c;margin-top:30px}</style>
</head><body>
<h1>Política de Privacidad — Spray Analyzer</h1>
<p><strong>Última actualización:</strong> 14 de febrero de 2026</p>

<h2>1. Información que recopilamos</h2>
<p>Spray Analyzer recopila únicamente las imágenes que los usuarios envían voluntariamente a través de WhatsApp para su análisis de cobertura de rociado agrícola. También recopilamos el número de teléfono del remitente para poder enviar la respuesta.</p>

<h2>2. Uso de la información</h2>
<p>Las imágenes recibidas se procesan exclusivamente para generar un análisis de cobertura de rociado mediante visión por computador. No utilizamos las imágenes para ningún otro propósito.</p>

<h2>3. Almacenamiento y retención</h2>
<p>Las imágenes se procesan en tiempo real y <strong>no se almacenan permanentemente</strong> en nuestros servidores. Los datos del análisis se mantienen temporalmente durante la sesión activa.</p>

<h2>4. Compartir información</h2>
<p>No compartimos, vendemos ni transferimos la información personal de los usuarios a terceros.</p>

<h2>5. Seguridad</h2>
<p>Utilizamos medidas de seguridad estándar de la industria, incluyendo cifrado en tránsito (HTTPS) y autenticación mediante tokens JWT.</p>

<h2>6. Derechos del usuario</h2>
<p>Los usuarios pueden solicitar la eliminación de cualquier dato asociado a su cuenta contactándonos directamente.</p>

<h2>7. Contacto</h2>
<p>Para consultas sobre privacidad: <strong>admin@almagricola.com</strong></p>

<h2>8. Cambios a esta política</h2>
<p>Nos reservamos el derecho de actualizar esta política. Los cambios serán publicados en esta misma página.</p>
</body></html>"""

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_single_image(
    file: UploadFile,
    current_user: str = Depends(verify_token)
):
    """
    Analiza una imagen para detectar cobertura de spray usando fluorescencia UV.
    Combina detección por brillo relativo y por color cian/azul.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="El archivo debe ser una imagen")
    try:
        contents = await file.read()
        coverage, total_area, sprayed_area, processed_image = SprayAnalyzer.analyze_image(
            contents,
            save_debug=True
        )
        return ImageAnalysisResponse(
            coverage_percentage=round(coverage, 2),
            total_area=total_area,
            sprayed_area=sprayed_area,
            image_id=SprayAnalyzer.generate_image_id(),
            file_name=file.filename,
            processed_image=processed_image
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"Error al analizar imagen: {str(e)}")

@app.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_multiple_images(
    files: List[UploadFile] = File(...),
    current_user: str = Depends(verify_token)
):
    """
    Analiza múltiples imágenes para detectar cobertura de spray usando fluorescencia UV.
    Combina detección por brillo relativo y por color cian/azul.
    """
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
                
            coverage, total_area, sprayed_area, processed_image = SprayAnalyzer.analyze_image(
                contents,
                save_debug=False
            )
            analysis = ImageAnalysisResponse(
                coverage_percentage=round(coverage, 2),
                total_area=total_area,
                sprayed_area=sprayed_area,
                image_id=SprayAnalyzer.generate_image_id(),
                file_name=file.filename,
                processed_image=processed_image
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
async def download_excel(
    analysis_id: str,
    current_user: str = Depends(verify_token)
):
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