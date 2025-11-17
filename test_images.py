#!/usr/bin/env python3
"""
Script de prueba para analizar imágenes de ejemplo
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from app.image_processing.analyzer import SprayAnalyzer

def analyze_image_file(image_path: str):
    """Analiza una imagen y muestra los resultados"""
    print(f"\n{'='*60}")
    print(f"Analizando: {image_path}")
    print(f"{'='*60}")
    
    # Leer imagen
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        print(f"❌ Error: No se encontró el archivo {image_path}")
        return None
    
    # Leer bytes de la imagen
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Analizar
    try:
        coverage, leaf_area, sprayed_area, processed_image_base64 = SprayAnalyzer.analyze_image(
            image_bytes, 
            save_debug=True
        )
        
        print(f"\n📊 RESULTADOS:")
        print(f"  • Coverage: {coverage}%")
        print(f"  • Área de hoja: {leaf_area:,} píxeles")
        print(f"  • Área con spray: {sprayed_area:,} píxeles")
        
        if coverage > 0:
            print(f"\n✅ Se detectaron gotas (coverage: {coverage}%)")
        else:
            print(f"\n❌ No se detectaron gotas (coverage: {coverage}%)")
        
        # Leer estadísticas de debug si existen
        debug_file = Path("debug_stats.txt")
        if debug_file.exists():
            print(f"\n📋 Estadísticas detalladas:")
            with open(debug_file, 'r') as f:
                print(f.read())
        
        return {
            'coverage': coverage,
            'leaf_area': leaf_area,
            'sprayed_area': sprayed_area,
            'has_droplets': coverage > 0
        }
        
    except Exception as e:
        print(f"❌ Error al analizar: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Buscar imágenes de ejemplo
    workspace = Path("/workspace")
    
    # Buscar imágenes con nombres comunes
    possible_names = [
        "ejemplo_sin_coverage.jpg",
        "ejemplo_con_coverage.jpg",
        "sin_gotas.jpg",
        "con_gotas.jpg",
        "sin_coverage.jpg",
        "con_coverage.jpg",
        "imagen_sin.jpg",
        "imagen_con.jpg",
    ]
    
    # También buscar cualquier .jpg o .png en el workspace
    image_files = list(workspace.glob("*.jpg")) + list(workspace.glob("*.png")) + list(workspace.glob("*.jpeg"))
    
    # Filtrar archivos de debug
    image_files = [f for f in image_files if not f.name.startswith("debug_")]
    
    if len(image_files) == 0:
        print("⚠️  No se encontraron imágenes en /workspace")
        print("\nPor favor, coloca las imágenes en /workspace con nombres:")
        print("  - ejemplo_sin_coverage.jpg (imagen sin gotas)")
        print("  - ejemplo_con_coverage.jpg (imagen con gotas)")
        print("\nO proporciona las rutas como argumentos:")
        print("  python3 test_images.py /ruta/imagen1.jpg /ruta/imagen2.jpg")
        sys.exit(1)
    
    print(f"🔍 Encontradas {len(image_files)} imagen(es) para analizar")
    
    results = []
    for img_file in sorted(image_files):
        result = analyze_image_file(str(img_file))
        if result:
            results.append((img_file.name, result))
    
    # Resumen
    if results:
        print(f"\n{'='*60}")
        print("📈 RESUMEN")
        print(f"{'='*60}")
        for name, result in results:
            status = "✅ CON GOTAS" if result['has_droplets'] else "❌ SIN GOTAS"
            print(f"{name:40s} | {status:15s} | Coverage: {result['coverage']:6.2f}%")
