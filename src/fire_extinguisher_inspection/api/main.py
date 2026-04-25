"""API FastAPI para inspección de imágenes de extintores."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline


try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:
    FastAPI = None  # type: ignore[assignment]
    File = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    UploadFile = Any  # type: ignore[misc, assignment]
    JSONResponse = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


def _nombre_seguro(nombre: str) -> str:
    nombre = Path(nombre).name
    nombre = re.sub(r"[^A-Za-z0-9_.-]", "_", nombre)
    return nombre or "imagen.jpg"


def crear_app():
    """Crea la aplicación FastAPI."""

    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError("No se puede importar FastAPI. Instala requirements.txt.") from _FASTAPI_IMPORT_ERROR

    config = cargar_configuracion(
        os.getenv("CONFIG_PATH", "config/default.yaml"),
        os.getenv("CLASSES_PATH", "config/classes.yaml"),
    )
    app = FastAPI(title="Inspección visual de extintores", version="0.1.0")
    pipeline = InspectionPipeline(config)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "modelo_yolo_existe": config.modelos.yolo.exists(),
            "modelo_cnn_existe": config.modelos.cnn.exists(),
            "clases": config.clases.classification,
        }

    @app.post("/inspect/image")
    async def inspect_image(
        file: UploadFile = File(...),  # type: ignore[misc]
        guardar_anotada: bool = config.api.guardar_anotada_por_defecto,
        guardar_crops: bool = False,
    ):
        contenido = await file.read()
        if not contenido:
            raise HTTPException(status_code=400, detail="La imagen recibida está vacía.")

        upload_dir = config.outputs.reports / "api_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        ruta_imagen = upload_dir / _nombre_seguro(file.filename or "imagen.jpg")
        ruta_imagen.write_bytes(contenido)

        resultado = pipeline.inspeccionar_imagen(
            ruta_imagen,
            guardar_crops=guardar_crops,
            guardar_anotada=guardar_anotada,
        )
        status_code = 200 if resultado.ok else 422
        return JSONResponse(content=resultado.to_dict(), status_code=status_code)

    return app


try:
    app = crear_app()
except RuntimeError:
    app = None
