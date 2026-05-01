"""API FastAPI para inspección de imágenes de extintores."""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Annotated, Any

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline


try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import JSONResponse, Response
except ImportError as exc:
    FastAPI = None  # type: ignore[assignment]
    File = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Query = None  # type: ignore[assignment]
    UploadFile = Any  # type: ignore[misc, assignment]
    JSONResponse = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AplicacionNoDisponible:
    """ASGI mínimo para informar que falta FastAPI sin romper imports."""

    def __init__(self, title: str, detalle: str) -> None:
        self.title = title
        self.version = "0.1.0"
        self.routes: list[Any] = []
        self.detalle = detalle

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Devuelve 503 si alguien intenta servir la API sin dependencias."""

        contenido = (
            '{"status":"error","detail":"'
            + self.detalle.replace('"', "'")
            + '"}'
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": contenido})


def _nombre_seguro(nombre: str) -> str:
    nombre = Path(nombre).name
    nombre = re.sub(r"[^A-Za-z0-9_.-]", "_", nombre)
    return nombre or "imagen.jpg"


def _nombre_seguro_desde_zip(nombre: str, indice: int) -> str:
    partes = [
        re.sub(r"[^A-Za-z0-9_.-]", "_", parte)
        for parte in Path(nombre).parts
        if parte not in {"", ".", ".."}
    ]
    nombre_limpio = "__".join(partes) or f"imagen_{indice}.jpg"
    return f"{indice:04d}_{nombre_limpio}"


async def _guardar_upload(file: UploadFile, upload_dir: Path) -> Path:
    """Guarda una imagen recibida por la API en una ruta local segura."""

    contenido = await file.read()
    if not contenido:
        raise HTTPException(
            status_code=400,
            detail=f"La imagen recibida está vacía: {file.filename}",
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    ruta_imagen = upload_dir / _nombre_seguro(file.filename or "imagen.jpg")
    ruta_imagen.write_bytes(contenido)
    return ruta_imagen


async def _guardar_zip(file: UploadFile, upload_dir: Path) -> list[Path]:
    """Guarda las imágenes incluidas en un ZIP recibido por la API."""

    contenido = await file.read()
    if not contenido:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo recibido está vacío: {file.filename}",
        )

    try:
        archivo = zipfile.ZipFile(io.BytesIO(contenido))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no es un ZIP válido.",
        ) from exc

    carpeta = upload_dir / _nombre_seguro(Path(file.filename or "imagenes.zip").stem)
    carpeta.mkdir(parents=True, exist_ok=True)

    rutas: list[Path] = []
    with archivo:
        for info in archivo.infolist():
            if (
                info.is_dir()
                or Path(info.filename).suffix.lower() not in EXTENSIONES_IMAGEN
            ):
                continue

            datos = archivo.read(info)
            if not datos:
                continue

            ruta_imagen = carpeta / _nombre_seguro_desde_zip(
                info.filename,
                len(rutas) + 1,
            )
            ruta_imagen.write_bytes(datos)
            rutas.append(ruta_imagen)

    if not rutas:
        raise HTTPException(
            status_code=400,
            detail="El ZIP no contiene imágenes compatibles.",
        )

    return rutas


def _contenido_zip_resultados(contenido: dict[str, Any]) -> bytes:
    """Crea un ZIP con el JSON de resultados y las imágenes anotadas."""

    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, mode="w", compression=zipfile.ZIP_DEFLATED) as archivo:
        archivo.writestr(
            "resultados.json",
            json.dumps(contenido, ensure_ascii=False, indent=2),
        )

        for indice, resultado in enumerate(contenido["resultados"], start=1):
            ruta_anotada = resultado.get("annotated_image_path")
            if not ruta_anotada:
                continue

            ruta = Path(ruta_anotada)
            if ruta.exists():
                nombre = f"annotated/{indice:04d}_{_nombre_seguro(ruta.name)}"
                archivo.write(ruta, arcname=nombre)

    memoria.seek(0)
    return memoria.read()


def crear_app():
    """Crea la aplicación FastAPI."""

    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "No se puede importar FastAPI. Instala requirements.txt."
        ) from _FASTAPI_IMPORT_ERROR

    config = cargar_configuracion(
        os.getenv("CONFIG_PATH", "config/default.yaml"),
        os.getenv("CLASSES_PATH", "config/classes.yaml"),
    )
    app = FastAPI(
        title="Inspección visual de extintores",
        version="0.1.0",
        description=(
            "API para detectar extintores y clasificar su accesibilidad. "
            "La interfaz Swagger está disponible en /docs."
        ),
        openapi_tags=[
            {"name": "Estado", "description": "Comprobación básica del servicio."},
            {
                "name": "Inspección",
                "description": "Inferencia sobre una o varias imágenes.",
            },
        ],
    )
    pipeline = InspectionPipeline(config)

    def ejecutar_lote(
        rutas_imagenes: list[Path],
        guardar_anotada: bool,
        guardar_crops: bool,
    ) -> dict[str, Any]:
        resultados = []
        for ruta_imagen in rutas_imagenes:
            resultado = pipeline.inspeccionar_imagen(
                ruta_imagen,
                guardar_crops=guardar_crops,
                guardar_anotada=guardar_anotada,
            )
            resultados.append(resultado.to_dict())

        errores = sum(1 for resultado in resultados if resultado.get("errors"))
        contenido = {
            "total": len(resultados),
            "ok": errores == 0,
            "imagenes_con_error": errores,
            "resultados": resultados,
        }
        return contenido

    def respuesta_lote_json(contenido: dict[str, Any]) -> JSONResponse:
        errores = int(contenido["imagenes_con_error"])
        status_code = 200 if errores == 0 else 207
        return JSONResponse(content=contenido, status_code=status_code)

    def respuesta_lote_zip(contenido: dict[str, Any], nombre_archivo: str) -> Response:
        zip_bytes = _contenido_zip_resultados(contenido)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{nombre_archivo}"',
            },
        )

    @app.get("/health", tags=["Estado"], summary="Comprobar el estado del servicio")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "modelo_yolo_existe": config.modelos.yolo.exists(),
            "modelo_cnn_existe": config.modelos.cnn.exists(),
            "clases": config.clases.classification,
        }

    @app.post(
        "/inspect/image",
        tags=["Inspección"],
        summary="Inspeccionar una imagen",
        description=(
            "Recibe una imagen, detecta los extintores y clasifica cada uno como "
            "visible, parcialmente tapado o bloqueado."
        ),
    )
    async def inspect_image(
        file: Annotated[
            UploadFile,
            File(description="Imagen que se quiere inspeccionar."),
        ],
        guardar_anotada: Annotated[
            bool,
            Query(description="Guarda una copia de la imagen con las detecciones dibujadas."),
        ] = config.api.guardar_anotada_por_defecto,
        guardar_crops: Annotated[
            bool,
            Query(description="Guarda los recortes usados por la CNN."),
        ] = False,
    ):
        upload_dir = config.outputs.reports / "api_uploads"
        ruta_imagen = await _guardar_upload(file, upload_dir)

        resultado = pipeline.inspeccionar_imagen(
            ruta_imagen,
            guardar_crops=guardar_crops,
            guardar_anotada=guardar_anotada,
        )
        status_code = 200 if resultado.ok else 422
        return JSONResponse(content=resultado.to_dict(), status_code=status_code)

    @app.post(
        "/inspect/images",
        tags=["Inspección"],
        summary="Inspeccionar varias imágenes",
        description=(
            "Recibe varias imágenes en una sola petición. Es el modo recomendado "
            "para ejecutar la demo completa desde Swagger o desde curl."
        ),
    )
    async def inspect_images(
        files: Annotated[
            list[UploadFile],
            File(description="Imágenes que se quieren inspeccionar."),
        ],
        guardar_anotada: Annotated[
            bool,
            Query(description="Guarda una copia anotada por cada imagen procesada."),
        ] = config.api.guardar_anotada_por_defecto,
        guardar_crops: Annotated[
            bool,
            Query(description="Guarda los recortes usados por la CNN."),
        ] = False,
    ):
        if not files:
            raise HTTPException(status_code=400, detail="No se han recibido imágenes.")

        upload_dir = config.outputs.reports / "api_uploads"
        rutas_imagenes = []
        for file in files:
            rutas_imagenes.append(await _guardar_upload(file, upload_dir))

        contenido = ejecutar_lote(rutas_imagenes, guardar_anotada, guardar_crops)
        return respuesta_lote_json(contenido)

    @app.post(
        "/inspect/folder",
        tags=["Inspección"],
        summary="Inspeccionar una carpeta comprimida",
        description=(
            "Recibe un ZIP con una carpeta de imágenes y procesa todas las imágenes "
            "compatibles. Este endpoint permite ejecutar la demo completa desde "
            "Swagger subiendo un solo archivo."
        ),
    )
    async def inspect_folder(
        archivo_zip: Annotated[
            UploadFile,
            File(description="Archivo .zip con la carpeta de imágenes."),
        ],
        guardar_anotada: Annotated[
            bool,
            Query(description="Guarda una copia anotada por cada imagen procesada."),
        ] = config.api.guardar_anotada_por_defecto,
        guardar_crops: Annotated[
            bool,
            Query(description="Guarda los recortes usados por la CNN."),
        ] = False,
    ):
        upload_dir = config.outputs.reports / "api_uploads"
        rutas_imagenes = await _guardar_zip(archivo_zip, upload_dir)
        contenido = ejecutar_lote(rutas_imagenes, guardar_anotada, guardar_crops)
        return respuesta_lote_json(contenido)

    @app.post(
        "/inspect/folder/zip",
        tags=["Inspección"],
        summary="Inspeccionar una carpeta y descargar resultados",
        description=(
            "Recibe un ZIP con una carpeta de imágenes y devuelve otro ZIP con "
            "resultados.json y las imágenes anotadas."
        ),
        response_class=Response,
        responses={
            200: {
                "description": "ZIP con el JSON y las imágenes anotadas.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"},
                    }
                },
            }
        },
    )
    async def inspect_folder_zip(
        archivo_zip: Annotated[
            UploadFile,
            File(description="Archivo .zip con la carpeta de imágenes."),
        ],
        guardar_crops: Annotated[
            bool,
            Query(description="Guarda los recortes usados por la CNN."),
        ] = False,
    ):
        upload_dir = config.outputs.reports / "api_uploads"
        rutas_imagenes = await _guardar_zip(archivo_zip, upload_dir)
        contenido = ejecutar_lote(
            rutas_imagenes,
            guardar_anotada=True,
            guardar_crops=guardar_crops,
        )
        return respuesta_lote_zip(contenido, "resultados_inspeccion.zip")

    return app


try:
    app = crear_app()
except RuntimeError as exc:
    app = AplicacionNoDisponible(
        title="Inspección visual de extintores",
        detalle=f"API no disponible hasta instalar dependencias: {exc}",
    )
