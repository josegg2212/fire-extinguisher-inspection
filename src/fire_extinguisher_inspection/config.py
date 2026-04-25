"""Carga centralizada de configuración del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RutasConfig:
    """Rutas generales de entrada."""

    imagenes_entrada: Path


@dataclass(frozen=True)
class ModelosConfig:
    """Rutas de modelos y pesos."""

    yolo: Path
    yolo_base: str
    cnn: Path


@dataclass(frozen=True)
class DatasetsConfig:
    """Rutas esperadas para datasets."""

    yolo_root: Path
    yolo_yaml: Path
    classifier_root: Path


@dataclass(frozen=True)
class OutputsConfig:
    """Rutas de salida generadas por el sistema."""

    detections: Path
    crops: Path
    reports: Path
    logs: Path


@dataclass(frozen=True)
class InferenciaConfig:
    """Parámetros de inferencia."""

    detection_confidence_threshold: float
    classification_confidence_threshold: float
    cnn_image_size: int
    crop_margin: float
    guardar_crops: bool
    guardar_anotada: bool


@dataclass(frozen=True)
class ApiConfig:
    """Parámetros de la API."""

    host: str
    port: int
    guardar_anotada_por_defecto: bool


@dataclass(frozen=True)
class ClasesConfig:
    """Nombres de clases usados por detector y clasificador."""

    detection: dict[int, str]
    classification: list[str]


@dataclass(frozen=True)
class Configuracion:
    """Configuración completa del proyecto."""

    rutas: RutasConfig
    modelos: ModelosConfig
    datasets: DatasetsConfig
    outputs: OutputsConfig
    inferencia: InferenciaConfig
    api: ApiConfig
    clases: ClasesConfig
    raiz_proyecto: Path = RAIZ_PROYECTO


def resolver_ruta(ruta: str | Path, base: Path = RAIZ_PROYECTO) -> Path:
    """Convierte una ruta relativa del YAML en una ruta absoluta del repositorio."""

    ruta_path = Path(ruta)
    if ruta_path.is_absolute():
        return ruta_path
    return (base / ruta_path).resolve()


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {ruta}")

    with ruta.open("r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo) or {}

    if not isinstance(datos, dict):
        raise ValueError(f"El archivo YAML debe contener un diccionario: {ruta}")
    return datos


def _convertir_clases_detection(datos: Any) -> dict[int, str]:
    if not isinstance(datos, dict):
        raise ValueError("La clave 'detection' de classes.yaml debe ser un diccionario.")
    return {int(indice): str(nombre) for indice, nombre in datos.items()}


def cargar_configuracion(
    config_path: str | Path | None = None,
    classes_path: str | Path | None = None,
) -> Configuracion:
    """Carga la configuración YAML y devuelve dataclasses con rutas resueltas."""

    ruta_config = resolver_ruta(config_path or "config/default.yaml")
    ruta_clases = resolver_ruta(classes_path or "config/classes.yaml")

    datos = _leer_yaml(ruta_config)
    clases = _leer_yaml(ruta_clases)

    try:
        rutas = datos["rutas"]
        modelos = datos["modelos"]
        datasets = datos["datasets"]
        outputs = datos["outputs"]
        inferencia = datos["inferencia"]
        api = datos["api"]
    except KeyError as exc:
        raise ValueError(f"Falta una sección obligatoria en la configuración: {exc}") from exc

    clases_clasificador = clases.get("classification", [])
    if not isinstance(clases_clasificador, list) or not clases_clasificador:
        raise ValueError("La clave 'classification' de classes.yaml debe ser una lista no vacía.")

    return Configuracion(
        rutas=RutasConfig(imagenes_entrada=resolver_ruta(rutas["imagenes_entrada"])),
        modelos=ModelosConfig(
            yolo=resolver_ruta(modelos["yolo"]),
            yolo_base=str(modelos.get("yolo_base", "yolo26n.pt")),
            cnn=resolver_ruta(modelos["cnn"]),
        ),
        datasets=DatasetsConfig(
            yolo_root=resolver_ruta(datasets["yolo_root"]),
            yolo_yaml=resolver_ruta(datasets["yolo_yaml"]),
            classifier_root=resolver_ruta(datasets["classifier_root"]),
        ),
        outputs=OutputsConfig(
            detections=resolver_ruta(outputs["detections"]),
            crops=resolver_ruta(outputs["crops"]),
            reports=resolver_ruta(outputs["reports"]),
            logs=resolver_ruta(outputs["logs"]),
        ),
        inferencia=InferenciaConfig(
            detection_confidence_threshold=float(inferencia["detection_confidence_threshold"]),
            classification_confidence_threshold=float(inferencia["classification_confidence_threshold"]),
            cnn_image_size=int(inferencia["cnn_image_size"]),
            crop_margin=float(inferencia["crop_margin"]),
            guardar_crops=bool(inferencia["guardar_crops"]),
            guardar_anotada=bool(inferencia["guardar_anotada"]),
        ),
        api=ApiConfig(
            host=str(api["host"]),
            port=int(api["port"]),
            guardar_anotada_por_defecto=bool(api["guardar_anotada_por_defecto"]),
        ),
        clases=ClasesConfig(
            detection=_convertir_clases_detection(clases["detection"]),
            classification=[str(nombre) for nombre in clases_clasificador],
        ),
    )
