"""Utilidades para recortar regiones de interés."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def recortar_con_margen(imagen: Any, bbox: list[int], margen: float = 0.05) -> Any:
    """Recorta una bounding box ampliada con un margen porcentual."""

    if imagen is None:
        raise ValueError("No se puede recortar una imagen vacía.")
    if len(bbox) != 4:
        raise ValueError(f"La bbox debe tener cuatro valores [x1, y1, x2, y2], recibido: {bbox}")

    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = [int(valor) for valor in bbox]
    caja_ancho = max(x2 - x1, 1)
    caja_alto = max(y2 - y1, 1)
    dx = int(caja_ancho * margen)
    dy = int(caja_alto * margen)

    x1m = max(0, x1 - dx)
    y1m = max(0, y1 - dy)
    x2m = min(ancho, x2 + dx)
    y2m = min(alto, y2 + dy)

    if x2m <= x1m or y2m <= y1m:
        raise ValueError(f"La bbox no genera un recorte válido: {bbox}")

    return imagen[y1m:y2m, x1m:x2m]


def guardar_crop(crop: Any, output_path: str | Path) -> Path:
    """Guarda un recorte en disco usando OpenCV."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("No se puede importar OpenCV. Instala requirements.txt.") from exc

    ruta_salida = Path(output_path)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    if crop is None or getattr(crop, "size", 0) == 0:
        raise ValueError(f"El crop está vacío y no se puede guardar en {ruta_salida}")

    ok = cv2.imwrite(str(ruta_salida), crop)
    if not ok:
        raise OSError(f"No se pudo guardar el crop en {ruta_salida}")
    return ruta_salida
