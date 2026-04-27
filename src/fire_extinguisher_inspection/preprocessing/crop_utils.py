"""Utilidades para recortar regiones de interés."""

from __future__ import annotations

from pathlib import Path
from typing import Any


Region = tuple[int, int, int, int]


def _validar_bbox(bbox: list[int] | tuple[int, int, int, int]) -> Region:
    if len(bbox) != 4:
        raise ValueError(f"La bbox debe tener cuatro valores [x1, y1, x2, y2], recibido: {bbox}")

    x1, y1, x2, y2 = [int(valor) for valor in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"La bbox no tiene area positiva: {bbox}")
    return x1, y1, x2, y2


def calcular_region_contextual(
    ancho_imagen: int,
    alto_imagen: int,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    context_padding: float = 0.75,
    square: bool = False,
) -> Region:
    """Calcula una region ampliada alrededor de una bbox y limitada a la imagen.

    `context_padding` se expresa como fraccion del ancho/alto de la bbox. Un valor
    de 0.75 deja bastante entorno alrededor del extintor sin convertirlo en un
    punto minusculo dentro del crop.
    """

    if ancho_imagen <= 0 or alto_imagen <= 0:
        raise ValueError("La imagen debe tener ancho y alto positivos.")
    if context_padding < 0:
        raise ValueError("--context-padding no puede ser negativo.")

    x1, y1, x2, y2 = _validar_bbox(bbox)
    caja_ancho = max(x2 - x1, 1)
    caja_alto = max(y2 - y1, 1)

    margen_x = caja_ancho * context_padding
    margen_y = caja_alto * context_padding
    x1c = max(0, int(round(x1 - margen_x)))
    y1c = max(0, int(round(y1 - margen_y)))
    x2c = min(ancho_imagen, int(round(x2 + margen_x)))
    y2c = min(alto_imagen, int(round(y2 + margen_y)))

    if x2c <= x1c or y2c <= y1c:
        raise ValueError(f"La bbox no genera una region contextual valida: {bbox}")

    if not square:
        return x1c, y1c, x2c, y2c

    region_ancho = x2c - x1c
    region_alto = y2c - y1c
    lado = max(region_ancho, region_alto)

    # Si la imagen no permite un cuadrado que contenga toda la region ampliada,
    # se conserva la region rectangular para no cortar el extintor.
    if lado > ancho_imagen or lado > alto_imagen:
        return x1c, y1c, x2c, y2c

    centro_x = (x1c + x2c) / 2
    centro_y = (y1c + y2c) / 2
    x1s = int(round(centro_x - lado / 2))
    y1s = int(round(centro_y - lado / 2))
    x2s = x1s + lado
    y2s = y1s + lado

    if x1s < 0:
        x2s -= x1s
        x1s = 0
    if y1s < 0:
        y2s -= y1s
        y1s = 0
    if x2s > ancho_imagen:
        desplazamiento = x2s - ancho_imagen
        x1s -= desplazamiento
        x2s = ancho_imagen
    if y2s > alto_imagen:
        desplazamiento = y2s - alto_imagen
        y1s -= desplazamiento
        y2s = alto_imagen

    x1s = max(0, x1s)
    y1s = max(0, y1s)
    x2s = min(ancho_imagen, x2s)
    y2s = min(alto_imagen, y2s)
    if x2s <= x1s or y2s <= y1s:
        raise ValueError(f"La bbox no genera un crop cuadrado valido: {bbox}")
    return x1s, y1s, x2s, y2s


def recortar_contextual(
    imagen: Any,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    context_padding: float = 0.75,
    square: bool = False,
) -> Any:
    """Recorta una region ampliada para que la CNN vea extintor y entorno."""

    if imagen is None:
        raise ValueError("No se puede recortar una imagen vacía.")

    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = calcular_region_contextual(
        ancho,
        alto,
        bbox,
        context_padding=context_padding,
        square=square,
    )
    return imagen[y1:y2, x1:x2]


def recortar_con_margen(imagen: Any, bbox: list[int], margen: float = 0.05) -> Any:
    """Recorta una bounding box ampliada con un margen porcentual."""

    if imagen is None:
        raise ValueError("No se puede recortar una imagen vacía.")

    alto, ancho = imagen.shape[:2]
    x1m, y1m, x2m, y2m = calcular_region_contextual(
        ancho,
        alto,
        bbox,
        context_padding=margen,
        square=False,
    )
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
