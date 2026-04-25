"""Dibujo de bounding boxes, estados y confidencias."""

from __future__ import annotations

from typing import Any

from fire_extinguisher_inspection.pipeline.result_schema import DetectionResult


COLORES_ESTADO = {
    "visible": (45, 180, 75),
    "partially_occluded": (0, 165, 255),
    "blocked": (45, 45, 220),
    "sin_estado": (180, 180, 180),
}


def _texto_etiqueta(deteccion: DetectionResult) -> str:
    etiqueta = f"{deteccion.class_name} {deteccion.detection_confidence:.2f}"
    if deteccion.status_prediction is not None and deteccion.status_confidence is not None:
        etiqueta += f" | {deteccion.status_prediction} {deteccion.status_confidence:.2f}"
    return etiqueta


def dibujar_resultados(imagen: Any, detecciones: list[DetectionResult]) -> Any:
    """Dibuja detecciones sobre una imagen BGR de OpenCV."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("No se puede importar OpenCV. Instala requirements.txt.") from exc

    for deteccion in detecciones:
        x1, y1, x2, y2 = [int(valor) for valor in deteccion.bbox]
        estado = deteccion.status_prediction or "sin_estado"
        color = COLORES_ESTADO.get(estado, COLORES_ESTADO["sin_estado"])
        etiqueta = _texto_etiqueta(deteccion)

        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)

        (texto_ancho, texto_alto), baseline = cv2.getTextSize(
            etiqueta,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        y_texto = max(y1 - 8, texto_alto + 8)
        cv2.rectangle(
            imagen,
            (x1, y_texto - texto_alto - baseline - 4),
            (x1 + texto_ancho + 6, y_texto + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            imagen,
            etiqueta,
            (x1 + 3, y_texto - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return imagen
