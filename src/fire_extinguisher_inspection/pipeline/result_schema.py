"""Esquema de resultados de inferencia."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """Resultado de una detección individual."""

    bbox: list[int]
    detection_confidence: float
    class_name: str
    class_id: int | None = None
    crop_path: str | None = None
    status_prediction: str | None = None
    status_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado a diccionario serializable."""

        return asdict(self)


@dataclass
class InspectionResult:
    """Resultado completo de una inspección sobre una imagen."""

    image_path: str
    detections: list[DetectionResult] = field(default_factory=list)
    annotated_image_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Indica si la inspección terminó sin errores."""

        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado completo a diccionario serializable."""

        return {
            "image_path": self.image_path,
            "detections": [deteccion.to_dict() for deteccion in self.detections],
            "annotated_image_path": self.annotated_image_path,
            "warnings": self.warnings,
            "errors": self.errors,
        }
