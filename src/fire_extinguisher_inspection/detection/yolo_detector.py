"""Detector YOLO para localizar extintores en imágenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeteccionYolo:
    """Resultado normalizado de una detección YOLO."""

    bbox: list[int]
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la detección como diccionario serializable."""

        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class YoloExtinguisherDetector:
    """Carga un modelo YOLO y devuelve detecciones de extintores."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        class_names: dict[int, str] | None = None,
        image_size: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.class_names = class_names or {0: "fire_extinguisher"}
        self.image_size = int(image_size) if image_size is not None else None

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo YOLO en {self.model_path}. "
                "Entrena el detector o ajusta 'modelos.yolo' en config/default.yaml."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "No se puede importar ultralytics. Instala las dependencias con "
                "'pip install -r requirements.txt'."
            ) from exc

        self.model = YOLO(str(self.model_path))

    def inferir_imagen(self, image_path: str | Path) -> list[DeteccionYolo]:
        """Ejecuta YOLO sobre una imagen y devuelve detecciones normalizadas."""

        ruta_imagen = Path(image_path)
        if not ruta_imagen.exists():
            raise FileNotFoundError(f"No existe la imagen de entrada: {ruta_imagen}")

        parametros: dict[str, Any] = {
            "source": str(ruta_imagen),
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.image_size is not None:
            parametros["imgsz"] = self.image_size

        resultados = self.model.predict(**parametros)
        if not resultados:
            return []

        return self._convertir_resultado(resultados[0])

    def obtener_bounding_boxes(self, image_path: str | Path) -> list[dict[str, Any]]:
        """Devuelve bounding boxes, confidencias y clases como diccionarios."""

        return [deteccion.to_dict() for deteccion in self.inferir_imagen(image_path)]

    def _convertir_resultado(self, resultado: Any) -> list[DeteccionYolo]:
        boxes = getattr(resultado, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confidencias = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else []
        clases = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else []

        detecciones: list[DeteccionYolo] = []
        for indice, bbox in enumerate(xyxy):
            class_id = int(clases[indice]) if indice < len(clases) else 0
            confidence = float(confidencias[indice]) if indice < len(confidencias) else 0.0
            detecciones.append(
                DeteccionYolo(
                    bbox=[int(round(valor)) for valor in bbox],
                    confidence=confidence,
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                )
            )
        return detecciones
