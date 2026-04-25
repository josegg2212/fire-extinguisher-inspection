"""Pipeline completo de inspección: YOLO, recorte, CNN y visualización."""

from __future__ import annotations

import logging
from pathlib import Path

from fire_extinguisher_inspection.classification.predict_classifier import CNNStatePredictor
from fire_extinguisher_inspection.config import Configuracion, cargar_configuracion
from fire_extinguisher_inspection.detection.yolo_detector import YoloExtinguisherDetector
from fire_extinguisher_inspection.pipeline.result_schema import DetectionResult, InspectionResult
from fire_extinguisher_inspection.preprocessing.crop_utils import guardar_crop, recortar_con_margen
from fire_extinguisher_inspection.visualization.draw_results import dibujar_resultados


logger = logging.getLogger(__name__)


class InspectionPipeline:
    """Orquesta la inferencia completa sobre una imagen."""

    def __init__(self, config: Configuracion | None = None) -> None:
        self.config = config or cargar_configuracion()
        self._clasificador: CNNStatePredictor | None = None

    def inspeccionar_imagen(
        self,
        image_path: str | Path,
        guardar_crops: bool | None = None,
        guardar_anotada: bool | None = None,
    ) -> InspectionResult:
        """Ejecuta el pipeline sobre una imagen y devuelve un resultado estructurado."""

        ruta_imagen = Path(image_path)
        resultado = InspectionResult(image_path=str(ruta_imagen))

        if not ruta_imagen.exists():
            resultado.errors.append(f"No existe la imagen de entrada: {ruta_imagen}")
            return resultado

        guardar_crops = self.config.inferencia.guardar_crops if guardar_crops is None else guardar_crops
        guardar_anotada = self.config.inferencia.guardar_anotada if guardar_anotada is None else guardar_anotada

        imagen = self._leer_imagen_si_hace_falta(resultado)
        if imagen is None:
            return resultado

        try:
            detector = YoloExtinguisherDetector(
                model_path=self.config.modelos.yolo,
                confidence_threshold=self.config.inferencia.detection_confidence_threshold,
                class_names=self.config.clases.detection,
            )
        except Exception as exc:
            resultado.errors.append(str(exc))
            return resultado

        try:
            detecciones_yolo = detector.inferir_imagen(ruta_imagen)
        except Exception as exc:
            resultado.errors.append(f"Error durante la inferencia YOLO: {exc}")
            return resultado

        if not detecciones_yolo:
            resultado.warnings.append("No se detectaron extintores en la imagen.")
            return resultado

        clasificador = self._cargar_clasificador(resultado)

        for indice, deteccion_yolo in enumerate(detecciones_yolo):
            deteccion = DetectionResult(
                bbox=deteccion_yolo.bbox,
                detection_confidence=deteccion_yolo.confidence,
                class_id=deteccion_yolo.class_id,
                class_name=deteccion_yolo.class_name,
            )

            try:
                crop = recortar_con_margen(
                    imagen,
                    deteccion_yolo.bbox,
                    margen=self.config.inferencia.crop_margin,
                )
            except Exception as exc:
                resultado.warnings.append(f"No se pudo recortar la detección {indice}: {exc}")
                resultado.detections.append(deteccion)
                continue

            if guardar_crops:
                crop_path = self._ruta_crop(ruta_imagen, indice)
                try:
                    deteccion.crop_path = str(guardar_crop(crop, crop_path))
                except Exception as exc:
                    resultado.warnings.append(f"No se pudo guardar el crop {indice}: {exc}")

            if clasificador is not None:
                try:
                    clase, confianza, _ = clasificador.predecir_array_bgr(crop)
                    deteccion.status_prediction = clase
                    deteccion.status_confidence = confianza
                    if confianza < self.config.inferencia.classification_confidence_threshold:
                        resultado.warnings.append(
                            f"La clasificación de la detección {indice} está por debajo del umbral: "
                            f"{confianza:.3f}"
                        )
                except Exception as exc:
                    resultado.warnings.append(f"No se pudo clasificar la detección {indice}: {exc}")

            resultado.detections.append(deteccion)

        if guardar_anotada:
            try:
                imagen_anotada = dibujar_resultados(imagen.copy(), resultado.detections)
                resultado.annotated_image_path = str(self._guardar_imagen_anotada(ruta_imagen, imagen_anotada))
            except Exception as exc:
                resultado.warnings.append(f"No se pudo guardar la imagen anotada: {exc}")

        return resultado

    def _leer_imagen_si_hace_falta(self, resultado: InspectionResult):
        try:
            import cv2
        except ImportError:
            resultado.errors.append("No se puede importar OpenCV. Instala requirements.txt.")
            return None

        imagen = cv2.imread(resultado.image_path)
        if imagen is None:
            resultado.errors.append(f"No se pudo leer la imagen con OpenCV: {resultado.image_path}")
            return None
        return imagen

    def _cargar_clasificador(self, resultado: InspectionResult) -> CNNStatePredictor | None:
        if self._clasificador is not None:
            return self._clasificador

        if not self.config.modelos.cnn.exists():
            resultado.warnings.append(
                f"No existe el modelo CNN en {self.config.modelos.cnn}. "
                "Se devuelven detecciones sin clasificación de estado."
            )
            return None

        try:
            self._clasificador = CNNStatePredictor(
                model_path=self.config.modelos.cnn,
                class_names=self.config.clases.classification,
                image_size=self.config.inferencia.cnn_image_size,
            )
            return self._clasificador
        except Exception as exc:
            resultado.warnings.append(f"No se pudo cargar el clasificador CNN: {exc}")
            return None

    def _ruta_crop(self, ruta_imagen: Path, indice: int) -> Path:
        self.config.outputs.crops.mkdir(parents=True, exist_ok=True)
        return self.config.outputs.crops / f"{ruta_imagen.stem}_det_{indice:03d}.jpg"

    def _guardar_imagen_anotada(self, ruta_imagen: Path, imagen_anotada) -> Path:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("No se puede importar OpenCV. Instala requirements.txt.") from exc

        self.config.outputs.detections.mkdir(parents=True, exist_ok=True)
        output_path = self.config.outputs.detections / f"{ruta_imagen.stem}_annotated.jpg"
        ok = cv2.imwrite(str(output_path), imagen_anotada)
        if not ok:
            raise OSError(f"No se pudo escribir la imagen anotada en {output_path}")
        return output_path
