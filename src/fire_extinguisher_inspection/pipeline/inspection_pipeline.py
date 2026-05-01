"""Pipeline completo de inspección: YOLO, recorte, CNN y visualización."""

from __future__ import annotations

import logging
from pathlib import Path

from fire_extinguisher_inspection.classification.predict_classifier import CNNStatePredictor
from fire_extinguisher_inspection.config import Configuracion, cargar_configuracion
from fire_extinguisher_inspection.detection.yolo_detector import YoloExtinguisherDetector
from fire_extinguisher_inspection.pipeline.result_schema import DetectionResult, InspectionResult
from fire_extinguisher_inspection.preprocessing.crop_utils import (
    calcular_region_contextual,
    guardar_crop,
)
from fire_extinguisher_inspection.visualization.draw_results import dibujar_resultados


logger = logging.getLogger(__name__)


class InspectionPipeline:
    """Orquesta la inferencia completa sobre una imagen."""

    def __init__(self, config: Configuracion | None = None) -> None:
        self.config = config or cargar_configuracion()
        self._detector: YoloExtinguisherDetector | None = None
        self._clasificador: CNNStatePredictor | None = None
        self._verificador_visibilidad: CNNStatePredictor | None = None

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

        detector = self._cargar_detector(resultado)
        if detector is None:
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
        verificador_visibilidad = (
            self._cargar_verificador_visibilidad(resultado) if clasificador is not None else None
        )

        for indice, deteccion_yolo in enumerate(detecciones_yolo):
            deteccion = DetectionResult(
                bbox=deteccion_yolo.bbox,
                detection_confidence=deteccion_yolo.confidence,
                class_id=deteccion_yolo.class_id,
                class_name=deteccion_yolo.class_name,
            )

            try:
                x1c, y1c, x2c, y2c = calcular_region_contextual(
                    imagen.shape[1],
                    imagen.shape[0],
                    deteccion_yolo.bbox,
                    context_padding=self.config.inferencia.classifier_context_padding,
                    square=self.config.inferencia.classifier_square_crop,
                )
                crop = imagen[y1c:y2c, x1c:x2c]
                deteccion.classifier_crop_bbox = [x1c, y1c, x2c, y2c]
            except Exception as exc:
                resultado.warnings.append(f"No se pudo generar el crop contextual de la detección {indice}: {exc}")
                resultado.detections.append(deteccion)
                continue

            if guardar_crops:
                crop_path = self._ruta_crop(ruta_imagen, indice)
                try:
                    ruta_crop = str(guardar_crop(crop, crop_path))
                    deteccion.crop_path = ruta_crop
                    deteccion.classifier_crop_path = ruta_crop
                except Exception as exc:
                    resultado.warnings.append(f"No se pudo guardar el crop {indice}: {exc}")

            if clasificador is not None:
                try:
                    clase_raw, confianza_raw, probabilidades_raw = clasificador.predecir_array_bgr(crop)
                    clase, confianza, ajuste = self._calibrar_estado_bloqueo_parcial(
                        clase_raw,
                        confianza_raw,
                        probabilidades_raw,
                    )
                    probabilidades = probabilidades_raw
                    clase, confianza, probabilidades, ajuste = self._aplicar_verificador_visibilidad(
                        crop,
                        clase,
                        confianza,
                        probabilidades,
                        ajuste,
                        verificador_visibilidad,
                    )
                    deteccion.status_prediction = clase
                    deteccion.status_confidence = confianza
                    deteccion.status_probabilities = probabilidades
                    if ajuste is not None:
                        deteccion.raw_status_prediction = clase_raw
                        deteccion.raw_status_confidence = confianza_raw
                        deteccion.raw_status_probabilities = probabilidades_raw
                        deteccion.status_adjustment = ajuste
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

    def _calibrar_estado_bloqueo_parcial(
        self,
        clase: str,
        confianza: float,
        probabilidades: dict[str, float],
    ) -> tuple[str, float, str | None]:
        """Rebaja bloqueos ambiguos a parcialmente ocultos cuando ambas clases estan cerca."""

        if clase != "blocked":
            return clase, confianza, None

        prob_blocked = float(probabilidades.get("blocked", 0.0))
        prob_partial = float(probabilidades.get("partially_occluded", 0.0))
        margen = self.config.inferencia.blocked_to_partial_margin
        minimo_partial = self.config.inferencia.blocked_to_partial_min_probability
        if prob_partial >= minimo_partial and prob_blocked - prob_partial <= margen:
            return "partially_occluded", prob_partial, "blocked_to_partial_calibration"
        return clase, confianza, None

    def _aplicar_verificador_visibilidad(
        self,
        crop,
        clase: str,
        confianza: float,
        probabilidades: dict[str, float],
        ajuste: str | None,
        verificador_visibilidad: CNNStatePredictor | None,
    ) -> tuple[str, float, dict[str, float], str | None]:
        """Promociona partial a visible solo cuando un segundo modelo lo confirma."""

        if verificador_visibilidad is None or clase != "partially_occluded":
            return clase, confianza, probabilidades, ajuste

        prob_visible_primaria = float(probabilidades.get("visible", 0.0))
        prob_blocked_primaria = float(probabilidades.get("blocked", 0.0))
        if prob_visible_primaria < self.config.inferencia.visibility_verifier_primary_min_visible_probability:
            return clase, confianza, probabilidades, ajuste
        if prob_blocked_primaria > self.config.inferencia.visibility_verifier_primary_max_blocked_probability:
            return clase, confianza, probabilidades, ajuste

        clase_verificador, _confianza_verificador, probabilidades_verificador = (
            verificador_visibilidad.predecir_array_bgr(crop)
        )
        prob_visible_verificador = float(probabilidades_verificador.get("visible", 0.0))
        if (
            clase_verificador == "visible"
            and prob_visible_verificador >= self.config.inferencia.visibility_verifier_min_visible_probability
        ):
            return "visible", prob_visible_verificador, probabilidades_verificador, "visibility_verifier"

        return clase, confianza, probabilidades, ajuste

    def _cargar_detector(self, resultado: InspectionResult) -> YoloExtinguisherDetector | None:
        """Carga YOLO una sola vez para procesar lotes de imágenes."""

        if self._detector is not None:
            return self._detector

        try:
            self._detector = YoloExtinguisherDetector(
                model_path=self.config.modelos.yolo,
                confidence_threshold=self.config.inferencia.detection_confidence_threshold,
                class_names=self.config.clases.detection,
                image_size=self.config.inferencia.yolo_imgsz,
            )
            return self._detector
        except Exception as exc:
            resultado.errors.append(str(exc))
            return None

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

    def _cargar_verificador_visibilidad(self, resultado: InspectionResult) -> CNNStatePredictor | None:
        if self._verificador_visibilidad is not None:
            return self._verificador_visibilidad

        ruta_modelo = self.config.modelos.visibility_verifier
        if ruta_modelo is None:
            return None

        if not ruta_modelo.exists():
            resultado.warnings.append(
                f"No existe el verificador de visibles en {ruta_modelo}. "
                "Se usa solo el clasificador principal."
            )
            return None

        try:
            self._verificador_visibilidad = CNNStatePredictor(
                model_path=ruta_modelo,
                class_names=self.config.clases.classification,
                image_size=self.config.inferencia.cnn_image_size,
            )
            return self._verificador_visibilidad
        except Exception as exc:
            resultado.warnings.append(f"No se pudo cargar el verificador de visibles: {exc}")
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
