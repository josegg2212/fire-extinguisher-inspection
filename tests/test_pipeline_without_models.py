"""Tests del pipeline sin modelos disponibles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline


class TestPipelineSinModelos(unittest.TestCase):
    """Comprueba errores controlados cuando faltan entradas o modelos."""

    def test_pipeline_imagen_inexistente_error_controlado(self) -> None:
        pipeline = InspectionPipeline(cargar_configuracion())
        resultado = pipeline.inspeccionar_imagen(Path("imagen_que_no_existe.jpg"))
        self.assertTrue(resultado.errors)
        self.assertIn("No existe la imagen", resultado.errors[0])

    def test_pipeline_sin_modelo_yolo_error_controlado(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            imagen = Path(tmpdir) / "dummy.jpg"
            imagen.write_bytes(b"no es una imagen real")

            pipeline = InspectionPipeline(cargar_configuracion())
            resultado = pipeline.inspeccionar_imagen(imagen)

        self.assertTrue(resultado.errors)
        self.assertTrue(
            "No existe el modelo YOLO" in resultado.errors[0]
            or "OpenCV" in resultado.errors[0]
            or "No se pudo leer la imagen" in resultado.errors[0]
        )
