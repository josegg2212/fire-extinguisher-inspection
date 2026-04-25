"""Tests mínimos de importación."""

from __future__ import annotations

import unittest


class TestImportaciones(unittest.TestCase):
    """Comprueba que los módulos principales importan."""

    def test_importaciones_principales(self) -> None:
        import fire_extinguisher_inspection
        from fire_extinguisher_inspection.config import cargar_configuracion
        from fire_extinguisher_inspection.pipeline.result_schema import InspectionResult

        self.assertTrue(fire_extinguisher_inspection.__version__)
        self.assertTrue(callable(cargar_configuracion))
        self.assertEqual(InspectionResult(image_path="x").to_dict()["image_path"], "x")
