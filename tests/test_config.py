"""Tests de configuración."""

from __future__ import annotations

import unittest

from fire_extinguisher_inspection.config import cargar_configuracion


class TestConfiguracion(unittest.TestCase):
    """Comprueba la configuración por defecto."""

    def test_carga_configuracion_por_defecto(self) -> None:
        config = cargar_configuracion()
        self.assertEqual(config.modelos.yolo.name, "extinguisher_yolo.pt")
        self.assertEqual(config.modelos.yolo_base, "yolo26n.pt")
        self.assertEqual(config.clases.classification, ["visible", "partially_occluded", "blocked"])
        self.assertEqual(config.clases.detection[0], "fire_extinguisher")
