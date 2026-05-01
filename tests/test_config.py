"""Tests de configuración."""

from __future__ import annotations

import unittest

from fire_extinguisher_inspection.config import cargar_configuracion


class TestConfiguracion(unittest.TestCase):
    """Comprueba la configuración por defecto."""

    def test_carga_configuracion_por_defecto(self) -> None:
        config = cargar_configuracion()
        self.assertEqual(config.modelos.yolo.name, "best.pt")
        self.assertTrue(config.modelos.yolo.as_posix().endswith("models/yolo/extinguisher_yolo_v1/weights/best.pt"))
        self.assertEqual(config.modelos.yolo_base, "models/yolo/base/yolo26n.pt")
        self.assertEqual(config.modelos.cnn.name, "state_classifier_context_v7_real_tuned_with_manual_tests.pt")
        self.assertIsNone(config.modelos.visibility_verifier)
        self.assertEqual(config.inferencia.detection_confidence_threshold, 0.05)
        self.assertEqual(config.inferencia.yolo_imgsz, 1920)
        self.assertEqual(config.inferencia.blocked_to_partial_margin, 0.20)
        self.assertEqual(config.inferencia.blocked_to_partial_min_probability, 0.35)
        self.assertEqual(config.inferencia.visibility_verifier_min_visible_probability, 0.55)
        self.assertEqual(config.inferencia.visibility_verifier_primary_min_visible_probability, 0.20)
        self.assertEqual(config.inferencia.visibility_verifier_primary_max_blocked_probability, 0.30)
        self.assertEqual(config.clases.classification, ["visible", "partially_occluded", "blocked"])
        self.assertEqual(config.clases.detection[0], "fire_extinguisher")
