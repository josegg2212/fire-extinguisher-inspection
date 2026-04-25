"""Tests mínimos de la API."""

from __future__ import annotations

import unittest

from fire_extinguisher_inspection.api.main import app


class TestAPI(unittest.TestCase):
    """Comprueba que la API puede importarse en esta fase."""

    def test_app_tiene_titulo(self) -> None:
        self.assertEqual(app.title, "Inspección visual de extintores")

    def test_health_si_fastapi_disponible(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI no está instalado en este entorno.")

        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
