"""Tests mínimos de la API."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from fire_extinguisher_inspection.api.main import _contenido_zip_resultados, app


class TestAPI(unittest.TestCase):
    """Comprueba que la API puede importarse en esta fase."""

    def test_app_tiene_titulo(self) -> None:
        self.assertEqual(app.title, "Inspección visual de extintores")

    def test_endpoints_principales_existen(self) -> None:
        rutas = {ruta.path for ruta in app.routes}
        self.assertIn("/health", rutas)
        self.assertIn("/inspect/image", rutas)
        self.assertIn("/inspect/images", rutas)
        self.assertIn("/inspect/folder", rutas)
        self.assertIn("/inspect/folder/zip", rutas)

    def test_swagger_esta_disponible(self) -> None:
        rutas = {ruta.path for ruta in app.routes}
        self.assertEqual(app.docs_url, "/docs")
        self.assertIn("/docs", rutas)
        self.assertIn("/openapi.json", rutas)

    def test_zip_de_resultados_incluye_json_e_imagenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta_imagen = Path(tmpdir) / "imagen_anotada.jpg"
            ruta_imagen.write_bytes(b"imagen de prueba")

            contenido = {
                "total": 1,
                "ok": True,
                "imagenes_con_error": 0,
                "resultados": [
                    {
                        "annotated_image_path": str(ruta_imagen),
                        "errors": [],
                    }
                ],
            }

            zip_bytes = _contenido_zip_resultados(contenido)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archivo:
            nombres = archivo.namelist()
            self.assertIn("resultados.json", nombres)
            self.assertTrue(any(nombre.startswith("annotated/") for nombre in nombres))
            datos = json.loads(archivo.read("resultados.json").decode("utf-8"))
            self.assertEqual(datos["total"], 1)

    def test_health_si_fastapi_disponible(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI no está instalado en este entorno.")

        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
