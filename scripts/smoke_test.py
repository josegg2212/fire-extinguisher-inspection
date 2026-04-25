#!/usr/bin/env python3
"""Prueba rápida para comprobar que el proyecto no nace roto."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.dataset import validar_dataset_clasificador
from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline


def main() -> int:
    config = cargar_configuracion()
    print(f"Configuración cargada desde: {config.raiz_proyecto}")
    print(f"Clases de estado: {', '.join(config.clases.classification)}")

    validacion = validar_dataset_clasificador(config.datasets.classifier_root)
    if validacion.errores:
        print("AVISO: estructura del dataset CNN incompleta.")
        validacion.imprimir()
    else:
        print("Estructura del dataset CNN presente.")

    pipeline = InspectionPipeline(config)
    resultado = pipeline.inspeccionar_imagen(config.raiz_proyecto / "no_existe.jpg")
    if not resultado.errors:
        print("ERROR: el pipeline no informó error con una imagen inexistente.")
        return 2

    print("Pipeline sin imagen real devuelve error controlado.")
    print("Smoke test completado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
