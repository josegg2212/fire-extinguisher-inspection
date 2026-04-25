#!/usr/bin/env python3
"""Ejecuta el pipeline de inspección sobre una imagen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia sobre una imagen.")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
    parser.add_argument("--config", default="config/default.yaml", help="Ruta a la configuración YAML.")
    parser.add_argument("--classes", default="config/classes.yaml", help="Ruta a classes.yaml.")
    parser.add_argument("--save-crops", action="store_true", help="Guarda los recortes detectados.")
    parser.add_argument("--no-save-annotated", action="store_true", help="No guarda la imagen anotada.")
    parser.add_argument("--json-output", default=None, help="Ruta opcional para guardar el JSON de resultado.")
    return parser


def main() -> int:
    args = crear_parser().parse_args()
    config = cargar_configuracion(args.config, args.classes)
    pipeline = InspectionPipeline(config)
    resultado = pipeline.inspeccionar_imagen(
        args.image,
        guardar_crops=args.save_crops,
        guardar_anotada=not args.no_save_annotated,
    )
    salida = resultado.to_dict()
    print(json.dumps(salida, indent=2, ensure_ascii=False))

    if args.json_output:
        ruta_salida = Path(args.json_output)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if resultado.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
