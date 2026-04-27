#!/usr/bin/env python3
"""Ejecuta el pipeline de inspección sobre una imagen."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
    parser.add_argument("--yolo-model", default=None, help="Ruta opcional al modelo YOLO.")
    parser.add_argument("--classifier-model", default=None, help="Ruta opcional al modelo CNN.")
    parser.add_argument("--output-dir", default=None, help="Directorio base para imágenes anotadas, crops y JSON.")
    parser.add_argument("--save-crops", action="store_true", help="Guarda los recortes detectados.")
    parser.add_argument("--save-json", action="store_true", help="Guarda JSON en --output-dir/json si no se indica --json-output.")
    parser.add_argument("--no-save-annotated", action="store_true", help="No guarda la imagen anotada.")
    parser.add_argument("--json-output", default=None, help="Ruta opcional para guardar el JSON de resultado.")
    parser.add_argument("--image-size", type=int, default=None, help="Tamaño de entrada usado por la CNN.")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Umbral de confianza YOLO.")
    parser.add_argument(
        "--classifier-context-padding",
        type=float,
        default=None,
        help="Padding contextual para el crop que recibe la CNN.",
    )
    parser.add_argument(
        "--classifier-square-crop",
        dest="classifier_square_crop",
        action="store_true",
        default=None,
        help="Activa crop contextual cuadrado para la CNN.",
    )
    parser.add_argument(
        "--no-classifier-square-crop",
        dest="classifier_square_crop",
        action="store_false",
        help="Desactiva crop contextual cuadrado para la CNN.",
    )
    return parser


def main() -> int:
    args = crear_parser().parse_args()
    config = cargar_configuracion(args.config, args.classes)
    cambios_modelos = {}
    cambios_outputs = {}
    cambios_inferencia = {}
    if args.yolo_model is not None:
        cambios_modelos["yolo"] = Path(args.yolo_model)
    if args.classifier_model is not None:
        cambios_modelos["cnn"] = Path(args.classifier_model)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        cambios_outputs["detections"] = output_dir / "annotated"
        cambios_outputs["crops"] = output_dir / "crops"
        cambios_outputs["reports"] = output_dir / "reports"
    if args.image_size is not None:
        if args.image_size <= 0:
            print("ERROR: --image-size debe ser mayor que 0.", file=sys.stderr)
            return 2
        cambios_inferencia["cnn_image_size"] = args.image_size
    if args.confidence_threshold is not None:
        if args.confidence_threshold < 0:
            print("ERROR: --confidence-threshold no puede ser negativo.", file=sys.stderr)
            return 2
        cambios_inferencia["detection_confidence_threshold"] = args.confidence_threshold
    if args.classifier_context_padding is not None:
        if args.classifier_context_padding < 0:
            print("ERROR: --classifier-context-padding no puede ser negativo.", file=sys.stderr)
            return 2
        cambios_inferencia["classifier_context_padding"] = args.classifier_context_padding
    if args.classifier_square_crop is not None:
        cambios_inferencia["classifier_square_crop"] = args.classifier_square_crop
    if cambios_modelos:
        config = replace(config, modelos=replace(config.modelos, **cambios_modelos))
    if cambios_outputs:
        config = replace(config, outputs=replace(config.outputs, **cambios_outputs))
    if cambios_inferencia:
        config = replace(config, inferencia=replace(config.inferencia, **cambios_inferencia))

    pipeline = InspectionPipeline(config)
    resultado = pipeline.inspeccionar_imagen(
        args.image,
        guardar_crops=args.save_crops,
        guardar_anotada=not args.no_save_annotated,
    )
    salida = resultado.to_dict()
    print(json.dumps(salida, indent=2, ensure_ascii=False))

    json_output = args.json_output
    if args.save_json and json_output is None:
        base_output = Path(args.output_dir) if args.output_dir else config.outputs.reports
        json_output = str(base_output / "json" / f"{Path(args.image).stem}.json")

    if json_output:
        ruta_salida = Path(json_output)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if resultado.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
