#!/usr/bin/env python3
"""Genera crops de extintores detectados con YOLO para preparar el dataset CNN."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.detection.yolo_detector import YoloExtinguisherDetector
from fire_extinguisher_inspection.preprocessing.crop_utils import guardar_crop, recortar_con_margen


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera crops a partir de detecciones YOLO.")
    parser.add_argument("--input-dir", required=True, help="Directorio con imágenes completas.")
    parser.add_argument("--output-dir", required=True, help="Directorio donde guardar crops.")
    parser.add_argument("--config", default="config/default.yaml", help="Ruta a la configuración YAML.")
    parser.add_argument("--classes", default="config/classes.yaml", help="Ruta a classes.yaml.")
    parser.add_argument("--metadata", default=None, help="CSV opcional con metadatos de los crops.")
    return parser


def iterar_imagenes(input_dir: Path):
    for ruta in sorted(input_dir.rglob("*")):
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN:
            yield ruta


def main() -> int:
    args = crear_parser().parse_args()
    try:
        import cv2
    except ImportError:
        print("ERROR: No se puede importar OpenCV. Instala requirements.txt.", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: No existe el directorio de entrada: {input_dir}", file=sys.stderr)
        return 2

    config = cargar_configuracion(args.config, args.classes)
    try:
        detector = YoloExtinguisherDetector(
            config.modelos.yolo,
            confidence_threshold=config.inferencia.detection_confidence_threshold,
            class_names=config.clases.detection,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata) if args.metadata else output_dir / "crops_metadata.csv"

    total_crops = 0
    with metadata_path.open("w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.DictWriter(
            archivo_csv,
            fieldnames=["crop_path", "source_image", "bbox", "detection_confidence", "class_name"],
        )
        writer.writeheader()

        for image_path in iterar_imagenes(input_dir):
            imagen = cv2.imread(str(image_path))
            if imagen is None:
                print(f"AVISO: no se pudo leer {image_path}")
                continue

            for indice, deteccion in enumerate(detector.inferir_imagen(image_path)):
                crop = recortar_con_margen(imagen, deteccion.bbox, margen=config.inferencia.crop_margin)
                crop_path = output_dir / f"{image_path.stem}_det_{indice:03d}.jpg"
                guardar_crop(crop, crop_path)
                writer.writerow(
                    {
                        "crop_path": str(crop_path),
                        "source_image": str(image_path),
                        "bbox": deteccion.bbox,
                        "detection_confidence": f"{deteccion.confidence:.6f}",
                        "class_name": deteccion.class_name,
                    }
                )
                total_crops += 1

    print(f"Crops generados: {total_crops}. Metadatos: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
