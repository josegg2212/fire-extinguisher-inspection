"""Entrenamiento del detector YOLO de extintores con Ultralytics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def crear_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de línea de comandos."""

    parser = argparse.ArgumentParser(description="Entrena un detector YOLO para extintores.")
    parser.add_argument("--data", required=True, help="Ruta al data.yaml del dataset YOLO.")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Modelo base de Ultralytics. Recomendado: yolo26n.pt; fallback compatible: yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen para entrenamiento.")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño de batch.")
    parser.add_argument("--workers", type=int, default=0, help="Workers del dataloader.")
    parser.add_argument("--project", default="models/yolo", help="Directorio de salida del experimento.")
    parser.add_argument("--name", default="extinguisher_yolo_v1", help="Nombre del experimento.")
    parser.add_argument("--device", default=None, help="Dispositivo opcional, por ejemplo 'cpu' o '0'.")
    return parser


def entrenar(args: argparse.Namespace) -> int:
    """Valida argumentos y lanza el entrenamiento YOLO."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ruta_data = Path(args.data)
    if not ruta_data.exists():
        logging.error("No existe el archivo data.yaml: %s", ruta_data)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        logging.error("No se puede importar ultralytics. Instala requirements.txt antes de entrenar.")
        return 2

    logging.info("Cargando modelo base: %s", args.model)
    modelo = YOLO(args.model)

    parametros = {
        "data": str(ruta_data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        parametros["device"] = args.device

    logging.info("Iniciando entrenamiento YOLO con data=%s", ruta_data)
    resultados = modelo.train(**parametros)
    save_dir = getattr(resultados, "save_dir", None)
    logging.info("Entrenamiento finalizado. Salida: %s", save_dir or args.project)
    return 0


def main() -> int:
    """Punto de entrada del script."""

    return entrenar(crear_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
