#!/usr/bin/env python3
"""Comprueba la estructura esperada de los datasets YOLO y CNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO, validar_dataset_clasificador


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comprueba estructura de datasets.")
    parser.add_argument("--tipo", choices=["yolo", "classifier"], required=True, help="Tipo de dataset.")
    parser.add_argument("--path", required=True, help="Ruta al data.yaml de YOLO o raíz del dataset CNN.")
    parser.add_argument("--require-images", action="store_true", help="Exige que haya imágenes.")
    return parser


def comprobar_yolo(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: no existe el YAML de YOLO: {path}")
        return 2

    try:
        import yaml
    except ImportError:
        print("ERROR: No se puede importar PyYAML. Instala requirements.txt.")
        return 2

    datos = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errores: list[str] = []
    for clave in ["train", "val", "names"]:
        if clave not in datos:
            errores.append(f"Falta la clave '{clave}' en {path}")

    names = datos.get("names", {})
    if isinstance(names, list):
        nombres = names
    elif isinstance(names, dict):
        nombres = list(names.values())
    else:
        nombres = []

    if "fire_extinguisher" not in [str(nombre) for nombre in nombres]:
        errores.append("El YAML debe incluir la clase 'fire_extinguisher'.")

    if errores:
        for error in errores:
            print(f"ERROR: {error}")
        return 2

    print(f"Dataset YOLO válido a nivel de configuración: {path}")
    return 0


def main() -> int:
    args = crear_parser().parse_args()
    ruta = Path(args.path)
    if args.tipo == "yolo":
        return comprobar_yolo(ruta)

    resultado = validar_dataset_clasificador(
        ruta,
        clases=CLASES_ESTADO,
        requerir_imagenes=args.require_images,
    )
    resultado.imprimir()
    return 0 if resultado.es_valido else 2


if __name__ == "__main__":
    raise SystemExit(main())
