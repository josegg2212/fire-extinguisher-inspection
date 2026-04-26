#!/usr/bin/env python3
"""Genera contact sheets del dataset CNN de estado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO, EXTENSIONES_IMAGEN


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualiza muestras del dataset classifier.")
    parser.add_argument("--dataset-dir", default="data/classifier", help="Raiz del dataset classifier.")
    parser.add_argument("--output", required=True, help="Ruta de salida de la contact sheet.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split a visualizar.")
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=8,
        help="Numero de muestras por clase.",
    )
    parser.add_argument("--tile-size", type=int, default=224, help="Tamano de cada imagen en la hoja.")
    return parser


def _asegurar_pillow() -> tuple[object, object, object, object]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc
    return Image, ImageDraw, ImageFont, ImageOps


def _recoger_muestras(dataset_dir: Path, split: str, num_samples_per_class: int) -> dict[str, list[Path]]:
    muestras: dict[str, list[Path]] = {}
    for clase in CLASES_ESTADO:
        ruta_clase = dataset_dir / split / clase
        if not ruta_clase.exists():
            raise RuntimeError(f"No existe la carpeta de clase: {ruta_clase}")
        imagenes = sorted(
            ruta
            for ruta in ruta_clase.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
        )
        muestras[clase] = imagenes[:num_samples_per_class]
    return muestras


def _crear_tile(path: Path | None, clase: str, tile_size: int) -> object:
    Image, ImageDraw, ImageFont, ImageOps = _asegurar_pillow()
    alto_cabecera = 32
    lienzo = Image.new("RGB", (tile_size, tile_size + alto_cabecera), (241, 245, 249))
    draw = ImageDraw.Draw(lienzo)
    draw.rectangle((0, 0, tile_size, alto_cabecera), fill=(31, 41, 55))
    etiqueta = clase if path is None else f"{clase} / {path.name[:28]}"
    draw.text((8, 9), etiqueta, fill=(255, 255, 255), font=ImageFont.load_default())

    if path is None:
        draw.text((8, alto_cabecera + 12), "sin muestras", fill=(127, 29, 29), font=ImageFont.load_default())
        return lienzo

    try:
        with Image.open(path) as imagen:
            tile = ImageOps.fit(imagen.convert("RGB"), (tile_size, tile_size))
    except Exception:
        tile = Image.new("RGB", (tile_size, tile_size), (254, 226, 226))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((8, 8), "no legible", fill=(127, 29, 29), font=ImageFont.load_default())
    lienzo.paste(tile, (0, alto_cabecera))
    return lienzo


def generar_contact_sheet(muestras: dict[str, list[Path]], output: Path, tile_size: int, num_samples_per_class: int) -> None:
    Image, _, _, _ = _asegurar_pillow()
    columnas = max(1, num_samples_per_class)
    filas = len(CLASES_ESTADO)
    alto_cabecera = 32
    hoja = Image.new("RGB", (columnas * tile_size, filas * (tile_size + alto_cabecera)), (226, 232, 240))

    for fila, clase in enumerate(CLASES_ESTADO):
        muestras_clase = muestras.get(clase, [])
        for columna in range(columnas):
            path = muestras_clase[columna] if columna < len(muestras_clase) else None
            tile = _crear_tile(path, clase, tile_size)
            hoja.paste(tile, (columna * tile_size, fila * (tile_size + alto_cabecera)))

    output.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(output, quality=92)


def main() -> int:
    args = crear_parser().parse_args()
    if args.num_samples_per_class <= 0:
        print("ERROR: --num-samples-per-class debe ser mayor que 0.")
        return 2
    if args.tile_size <= 0:
        print("ERROR: --tile-size debe ser mayor que 0.")
        return 2

    try:
        dataset_dir = Path(args.dataset_dir)
        muestras = _recoger_muestras(dataset_dir, args.split, args.num_samples_per_class)
        if not any(muestras.values()):
            print(f"ERROR: no hay muestras en el split {args.split}.")
            return 2
        generar_contact_sheet(muestras, Path(args.output), args.tile_size, args.num_samples_per_class)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    total = sum(len(paths) for paths in muestras.values())
    print(f"Contact sheet generada en: {args.output}")
    print(f"Muestras incluidas: {total} | split: {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
