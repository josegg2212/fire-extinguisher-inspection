#!/usr/bin/env python3
"""Genera inferencias YOLO anotadas sobre un lote de imágenes."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia YOLO sobre un directorio de imágenes.")
    parser.add_argument("--images-dir", required=True, help="Directorio con imágenes de entrada.")
    parser.add_argument("--model", required=True, help="Ruta al peso YOLO .pt.")
    parser.add_argument("--output-dir", required=True, help="Directorio para imágenes anotadas.")
    parser.add_argument("--max-images", type=int, default=30, help="Número máximo de imágenes a procesar.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Umbral de confianza para mostrar detecciones.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen para inferencia.")
    parser.add_argument("--device", default=None, help="Dispositivo opcional: cpu, 0, cuda:0.")
    parser.add_argument(
        "--contact-sheet-output",
        default=None,
        help="Ruta opcional para guardar una contact sheet con las imágenes anotadas.",
    )
    parser.add_argument("--tile-size", type=int, default=360, help="Tamaño de celda para la contact sheet.")
    return parser


def _resolver(path: str | Path) -> Path:
    ruta = Path(path)
    if ruta.is_absolute():
        return ruta
    return (Path.cwd() / ruta).resolve()


def _recoger_imagenes(images_dir: Path, max_images: int) -> list[Path]:
    if max_images <= 0:
        raise RuntimeError("--max-images debe ser mayor que 0.")
    if not images_dir.exists():
        raise RuntimeError(f"No existe el directorio de imágenes: {images_dir}")

    imagenes = sorted(
        ruta for ruta in images_dir.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    )
    if not imagenes:
        raise RuntimeError(f"No se encontraron imágenes en: {images_dir}")
    return imagenes[:max_images]


def _guardar_imagen_anotada(resultado, output_dir: Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Usa el Docker del proyecto.") from exc

    imagen_anotada = resultado.plot()
    if getattr(imagen_anotada, "ndim", 0) == 3 and imagen_anotada.shape[2] >= 3:
        imagen_anotada = imagen_anotada[..., :3][..., ::-1]

    ruta_origen = Path(str(resultado.path))
    ruta_salida = output_dir / f"{ruta_origen.stem}_yolo.jpg"
    Image.fromarray(imagen_anotada).save(ruta_salida, quality=92)
    return ruta_salida


def _redimensionar_para_tile(imagen, tile_size: int):
    from PIL import Image

    margen = 16
    alto_cabecera = 30
    lienzo = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
    ancho_original, alto_original = imagen.size
    max_ancho = tile_size - margen * 2
    max_alto = tile_size - alto_cabecera - margen * 2
    escala = min(max_ancho / ancho_original, max_alto / alto_original)
    nuevo_tamano = (max(1, int(ancho_original * escala)), max(1, int(alto_original * escala)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    imagen_redimensionada = imagen.resize(nuevo_tamano, resampling)
    offset_x = (tile_size - nuevo_tamano[0]) // 2
    offset_y = alto_cabecera + (max_alto - nuevo_tamano[1]) // 2
    lienzo.paste(imagen_redimensionada, (offset_x, offset_y))
    return lienzo


def generar_contact_sheet(imagenes_anotadas: Iterable[Path], output: Path, tile_size: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Usa el Docker del proyecto.") from exc

    rutas = list(imagenes_anotadas)
    if not rutas:
        raise RuntimeError("No hay imágenes anotadas para generar la contact sheet.")

    columnas = min(5, max(1, math.ceil(math.sqrt(len(rutas)))))
    filas = math.ceil(len(rutas) / columnas)
    contact_sheet = Image.new("RGB", (columnas * tile_size, filas * tile_size), (229, 231, 235))

    for indice, ruta in enumerate(rutas):
        imagen = Image.open(ruta).convert("RGB")
        tile = _redimensionar_para_tile(imagen, tile_size)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_size, 30), fill=(31, 41, 55))
        draw.text((8, 8), ruta.name[:42], fill=(255, 255, 255))
        x = (indice % columnas) * tile_size
        y = (indice // columnas) * tile_size
        contact_sheet.paste(tile, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output, quality=92)


def main() -> int:
    args = crear_parser().parse_args()
    images_dir = _resolver(args.images_dir)
    model_path = _resolver(args.model)
    output_dir = _resolver(args.output_dir)

    if not model_path.exists():
        print(f"ERROR: no existe el modelo YOLO: {model_path}")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: no se puede importar ultralytics. Usa el Docker del proyecto.")
        return 2

    try:
        imagenes = _recoger_imagenes(images_dir, args.max_images)
        output_dir.mkdir(parents=True, exist_ok=True)

        modelo = YOLO(str(model_path))
        parametros = {
            "source": [str(ruta) for ruta in imagenes],
            "conf": args.confidence_threshold,
            "imgsz": args.imgsz,
            "save": False,
            "verbose": False,
        }
        if args.device is not None:
            parametros["device"] = args.device

        rutas_anotadas = [_guardar_imagen_anotada(resultado, output_dir) for resultado in modelo.predict(**parametros)]

        if args.contact_sheet_output:
            generar_contact_sheet(rutas_anotadas, _resolver(args.contact_sheet_output), args.tile_size)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Imágenes anotadas guardadas en: {output_dir}")
    print(f"Imágenes procesadas: {len(rutas_anotadas)}")
    if args.contact_sheet_output:
        print(f"Contact sheet guardada en: {args.contact_sheet_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
