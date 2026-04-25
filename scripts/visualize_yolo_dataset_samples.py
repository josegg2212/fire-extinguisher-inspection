#!/usr/bin/env python3
"""Genera una contact sheet con muestras anotadas de un dataset YOLO."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORES = [
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (202, 138, 4),
    (147, 51, 234),
]


@dataclass(frozen=True)
class MuestraYolo:
    """Imagen y label asociados a una muestra YOLO."""

    split: str
    imagen: Path
    label: Path


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dibuja muestras de un dataset YOLO en una contact sheet."
    )
    parser.add_argument("--data-yaml", required=True, help="Ruta al data.yaml del dataset YOLO.")
    parser.add_argument(
        "--output",
        default="outputs/reports/contact_sheet_yolo_dataset.jpg",
        help="Ruta de salida de la imagen generada.",
    )
    parser.add_argument("--num-samples", type=int, default=16, help="Número máximo de muestras.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "valid", "val", "test"],
        help="Split a visualizar.",
    )
    parser.add_argument("--tile-size", type=int, default=360, help="Tamaño de cada celda en píxeles.")
    return parser


def _leer_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("No se puede importar PyYAML. Instala requirements.txt.") from exc

    datos = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(datos, dict):
        raise RuntimeError(f"El YAML {path} no contiene un diccionario válido.")
    return datos


def _normalizar_nombres(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(nombre) for nombre in names]
    if isinstance(names, dict):
        return [str(names[indice]) for indice in sorted(names, key=lambda valor: int(valor))]
    return []


def _resolver_base_dataset(ruta_yaml: Path, datos: dict[str, Any]) -> Path:
    valor_path = datos.get("path")
    if valor_path is None:
        return ruta_yaml.parent.resolve()

    ruta_base = Path(str(valor_path))
    if not ruta_base.is_absolute():
        ruta_base = ruta_yaml.parent / ruta_base
    return ruta_base.resolve()


def _resolver_ruta(base_dataset: Path, valor: Any) -> Path:
    if isinstance(valor, list):
        if not valor:
            return base_dataset / "__split_vacio__"
        valor = valor[0]
    ruta = Path(str(valor))
    if not ruta.is_absolute():
        ruta = base_dataset / ruta
    return ruta.resolve()


def _ruta_labels(ruta_imagenes: Path) -> Path:
    if ruta_imagenes.name == "images":
        return ruta_imagenes.parent / "labels"
    return ruta_imagenes.parent / "labels"


def _splits_disponibles(datos: dict[str, Any], split_solicitado: str) -> list[tuple[str, str]]:
    val_key = "val" if "val" in datos else "valid"
    todos = [("train", "train"), ("valid", val_key)]
    if datos.get("test"):
        todos.append(("test", "test"))

    if split_solicitado == "all":
        return [(nombre, clave) for nombre, clave in todos if clave in datos]
    if split_solicitado == "val":
        split_solicitado = "valid"
    return [(nombre, clave) for nombre, clave in todos if nombre == split_solicitado and clave in datos]


def _recoger_muestras(
    ruta_yaml: Path,
    datos: dict[str, Any],
    split_solicitado: str,
    limite: int,
) -> list[MuestraYolo]:
    base_dataset = _resolver_base_dataset(ruta_yaml, datos)
    muestras_por_split: list[list[MuestraYolo]] = []

    for split, clave_yaml in _splits_disponibles(datos, split_solicitado):
        ruta_imagenes = _resolver_ruta(base_dataset, datos[clave_yaml])
        ruta_labels = _ruta_labels(ruta_imagenes)
        if not ruta_imagenes.exists() or not ruta_labels.exists():
            continue

        imagenes = sorted(
            ruta
            for ruta in ruta_imagenes.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
        )
        muestras_split = [MuestraYolo(split, imagen, ruta_labels / f"{imagen.stem}.txt") for imagen in imagenes]
        muestras_por_split.append(muestras_split)

    seleccionadas: list[MuestraYolo] = []
    indice = 0
    while len(seleccionadas) < limite:
        hubo_muestra = False
        for muestras_split in muestras_por_split:
            if indice < len(muestras_split):
                seleccionadas.append(muestras_split[indice])
                hubo_muestra = True
                if len(seleccionadas) >= limite:
                    break
        if not hubo_muestra:
            break
        indice += 1

    return seleccionadas


def _leer_anotaciones(label: Path, numero_clases: int) -> list[tuple[int, float, float, float, float]]:
    if not label.exists():
        return []

    anotaciones: list[tuple[int, float, float, float, float]] = []
    for linea in label.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) != 5:
            continue
        try:
            clase_id = int(float(partes[0]))
            x_centro, y_centro, ancho, alto = [float(valor) for valor in partes[1:]]
        except ValueError:
            continue
        if 0 <= clase_id < numero_clases:
            anotaciones.append((clase_id, x_centro, y_centro, ancho, alto))
    return anotaciones


def _dibujar_muestra(muestra: MuestraYolo, nombres: list[str], tile_size: int):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    margen = 18
    alto_cabecera = 30
    lienzo = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
    draw = ImageDraw.Draw(lienzo)

    imagen = Image.open(muestra.imagen).convert("RGB")
    ancho_original, alto_original = imagen.size
    max_ancho = tile_size - margen * 2
    max_alto = tile_size - alto_cabecera - margen * 2
    escala = min(max_ancho / ancho_original, max_alto / alto_original)
    nuevo_tamano = (max(1, int(ancho_original * escala)), max(1, int(alto_original * escala)))
    imagen_redimensionada = imagen.resize(nuevo_tamano)
    offset_x = (tile_size - nuevo_tamano[0]) // 2
    offset_y = alto_cabecera + (max_alto - nuevo_tamano[1]) // 2
    lienzo.paste(imagen_redimensionada, (offset_x, offset_y))

    draw.rectangle((0, 0, tile_size, alto_cabecera), fill=(31, 41, 55))
    titulo = f"{muestra.split} / {muestra.imagen.name[:34]}"
    draw.text((8, 8), titulo, fill=(255, 255, 255))

    anotaciones = _leer_anotaciones(muestra.label, len(nombres))
    for clase_id, x_centro, y_centro, ancho, alto in anotaciones:
        color = COLORES[clase_id % len(COLORES)]
        x1 = (x_centro - ancho / 2) * ancho_original * escala + offset_x
        y1 = (y_centro - alto / 2) * alto_original * escala + offset_y
        x2 = (x_centro + ancho / 2) * ancho_original * escala + offset_x
        y2 = (y_centro + alto / 2) * alto_original * escala + offset_y
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        etiqueta = nombres[clase_id]
        bbox_texto = draw.textbbox((x1, max(offset_y, y1 - 18)), etiqueta)
        draw.rectangle(bbox_texto, fill=color)
        draw.text((x1, max(offset_y, y1 - 18)), etiqueta, fill=(255, 255, 255))

    if not anotaciones:
        draw.text((8, tile_size - 22), "sin anotaciones", fill=(127, 29, 29))

    return lienzo


def generar_contact_sheet(muestras: list[MuestraYolo], nombres: list[str], output: Path, tile_size: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    columnas = min(4, max(1, math.ceil(math.sqrt(len(muestras)))))
    filas = math.ceil(len(muestras) / columnas)
    contact_sheet = Image.new("RGB", (columnas * tile_size, filas * tile_size), (229, 231, 235))

    for indice, muestra in enumerate(muestras):
        tile = _dibujar_muestra(muestra, nombres, tile_size)
        x = (indice % columnas) * tile_size
        y = (indice // columnas) * tile_size
        contact_sheet.paste(tile, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output, quality=92)


def main() -> int:
    args = crear_parser().parse_args()
    ruta_yaml = Path(args.data_yaml)
    if not ruta_yaml.exists():
        print(f"ERROR: no existe el data.yaml: {ruta_yaml}")
        return 2
    if args.num_samples <= 0:
        print("ERROR: --num-samples debe ser mayor que 0.")
        return 2

    try:
        datos = _leer_yaml(ruta_yaml)
        nombres = _normalizar_nombres(datos.get("names", {}))
        if not nombres:
            print("ERROR: data.yaml no define clases en 'names'.")
            return 2
        muestras = _recoger_muestras(ruta_yaml, datos, args.split, args.num_samples)
        if not muestras:
            print("ERROR: no se encontraron muestras para visualizar.")
            return 2
        generar_contact_sheet(muestras, nombres, Path(args.output), args.tile_size)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    splits = sorted({muestra.split for muestra in muestras})
    print(f"Contact sheet generada en: {args.output}")
    print(f"Muestras: {len(muestras)} | splits: {', '.join(splits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
