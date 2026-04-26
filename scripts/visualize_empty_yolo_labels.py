#!/usr/bin/env python3
"""Lista y visualiza imágenes asociadas a labels YOLO vacíos."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabelVacio:
    """Relación entre un label vacío y su imagen correspondiente."""

    split: str
    imagen: Path | None
    label: Path


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Busca labels YOLO vacíos y genera una contact sheet de revisión."
    )
    parser.add_argument("--data-yaml", required=True, help="Ruta al data.yaml del dataset YOLO.")
    parser.add_argument(
        "--output",
        default="outputs/reports/empty_labels_contact_sheet.jpg",
        help="Ruta de salida para la contact sheet.",
    )
    parser.add_argument(
        "--report",
        default="docs/empty_labels_review.md",
        help="Ruta del informe Markdown.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "valid", "val", "test"],
        help="Split a revisar.",
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
    return ruta_imagenes.parent / "labels"


def _splits(datos: dict[str, Any], split_solicitado: str) -> list[tuple[str, str]]:
    val_key = "val" if "val" in datos else "valid"
    todos = [("train", "train"), ("valid", val_key)]
    if datos.get("test"):
        todos.append(("test", "test"))

    if split_solicitado == "all":
        return [(nombre, clave) for nombre, clave in todos if clave in datos]
    if split_solicitado == "val":
        split_solicitado = "valid"
    return [(nombre, clave) for nombre, clave in todos if nombre == split_solicitado and clave in datos]


def _buscar_imagen_por_stem(ruta_imagenes: Path, stem: str) -> Path | None:
    for extension in sorted(EXTENSIONES_IMAGEN):
        candidata = ruta_imagenes / f"{stem}{extension}"
        if candidata.exists():
            return candidata
    return None


def buscar_labels_vacios(ruta_yaml: Path, split_solicitado: str) -> list[LabelVacio]:
    datos = _leer_yaml(ruta_yaml)
    base_dataset = _resolver_base_dataset(ruta_yaml, datos)
    labels_vacios: list[LabelVacio] = []

    for split, clave_yaml in _splits(datos, split_solicitado):
        ruta_imagenes = _resolver_ruta(base_dataset, datos[clave_yaml])
        ruta_labels = _ruta_labels(ruta_imagenes)
        if not ruta_labels.exists():
            continue

        for ruta_label in sorted(ruta_labels.glob("*.txt")):
            if ruta_label.name == ".gitkeep":
                continue
            if ruta_label.read_text(encoding="utf-8").strip():
                continue
            labels_vacios.append(
                LabelVacio(
                    split=split,
                    imagen=_buscar_imagen_por_stem(ruta_imagenes, ruta_label.stem),
                    label=ruta_label,
                )
            )

    return labels_vacios


def _relativa(ruta: Path | None, repo_root: Path) -> str:
    if ruta is None:
        return "NO_ENCONTRADA"
    try:
        return str(ruta.resolve().relative_to(repo_root))
    except ValueError:
        return str(ruta)


def generar_informe(labels_vacios: list[LabelVacio], report: Path, output: Path, repo_root: Path) -> None:
    conteo_por_split = Counter(label.split for label in labels_vacios)
    lineas = [
        "# Revisión de labels YOLO vacíos",
        "",
        f"Fecha: {date.today().isoformat()}",
        "",
        f"Total de labels vacíos: {len(labels_vacios)}",
        "",
        "## Distribución por split",
        "",
    ]

    if conteo_por_split:
        for split, total in sorted(conteo_por_split.items()):
            lineas.append(f"- `{split}`: {total}")
    else:
        lineas.append("- No se encontraron labels vacíos.")

    lineas.extend(
        [
            "",
            "## Imágenes asociadas",
            "",
        ]
    )

    for label_vacio in labels_vacios:
        imagen_relativa = _relativa(label_vacio.imagen, repo_root)
        label_relativo = _relativa(label_vacio.label, repo_root)
        lineas.append(f"- `{label_vacio.split}`: imagen `{imagen_relativa}` | label `{label_relativo}`")

    lineas.extend(
        [
            "",
            "## Interpretación",
            "",
            "El script solo detecta labels vacíos y genera una contact sheet; no decide de forma automática si la imagen contiene o no un extintor.",
            "",
            "- Si las imágenes parecen negativas sin extintor visible, los labels vacíos son aceptables para YOLO.",
            "- Si alguna imagen contiene un extintor sin caja, hay que corregir la anotación antes de entrenar.",
            "",
            "Contact sheet generada:",
            "",
            f"```text\n{_relativa(output, repo_root)}\n```",
        ]
    )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def generar_contact_sheet(labels_vacios: list[LabelVacio], output: Path, tile_size: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    if not labels_vacios:
        imagen = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
        draw = ImageDraw.Draw(imagen)
        draw.text((20, 20), "No hay labels vacios", fill=(31, 41, 55))
        output.parent.mkdir(parents=True, exist_ok=True)
        imagen.save(output, quality=92)
        return

    columnas = min(4, max(1, math.ceil(math.sqrt(len(labels_vacios)))))
    filas = math.ceil(len(labels_vacios) / columnas)
    contact_sheet = Image.new("RGB", (columnas * tile_size, filas * tile_size), (229, 231, 235))

    for indice, label_vacio in enumerate(labels_vacios):
        tile = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
        draw = ImageDraw.Draw(tile)
        alto_cabecera = 34
        draw.rectangle((0, 0, tile_size, alto_cabecera), fill=(31, 41, 55))
        titulo = f"{label_vacio.split} / {label_vacio.label.stem[:34]}"
        draw.text((8, 9), titulo, fill=(255, 255, 255))

        if label_vacio.imagen is None:
            draw.text((12, 58), "Imagen no encontrada", fill=(127, 29, 29))
        else:
            imagen = Image.open(label_vacio.imagen).convert("RGB")
            ancho_original, alto_original = imagen.size
            margen = 16
            max_ancho = tile_size - margen * 2
            max_alto = tile_size - alto_cabecera - margen * 2
            escala = min(max_ancho / ancho_original, max_alto / alto_original)
            nuevo_tamano = (max(1, int(ancho_original * escala)), max(1, int(alto_original * escala)))
            imagen = imagen.resize(nuevo_tamano)
            offset_x = (tile_size - nuevo_tamano[0]) // 2
            offset_y = alto_cabecera + (max_alto - nuevo_tamano[1]) // 2
            tile.paste(imagen, (offset_x, offset_y))
            draw.rectangle(
                (offset_x, offset_y, offset_x + nuevo_tamano[0], offset_y + nuevo_tamano[1]),
                outline=(220, 38, 38),
                width=3,
            )
            draw.text((8, tile_size - 22), "label vacio", fill=(127, 29, 29))

        x = (indice % columnas) * tile_size
        y = (indice // columnas) * tile_size
        contact_sheet.paste(tile, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output, quality=92)


def main() -> int:
    args = crear_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    ruta_yaml = Path(args.data_yaml)
    output = Path(args.output)
    report = Path(args.report)

    if not ruta_yaml.exists():
        print(f"ERROR: no existe el data.yaml: {ruta_yaml}")
        return 2

    try:
        labels_vacios = buscar_labels_vacios(ruta_yaml, args.split)
        generar_contact_sheet(labels_vacios, output, args.tile_size)
        generar_informe(labels_vacios, report, output, repo_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Labels vacíos encontrados: {len(labels_vacios)}")
    for label_vacio in labels_vacios:
        print(
            f"- {label_vacio.split}: "
            f"{_relativa(label_vacio.imagen, repo_root)} | {_relativa(label_vacio.label, repo_root)}"
        )
    print(f"Contact sheet: {output}")
    print(f"Informe: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
