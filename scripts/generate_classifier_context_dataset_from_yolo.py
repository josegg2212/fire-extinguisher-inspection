#!/usr/bin/env python3
"""Genera el dataset CNN contextual v2 desde anotaciones YOLO."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO
from fire_extinguisher_inspection.preprocessing.crop_utils import calcular_region_contextual
from fire_extinguisher_inspection.preprocessing.occlusion_utils import (
    aplicar_oclusion_fuerte,
    aplicar_oclusion_parcial,
)


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS_SALIDA = ("train", "val", "test")


@dataclass(frozen=True)
class SplitYolo:
    """Relacion entre un split YOLO y su salida ImageFolder."""

    nombre_yolo: str
    nombre_salida: str
    ruta_imagenes: Path
    ruta_labels: Path


@dataclass(frozen=True)
class AnotacionYolo:
    """Bounding box YOLO normalizada."""

    clase_id: int
    x_centro: float
    y_centro: float
    ancho: float
    alto: float


@dataclass
class ResumenGeneracion:
    """Contadores de generacion del dataset contextual."""

    imagenes_procesadas: Counter[str]
    objetos_procesados: Counter[str]
    crops_generados: dict[str, Counter[str]]
    imagenes_sin_label: int = 0
    labels_vacios: int = 0
    anotaciones_invalidas: int = 0
    crops_descartados: int = 0
    imagenes_no_legibles: int = 0


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un dataset CNN contextual v2 desde labels YOLO."
    )
    parser.add_argument("--data-yaml", default="data/yolo/data.yaml", help="Ruta al data.yaml YOLO.")
    parser.add_argument(
        "--output-dir",
        default="data/classifier_context_v2",
        help="Directorio de salida del dataset contextual.",
    )
    parser.add_argument(
        "--context-padding",
        type=float,
        default=0.75,
        help="Margen contextual alrededor de la bbox, relativo al tamano de la bbox.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Tamano final cuadrado de cada crop.")
    parser.add_argument(
        "--visible-crops-per-object",
        type=int,
        default=1,
        help="Numero de crops visibles por objeto.",
    )
    parser.add_argument(
        "--partial-occlusions-per-object",
        type=int,
        default=1,
        help="Numero de variantes parcialmente ocultas por objeto.",
    )
    parser.add_argument(
        "--blocked-occlusions-per-object",
        type=int,
        default=1,
        help="Numero de variantes bloqueadas por objeto.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Maximo opcional de imagenes originales a procesar por split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--overwrite", action="store_true", help="Limpia el dataset contextual anterior.")
    parser.add_argument("--min-crop-size", type=int, default=32, help="Tamano minimo aceptado del crop.")
    parser.add_argument(
        "--square-crop",
        action="store_true",
        help="Intenta convertir la region contextual a cuadrada antes de redimensionar.",
    )
    return parser


def _leer_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("No se puede importar PyYAML. Instala requirements.txt.") from exc

    try:
        datos = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer el YAML {path}: {exc}") from exc

    if not isinstance(datos, dict):
        raise RuntimeError(f"El YAML {path} no contiene un diccionario valido.")
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


def _ruta_labels_desde_imagenes(ruta_imagenes: Path) -> Path:
    return ruta_imagenes.parent / "labels"


def _obtener_splits_yolo(ruta_yaml: Path, datos: dict[str, Any]) -> list[SplitYolo]:
    base_dataset = _resolver_base_dataset(ruta_yaml, datos)
    claves = [("train", "train", "train")]
    if "val" in datos:
        claves.append(("valid", "val", "val"))
    elif "valid" in datos:
        claves.append(("valid", "val", "valid"))
    else:
        raise RuntimeError("El data.yaml debe definir 'val' o 'valid'.")

    if datos.get("test"):
        claves.append(("test", "test", "test"))
    else:
        raise RuntimeError("El data.yaml debe definir el split 'test'.")

    splits: list[SplitYolo] = []
    for nombre_yolo, nombre_salida, clave_yaml in claves:
        ruta_imagenes = _resolver_ruta(base_dataset, datos[clave_yaml])
        ruta_labels = _ruta_labels_desde_imagenes(ruta_imagenes)
        if not ruta_imagenes.exists():
            raise RuntimeError(f"No existe la carpeta de imagenes del split {nombre_yolo}: {ruta_imagenes}")
        if not ruta_labels.exists():
            raise RuntimeError(f"No existe la carpeta de labels del split {nombre_yolo}: {ruta_labels}")
        splits.append(SplitYolo(nombre_yolo, nombre_salida, ruta_imagenes, ruta_labels))
    return splits


def _validar_args(args: argparse.Namespace) -> None:
    if args.max_per_split is not None and args.max_per_split <= 0:
        raise RuntimeError("--max-per-split debe ser mayor que 0.")
    if args.visible_crops_per_object < 0:
        raise RuntimeError("--visible-crops-per-object no puede ser negativo.")
    if args.partial_occlusions_per_object < 0:
        raise RuntimeError("--partial-occlusions-per-object no puede ser negativo.")
    if args.blocked_occlusions_per_object < 0:
        raise RuntimeError("--blocked-occlusions-per-object no puede ser negativo.")
    if (
        args.visible_crops_per_object
        + args.partial_occlusions_per_object
        + args.blocked_occlusions_per_object
        == 0
    ):
        raise RuntimeError("Debe generarse al menos una variante por objeto.")
    if args.context_padding < 0:
        raise RuntimeError("--context-padding no puede ser negativo.")
    if args.min_crop_size <= 0:
        raise RuntimeError("--min-crop-size debe ser mayor que 0.")
    if args.image_size <= 0:
        raise RuntimeError("--image-size debe ser mayor que 0.")


def _limpiar_salida(output_dir: Path) -> None:
    ruta_resuelta = output_dir.resolve()
    rutas_protegidas = {
        Path("/").resolve(),
        Path.home().resolve(),
        REPO_ROOT.resolve(),
        (REPO_ROOT / "data").resolve(),
    }
    if ruta_resuelta in rutas_protegidas:
        raise RuntimeError(f"Ruta de salida demasiado amplia para limpiar: {ruta_resuelta}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _crear_estructura_salida(output_dir: Path) -> None:
    for split in SPLITS_SALIDA:
        for clase in CLASES_ESTADO:
            (output_dir / split / clase).mkdir(parents=True, exist_ok=True)


def _iterar_imagenes(ruta_imagenes: Path) -> list[Path]:
    return sorted(
        ruta
        for ruta in ruta_imagenes.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    )


def _leer_anotaciones(ruta_label: Path, resumen: ResumenGeneracion) -> list[AnotacionYolo]:
    if not ruta_label.exists():
        resumen.imagenes_sin_label += 1
        return []

    contenido = ruta_label.read_text(encoding="utf-8").strip()
    if not contenido:
        resumen.labels_vacios += 1
        return []

    anotaciones: list[AnotacionYolo] = []
    for linea in contenido.splitlines():
        partes = linea.split()
        if len(partes) != 5:
            resumen.anotaciones_invalidas += 1
            continue
        try:
            clase_id = int(float(partes[0]))
            x_centro, y_centro, ancho, alto = [float(valor) for valor in partes[1:]]
        except ValueError:
            resumen.anotaciones_invalidas += 1
            continue
        if ancho <= 0 or alto <= 0 or any(valor < 0 or valor > 1 for valor in (x_centro, y_centro, ancho, alto)):
            resumen.anotaciones_invalidas += 1
            continue
        anotaciones.append(AnotacionYolo(clase_id, x_centro, y_centro, ancho, alto))
    return anotaciones


def _bbox_yolo_a_pixeles(
    anotacion: AnotacionYolo,
    ancho_imagen: int,
    alto_imagen: int,
) -> tuple[int, int, int, int]:
    bbox_ancho = anotacion.ancho * ancho_imagen
    bbox_alto = anotacion.alto * alto_imagen
    x1 = (anotacion.x_centro * ancho_imagen) - bbox_ancho / 2
    y1 = (anotacion.y_centro * alto_imagen) - bbox_alto / 2
    x2 = x1 + bbox_ancho
    y2 = y1 + bbox_alto
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(ancho_imagen, int(round(x2))),
        min(alto_imagen, int(round(y2))),
    )


def _bbox_relativa_redimensionada(
    bbox: tuple[int, int, int, int],
    region: tuple[int, int, int, int],
    image_size: int,
) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = region
    escala_x = image_size / max(1, rx2 - rx1)
    escala_y = image_size / max(1, ry2 - ry1)
    return (
        max(0, min(image_size - 1, int(round((bx1 - rx1) * escala_x)))),
        max(0, min(image_size - 1, int(round((by1 - ry1) * escala_y)))),
        max(1, min(image_size, int(round((bx2 - rx1) * escala_x)))),
        max(1, min(image_size, int(round((by2 - ry1) * escala_y)))),
    )


def _normalizar_nombre(nombre: str) -> str:
    limpio = re.sub(r"[^A-Za-z0-9_-]+", "_", nombre).strip("_")
    return limpio or "image"


def _ruta_unica(ruta: Path) -> Path:
    if not ruta.exists():
        return ruta
    for indice in range(1, 10000):
        candidata = ruta.with_name(f"{ruta.stem}_{indice}{ruta.suffix}")
        if not candidata.exists():
            return candidata
    raise RuntimeError(f"No se pudo crear un nombre unico para {ruta}")


def _guardar_imagen(imagen: object, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imagen.convert("RGB").save(_ruta_unica(ruta), quality=92)


def _resize_crop(crop: object, image_size: int) -> object:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    filtro = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return crop.resize((image_size, image_size), filtro)


def _procesar_objeto(
    imagen: object,
    anotacion: AnotacionYolo,
    split_salida: str,
    stem_imagen: str,
    indice_objeto: int,
    args: argparse.Namespace,
    rng: random.Random,
    output_dir: Path,
    resumen: ResumenGeneracion,
) -> None:
    ancho_imagen, alto_imagen = imagen.size
    bbox = _bbox_yolo_a_pixeles(anotacion, ancho_imagen, alto_imagen)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        resumen.crops_descartados += 1
        return

    try:
        region = calcular_region_contextual(
            ancho_imagen,
            alto_imagen,
            bbox,
            context_padding=args.context_padding,
            square=args.square_crop,
        )
    except ValueError:
        resumen.crops_descartados += 1
        return

    if region[2] - region[0] < args.min_crop_size or region[3] - region[1] < args.min_crop_size:
        resumen.crops_descartados += 1
        return

    bbox_objeto = _bbox_relativa_redimensionada(bbox, region, args.image_size)
    if bbox_objeto[2] <= bbox_objeto[0] or bbox_objeto[3] <= bbox_objeto[1]:
        resumen.crops_descartados += 1
        return

    crop_visible = _resize_crop(imagen.crop(region), args.image_size)
    nombre_base = f"{split_salida}_{_normalizar_nombre(stem_imagen)}_obj{indice_objeto}"

    for indice in range(args.visible_crops_per_object):
        sufijo = "visible" if args.visible_crops_per_object == 1 else f"visible_{indice}"
        ruta = output_dir / split_salida / "visible" / f"{nombre_base}_{sufijo}.jpg"
        _guardar_imagen(crop_visible, ruta)
        resumen.crops_generados[split_salida]["visible"] += 1

    for indice in range(args.partial_occlusions_per_object):
        crop_partial = aplicar_oclusion_parcial(crop_visible, rng, bbox_objeto=bbox_objeto)
        ruta = output_dir / split_salida / "partially_occluded" / f"{nombre_base}_partial_{indice}.jpg"
        _guardar_imagen(crop_partial, ruta)
        resumen.crops_generados[split_salida]["partially_occluded"] += 1

    for indice in range(args.blocked_occlusions_per_object):
        crop_blocked = aplicar_oclusion_fuerte(crop_visible, rng, bbox_objeto=bbox_objeto)
        ruta = output_dir / split_salida / "blocked" / f"{nombre_base}_blocked_{indice}.jpg"
        _guardar_imagen(crop_blocked, ruta)
        resumen.crops_generados[split_salida]["blocked"] += 1

    resumen.objetos_procesados[split_salida] += 1


def _procesar_split(
    split: SplitYolo,
    output_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
    resumen: ResumenGeneracion,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    imagenes = _iterar_imagenes(split.ruta_imagenes)
    if args.max_per_split is not None:
        imagenes = imagenes[: args.max_per_split]

    for ruta_imagen in imagenes:
        ruta_label = split.ruta_labels / f"{ruta_imagen.stem}.txt"
        anotaciones = _leer_anotaciones(ruta_label, resumen)
        resumen.imagenes_procesadas[split.nombre_salida] += 1
        if not anotaciones:
            continue

        try:
            with Image.open(ruta_imagen) as imagen_original:
                imagen = imagen_original.convert("RGB")
        except Exception:
            resumen.imagenes_no_legibles += 1
            continue

        for indice_objeto, anotacion in enumerate(anotaciones):
            _procesar_objeto(
                imagen=imagen,
                anotacion=anotacion,
                split_salida=split.nombre_salida,
                stem_imagen=ruta_imagen.stem,
                indice_objeto=indice_objeto,
                args=args,
                rng=rng,
                output_dir=output_dir,
                resumen=resumen,
            )


def _imprimir_resumen(resumen: ResumenGeneracion, output_dir: Path) -> None:
    print(f"Dataset CNN contextual generado en: {output_dir}")
    for split in SPLITS_SALIDA:
        total_split = sum(resumen.crops_generados[split].values())
        print(
            f"- {split}: {resumen.imagenes_procesadas[split]} imagenes YOLO, "
            f"{resumen.objetos_procesados[split]} objetos, {total_split} crops"
        )
        for clase in CLASES_ESTADO:
            print(f"  {clase}: {resumen.crops_generados[split][clase]}")

    avisos = {
        "imagenes_sin_label": resumen.imagenes_sin_label,
        "labels_vacios": resumen.labels_vacios,
        "anotaciones_invalidas": resumen.anotaciones_invalidas,
        "crops_descartados": resumen.crops_descartados,
        "imagenes_no_legibles": resumen.imagenes_no_legibles,
    }
    for nombre, total in avisos.items():
        if total:
            print(f"AVISO: {nombre}={total}")


def main() -> int:
    args = crear_parser().parse_args()
    try:
        _validar_args(args)
        ruta_yaml = Path(args.data_yaml)
        if not ruta_yaml.exists():
            print(f"ERROR: no existe el data.yaml YOLO: {ruta_yaml}")
            return 2

        datos = _leer_yaml(ruta_yaml)
        splits = _obtener_splits_yolo(ruta_yaml, datos)
        output_dir = Path(args.output_dir)

        if args.overwrite:
            _limpiar_salida(output_dir)
        _crear_estructura_salida(output_dir)

        resumen = ResumenGeneracion(
            imagenes_procesadas=Counter(),
            objetos_procesados=Counter(),
            crops_generados=defaultdict(Counter),
        )
        rng = random.Random(args.seed)
        for split in splits:
            _procesar_split(split, output_dir, args, rng, resumen)
        _imprimir_resumen(resumen, output_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
