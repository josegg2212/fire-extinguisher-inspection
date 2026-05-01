#!/usr/bin/env python3
"""Comprueba la estructura esperada de los datasets YOLO y CNN."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO, validar_dataset_clasificador

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EstadisticasSplitYolo:
    """Resumen de validación para un split YOLO."""

    nombre: str
    ruta_imagenes: Path
    ruta_labels: Path
    imagenes: int = 0
    labels: int = 0
    anotaciones_por_clase: Counter[int] = field(default_factory=Counter)
    labels_vacios: int = 0
    imagenes_sin_label: int = 0
    labels_sin_imagen: int = 0
    extensiones_no_soportadas: int = 0

    @property
    def anotaciones(self) -> int:
        return sum(self.anotaciones_por_clase.values())


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


def _resolver_ruta_split(base_dataset: Path, valor: Any) -> Path:
    if isinstance(valor, list):
        if not valor:
            return base_dataset / "__split_vacio__"
        valor = valor[0]
    ruta = Path(str(valor))
    if not ruta.is_absolute():
        ruta = base_dataset / ruta
    return ruta.resolve()


def _ruta_labels_desde_imagenes(ruta_imagenes: Path) -> Path:
    if ruta_imagenes.name == "images":
        return ruta_imagenes.parent / "labels"
    return ruta_imagenes.parent / "labels"


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
        raise RuntimeError(f"El YAML {path} no contiene un diccionario válido.")
    return datos


def _validar_linea_yolo(
    linea: str,
    ruta_label: Path,
    numero_linea: int,
    numero_clases: int,
    errores: list[str],
) -> int | None:
    partes = linea.split()
    if len(partes) != 5:
        errores.append(f"{ruta_label}:{numero_linea}: se esperaban 5 campos y hay {len(partes)}.")
        return None

    try:
        clase_float = float(partes[0])
        clase_id = int(clase_float)
        coordenadas = [float(valor) for valor in partes[1:]]
    except ValueError:
        errores.append(f"{ruta_label}:{numero_linea}: hay campos no numéricos.")
        return None

    if clase_id != clase_float:
        errores.append(f"{ruta_label}:{numero_linea}: el id de clase debe ser entero.")
        return None

    if clase_id < 0 or clase_id >= numero_clases:
        errores.append(
            f"{ruta_label}:{numero_linea}: clase {clase_id} fuera de rango 0..{numero_clases - 1}."
        )

    x_centro, y_centro, ancho, alto = coordenadas
    if any(valor < 0 or valor > 1 for valor in coordenadas):
        errores.append(f"{ruta_label}:{numero_linea}: coordenadas fuera del rango [0, 1].")
    if ancho <= 0 or alto <= 0:
        errores.append(f"{ruta_label}:{numero_linea}: ancho y alto deben ser mayores que 0.")

    return clase_id


def _validar_split_yolo(
    nombre: str,
    ruta_imagenes: Path,
    numero_clases: int,
    require_images: bool,
    errores: list[str],
    advertencias: list[str],
) -> EstadisticasSplitYolo:
    ruta_labels = _ruta_labels_desde_imagenes(ruta_imagenes)
    stats = EstadisticasSplitYolo(nombre=nombre, ruta_imagenes=ruta_imagenes, ruta_labels=ruta_labels)

    if not ruta_imagenes.exists():
        errores.append(f"El split '{nombre}' no tiene carpeta de imágenes: {ruta_imagenes}")
        return stats
    if not ruta_labels.exists():
        errores.append(f"El split '{nombre}' no tiene carpeta de labels: {ruta_labels}")
        return stats

    imagenes = [
        ruta
        for ruta in ruta_imagenes.iterdir()
        if ruta.is_file() and ruta.name != ".gitkeep"
    ]
    labels = [
        ruta
        for ruta in ruta_labels.iterdir()
        if ruta.is_file() and ruta.name != ".gitkeep"
    ]
    imagenes_soportadas = [ruta for ruta in imagenes if ruta.suffix.lower() in EXTENSIONES_IMAGEN]
    imagenes_no_soportadas = [ruta for ruta in imagenes if ruta.suffix.lower() not in EXTENSIONES_IMAGEN]
    labels_txt = [ruta for ruta in labels if ruta.suffix.lower() == ".txt"]
    labels_no_txt = [ruta for ruta in labels if ruta.suffix.lower() != ".txt"]

    stats.imagenes = len(imagenes_soportadas)
    stats.labels = len(labels_txt)
    stats.extensiones_no_soportadas = len(imagenes_no_soportadas)

    if require_images and stats.imagenes == 0:
        errores.append(f"El split '{nombre}' no contiene imágenes.")
    for ruta in imagenes_no_soportadas[:10]:
        errores.append(f"Imagen con extensión no soportada en '{nombre}': {ruta.name}")
    for ruta in labels_no_txt[:10]:
        errores.append(f"Label con extensión no soportada en '{nombre}': {ruta.name}")

    imagenes_por_stem = {ruta.stem: ruta for ruta in imagenes_soportadas}
    labels_por_stem = {ruta.stem: ruta for ruta in labels_txt}
    sin_label = sorted(set(imagenes_por_stem) - set(labels_por_stem))
    sin_imagen = sorted(set(labels_por_stem) - set(imagenes_por_stem))
    stats.imagenes_sin_label = len(sin_label)
    stats.labels_sin_imagen = len(sin_imagen)

    if sin_label:
        advertencias.append(
            f"Split '{nombre}': {len(sin_label)} imágenes no tienen label. "
            "Puede ser válido si son ejemplos negativos."
        )
    if sin_imagen:
        errores.append(f"Split '{nombre}': {len(sin_imagen)} labels no tienen imagen correspondiente.")

    for ruta_label in labels_txt:
        contenido = ruta_label.read_text(encoding="utf-8").strip()
        if not contenido:
            stats.labels_vacios += 1
            continue
        for numero_linea, linea in enumerate(contenido.splitlines(), start=1):
            clase_id = _validar_linea_yolo(
                linea=linea,
                ruta_label=ruta_label,
                numero_linea=numero_linea,
                numero_clases=numero_clases,
                errores=errores,
            )
            if clase_id is not None:
                stats.anotaciones_por_clase[clase_id] += 1

    if stats.labels_vacios:
        advertencias.append(
            f"Split '{nombre}': {stats.labels_vacios} labels están vacíos. "
            "YOLO lo admite para imágenes sin objetos, pero conviene revisarlo."
        )

    return stats


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
        datos = _leer_yaml(path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    errores: list[str] = []
    advertencias: list[str] = []

    for clave in ["train", "names"]:
        if clave not in datos:
            errores.append(f"Falta la clave '{clave}' en {path}")
    if "val" not in datos and "valid" not in datos:
        errores.append(f"Falta la clave 'val' o 'valid' en {path}")

    nombres = _normalizar_nombres(datos.get("names", {}))
    numero_clases = len(nombres)
    nc = datos.get("nc", numero_clases)
    try:
        nc = int(nc)
    except (TypeError, ValueError):
        errores.append("La clave 'nc' debe ser un entero.")
        nc = numero_clases

    if not nombres:
        errores.append("La clave 'names' no define ninguna clase.")
    if numero_clases != nc:
        errores.append(f"El número de clases no coincide: nc={nc}, len(names)={numero_clases}.")

    if "fire_extinguisher" not in [str(nombre) for nombre in nombres]:
        errores.append("El YAML debe incluir la clase 'fire_extinguisher'.")

    if errores:
        for error in errores:
            print(f"ERROR: {error}")
        return 2

    base_dataset = _resolver_base_dataset(path, datos)
    splits: list[tuple[str, str]] = [("train", "train")]
    splits.append(("valid", "val" if "val" in datos else "valid"))
    if "test" in datos and datos["test"]:
        splits.append(("test", "test"))

    estadisticas: list[EstadisticasSplitYolo] = []
    for nombre_visible, clave_yaml in splits:
        ruta_imagenes = _resolver_ruta_split(base_dataset, datos[clave_yaml])
        estadisticas.append(
            _validar_split_yolo(
                nombre=nombre_visible,
                ruta_imagenes=ruta_imagenes,
                numero_clases=numero_clases,
                require_images=True,
                errores=errores,
                advertencias=advertencias,
            )
        )

    if errores:
        for error in errores:
            print(f"ERROR: {error}")
        for advertencia in advertencias:
            print(f"AVISO: {advertencia}")
        return 2

    print(f"Dataset YOLO válido: {path}")
    print(f"Base del dataset: {base_dataset}")
    print(f"Clases ({numero_clases}): {', '.join(nombres)}")
    for stats in estadisticas:
        print(
            f"- {stats.nombre}: {stats.imagenes} imágenes, {stats.labels} labels, "
            f"{stats.anotaciones} anotaciones"
        )
        for clase_id, total in sorted(stats.anotaciones_por_clase.items()):
            print(f"  clase {clase_id} ({nombres[clase_id]}): {total}")
    for advertencia in advertencias:
        print(f"AVISO: {advertencia}")
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
