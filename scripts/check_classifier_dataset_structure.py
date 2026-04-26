#!/usr/bin/env python3
"""Valida la estructura ImageFolder del dataset CNN de estado."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO, EXTENSIONES_IMAGEN


SPLITS = ("train", "val", "test")


@dataclass
class ResultadoClassifier:
    """Resultado detallado de la validacion del dataset CNN."""

    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    conteos: dict[str, Counter[str]] = field(default_factory=dict)
    corruptas: list[Path] = field(default_factory=list)
    extensiones_no_soportadas: list[Path] = field(default_factory=list)

    @property
    def es_valido(self) -> bool:
        return not self.errores


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comprueba un dataset CNN de estado.")
    parser.add_argument("--dataset-dir", default="data/classifier", help="Raiz del dataset classifier.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permite clases vacias sin devolver error.",
    )
    return parser


def _iterar_archivos_clase(ruta_clase: Path) -> tuple[list[Path], list[Path]]:
    soportadas: list[Path] = []
    no_soportadas: list[Path] = []
    for ruta in sorted(ruta_clase.iterdir()):
        if not ruta.is_file() or ruta.name == ".gitkeep":
            continue
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
            soportadas.append(ruta)
        else:
            no_soportadas.append(ruta)
    return soportadas, no_soportadas


def _imagen_es_legible(path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    try:
        with Image.open(path) as imagen:
            imagen.verify()
        with Image.open(path) as imagen:
            imagen.convert("RGB").load()
    except Exception:
        return False
    return True


def validar_dataset(dataset_dir: Path, allow_empty: bool) -> ResultadoClassifier:
    resultado = ResultadoClassifier()
    if not dataset_dir.exists():
        resultado.errores.append(f"No existe el directorio del dataset: {dataset_dir}")
        return resultado

    for split in SPLITS:
        ruta_split = dataset_dir / split
        resultado.conteos[split] = Counter()
        if not ruta_split.exists():
            resultado.errores.append(f"Falta el split '{split}': {ruta_split}")
            continue

        for clase in CLASES_ESTADO:
            ruta_clase = ruta_split / clase
            if not ruta_clase.exists():
                resultado.errores.append(f"Falta la carpeta de clase: {ruta_clase}")
                resultado.conteos[split][clase] = 0
                continue

            imagenes, no_soportadas = _iterar_archivos_clase(ruta_clase)
            resultado.extensiones_no_soportadas.extend(no_soportadas)
            resultado.conteos[split][clase] = len(imagenes)

            if not imagenes:
                mensaje = f"Clase vacia: {ruta_clase}"
                if allow_empty:
                    resultado.advertencias.append(mensaje)
                else:
                    resultado.errores.append(mensaje)

            for imagen in imagenes:
                if not _imagen_es_legible(imagen):
                    resultado.corruptas.append(imagen)

    for ruta in resultado.extensiones_no_soportadas:
        resultado.errores.append(f"Archivo con extension no soportada: {ruta}")
    for ruta in resultado.corruptas:
        resultado.errores.append(f"Imagen corrupta o no legible: {ruta}")

    for split, conteo in resultado.conteos.items():
        valores = [conteo[clase] for clase in CLASES_ESTADO]
        total = sum(valores)
        if total == 0:
            resultado.errores.append(f"El split '{split}' no contiene imagenes.")
            continue
        positivos = [valor for valor in valores if valor > 0]
        if len(positivos) >= 2 and max(positivos) / min(positivos) > 2.0:
            resultado.advertencias.append(
                f"Distribucion desbalanceada en '{split}': "
                + ", ".join(f"{clase}={conteo[clase]}" for clase in CLASES_ESTADO)
            )

    return resultado


def imprimir_resultado(resultado: ResultadoClassifier, dataset_dir: Path) -> None:
    print(f"Dataset classifier: {dataset_dir}")
    for split in SPLITS:
        conteo = resultado.conteos.get(split, Counter())
        total = sum(conteo.values())
        print(f"- {split}: total={total}")
        for clase in CLASES_ESTADO:
            porcentaje = (conteo[clase] / total * 100) if total else 0.0
            print(f"  {clase}: {conteo[clase]} ({porcentaje:.1f}%)")

    total_por_clase = Counter()
    for conteo in resultado.conteos.values():
        total_por_clase.update(conteo)
    total_global = sum(total_por_clase.values())
    print("Distribucion global:")
    for clase in CLASES_ESTADO:
        porcentaje = (total_por_clase[clase] / total_global * 100) if total_global else 0.0
        print(f"- {clase}: {total_por_clase[clase]} ({porcentaje:.1f}%)")

    for advertencia in resultado.advertencias:
        print(f"AVISO: {advertencia}")
    for error in resultado.errores:
        print(f"ERROR: {error}")


def main() -> int:
    args = crear_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    try:
        resultado = validar_dataset(dataset_dir, allow_empty=args.allow_empty)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    imprimir_resultado(resultado, dataset_dir)
    return 0 if resultado.es_valido else 2


if __name__ == "__main__":
    raise SystemExit(main())
