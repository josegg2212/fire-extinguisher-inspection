"""Entrenamiento del detector YOLO de extintores con Ultralytics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def crear_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de línea de comandos."""

    parser = argparse.ArgumentParser(description="Entrena un detector YOLO para extintores.")
    parser.add_argument("--data", required=True, help="Ruta al data.yaml del dataset YOLO.")
    parser.add_argument(
        "--model",
        default="models/yolo/base/yolo26n.pt",
        help=(
            "Modelo base de Ultralytics. Por defecto usa el peso local limpio; "
            "tambien acepta nombres como yolo26n.pt o yolo11n.pt."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen para entrenamiento.")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño de batch.")
    parser.add_argument("--workers", type=int, default=0, help="Workers del dataloader.")
    parser.add_argument("--project", default="models/yolo", help="Directorio de salida del experimento.")
    parser.add_argument("--name", default="extinguisher_yolo_v1", help="Nombre del experimento.")
    parser.add_argument("--device", default=None, help="Dispositivo opcional, por ejemplo 'cpu' o '0'.")
    return parser


def _resolver_path_dataset(ruta_data: Path, datos: dict[str, Any]) -> Path:
    """Resuelve la raíz del dataset de forma estable para Ultralytics."""

    valor_path = datos.get("path")
    if valor_path is None:
        return ruta_data.parent.resolve()

    ruta_path = Path(str(valor_path))
    if ruta_path.is_absolute():
        return ruta_path

    candidatos = [
        (ruta_data.parent / ruta_path).resolve(),
        (Path.cwd() / ruta_path).resolve(),
        (REPO_ROOT / ruta_path).resolve(),
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    return candidatos[0]


def preparar_yaml_para_ultralytics(ruta_data: Path) -> Path:
    """Crea una copia temporal del YAML con `path` absoluto.

    Ultralytics puede interpretar `path: .` relativo al directorio de ejecución.
    Para evitar errores al lanzar el script desde Docker u otra carpeta, se genera
    un YAML equivalente en `outputs/logs/` con la raíz del dataset ya resuelta.
    """

    try:
        import yaml
    except ImportError:
        logging.warning("PyYAML no está disponible; se usará el data.yaml original.")
        return ruta_data

    try:
        datos = yaml.safe_load(ruta_data.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logging.warning("No se pudo normalizar %s: %s", ruta_data, exc)
        return ruta_data

    if not isinstance(datos, dict):
        logging.warning("El archivo %s no contiene un diccionario YAML válido.", ruta_data)
        return ruta_data

    datos["path"] = str(_resolver_path_dataset(ruta_data, datos))
    ruta_salida = REPO_ROOT / "outputs" / "logs" / f"{ruta_data.stem}_ultralytics_resuelto.yaml"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(yaml.safe_dump(datos, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logging.info("YAML normalizado para Ultralytics: %s", ruta_salida)
    return ruta_salida


def resolver_directorio_proyecto(project: str) -> str:
    """Resuelve el directorio de salida para evitar `runs/detect` inesperados."""

    ruta_project = Path(project)
    if not ruta_project.is_absolute():
        ruta_project = REPO_ROOT / ruta_project
    return str(ruta_project.resolve())


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
    ruta_data_entrenamiento = preparar_yaml_para_ultralytics(ruta_data)

    parametros = {
        "data": str(ruta_data_entrenamiento),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": resolver_directorio_proyecto(args.project),
        "name": args.name,
    }
    if args.device is not None:
        parametros["device"] = args.device

    logging.info("Iniciando entrenamiento YOLO con data=%s", ruta_data_entrenamiento)
    resultados = modelo.train(**parametros)
    save_dir = getattr(resultados, "save_dir", None)
    logging.info("Entrenamiento finalizado. Salida: %s", save_dir or args.project)
    return 0


def main() -> int:
    """Punto de entrada del script."""

    return entrenar(crear_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
