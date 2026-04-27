#!/usr/bin/env python3
"""Evaluación preliminar de YOLO sobre el split test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.detection.train_yolo import preparar_yaml_para_ultralytics


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa preliminarmente un modelo YOLO.")
    parser.add_argument(
        "--model",
        default="models/yolo/extinguisher_yolo_test_gpu/weights/best.pt",
        help="Peso YOLO a evaluar.",
    )
    parser.add_argument("--data", default="data/yolo/data.yaml", help="data.yaml del dataset YOLO.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split a evaluar.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen.")
    parser.add_argument(
        "--output-dir",
        default="outputs/reports/yolo_test_preliminary",
        help="Directorio de salidas de evaluación.",
    )
    parser.add_argument("--device", default=None, help="Dispositivo opcional: cpu, 0, cuda:0.")
    parser.add_argument("--batch", type=int, default=8, help="Batch de validación.")
    parser.add_argument("--workers", type=int, default=0, help="Workers del dataloader. 0 evita problemas de /dev/shm.")
    return parser


def _resolver(path: str | Path) -> Path:
    ruta = Path(path)
    if ruta.is_absolute():
        return ruta
    return (REPO_ROOT / ruta).resolve()


def _serializable(valor: Any) -> Any:
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, dict):
        return {str(k): _serializable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_serializable(v) for v in valor]
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except Exception:
            pass
    if hasattr(valor, "tolist"):
        try:
            return valor.tolist()
        except Exception:
            pass
    return str(valor)


def _extraer_metricas(metrics: Any) -> dict[str, Any]:
    datos: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        datos["results_dict"] = _serializable(results_dict)

    box = getattr(metrics, "box", None)
    if box is not None:
        for nombre in ["mp", "mr", "map50", "map75", "map", "maps"]:
            if hasattr(box, nombre):
                datos[f"box_{nombre}"] = _serializable(getattr(box, nombre))
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        datos["save_dir"] = str(save_dir)
    return datos


def main() -> int:
    args = crear_parser().parse_args()
    model_path = _resolver(args.model)
    data_path = _resolver(args.data)
    output_dir = _resolver(args.output_dir)

    if not model_path.exists():
        print(f"ERROR: no existe el modelo YOLO: {model_path}")
        return 2
    if not data_path.exists():
        print(f"ERROR: no existe el data.yaml: {data_path}")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: no se puede importar ultralytics. Usa el Docker del proyecto.")
        return 2

    try:
        data_resuelto = preparar_yaml_para_ultralytics(data_path)
        modelo = YOLO(str(model_path))
        parametros: dict[str, Any] = {
            "data": str(data_resuelto),
            "split": args.split,
            "imgsz": args.imgsz,
            "project": str(output_dir),
            "name": "val",
            "exist_ok": True,
            "batch": args.batch,
            "workers": args.workers,
        }
        if args.device is not None:
            parametros["device"] = args.device
        metrics = modelo.val(**parametros)
        datos = {
            "tipo": "evaluacion_preliminar_yolo_test",
            "model": str(model_path),
            "data": str(data_path),
            "data_resuelto": str(data_resuelto),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "metrics": _extraer_metricas(metrics),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: evaluación YOLO fallida: {exc}")
        return 2

    print(f"Métricas YOLO guardadas en: {metrics_path}")
    print(json.dumps(datos["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
