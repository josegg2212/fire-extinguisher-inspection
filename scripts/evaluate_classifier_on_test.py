#!/usr/bin/env python3
"""Evalúa de forma preliminar la CNN sobre el split test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.classification.cnn_model import construir_modelo
from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO, EXTENSIONES_IMAGEN


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa la CNN en data/classifier/test.")
    parser.add_argument("--dataset-dir", default="data/classifier/test", help="Directorio del split test.")
    parser.add_argument(
        "--model-path",
        default="models/classifier/extinguisher_status_cnn_test.pth",
        help="Checkpoint CNN a evaluar.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de entrada de la CNN.")
    parser.add_argument(
        "--output-dir",
        default="outputs/reports/classifier_test_preliminary",
        help="Directorio de métricas de salida.",
    )
    parser.add_argument("--device", default=None, help="Dispositivo opcional: cpu, cuda o cuda:0.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch para evaluación.")
    return parser


def _resolver(path: str | Path) -> Path:
    ruta = Path(path)
    if ruta.is_absolute():
        return ruta
    return (REPO_ROOT / ruta).resolve()


def _recoger_muestras(dataset_dir: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    muestras: list[tuple[Path, int]] = []
    for indice, clase in enumerate(class_names):
        ruta_clase = dataset_dir / clase
        if not ruta_clase.exists():
            raise RuntimeError(f"Falta la carpeta de clase: {ruta_clase}")
        imagenes = sorted(
            ruta for ruta in ruta_clase.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
        )
        if not imagenes:
            raise RuntimeError(f"No hay imágenes para la clase {clase}: {ruta_clase}")
        muestras.extend((ruta, indice) for ruta in imagenes)
    return muestras


def _crear_dataset(muestras: list[tuple[Path, int]], image_size: int) -> Any:
    try:
        from PIL import Image
        from torch.utils.data import Dataset
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("No se pueden importar Pillow/torchvision. Usa el Docker del proyecto.") from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class DatasetClasificador(Dataset):
        """Dataset mínimo con orden de clases fijo."""

        def __len__(self) -> int:
            return len(muestras)

        def __getitem__(self, indice: int) -> tuple[Any, int, str]:
            ruta, etiqueta = muestras[indice]
            with Image.open(ruta) as imagen:
                tensor = transform(imagen.convert("RGB"))
            return tensor, etiqueta, str(ruta)

    return DatasetClasificador()


def _resolver_dispositivo(device_arg: str | None) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("No se puede importar PyTorch. Usa el Docker del proyecto.") from exc
    return torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))


def _cargar_modelo(model_path: Path, device: Any, fallback_classes: list[str]) -> tuple[Any, list[str]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("No se puede importar PyTorch. Usa el Docker del proyecto.") from exc

    if not model_path.exists():
        raise RuntimeError(f"No existe el modelo CNN: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    class_names = list(checkpoint.get("class_names", fallback_classes))
    arquitectura = str(checkpoint.get("architecture", "simple_cnn"))
    modelo = construir_modelo(num_classes=len(class_names), arquitectura=arquitectura).to(device)
    modelo.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    modelo.eval()
    return modelo, class_names


def _calcular_metricas(confusion: list[list[int]], class_names: list[str]) -> dict[str, Any]:
    total = sum(sum(fila) for fila in confusion)
    correctos = sum(confusion[i][i] for i in range(len(class_names)))
    metricas_clase: dict[str, dict[str, float | int]] = {}

    for indice, clase in enumerate(class_names):
        tp = confusion[indice][indice]
        fp = sum(confusion[fila][indice] for fila in range(len(class_names)) if fila != indice)
        fn = sum(confusion[indice][columna] for columna in range(len(class_names)) if columna != indice)
        soporte = sum(confusion[indice])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metricas_clase[clase] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": soporte,
        }

    return {
        "accuracy": correctos / total if total else 0.0,
        "total_samples": total,
        "per_class": metricas_clase,
        "confusion_matrix": confusion,
    }


def _guardar_confusion_csv(confusion: list[list[int]], class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["true\\pred", *class_names])
        for clase, fila in zip(class_names, confusion):
            writer.writerow([clase, *fila])


def evaluar(args: argparse.Namespace) -> int:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: No se puede importar PyTorch. Usa el Docker del proyecto.")
        return 2

    try:
        dataset_dir = _resolver(args.dataset_dir)
        model_path = _resolver(args.model_path)
        output_dir = _resolver(args.output_dir)
        device = _resolver_dispositivo(args.device)
        modelo, class_names = _cargar_modelo(model_path, device, CLASES_ESTADO)
        muestras = _recoger_muestras(dataset_dir, class_names)
        dataset = _crear_dataset(muestras, args.image_size)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    confusion = [[0 for _ in class_names] for _ in class_names]
    soportes = Counter(etiqueta for _, etiqueta in muestras)
    errores: list[dict[str, Any]] = []

    with torch.no_grad():
        for imagenes, etiquetas, rutas in loader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)
            logits = modelo(imagenes)
            probabilidades = torch.softmax(logits, dim=1)
            predicciones = probabilidades.argmax(dim=1)
            confianzas = probabilidades.max(dim=1).values

            for etiqueta, prediccion, confianza, ruta in zip(
                etiquetas.cpu().tolist(),
                predicciones.cpu().tolist(),
                confianzas.cpu().tolist(),
                rutas,
            ):
                confusion[int(etiqueta)][int(prediccion)] += 1
                if etiqueta != prediccion and len(errores) < 50:
                    errores.append(
                        {
                            "image_path": ruta,
                            "true_class": class_names[int(etiqueta)],
                            "predicted_class": class_names[int(prediccion)],
                            "confidence": float(confianza),
                        }
                    )

    metricas = _calcular_metricas(confusion, class_names)
    metricas.update(
        {
            "tipo": "evaluacion_preliminar_cnn_test",
            "dataset_dir": str(dataset_dir),
            "model_path": str(model_path),
            "device": str(device),
            "image_size": args.image_size,
            "samples_per_class": {class_names[indice]: soportes[indice] for indice in range(len(class_names))},
            "misclassified_examples": errores,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    errors_path = output_dir / "misclassified_examples.json"
    metrics_path.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    _guardar_confusion_csv(confusion, class_names, confusion_path)
    errors_path.write_text(json.dumps(errores, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Accuracy global: {metricas['accuracy']:.6f}")
    for clase in class_names:
        datos = metricas["per_class"][clase]
        print(
            f"- {clase}: precision={datos['precision']:.6f} "
            f"recall={datos['recall']:.6f} f1={datos['f1']:.6f} support={datos['support']}"
        )
    print(f"Métricas JSON: {metrics_path}")
    print(f"Matriz de confusión: {confusion_path}")
    print(f"Errores de ejemplo: {errors_path}")
    return 0


def main() -> int:
    return evaluar(crear_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
