"""Entrenamiento de la CNN de clasificación de estado visual."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from fire_extinguisher_inspection.classification.cnn_model import construir_modelo
from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO, crear_dataloaders


def crear_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos para entrenamiento."""

    parser = argparse.ArgumentParser(description="Entrena la CNN de estado de extintores.")
    parser.add_argument("--dataset-path", required=True, help="Ruta al dataset con train/val/test.")
    parser.add_argument("--epochs", type=int, default=30, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de entrada de la CNN.")
    parser.add_argument(
        "--output-model-path",
        default="models/classifier/state_classifier.pt",
        help="Ruta donde guardar el mejor modelo.",
    )
    parser.add_argument("--device", default=None, help="Dispositivo opcional: cpu, cuda o cuda:0.")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers para dataloaders.")
    return parser


def resolver_dispositivo(device_arg: str | None) -> Any:
    """Elige CPU o GPU automáticamente si no se indica dispositivo."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("No se puede importar PyTorch. Instala requirements.txt.") from exc

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entrenar_epoca(modelo: Any, loader: Any, criterio: Any, optimizador: Any, device: Any) -> tuple[float, float]:
    """Entrena una época y devuelve loss y accuracy."""

    modelo.train()
    total_loss = 0.0
    correctos = 0
    total = 0

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()
        logits = modelo(imagenes)
        loss = criterio(logits, etiquetas)
        loss.backward()
        optimizador.step()

        total_loss += float(loss.item()) * imagenes.size(0)
        predicciones = logits.argmax(dim=1)
        correctos += int((predicciones == etiquetas).sum().item())
        total += int(etiquetas.size(0))

    return total_loss / max(total, 1), correctos / max(total, 1)


def evaluar(modelo: Any, loader: Any, criterio: Any, device: Any) -> tuple[float, float]:
    """Evalúa el modelo y devuelve loss y accuracy."""

    import torch

    modelo.eval()
    total_loss = 0.0
    correctos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)
            logits = modelo(imagenes)
            loss = criterio(logits, etiquetas)

            total_loss += float(loss.item()) * imagenes.size(0)
            predicciones = logits.argmax(dim=1)
            correctos += int((predicciones == etiquetas).sum().item())
            total += int(etiquetas.size(0))

    return total_loss / max(total, 1), correctos / max(total, 1)


def entrenar(args: argparse.Namespace) -> int:
    """Ejecuta el entrenamiento completo."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        import torch
        from torch import nn
    except ImportError:
        logging.error("No se puede importar PyTorch. Instala requirements.txt antes de entrenar.")
        return 2

    try:
        train_loader, val_loader, class_names = crear_dataloaders(
            dataset_path=args.dataset_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            clases=CLASES_ESTADO,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        logging.error("%s", exc)
        return 2

    device = resolver_dispositivo(args.device)
    modelo = construir_modelo(num_classes=len(class_names)).to(device)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.learning_rate)

    output_model_path = Path(args.output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_model_path.with_suffix(".metrics.json")

    mejor_val_acc = -1.0
    historial: list[dict[str, float | int]] = []

    logging.info("Entrenando en %s con clases %s", device, class_names)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = entrenar_epoca(modelo, train_loader, criterio, optimizador, device)
        val_loss, val_acc = evaluar(modelo, val_loader, criterio, device)

        registro = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        historial.append(registro)

        logging.info(
            "Época %03d/%03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": modelo.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "architecture": "simple_cnn",
                    "best_val_accuracy": mejor_val_acc,
                },
                output_model_path,
            )
            logging.info("Nuevo mejor modelo guardado en %s", output_model_path)

    metrics_path.write_text(json.dumps({"history": historial}, indent=2), encoding="utf-8")
    logging.info("Métricas guardadas en %s", metrics_path)
    return 0


def main() -> int:
    """Punto de entrada del script."""

    return entrenar(crear_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
