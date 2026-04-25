"""Modelo CNN base para clasificar el estado visual de un extintor."""

from __future__ import annotations

from typing import Any


def _importar_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "No se puede importar PyTorch. Instala las dependencias con "
            "'pip install -r requirements.txt'."
        ) from exc
    return torch, nn


def construir_cnn_simple(num_classes: int = 3, dropout: float = 0.25) -> Any:
    """Construye una CNN pequeña, clara y suficiente como baseline."""

    _, nn = _importar_torch()

    class CNNEstadoExtintor(nn.Module):
        """CNN baseline para clasificación de crops de extintores."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: Any) -> Any:
            """Propagación hacia delante."""

            return self.classifier(self.features(x))

    return CNNEstadoExtintor()


def construir_modelo(num_classes: int = 3, arquitectura: str = "simple_cnn") -> Any:
    """Construye el modelo solicitado.

    Por ahora la arquitectura soportada y recomendada es `simple_cnn`.
    Más adelante se puede añadir transfer learning con torchvision sin cambiar
    el resto del pipeline.
    """

    if arquitectura != "simple_cnn":
        raise ValueError(
            f"Arquitectura no soportada todavía: {arquitectura}. "
            "Usa 'simple_cnn' para la primera versión."
        )
    return construir_cnn_simple(num_classes=num_classes)
