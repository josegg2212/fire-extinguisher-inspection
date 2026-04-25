"""Validación y carga del dataset de clasificación de estado."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CLASES_ESTADO = ["visible", "partially_occluded", "blocked"]
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ResultadoValidacionDataset:
    """Resultado de una comprobación de estructura de dataset."""

    es_valido: bool
    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    conteos: dict[str, dict[str, int]] = field(default_factory=dict)

    def imprimir(self) -> None:
        """Imprime un resumen legible por consola."""

        for error in self.errores:
            print(f"ERROR: {error}")
        for advertencia in self.advertencias:
            print(f"AVISO: {advertencia}")
        for split, conteo_split in self.conteos.items():
            detalle = ", ".join(f"{clase}={total}" for clase, total in conteo_split.items())
            print(f"{split}: {detalle}")


def contar_imagenes(carpeta: Path) -> int:
    """Cuenta imágenes por extensión conocida dentro de una carpeta."""

    if not carpeta.exists():
        return 0
    return sum(1 for ruta in carpeta.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN)


def validar_dataset_clasificador(
    dataset_path: str | Path,
    clases: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    requerir_imagenes: bool = False,
) -> ResultadoValidacionDataset:
    """Valida carpetas tipo ImageFolder sin exigir que ya exista un dataset real."""

    ruta_dataset = Path(dataset_path)
    clases = clases or CLASES_ESTADO
    resultado = ResultadoValidacionDataset(es_valido=True)

    if not ruta_dataset.exists():
        resultado.es_valido = False
        resultado.errores.append(f"No existe el directorio del dataset: {ruta_dataset}")
        return resultado

    for split in splits:
        ruta_split = ruta_dataset / split
        resultado.conteos[split] = {}
        if not ruta_split.exists():
            resultado.es_valido = False
            resultado.errores.append(f"Falta el split '{split}': {ruta_split}")
            continue

        for clase in clases:
            ruta_clase = ruta_split / clase
            if not ruta_clase.exists():
                resultado.es_valido = False
                resultado.errores.append(f"Falta la carpeta de clase: {ruta_clase}")
                resultado.conteos[split][clase] = 0
                continue

            total = contar_imagenes(ruta_clase)
            resultado.conteos[split][clase] = total
            if total == 0:
                mensaje = f"No hay imágenes en {ruta_clase}"
                if requerir_imagenes:
                    resultado.es_valido = False
                    resultado.errores.append(mensaje)
                else:
                    resultado.advertencias.append(mensaje)

    return resultado


def crear_dataloaders(
    dataset_path: str | Path,
    image_size: int,
    batch_size: int,
    clases: list[str] | None = None,
    num_workers: int = 0,
) -> tuple[Any, Any, list[str]]:
    """Crea dataloaders de train y val preservando el orden de clases configurado."""

    clases = clases or CLASES_ESTADO
    validacion = validar_dataset_clasificador(
        dataset_path,
        clases=clases,
        splits=("train", "val"),
        requerir_imagenes=True,
    )
    if not validacion.es_valido:
        detalle = "; ".join(validacion.errores)
        raise ValueError(f"El dataset de clasificación no está listo para entrenar: {detalle}")

    try:
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise RuntimeError(
            "No se pueden importar torch/torchvision. Instala requirements.txt antes de entrenar."
        ) from exc

    class ImageFolderConOrden(ImageFolder):
        """ImageFolder con orden de clases fijado por configuración."""

        def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
            class_to_idx = {nombre: indice for indice, nombre in enumerate(clases)}
            return list(clases), class_to_idx

    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ruta_dataset = Path(dataset_path)
    train_dataset = ImageFolderConOrden(str(ruta_dataset / "train"), transform=transform_train)
    val_dataset = ImageFolderConOrden(str(ruta_dataset / "val"), transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, list(clases)
