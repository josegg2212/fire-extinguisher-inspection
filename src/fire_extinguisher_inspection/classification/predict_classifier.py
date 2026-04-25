"""Inferencia del clasificador CNN sobre crops de extintores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fire_extinguisher_inspection.classification.cnn_model import construir_modelo
from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO


class CNNStatePredictor:
    """Carga un checkpoint CNN y predice la clase visual de un crop."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str] | None = None,
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo CNN en {self.model_path}. "
                "Entrena el clasificador o ajusta 'modelos.cnn' en config/default.yaml."
            )

        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "No se pueden importar torch/torchvision. Instala requirements.txt antes de inferir."
            ) from exc

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = list(checkpoint.get("class_names", class_names or CLASES_ESTADO))
        self.image_size = int(checkpoint.get("image_size", image_size))
        arquitectura = str(checkpoint.get("architecture", "simple_cnn"))

        self.model = construir_modelo(num_classes=len(self.class_names), arquitectura=arquitectura).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predecir_imagen(self, image_path: str | Path) -> tuple[str, float, dict[str, float]]:
        """Predice la clase de estado desde una ruta de imagen."""

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

        ruta = Path(image_path)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el crop de entrada: {ruta}")

        with Image.open(ruta) as imagen:
            imagen_rgb = imagen.convert("RGB")
            return self._predecir_pil(imagen_rgb)

    def predecir_array_bgr(self, imagen_bgr: Any) -> tuple[str, float, dict[str, float]]:
        """Predice la clase desde un array BGR de OpenCV."""

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

        imagen_rgb = imagen_bgr[:, :, ::-1]
        imagen_pil = Image.fromarray(imagen_rgb)
        return self._predecir_pil(imagen_pil)

    def _predecir_pil(self, imagen_pil: Any) -> tuple[str, float, dict[str, float]]:
        tensor = self.transform(imagen_pil).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilidades = self.torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        indice = int(max(range(len(probabilidades)), key=lambda idx: probabilidades[idx]))
        probabilidades_por_clase = {
            clase: float(probabilidades[posicion]) for posicion, clase in enumerate(self.class_names)
        }
        return self.class_names[indice], float(probabilidades[indice]), probabilidades_por_clase


def predecir_crop(
    image_path: str | Path,
    model_path: str | Path,
    class_names: list[str] | None = None,
    image_size: int = 224,
    device: str | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Función de conveniencia para inferir sobre un crop."""

    predictor = CNNStatePredictor(
        model_path=model_path,
        class_names=class_names,
        image_size=image_size,
        device=device,
    )
    return predictor.predecir_imagen(image_path)


def crear_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos para inferencia sobre un crop."""

    parser = argparse.ArgumentParser(description="Clasifica el estado visual de un crop de extintor.")
    parser.add_argument("--image", required=True, help="Ruta al crop de entrada.")
    parser.add_argument("--model-path", required=True, help="Ruta al checkpoint CNN.")
    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de entrada usado por la CNN.")
    parser.add_argument("--device", default=None, help="Dispositivo opcional: cpu, cuda o cuda:0.")
    return parser


def main() -> int:
    """Punto de entrada CLI para predicción de un crop."""

    args = crear_parser().parse_args()
    try:
        clase, confianza, probabilidades = predecir_crop(
            image_path=args.image,
            model_path=args.model_path,
            image_size=args.image_size,
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(
        json.dumps(
            {
                "image_path": args.image,
                "status_prediction": clase,
                "status_confidence": confianza,
                "probabilities": probabilidades,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
