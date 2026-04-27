#!/usr/bin/env python3
"""Evaluación preliminar del pipeline completo YOLO + CNN."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.classification.dataset import CLASES_ESTADO
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline
from fire_extinguisher_inspection.pipeline.result_schema import InspectionResult


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta una evaluación preliminar del pipeline completo.")
    parser.add_argument("--images-dir", default="data/yolo/test/images", help="Directorio con imágenes completas.")
    parser.add_argument(
        "--yolo-model",
        default="models/yolo/extinguisher_yolo_test_gpu/weights/best.pt",
        help="Modelo YOLO de prueba.",
    )
    parser.add_argument(
        "--classifier-model",
        default="models/classifier/extinguisher_status_cnn_test.pth",
        help="Checkpoint CNN de prueba.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/detections/preliminary_pipeline",
        help="Directorio de salidas del pipeline.",
    )
    parser.add_argument("--max-images", type=int, default=30, help="Máximo de imágenes a procesar.")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Máximo de imágenes por clase si --images-dir contiene subcarpetas de clase.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25, help="Umbral de detección YOLO.")
    parser.add_argument("--save-crops", action="store_true", help="Guarda crops de las detecciones.")
    parser.add_argument("--save-json", action="store_true", help="Guarda JSON por imagen.")
    parser.add_argument("--image-size", type=int, default=224, help="Tamaño esperado por la CNN.")
    parser.add_argument(
        "--classifier-context-padding",
        type=float,
        default=0.75,
        help="Padding contextual para el crop que recibe la CNN.",
    )
    parser.add_argument(
        "--classifier-square-crop",
        dest="classifier_square_crop",
        action="store_true",
        default=True,
        help="Usa crop contextual cuadrado para la CNN.",
    )
    parser.add_argument(
        "--no-classifier-square-crop",
        dest="classifier_square_crop",
        action="store_false",
        help="Desactiva el crop contextual cuadrado para la CNN.",
    )
    parser.add_argument(
        "--contact-sheet-output",
        default="outputs/reports/preliminary_pipeline_contact_sheet.jpg",
        help="Ruta de salida de la contact sheet.",
    )
    return parser


def _resolver(path: str | Path) -> Path:
    ruta = Path(path)
    if ruta.is_absolute():
        return ruta
    return (REPO_ROOT / ruta).resolve()


def _validar_entradas(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    images_dir = _resolver(args.images_dir)
    yolo_model = _resolver(args.yolo_model)
    classifier_model = _resolver(args.classifier_model)
    output_dir = _resolver(args.output_dir)

    faltantes = []
    if not images_dir.exists():
        faltantes.append(f"directorio de imágenes: {images_dir}")
    if not yolo_model.exists():
        faltantes.append(f"modelo YOLO: {yolo_model}")
    if not classifier_model.exists():
        faltantes.append(f"modelo CNN: {classifier_model}")
    if faltantes:
        detalle = "\n- ".join(faltantes)
        raise RuntimeError(f"Faltan entradas para la evaluación preliminar:\n- {detalle}")
    if args.max_images <= 0:
        raise RuntimeError("--max-images debe ser mayor que 0.")
    if args.max_images_per_class is not None and args.max_images_per_class <= 0:
        raise RuntimeError("--max-images-per-class debe ser mayor que 0.")
    if args.image_size <= 0:
        raise RuntimeError("--image-size debe ser mayor que 0.")
    if args.classifier_context_padding < 0:
        raise RuntimeError("--classifier-context-padding no puede ser negativo.")
    return images_dir, yolo_model, classifier_model, output_dir


def _iterar_imagenes(images_dir: Path, max_images: int, max_images_per_class: int | None) -> list[Path]:
    if max_images_per_class is not None:
        seleccionadas: list[Path] = []
        for clase in CLASES_ESTADO:
            ruta_clase = images_dir / clase
            if not ruta_clase.exists():
                continue
            imagenes_clase = sorted(
                ruta for ruta in ruta_clase.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
            )
            seleccionadas.extend(imagenes_clase[:max_images_per_class])
        return seleccionadas[:max_images]

    imagenes = sorted(
        ruta for ruta in images_dir.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    )
    return imagenes[:max_images]


def _clase_esperada(images_dir: Path, image_path: Path) -> str | None:
    """Obtiene la clase desde la subcarpeta padre si existe."""

    try:
        relativa = image_path.relative_to(images_dir)
    except ValueError:
        return None
    if len(relativa.parts) >= 2 and relativa.parts[0] in CLASES_ESTADO:
        return relativa.parts[0]
    return None


def _crear_config_pipeline(
    yolo_model: Path,
    classifier_model: Path,
    output_dir: Path,
    confidence_threshold: float,
    image_size: int,
    classifier_context_padding: float,
    classifier_square_crop: bool,
) -> Any:
    config = cargar_configuracion()
    modelos = replace(config.modelos, yolo=yolo_model, cnn=classifier_model)
    outputs = replace(
        config.outputs,
        detections=output_dir / "annotated",
        crops=output_dir / "crops",
        reports=REPO_ROOT / "outputs" / "reports",
    )
    inferencia = replace(
        config.inferencia,
        detection_confidence_threshold=confidence_threshold,
        cnn_image_size=image_size,
        classifier_context_padding=classifier_context_padding,
        classifier_square_crop=classifier_square_crop,
        guardar_anotada=True,
    )
    return replace(config, modelos=modelos, outputs=outputs, inferencia=inferencia)


def _guardar_json(resultado: InspectionResult, output_dir: Path) -> Path:
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    ruta = json_dir / f"{Path(resultado.image_path).stem}.json"
    ruta.write_text(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def _generar_contact_sheet(resultados: list[InspectionResult], output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc

    if not resultados:
        return

    tile_size = 320
    header = 42
    columnas = min(5, max(1, len(resultados)))
    filas = (len(resultados) + columnas - 1) // columnas
    hoja = Image.new("RGB", (columnas * tile_size, filas * (tile_size + header)), (226, 232, 240))
    font = ImageFont.load_default()

    for indice, resultado in enumerate(resultados):
        ruta_imagen = Path(resultado.annotated_image_path or resultado.image_path)
        x = (indice % columnas) * tile_size
        y = (indice // columnas) * (tile_size + header)
        tile = Image.new("RGB", (tile_size, tile_size + header), (241, 245, 249))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_size, header), fill=(31, 41, 55))
        texto = f"{Path(resultado.image_path).name[:24]} | det={len(resultado.detections)}"
        if resultado.errors:
            texto += " | ERROR"
        draw.text((8, 12), texto, fill=(255, 255, 255), font=font)
        try:
            with Image.open(ruta_imagen) as imagen:
                preparada = ImageOps.fit(imagen.convert("RGB"), (tile_size, tile_size))
        except Exception:
            preparada = Image.new("RGB", (tile_size, tile_size), (254, 226, 226))
            ImageDraw.Draw(preparada).text((8, 8), "no legible", fill=(127, 29, 29), font=font)
        tile.paste(preparada, (0, header))
        hoja.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(output_path, quality=92)


def _crear_resumen(
    resultados: list[InspectionResult],
    json_paths: list[Path],
    output_dir: Path,
    contact_sheet_path: Path,
    args: argparse.Namespace,
    expected_by_image: dict[str, str | None],
) -> dict[str, Any]:
    estados = Counter()
    esperadas = Counter()
    detectadas_por_clase = Counter()
    sin_deteccion_por_clase = Counter()
    confusion_estado = {clase: Counter() for clase in CLASES_ESTADO}
    total_detecciones = 0
    imagenes_con_errores = 0
    imagenes_sin_detecciones = 0

    for resultado in resultados:
        clase_esperada = expected_by_image.get(resultado.image_path)
        if clase_esperada is not None:
            esperadas[clase_esperada] += 1
        if resultado.errors:
            imagenes_con_errores += 1
        if not resultado.detections:
            imagenes_sin_detecciones += 1
            if clase_esperada is not None:
                sin_deteccion_por_clase[clase_esperada] += 1
        elif clase_esperada is not None:
            detectadas_por_clase[clase_esperada] += 1
            mejor = max(resultado.detections, key=lambda deteccion: deteccion.detection_confidence)
            confusion_estado[clase_esperada][mejor.status_prediction or "sin_estado"] += 1
        total_detecciones += len(resultado.detections)
        for deteccion in resultado.detections:
            estados[deteccion.status_prediction or "sin_estado"] += 1

    total_con_esperada = sum(sum(fila.values()) for fila in confusion_estado.values())
    aciertos_estado = sum(confusion_estado[clase][clase] for clase in CLASES_ESTADO)
    return {
        "tipo": "evaluacion_preliminar_pipeline",
        "modelos_de_prueba": {
            "yolo": args.yolo_model,
            "classifier": args.classifier_model,
        },
        "images_dir": args.images_dir,
        "output_dir": str(output_dir),
        "classifier_context_padding": args.classifier_context_padding,
        "classifier_square_crop": args.classifier_square_crop,
        "imagenes_procesadas": len(resultados),
        "imagenes_con_errores": imagenes_con_errores,
        "imagenes_sin_detecciones": imagenes_sin_detecciones,
        "detecciones_totales": total_detecciones,
        "detecciones_por_estado": dict(estados),
        "clases_esperadas": dict(esperadas),
        "imagenes_con_deteccion_por_clase_esperada": dict(detectadas_por_clase),
        "imagenes_sin_deteccion_por_clase_esperada": dict(sin_deteccion_por_clase),
        "matriz_estado_primer_match": {
            clase: dict(confusion_estado[clase]) for clase in CLASES_ESTADO if confusion_estado[clase]
        },
        "accuracy_estado_sobre_imagenes_detectadas": (
            aciertos_estado / total_con_esperada if total_con_esperada else None
        ),
        "json_generados": [str(path) for path in json_paths],
        "contact_sheet": str(contact_sheet_path),
    }


def main() -> int:
    args = crear_parser().parse_args()
    try:
        images_dir, yolo_model, classifier_model, output_dir = _validar_entradas(args)
        imagenes = _iterar_imagenes(images_dir, args.max_images, args.max_images_per_class)
        if not imagenes:
            print(f"ERROR: no se encontraron imágenes en {images_dir}")
            return 2

        output_dir.mkdir(parents=True, exist_ok=True)
        config = _crear_config_pipeline(
            yolo_model=yolo_model,
            classifier_model=classifier_model,
            output_dir=output_dir,
            confidence_threshold=args.confidence_threshold,
            image_size=args.image_size,
            classifier_context_padding=args.classifier_context_padding,
            classifier_square_crop=args.classifier_square_crop,
        )
        pipeline = InspectionPipeline(config)

        resultados: list[InspectionResult] = []
        json_paths: list[Path] = []
        expected_by_image: dict[str, str | None] = {}
        for imagen in imagenes:
            resultado = pipeline.inspeccionar_imagen(
                imagen,
                guardar_crops=args.save_crops,
                guardar_anotada=True,
            )
            resultados.append(resultado)
            expected_by_image[resultado.image_path] = _clase_esperada(images_dir, imagen)
            if args.save_json:
                json_paths.append(_guardar_json(resultado, output_dir))

        contact_sheet_path = _resolver(args.contact_sheet_output)
        _generar_contact_sheet(resultados, contact_sheet_path)
        resumen = _crear_resumen(resultados, json_paths, output_dir, contact_sheet_path, args, expected_by_image)
        resumen_path = output_dir / "preliminary_summary.json"
        resumen_path.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Imágenes procesadas: {resumen['imagenes_procesadas']}")
    print(f"Detecciones totales: {resumen['detecciones_totales']}")
    print(f"Imágenes sin detecciones: {resumen['imagenes_sin_detecciones']}")
    print("Detecciones por estado:")
    for estado, total in sorted(resumen["detecciones_por_estado"].items()):
        print(f"- {estado}: {total}")
    if resumen["clases_esperadas"]:
        print("Clases esperadas:")
        for clase, total in sorted(resumen["clases_esperadas"].items()):
            print(f"- {clase}: {total}")
        print(f"Accuracy de estado sobre imágenes con detección: {resumen['accuracy_estado_sobre_imagenes_detectadas']}")
    print(f"Resumen JSON: {resumen_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
