#!/usr/bin/env python3
"""Ejecuta detección y clasificación sobre vídeo o webcam."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia sobre vídeo o webcam.")
    parser.add_argument("--source", required=True, help="Ruta de vídeo o índice de webcam, por ejemplo 0.")
    parser.add_argument("--config", default="config/default.yaml", help="Ruta a la configuración YAML.")
    parser.add_argument("--classes", default="config/classes.yaml", help="Ruta a classes.yaml.")
    parser.add_argument("--yolo-model", default=None, help="Ruta opcional al modelo YOLO.")
    parser.add_argument("--classifier-model", default=None, help="Ruta opcional al modelo CNN.")
    parser.add_argument("--output-dir", default=None, help="Directorio base para frames anotados, crops y JSON.")
    parser.add_argument("--output", default=None, help="Ruta opcional para guardar vídeo anotado.")
    parser.add_argument("--show", action="store_true", help="Muestra los frames anotados en una ventana.")
    parser.add_argument("--frame-step", type=int, default=1, help="Procesa uno de cada N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite opcional de frames procesados.")
    parser.add_argument("--save-crops", action="store_true", help="Guarda crops contextuales por detección.")
    parser.add_argument("--save-json", action="store_true", help="Guarda JSON por frame procesado en --output-dir/json.")
    parser.add_argument("--image-size", type=int, default=None, help="Tamaño de entrada usado por la CNN.")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Umbral de confianza YOLO.")
    parser.add_argument(
        "--classifier-context-padding",
        type=float,
        default=None,
        help="Padding contextual para el crop que recibe la CNN.",
    )
    parser.add_argument(
        "--classifier-square-crop",
        dest="classifier_square_crop",
        action="store_true",
        default=None,
        help="Activa crop contextual cuadrado para la CNN.",
    )
    parser.add_argument(
        "--no-classifier-square-crop",
        dest="classifier_square_crop",
        action="store_false",
        help="Desactiva crop contextual cuadrado para la CNN.",
    )
    return parser


def _parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def main() -> int:
    args = crear_parser().parse_args()
    try:
        import cv2
    except ImportError:
        print("ERROR: No se puede importar OpenCV. Instala requirements.txt.", file=sys.stderr)
        return 2

    config = cargar_configuracion(args.config, args.classes)
    cambios_modelos = {}
    cambios_outputs = {}
    cambios_inferencia = {}
    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.yolo_model is not None:
        cambios_modelos["yolo"] = Path(args.yolo_model)
    if args.classifier_model is not None:
        cambios_modelos["cnn"] = Path(args.classifier_model)
    if output_dir is not None:
        cambios_outputs["detections"] = output_dir / "annotated"
        cambios_outputs["crops"] = output_dir / "crops"
        cambios_outputs["reports"] = output_dir / "reports"
    if args.image_size is not None:
        if args.image_size <= 0:
            print("ERROR: --image-size debe ser mayor que 0.", file=sys.stderr)
            return 2
        cambios_inferencia["cnn_image_size"] = args.image_size
    if args.confidence_threshold is not None:
        if args.confidence_threshold < 0:
            print("ERROR: --confidence-threshold no puede ser negativo.", file=sys.stderr)
            return 2
        cambios_inferencia["detection_confidence_threshold"] = args.confidence_threshold
    if args.classifier_context_padding is not None:
        if args.classifier_context_padding < 0:
            print("ERROR: --classifier-context-padding no puede ser negativo.", file=sys.stderr)
            return 2
        cambios_inferencia["classifier_context_padding"] = args.classifier_context_padding
    if args.classifier_square_crop is not None:
        cambios_inferencia["classifier_square_crop"] = args.classifier_square_crop
    if cambios_modelos:
        config = replace(config, modelos=replace(config.modelos, **cambios_modelos))
    if cambios_outputs:
        config = replace(config, outputs=replace(config.outputs, **cambios_outputs))
    if cambios_inferencia:
        config = replace(config, inferencia=replace(config.inferencia, **cambios_inferencia))
    if args.save_json and output_dir is None:
        print("ERROR: --save-json requiere --output-dir para vídeo.", file=sys.stderr)
        return 2

    pipeline = InspectionPipeline(config)

    cap = cv2.VideoCapture(_parse_source(args.source))
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la fuente de vídeo: {args.source}", file=sys.stderr)
        return 2

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (ancho, alto))

    procesados = 0
    leidos = 0
    with tempfile.TemporaryDirectory(prefix="extinguisher_frames_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            leidos += 1
            if leidos % max(args.frame_step, 1) != 0:
                continue

            frame_path = tmpdir_path / f"frame_{leidos:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            resultado = pipeline.inspeccionar_imagen(frame_path, guardar_crops=args.save_crops, guardar_anotada=True)
            if args.save_json and output_dir is not None:
                json_dir = output_dir / "json"
                json_dir.mkdir(parents=True, exist_ok=True)
                ruta_json = json_dir / f"{frame_path.stem}.json"
                ruta_json.write_text(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

            frame_salida = frame
            if resultado.annotated_image_path:
                anotada = cv2.imread(resultado.annotated_image_path)
                if anotada is not None:
                    frame_salida = anotada

            if writer is not None:
                writer.write(frame_salida)
            if args.show:
                cv2.imshow("Inspección de extintores", frame_salida)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            procesados += 1
            if args.max_frames is not None and procesados >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Frames leídos: {leidos}. Frames procesados: {procesados}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
