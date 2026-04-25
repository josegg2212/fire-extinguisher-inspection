#!/usr/bin/env python3
"""Ejecuta detección y clasificación sobre vídeo o webcam."""

from __future__ import annotations

import argparse
import sys
import tempfile
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
    parser.add_argument("--output", default=None, help="Ruta opcional para guardar vídeo anotado.")
    parser.add_argument("--show", action="store_true", help="Muestra los frames anotados en una ventana.")
    parser.add_argument("--frame-step", type=int, default=1, help="Procesa uno de cada N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite opcional de frames procesados.")
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
            resultado = pipeline.inspeccionar_imagen(frame_path, guardar_crops=False, guardar_anotada=True)

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
