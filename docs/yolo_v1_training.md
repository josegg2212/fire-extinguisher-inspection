# Entrenamiento YOLO v1

Fecha: 2026-04-27

## Objetivo

Entrenar el primer detector YOLO serio para localizar extintores. Esta fase no entrena la CNN y no modifica los datasets de clasificacion.

Este resultado se considera el detector YOLO v1 del proyecto, no el resultado final absoluto del sistema completo.

## Dataset

Dataset usado: `data/yolo/data.yaml`

Clase: `fire_extinguisher`

Conteos validados con `scripts/check_dataset_structure.py`:

| Split | Imagenes | Labels | Anotaciones |
| --- | ---: | ---: | ---: |
| train | 3791 | 3791 | 7355 |
| valid | 837 | 837 | 1566 |
| test | 397 | 397 | 763 |

La validacion no encontro errores criticos. En `train` hay 12 labels vacios ya conocidos; YOLO los admite como posibles imagenes negativas/background y no bloquearon el entrenamiento.

Comando de validacion:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/check_dataset_structure.py \
  --tipo yolo \
  --path data/yolo/data.yaml
```

## Modelo y entorno

Modelo base usado realmente: `yolo26n.pt`

No fue necesario usar el fallback `yolo11n.pt`.

Entorno usado:

- Docker GPU: servicio `app-gpu`
- GPU: NVIDIA GeForce GTX 1650
- PyTorch: `2.8.0+cu126`
- Ultralytics: `8.4.41`
- CUDA disponible: si

Ultralytics desactivo AMP automaticamente durante el arranque en esta GPU; el entrenamiento continuo correctamente en CUDA.

## Entrenamiento

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_v1
```

Hiperparametros principales:

| Parametro | Valor |
| --- | --- |
| epocas | 50 |
| imgsz | 640 |
| batch | 16 |
| workers | 0 |
| device | CUDA automatico |

No hubo problemas de memoria con `batch=16`, asi que no fue necesario reducir a `batch=8`. `workers=0` se mantuvo para evitar problemas de multiproceso o `/dev/shm` en Docker.

Pesos generados localmente:

- `models/yolo/extinguisher_yolo_v1/weights/best.pt`
- `models/yolo/extinguisher_yolo_v1/weights/last.pt`

Estos pesos son artefactos locales y no se versionan en Git.

## Metricas de validacion durante entrenamiento

Mejor epoca por `metrics/mAP50-95(B)`: epoca 49.

| Epoca | Precision(B) | Recall(B) | mAP50(B) | mAP50-95(B) |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 0.97058 | 0.91699 | 0.97706 | 0.86173 |
| 50 | 0.95835 | 0.92976 | 0.97730 | 0.86163 |

Archivos de entrenamiento generados por Ultralytics:

- `models/yolo/extinguisher_yolo_v1/results.csv`
- `models/yolo/extinguisher_yolo_v1/results.png`
- `models/yolo/extinguisher_yolo_v1/confusion_matrix.png`
- `models/yolo/extinguisher_yolo_v1/confusion_matrix_normalized.png`
- `models/yolo/extinguisher_yolo_v1/BoxPR_curve.png`
- `models/yolo/extinguisher_yolo_v1/BoxF1_curve.png`

## Evaluacion en test

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluate_yolo_on_test.py \
  --model models/yolo/extinguisher_yolo_v1/weights/best.pt \
  --data data/yolo/data.yaml \
  --split test \
  --imgsz 640 \
  --output-dir outputs/reports/yolo_v1_test \
  --device 0 \
  --batch 8 \
  --workers 0
```

Metricas reales en `test`:

| Metrica | Valor |
| --- | ---: |
| precision(B) | 0.9455517560255178 |
| recall(B) | 0.9187418086500655 |
| mAP50(B) | 0.9737560421353847 |
| mAP75(B) | 0.8939457956072638 |
| mAP50-95(B) | 0.838757968330097 |

Ruta del informe local:

- `outputs/reports/yolo_v1_test/metrics.json`
- `outputs/reports/yolo_v1_test/val/`

## Ejemplos visuales

Se generaron 30 inferencias sobre imagenes de `data/yolo/test/images` con el modelo `best.pt`.

Rutas locales:

- `outputs/detections/yolo_v1_examples/`
- `outputs/reports/yolo_v1_test_contact_sheet.jpg`

Observaciones visuales:

- Las muestras revisadas muestran detecciones sobre extintores con confianzas altas, habitualmente entre 0.91 y 0.98.
- El detector cubre ejemplos verticales, inclinados, parcialmente recortados y escenas con dos extintores.
- Algunas cajas incluyen parte del soporte o la manguera, algo aceptable para esta version del detector.
- En la contact sheet no se observaron falsos positivos evidentes, aunque la revision visual de 30 muestras no sustituye la evaluacion cuantitativa.

## Problemas y decisiones

- `yolo26n.pt` estuvo disponible y entreno correctamente; no se uso `yolo11n.pt`.
- No hubo errores de memoria; se mantuvo `batch=16`.
- Se uso `workers=0` durante entrenamiento y evaluacion.
- AMP fue desactivado automaticamente por Ultralytics en la GTX 1650, sin bloquear el entrenamiento.
- Los labels vacios conocidos del split `train` se trataron como posibles negativos/background y no bloquearon el flujo.

## Conclusion

El entrenamiento termino correctamente y el modelo queda aceptado como detector YOLO v1. El mejor peso local es:

```text
models/yolo/extinguisher_yolo_v1/weights/best.pt
```

El siguiente paso recomendado es probar el pipeline completo usando este `best.pt` junto con la CNN contextual v2.
