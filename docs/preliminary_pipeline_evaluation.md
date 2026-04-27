# Evaluacion preliminar del pipeline completo

Fecha: 2026-04-26

## Alcance

Esta evaluacion es preliminar y usa modelos de prueba. No representa metricas finales del proyecto.

Objetivo: comprobar que el flujo YOLO + crop + CNN + visualizacion + JSON funciona de extremo a extremo antes de lanzar entrenamientos largos.

Nota posterior: esta evaluacion preliminar ayudo a detectar que el crop ajustado usado por la CNN perdia contexto. Por eso se creo `data/classifier_context_v2/` y el pipeline se actualizo para clasificar con crops contextuales ampliados.

## Modelos usados

YOLO de prueba:

```text
models/yolo/extinguisher_yolo_test_gpu/weights/best.pt
```

CNN de prueba:

```text
models/classifier/extinguisher_status_cnn_test.pth
```

Ambos modelos cargaron correctamente en Docker `app-gpu`.

## Datasets usados

YOLO test:

```text
data/yolo/test/images
data/yolo/data.yaml
```

CNN test:

```text
data/classifier/test
```

## Evaluacion preliminar CNN en test

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluate_classifier_on_test.py \
  --dataset-dir data/classifier/test \
  --model-path models/classifier/extinguisher_status_cnn_test.pth \
  --image-size 224 \
  --output-dir outputs/reports/classifier_test_preliminary
```

Resultados preliminares:

- Accuracy global: 0.9578853046594982
- Muestras evaluadas: 2232
- Dispositivo: `cuda`

Metricas por clase:

| Clase | precision | recall | F1 | muestras |
| --- | ---: | ---: | ---: | ---: |
| visible | 0.968338 | 0.986559 | 0.977364 | 744 |
| partially_occluded | 0.918919 | 0.959677 | 0.938856 | 744 |
| blocked | 0.989957 | 0.927419 | 0.957668 | 744 |

Matriz de confusion, filas = clase real, columnas = prediccion:

| real \ pred | visible | partially_occluded | blocked |
| --- | ---: | ---: | ---: |
| visible | 734 | 9 | 1 |
| partially_occluded | 24 | 714 | 6 |
| blocked | 0 | 54 | 690 |

Salidas locales:

- `outputs/reports/classifier_test_preliminary/metrics.json`
- `outputs/reports/classifier_test_preliminary/confusion_matrix.csv`
- `outputs/reports/classifier_test_preliminary/misclassified_examples.json`

## Evaluacion preliminar YOLO en test

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluate_yolo_on_test.py \
  --model models/yolo/extinguisher_yolo_test_gpu/weights/best.pt \
  --data data/yolo/data.yaml \
  --split test \
  --imgsz 640 \
  --output-dir outputs/reports/yolo_test_preliminary \
  --device 0 \
  --batch 8 \
  --workers 0
```

Resultados preliminares:

| metrica | valor |
| --- | ---: |
| precision(B) | 0.8905435054400116 |
| recall(B) | 0.8104064457970755 |
| mAP50(B) | 0.9199844597980692 |
| mAP50-95(B) | 0.7167372204936405 |
| mAP75(B) | 0.7899239979249311 |

Salidas locales:

- `outputs/reports/yolo_test_preliminary/metrics.json`
- `outputs/reports/yolo_test_preliminary/val/`

Problema encontrado: el primer intento de validacion YOLO fallo por memoria compartida de Docker (`/dev/shm`) al usar workers del DataLoader. Se corrigio ejecutando la evaluacion con `--workers 0`.

## Pipeline completo preliminar

### Imagenes completas YOLO test

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/run_preliminary_pipeline_evaluation.py \
  --images-dir data/yolo/test/images \
  --yolo-model models/yolo/extinguisher_yolo_test_gpu/weights/best.pt \
  --classifier-model models/classifier/extinguisher_status_cnn_test.pth \
  --output-dir outputs/detections/preliminary_pipeline \
  --max-images 30 \
  --confidence-threshold 0.25 \
  --save-crops \
  --save-json \
  --image-size 224
```

Resumen preliminar:

- Imagenes completas procesadas: 30
- Imagenes con errores: 0
- Imagenes sin detecciones: 1
- Detecciones totales: 31
- Estados predichos por la CNN:
  - `visible`: 31

Salidas locales:

- Imagenes anotadas: `outputs/detections/preliminary_pipeline/annotated/`
- Crops: `outputs/detections/preliminary_pipeline/crops/`
- JSON por imagen: `outputs/detections/preliminary_pipeline/json/`
- Resumen JSON: `outputs/detections/preliminary_pipeline/preliminary_summary.json`
- Contact sheet: `outputs/reports/preliminary_pipeline_contact_sheet.jpg`

Este lote contiene imagenes reales del split YOLO test, por lo que la mayoria de casos representan extintores visibles. Sirve para comprobar integracion sobre imagen completa, no para medir robustez de estado ante oclusiones.

### Prueba balanceada con casos visibles, parciales y bloqueados

Para probar las tres clases de estado dentro del pipeline, se ejecuto una segunda prueba preliminar usando crops del split `data/classifier/test` como imagen de entrada al pipeline. Esto no sustituye una evaluacion final con imagenes reales completas, pero permite comprobar como se comporta la combinacion YOLO + crop + CNN ante ejemplos `visible`, `partially_occluded` y `blocked`.

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/run_preliminary_pipeline_evaluation.py \
  --images-dir data/classifier/test \
  --yolo-model models/yolo/extinguisher_yolo_test_gpu/weights/best.pt \
  --classifier-model models/classifier/extinguisher_status_cnn_test.pth \
  --output-dir outputs/detections/preliminary_pipeline_balanced_classifier \
  --max-images 30 \
  --max-images-per-class 10 \
  --confidence-threshold 0.25 \
  --save-crops \
  --save-json \
  --image-size 224 \
  --contact-sheet-output outputs/reports/preliminary_pipeline_balanced_contact_sheet.jpg
```

Resumen preliminar balanceado:

- Imagenes procesadas: 30
- Clases esperadas: `visible`=10, `partially_occluded`=10, `blocked`=10
- Imagenes sin deteccion YOLO: 11
- Detecciones totales: 21
- Predicciones de estado en detecciones:
  - `visible`: 18
  - `partially_occluded`: 3
  - `blocked`: 0
- Accuracy de estado sobre imagenes con deteccion: 0.6842105263157895

Cobertura de deteccion por clase esperada:

| Clase esperada | imagenes | con deteccion | sin deteccion |
| --- | ---: | ---: | ---: |
| visible | 10 | 10 | 0 |
| partially_occluded | 10 | 6 | 4 |
| blocked | 10 | 3 | 7 |

Matriz preliminar de estado usando la deteccion principal por imagen:

| esperado \ predicho | visible | partially_occluded | blocked |
| --- | ---: | ---: | ---: |
| visible | 10 | 0 | 0 |
| partially_occluded | 3 | 3 | 0 |
| blocked | 3 | 0 | 0 |

Salidas locales:

- Imagenes anotadas: `outputs/detections/preliminary_pipeline_balanced_classifier/annotated/`
- Crops: `outputs/detections/preliminary_pipeline_balanced_classifier/crops/`
- JSON por imagen: `outputs/detections/preliminary_pipeline_balanced_classifier/json/`
- Resumen JSON: `outputs/detections/preliminary_pipeline_balanced_classifier/preliminary_summary.json`
- Contact sheet: `outputs/reports/preliminary_pipeline_balanced_contact_sheet.jpg`

## Observaciones visuales

- La contact sheet preliminar muestra cajas YOLO dibujadas sobre extintores y etiquetas con clase de deteccion y estado CNN.
- En las 30 imagenes procesadas, la CNN clasifico todas las detecciones como `visible`, algo esperable porque las imagenes completas seleccionadas son ejemplos reales del dataset YOLO test y no necesariamente contienen oclusiones sinteticas.
- Hubo una imagen sin detecciones en este subconjunto.
- Algunas cajas son amplias o cortan parte del contexto, pero el pipeline completo genera salida estructurada y visual sin errores.
- La prueba balanceada muestra que el YOLO de prueba detecta todos los casos `visible`, pero pierde bastantes casos `partially_occluded` y `blocked`.
- En los casos bloqueados donde YOLO aun detecta algo, la CNN de prueba no predijo `blocked`; tendio a `visible`. Esto es coherente con modelos de prueba cortos y debe revisarse tras entrenamientos definitivos.

## Decision

El pipeline completo funciona de forma preliminar:

- YOLO de prueba carga y evalua.
- CNN de prueba carga y evalua.
- Las imagenes completas pasan por deteccion, crop, clasificacion, JSON y visualizacion.
- Las salidas se generan en carpetas ignoradas por Git.

Pendiente antes de entrenamientos definitivos:

- Revisar visualmente `data/classifier_context_v2/` y usarlo como base para la siguiente CNN baseline.
- Entrenar YOLO y CNN con configuraciones finales.
- Repetir la prueba balanceada para comprobar si mejoran los casos `partially_occluded` y `blocked`.
- Repetir evaluaciones con modelos definitivos.
- Calcular y documentar metricas finales solo despues de esos entrenamientos.
