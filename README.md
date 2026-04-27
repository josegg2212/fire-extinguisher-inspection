# Inspección visual automática de extintores

Sistema base para detectar extintores en imágenes o vídeo con YOLO y clasificar su estado visual con una CNN en tres clases:

- `visible`
- `partially_occluded`
- `blocked`

El proyecto está preparado para desarrollarse aunque todavía no existan pesos entrenados. El dataset YOLO puede prepararse localmente en `data/yolo/`, pero no se versiona en Git; el dataset del clasificador CNN y los entrenamientos reales siguen pendientes.

## Arquitectura

El flujo sigue la versión equilibrada definida en el documento del proyecto:

1. Leer imagen o frame de vídeo.
2. Detectar extintores con YOLO usando una única clase de detección: `fire_extinguisher`.
3. Generar para la CNN un crop contextual ampliado alrededor de cada bbox YOLO.
4. Clasificar ese crop contextual en `visible`, `partially_occluded` o `blocked`.
5. Devolver una salida estructurada y, opcionalmente, una imagen anotada.

La estructura se inspira en el repositorio `inspeccion_inteligente`, manteniendo separación entre API, lógica de inferencia, modelos, outputs y scripts, pero evitando una API monolítica.

## Estructura

```text
fire-extinguisher-inspection/
├── config/                         # Configuración YAML
├── data/                           # Datos locales no versionados
│   ├── raw/
│   ├── yolo/
│   ├── classifier/                 # v1 historica, crops ajustados
│   └── classifier_context_v2/       # v2 recomendada, crops contextuales
├── models/                         # Pesos YOLO y CNN no versionados
├── outputs/                        # Salidas de inferencia, crops, informes y logs
├── src/fire_extinguisher_inspection/
│   ├── detection/                  # Detector YOLO y entrenamiento
│   ├── classification/             # CNN, dataset, entrenamiento y predicción
│   ├── pipeline/                   # Orquestación extremo a extremo
│   ├── preprocessing/              # Recortes y utilidades de imagen
│   ├── visualization/              # Dibujo de resultados
│   └── api/                        # FastAPI
├── scripts/                        # Puntos de entrada de uso práctico
├── tests/                          # Tests mínimos
└── docker/                         # Ejecución en contenedor
```

## Instalación local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Smoke test sin modelos entrenados:

```bash
python3 scripts/smoke_test.py
```

Tests mínimos:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Informe de verificación inicial:

```text
docs/verificacion_inicial.md
```

El informe recoge las comprobaciones realizadas antes de pasar a la fase de datasets. Desde entonces puede existir un dataset YOLO local en `data/yolo/`, pero sigue sin versionarse.

## Configuración

La configuración principal está en `config/default.yaml` y las clases en `config/classes.yaml`.

Rutas importantes por defecto:

- YOLO: `models/yolo/extinguisher_yolo.pt`
- modelo base recomendado para entrenamiento YOLO: `yolo26n.pt`
- fallback compatible si YOLO26 da problemas en el entorno: `yolo11n.pt`
- CNN: `models/classifier/state_classifier.pt`
- dataset YOLO: `data/yolo/data.yaml`
- dataset CNN v1 historico: `data/classifier`
- dataset CNN contextual recomendado: `data/classifier_context_v2`
- salidas: `outputs/`

Estas rutas son convenciones de trabajo. Si los modelos o datasets aún no existen, el código devuelve errores o avisos claros.

## Dataset YOLO

La clase de detección esperada es:

```yaml
names:
  - fire_extinguisher
```

Hay un ejemplo en `data/yolo/data.yaml.example`. El dataset real y `data/yolo/data.yaml` están ignorados por Git para evitar subir imágenes, labels y rutas locales.

Estructura final esperada:

```text
data/yolo/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Si recibes un ZIP de Roboflow, colócalo temporalmente en `fire-extinguisher-inspection-docs/`, extráelo primero en `data/raw/extracted_yolo_dataset_tmp/`, revisa su estructura y copia solo los splits YOLO finales a `data/yolo/`. No mezcles contenido desconocido directamente en `data/yolo/`.

Comprobar estructura:

```bash
PYTHONPATH=src python3 scripts/check_dataset_structure.py --tipo yolo --path data/yolo/data.yaml
```

Generar una contact sheet para revisar cajas de forma visual:

```bash
PYTHONPATH=src python3 scripts/visualize_yolo_dataset_samples.py \
  --data-yaml data/yolo/data.yaml \
  --output outputs/reports/contact_sheet_yolo_dataset.jpg \
  --num-samples 16
```

## Verificación previa al entrenamiento YOLO

Antes de entrenar conviene validar de nuevo el dataset:

```bash
PYTHONPATH=src python3 scripts/check_dataset_structure.py --tipo yolo --path data/yolo/data.yaml
```

Revisar labels vacíos:

```bash
PYTHONPATH=src python3 scripts/visualize_empty_yolo_labels.py \
  --data-yaml data/yolo/data.yaml \
  --output outputs/reports/empty_labels_contact_sheet.jpg \
  --report docs/empty_labels_review.md
```

Si el Python local no tiene Pillow, usa el contenedor:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app \
  python3 scripts/visualize_empty_yolo_labels.py \
  --data-yaml data/yolo/data.yaml \
  --output outputs/reports/empty_labels_contact_sheet.jpg \
  --report docs/empty_labels_review.md
```

Entrenamiento corto de prueba, solo después de corregir o aceptar explícitamente los labels vacíos:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo26n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_test \
  --device 0
```

El script genera una copia temporal del YAML en `outputs/logs/` con la ruta del dataset resuelta para Ultralytics. Esto evita problemas con `path: .` cuando se lanza desde Docker o desde la raíz del repositorio.

Si el Python local no tiene Ultralytics instalado, se puede usar Docker. Para CPU:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app \
  python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo26n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_test
```

Para GPU, el Compose incluye el servicio `app-gpu`, que instala PyTorch con CUDA y expone la GPU del host:

```bash
docker compose -f docker/docker-compose.yml build app-gpu
docker compose -f docker/docker-compose.yml run --rm --no-deps app-gpu \
  python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo26n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_test_gpu \
  --device 0
```

En una GTX 1650 se verificó un entrenamiento corto de 3 épocas con YOLO26. Ultralytics desactivó AMP automáticamente, pero el entrenamiento terminó y guardó resultados en `models/yolo/extinguisher_yolo_test_gpu/`. Esos pesos y salidas están ignorados por Git.

Fallback si YOLO26 no está disponible en la versión instalada de Ultralytics:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo11_test \
  --device 0
```

## Entrenamiento YOLO v1

Esta fase entrena el primer detector YOLO serio de extintores. No entrena la CNN ni modifica los datasets de clasificación.

Validar el dataset YOLO antes de entrenar:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/check_dataset_structure.py \
  --tipo yolo \
  --path data/yolo/data.yaml
```

Entrenar 50 épocas en Docker GPU:

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

El script valida que el YAML exista antes de entrenar y no espera en bucle a que aparezcan datos. `--model` acepta cualquier modelo válido de Ultralytics; se recomienda empezar con `yolo26n.pt` y usar `yolo11n.pt` como fallback de compatibilidad si el entorno todavía no soporta YOLO26:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_v1
```

Evaluar el mejor peso en el split `test`:

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

Ruta esperada del mejor modelo:

```text
models/yolo/extinguisher_yolo_v1/weights/best.pt
```

Los pesos, logs, datasets, outputs e imágenes generadas son artefactos locales y no se versionan en Git. El informe del entrenamiento queda en `docs/yolo_v1_training.md`.

Siguiente paso tras revisar resultados: probar el pipeline completo con `models/yolo/extinguisher_yolo_v1/weights/best.pt` y la CNN contextual v2.

## Dataset CNN

La CNN de estado trabaja con crops de extintores en formato `ImageFolder` y tres clases:

- `visible`
- `partially_occluded`
- `blocked`

La v1 en `data/classifier/` queda como historico de validacion del flujo. Para el siguiente entrenamiento se recomienda usar la v2 contextual en `data/classifier_context_v2/`, porque conserva entorno alrededor del extintor.

Estructura v1:

```text
data/classifier/
├── train/
│   ├── visible/
│   ├── partially_occluded/
│   └── blocked/
├── val/
│   ├── visible/
│   ├── partially_occluded/
│   └── blocked/
└── test/
    ├── visible/
    ├── partially_occluded/
    └── blocked/
```

### Generación del dataset de clasificación de estado

El dataset inicial se puede generar desde las anotaciones YOLO ya revisadas. El script recorta cada bbox real como `visible` y crea variantes semi-sintéticas para `partially_occluded` y `blocked`, manteniendo siempre separados los splits de origen: YOLO `train` va a classifier `train`, YOLO `valid`/`val` va a classifier `val` y YOLO `test` va a classifier `test`.

Generación limitada para prueba:

```bash
PYTHONPATH=src python3 scripts/generate_classifier_dataset_from_yolo.py \
  --data-yaml data/yolo/data.yaml \
  --output-dir data/classifier \
  --max-per-split 20 \
  --partial-occlusions-per-object 1 \
  --blocked-occlusions-per-object 1 \
  --image-size 224 \
  --overwrite
```

Generación completa, cuando la prueba visual sea satisfactoria:

```bash
PYTHONPATH=src python3 scripts/generate_classifier_dataset_from_yolo.py \
  --data-yaml data/yolo/data.yaml \
  --output-dir data/classifier \
  --partial-occlusions-per-object 1 \
  --blocked-occlusions-per-object 1 \
  --visible-crops-per-object 1 \
  --image-size 224 \
  --overwrite
```

En la generación completa v1 se obtuvieron 28.797 crops: 21.909 en `train`, 4.656 en `val` y 2.232 en `test`. La diferencia respecto a las 29.052 imágenes esperadas viene de 85 anotaciones descartadas por crop menor que `--min-crop-size 32`; los 12 labels vacíos de `train` ya estaban documentados.

Los crops generados son datos locales y no deben subirse a Git. `data/classifier/train/`, `data/classifier/val/`, `data/classifier/test/` y `outputs/` están ignorados, salvo los `.gitkeep`.

Validar estructura e imágenes:

```bash
PYTHONPATH=src python3 scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier
```

Generar contact sheets para revisión visual:

```bash
PYTHONPATH=src python3 scripts/visualize_classifier_dataset_samples.py \
  --dataset-dir data/classifier \
  --split train \
  --output outputs/reports/contact_sheet_classifier_train_full.jpg \
  --num-samples-per-class 10

PYTHONPATH=src python3 scripts/visualize_classifier_dataset_samples.py \
  --dataset-dir data/classifier \
  --split val \
  --output outputs/reports/contact_sheet_classifier_val_full.jpg \
  --num-samples-per-class 10

PYTHONPATH=src python3 scripts/visualize_classifier_dataset_samples.py \
  --dataset-dir data/classifier \
  --split test \
  --output outputs/reports/contact_sheet_classifier_test_full.jpg \
  --num-samples-per-class 10
```

Las clases `partially_occluded` y `blocked` son semi-sintéticas en esta primera versión. Antes de entrenar la CNN, revisa las contact sheets y descarta o ajusta el generador si las oclusiones no resultan útiles.

Comprobar estructura con el validador general:

```bash
PYTHONPATH=src python3 scripts/check_dataset_structure.py --tipo classifier --path data/classifier
```

### Generación del dataset CNN contextual v2

La v2 corrige la falta de contexto detectada en la evaluación preliminar del pipeline. En vez de recortar solo el bbox ajustado, genera una región ampliada con `--context-padding 0.75` y `--square-crop`.

Prueba limitada:

```bash
PYTHONPATH=src python3 scripts/generate_classifier_context_dataset_from_yolo.py \
  --data-yaml data/yolo/data.yaml \
  --output-dir data/classifier_context_v2 \
  --max-per-split 20 \
  --context-padding 0.75 \
  --partial-occlusions-per-object 1 \
  --blocked-occlusions-per-object 1 \
  --visible-crops-per-object 1 \
  --image-size 224 \
  --square-crop \
  --overwrite
```

Generación completa:

```bash
PYTHONPATH=src python3 scripts/generate_classifier_context_dataset_from_yolo.py \
  --data-yaml data/yolo/data.yaml \
  --output-dir data/classifier_context_v2 \
  --context-padding 0.75 \
  --partial-occlusions-per-object 1 \
  --blocked-occlusions-per-object 1 \
  --visible-crops-per-object 1 \
  --image-size 224 \
  --square-crop \
  --overwrite
```

La generación completa v2 produjo 29.052 crops equilibrados: 22.065 en `train`, 4.698 en `val` y 2.289 en `test`.

Validar:

```bash
PYTHONPATH=src python3 scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier_context_v2
```

Contact sheets:

```bash
PYTHONPATH=src python3 scripts/visualize_classifier_dataset_samples.py \
  --dataset-dir data/classifier_context_v2 \
  --split train \
  --output outputs/reports/contact_sheet_classifier_context_v2_train.jpg \
  --num-samples-per-class 10
```

Repite cambiando `--split val` y `--split test`. `data/classifier_context_v2/` y `outputs/` no se versionan.

## Entrenamiento corto de prueba de la CNN

Antes de entrenar el modelo definitivo, se valido el pipeline con 5 épocas usando la v1 (`data/classifier/`). Esa prueba queda documentada como historica; la siguiente baseline debe usar `data/classifier_context_v2/`. En el entorno Docker GPU:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier
```

Entrenamiento corto:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m fire_extinguisher_inspection.classification.train_classifier \
  --dataset-path data/classifier \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --image-size 224 \
  --output-model-path models/classifier/extinguisher_status_cnn_test.pth
```

La salida esperada es un checkpoint local en `models/classifier/extinguisher_status_cnn_test.pth` y métricas en `models/classifier/extinguisher_status_cnn_test.metrics.json`. Ambos archivos están ignorados por Git. Esta prueba queda como histórico de v1.

## Entrenamiento de la CNN contextual v2

La baseline recomendada usa `data/classifier_context_v2` porque conserva contexto alrededor del extintor y está alineada con el pipeline de inferencia.

Validar dataset:

```bash
PYTHONPATH=src python3 scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier_context_v2
```

Entrenar:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m fire_extinguisher_inspection.classification.train_classifier \
  --dataset-path data/classifier_context_v2 \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --image-size 224 \
  --output-model-path models/classifier/state_classifier_context_v2.pt
```

El entrenamiento usa `train/val`, guarda el mejor modelo por accuracy de validación y escribe métricas por época en `models/classifier/state_classifier_context_v2.metrics.json`. Los pesos y métricas locales no se versionan.

Evaluar en test:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluate_classifier_on_test.py \
  --dataset-dir data/classifier_context_v2/test \
  --model-path models/classifier/state_classifier_context_v2.pt \
  --image-size 224 \
  --output-dir outputs/reports/classifier_context_v2_test
```

Resultado de la baseline contextual v2 entrenada: mejor época 27, `val_accuracy=0.8842`, `test_accuracy=0.8755`. No son métricas finales del sistema completo: `partially_occluded` y `blocked` siguen siendo clases semi-sintéticas. El siguiente paso es probar el pipeline completo con YOLO + CNN contextual v2.

## Evaluación preliminar del pipeline completo

Esta fase usa modelos de prueba y no representa resultados finales. Sirve para validar que YOLO, crops contextuales, CNN, JSON e imágenes anotadas funcionan juntos antes de entrenar modelos definitivos.

Evaluar preliminarmente la CNN en `data/classifier/test`:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluate_classifier_on_test.py \
  --dataset-dir data/classifier/test \
  --model-path models/classifier/extinguisher_status_cnn_test.pth \
  --image-size 224 \
  --output-dir outputs/reports/classifier_test_preliminary
```

Evaluar preliminarmente YOLO en el split `test`:

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

Ejecutar el pipeline completo sobre imágenes de test:

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
  --image-size 224 \
  --classifier-context-padding 0.75 \
  --classifier-square-crop
```

Prueba preliminar balanceada con 10 ejemplos por clase desde `data/classifier/test`:

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
  --classifier-context-padding 0.75 \
  --classifier-square-crop \
  --contact-sheet-output outputs/reports/preliminary_pipeline_balanced_contact_sheet.jpg
```

Las salidas se guardan en `outputs/reports/classifier_test_preliminary/`, `outputs/reports/yolo_test_preliminary/`, `outputs/detections/preliminary_pipeline/`, `outputs/detections/preliminary_pipeline_balanced_classifier/` y sus contact sheets en `outputs/reports/`. Los outputs, modelos y datasets reales están ignorados por Git.

## Pipeline integrado YOLO v1 + CNN contextual v2

El pipeline integrado v1 usa:

- YOLO v1: `models/yolo/extinguisher_yolo_v1/weights/best.pt`
- CNN contextual v2: `models/classifier/state_classifier_context_v2.pt`

La bbox original de YOLO se conserva para visualizacion y JSON. Para clasificar el estado, el pipeline amplia esa bbox con `classifier_context_padding=0.75`, genera un crop contextual cuadrado y pasa ese crop a la CNN contextual v2. Esto mantiene el contexto usado durante el entrenamiento de la CNN.

Evaluacion sobre imagenes completas de `data/yolo/test/images`:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/run_integrated_pipeline_evaluation.py \
  --images-dir data/yolo/test/images \
  --yolo-model models/yolo/extinguisher_yolo_v1/weights/best.pt \
  --classifier-model models/classifier/state_classifier_context_v2.pt \
  --output-dir outputs/detections/integrated_pipeline_v1 \
  --max-images 50 \
  --confidence-threshold 0.25 \
  --classifier-context-padding 0.75 \
  --classifier-square-crop \
  --image-size 224 \
  --save-crops \
  --save-json \
  --contact-sheet-output outputs/reports/integrated_pipeline_v1_contact_sheet.jpg
```

Prueba balanceada de estados con `data/classifier_context_v2/test`:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/run_integrated_pipeline_evaluation.py \
  --images-dir data/classifier_context_v2/test \
  --yolo-model models/yolo/extinguisher_yolo_v1/weights/best.pt \
  --classifier-model models/classifier/state_classifier_context_v2.pt \
  --output-dir outputs/detections/integrated_pipeline_v1_balanced \
  --max-images 60 \
  --max-images-per-class 20 \
  --confidence-threshold 0.25 \
  --classifier-context-padding 0.75 \
  --classifier-square-crop \
  --image-size 224 \
  --save-crops \
  --save-json \
  --contact-sheet-output outputs/reports/integrated_pipeline_v1_balanced_contact_sheet.jpg
```

Las imagenes anotadas, crops contextuales, JSON, resumenes y contact sheets se guardan en `outputs/`. Los modelos, datasets y outputs generados no se versionan en Git. El informe de esta evaluacion queda en `docs/integrated_pipeline_v1_evaluation.md`.

## Inferencia sobre imagen

```bash
python3 scripts/run_image_inference.py --image path/a/imagen.jpg --save-crops
```

Por defecto la CNN recibe un crop contextual con `classifier_context_padding=0.75` y crop cuadrado. Se puede ajustar con `--classifier-context-padding`, `--classifier-square-crop` o `--no-classifier-square-crop`.

Si falta el modelo YOLO, el resultado incluirá un error controlado. Si falta el modelo CNN pero sí existe YOLO, el pipeline devolverá detecciones sin clasificación de estado.

## Inferencia sobre vídeo o webcam

Vídeo:

```bash
python3 scripts/run_video_inference.py --source path/a/video.mp4 --output outputs/detections/video_anotado.mp4
```

Webcam:

```bash
python3 scripts/run_video_inference.py --source 0 --show
```

## Generación de crops

Cuando exista un detector YOLO entrenado, se pueden generar recortes sin etiqueta para construir el dataset del clasificador:

```bash
python3 scripts/generate_crops_from_yolo.py \
  --input-dir data/raw \
  --output-dir outputs/crops/generated
```

El script no inventa etiquetas; guarda crops y metadatos para revisión manual.

## API

Arranque local:

```bash
PYTHONPATH=src uvicorn fire_extinguisher_inspection.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /inspect/image`

Ejemplo:

```bash
curl -X POST "http://localhost:8000/inspect/image?guardar_anotada=true" \
  -F "file=@path/a/imagen.jpg"
```

La respuesta es JSON con `image_path`, `detections`, `warnings` y `errors`.

## Docker

Construir y abrir shell:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm app
```

Levantar API:

```bash
docker compose -f docker/docker-compose.yml up api
```

El contenedor monta el repositorio en `/workspace` y define `PYTHONPATH=/workspace/src`. No asume GPU.
La imagen instala PyTorch en variante CPU para evitar dependencias CUDA innecesarias.

Construir shell con soporte GPU:

```bash
docker compose -f docker/docker-compose.yml build app-gpu
docker compose -f docker/docker-compose.yml run --rm --no-deps app-gpu
```

Comprobar CUDA dentro del contenedor GPU:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps app-gpu \
  python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

Para revisar la configuración de Compose sin construir la imagen:

```bash
docker compose -f docker/docker-compose.yml config
```

Si el entorno no puede descargar `python:3.12-slim` pero ya existe una imagen Python local, se puede indicar una base alternativa:

```bash
PYTHON_IMAGE=<imagen-python-local> docker compose -f docker/docker-compose.yml build
```

## Estado actual

Incluido:

- estructura profesional del proyecto
- configuración centralizada
- detector YOLO preparado
- entrenamiento YOLO por CLI
- CNN baseline en PyTorch
- entrenamiento e inferencia del clasificador
- pipeline completo con manejo de modelos ausentes
- visualización de resultados
- API FastAPI
- scripts de uso
- Docker
- tests mínimos y smoke test
- dataset YOLO inicial preparado localmente cuando existe `data/yolo/data.yaml`
- entrenamiento corto YOLO26 de 3 épocas verificado en GPU local; resultados ignorados por Git

Pendiente antes de entrenar:

- lanzar entrenamiento YOLO real de 50 épocas si se acepta el dataset actual
- generar y etiquetar crops para `data/classifier`
- entrenar pesos reales definitivos
- evaluar detector, clasificador y pipeline completo con métricas medidas
