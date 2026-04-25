# Inspección visual automática de extintores

Sistema base para detectar extintores en imágenes o vídeo con YOLO y clasificar su estado visual con una CNN en tres clases:

- `visible`
- `partially_occluded`
- `blocked`

El proyecto está preparado para desarrollarse sin dataset definitivo todavía. Las rutas, scripts y validaciones ya existen, pero el entrenamiento real queda pendiente hasta disponer de imágenes anotadas y recortes etiquetados.

## Arquitectura

El flujo sigue la versión equilibrada definida en el documento del proyecto:

1. Leer imagen o frame de vídeo.
2. Detectar extintores con YOLO usando una única clase de detección: `fire_extinguisher`.
3. Recortar cada bounding box como ROI, con un margen configurable.
4. Clasificar cada ROI con una CNN en `visible`, `partially_occluded` o `blocked`.
5. Devolver una salida estructurada y, opcionalmente, una imagen anotada.

La estructura se inspira en el repositorio `inspeccion_inteligente`, manteniendo separación entre API, lógica de inferencia, modelos, outputs y scripts, pero evitando una API monolítica.

## Estructura

```text
fire-extinguisher-inspection/
├── config/                         # Configuración YAML
├── data/                           # Datos locales no versionados
│   ├── raw/
│   ├── yolo/
│   └── classifier/
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

Smoke test sin modelos ni dataset:

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

El informe recoge las comprobaciones realizadas antes de pasar a la fase de datasets. En esta fase no hay pesos entrenados ni datasets reales.

## Configuración

La configuración principal está en `config/default.yaml` y las clases en `config/classes.yaml`.

Rutas importantes por defecto:

- YOLO: `models/yolo/extinguisher_yolo.pt`
- CNN: `models/classifier/state_classifier.pt`
- dataset YOLO: `data/yolo/data.yaml`
- dataset CNN: `data/classifier`
- salidas: `outputs/`

Estas rutas son convenciones de trabajo. Si los modelos o datasets aún no existen, el código devuelve errores o avisos claros.

## Dataset YOLO

La clase de detección esperada es:

```yaml
names:
  0: fire_extinguisher
```

Hay un ejemplo en `data/yolo/data.yaml.example`. Cuando exista el dataset real, crea un `data.yaml` propio apuntando a tus carpetas reales.

Comprobar estructura:

```bash
python3 scripts/check_dataset_structure.py --tipo yolo --path data/yolo/data.yaml
```

## Entrenamiento YOLO

Ejemplo:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_v1
```

El script valida que el YAML exista antes de entrenar y no espera en bucle a que aparezcan datos.

## Dataset CNN

Estructura esperada:

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

Comprobar estructura:

```bash
python3 scripts/check_dataset_structure.py --tipo classifier --path data/classifier
```

## Entrenamiento CNN

Baseline con una CNN sencilla en PyTorch:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.classification.train_classifier \
  --dataset-path data/classifier \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --image-size 224 \
  --output-model-path models/classifier/state_classifier.pt
```

El entrenamiento usa `train/val`, guarda el mejor modelo por accuracy de validación y escribe métricas básicas junto al checkpoint. Cuando haya dataset real se podrá ampliar con transfer learning, pesos de clase o métricas F1.

## Inferencia sobre imagen

```bash
python3 scripts/run_image_inference.py --image path/a/imagen.jpg --save-crops
```

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

Pendiente cuando haya dataset:

- crear `data/yolo/data.yaml` real
- anotar imágenes completas para YOLO
- generar y etiquetar crops para `data/classifier`
- entrenar pesos reales
- evaluar detector, clasificador y pipeline completo con métricas medidas
