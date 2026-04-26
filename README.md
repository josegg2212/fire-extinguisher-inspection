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
- dataset CNN: `data/classifier`
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

## Entrenamiento YOLO

Entrenamiento real recomendado de 50 épocas, una vez validado el dataset:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_v1 \
  --device 0
```

El script valida que el YAML exista antes de entrenar y no espera en bucle a que aparezcan datos. `--model` acepta cualquier modelo válido de Ultralytics; se recomienda empezar con `yolo26n.pt` y usar `yolo11n.pt` como fallback de compatibilidad si el entorno todavía no soporta YOLO26.

Fallback para entrenamiento real:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo11_v1 \
  --device 0
```

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
PYTHONPATH=src python3 scripts/check_dataset_structure.py --tipo classifier --path data/classifier
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
