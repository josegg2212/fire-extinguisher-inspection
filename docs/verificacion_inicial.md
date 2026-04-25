# Verificación inicial del repositorio

Fecha: 2026-04-25

## Alcance

Verificación del repositorio `fire-extinguisher-inspection` antes de pasar a la fase de datasets.

No se han descargado datasets, no se han creado imágenes falsas, no se han añadido pesos y no se ha lanzado ningún entrenamiento.

## Comprobaciones realizadas

### Estructura base

Resultado: correcto.

Se comprobó la presencia de:

- `README.md`
- `.gitignore`
- `.env.example`
- `requirements.txt`
- `config/default.yaml`
- `config/classes.yaml`
- `data/`
- `models/`
- `outputs/`
- `src/fire_extinguisher_inspection/`
- `scripts/`
- `tests/`
- `docker/`

Las carpetas necesarias sin contenido real se conservan con `.gitkeep`.

### `.gitignore`

Resultado: correcto.

Se verificó que se ignoran:

- imágenes y vídeos en `data/`
- pesos `.pt`, `.pth`, `.onnx` y `.engine`
- salidas generadas en `outputs/`
- logs
- cachés de Python
- entornos virtuales
- `.env`

También se comprobó que los `.gitkeep` no quedan ignorados.

Comando usado:

```bash
git check-ignore -v data/raw/prueba.jpg data/raw/.gitkeep models/yolo/modelo.pt outputs/reports/verificacion_inicial.md outputs/reports/.gitkeep .env .venv/lib.py || true
```

Como `outputs/` está ignorado, este informe se guarda en `docs/verificacion_inicial.md`.

### Configuración

Resultado: correcto.

Comandos usados:

```bash
PYTHONPATH=src python3 - <<'PY'
from fire_extinguisher_inspection.config import cargar_configuracion
print(cargar_configuracion())
PY
```

```bash
PYTHONPATH=src python3 - <<'PY'
from fire_extinguisher_inspection.config import cargar_configuracion
try:
    cargar_configuracion('config/no_existe.yaml')
except Exception as exc:
    print(type(exc).__name__ + ':', exc)
PY
```

Resultado observado:

- `config/default.yaml` carga correctamente.
- `config/classes.yaml` carga correctamente.
- Las rutas del YAML son relativas al repositorio.
- La configuración inexistente devuelve `FileNotFoundError` con mensaje claro.

### Imports principales

Resultado: correcto.

Se importaron correctamente:

- detector YOLO
- modelo CNN
- dataset CNN
- pipeline
- esquema de resultados
- API
- utilidades de crops
- visualización

Comando usado:

```bash
PYTHONPATH=src python3 - <<'PY'
mods = [
    'fire_extinguisher_inspection.detection.yolo_detector',
    'fire_extinguisher_inspection.classification.cnn_model',
    'fire_extinguisher_inspection.classification.dataset',
    'fire_extinguisher_inspection.pipeline.inspection_pipeline',
    'fire_extinguisher_inspection.pipeline.result_schema',
    'fire_extinguisher_inspection.api.main',
    'fire_extinguisher_inspection.preprocessing.crop_utils',
    'fire_extinguisher_inspection.visualization.draw_results',
]
for name in mods:
    __import__(name)
    print('OK', name)
PY
```

Nota del entorno local: no están instalados `fastapi`, `cv2`, `torch`, `torchvision`, `ultralytics` ni `Pillow`. Los imports principales funcionan porque las dependencias pesadas se cargan de forma perezosa cuando realmente se usan.

### Smoke test

Resultado: correcto.

Comando usado:

```bash
PYTHONPATH=src python3 scripts/smoke_test.py
```

Resultado observado:

```text
Smoke test completado.
```

### Tests mínimos

Resultado: correcto.

Comando usado:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Resultado observado:

```text
Ran 7 tests
OK (skipped=1)
```

El test omitido corresponde a `/health` con `TestClient`, porque FastAPI no está instalado en el entorno local de verificación.

### Scripts en modo ayuda

Resultado: correcto.

Comandos usados:

```bash
PYTHONPATH=src python3 scripts/run_image_inference.py --help
PYTHONPATH=src python3 scripts/run_video_inference.py --help
PYTHONPATH=src python3 scripts/generate_crops_from_yolo.py --help
PYTHONPATH=src python3 scripts/check_dataset_structure.py --help
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo --help
PYTHONPATH=src python3 -m fire_extinguisher_inspection.classification.train_classifier --help
PYTHONPATH=src python3 -m fire_extinguisher_inspection.classification.predict_classifier --help
```

Todos muestran ayuda por CLI.

### Comportamiento sin modelos ni imagen

Resultado: correcto.

Comando usado:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from fire_extinguisher_inspection.config import cargar_configuracion
from fire_extinguisher_inspection.detection.yolo_detector import YoloExtinguisherDetector
from fire_extinguisher_inspection.classification.predict_classifier import CNNStatePredictor
from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline
config = cargar_configuracion()
for label, fn in [
    ('modelo_yolo', lambda: YoloExtinguisherDetector(config.modelos.yolo)),
    ('modelo_cnn', lambda: CNNStatePredictor(config.modelos.cnn)),
]:
    try:
        fn()
    except Exception as exc:
        print(label + ':', type(exc).__name__ + ':', exc)
resultado = InspectionPipeline(config).inspeccionar_imagen(Path('no_existe.jpg'))
print('imagen_inexistente:', resultado.errors)
PY
```

Resultado observado:

- Modelo YOLO ausente: `FileNotFoundError` claro.
- Modelo CNN ausente: `FileNotFoundError` claro.
- Imagen inexistente: error controlado en el resultado del pipeline.

### API

Resultado: correcto a nivel de importación.

Comando usado:

```bash
PYTHONPATH=src python3 -c "from fire_extinguisher_inspection.api.main import app; print(app.title)"
```

Resultado observado:

```text
Inspección visual de extintores
```

La API real con `/health` queda lista para verificarse con FastAPI instalado mediante:

```bash
PYTHONPATH=src uvicorn fire_extinguisher_inspection.api.main:app --host 0.0.0.0 --port 8000
```

### Docker

Resultado inicial: correcto a nivel de configuración Compose.

Comando usado:

```bash
docker compose -f docker/docker-compose.yml config
```

Resultado observado:

- `app` y `api` montan el repositorio como volumen en `/workspace`.
- `working_dir` es `/workspace`.
- `PYTHONPATH` apunta a `/workspace/src`.
- No se asume GPU obligatoria.
- No hay rutas absolutas escritas en los archivos Docker; las rutas absolutas vistas en `docker compose config` son la expansión normal del host.

Verificación posterior en contenedor:

```bash
PYTHON_IMAGE=mqtt-client-mqtt-client:latest docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm --no-deps app python3 scripts/smoke_test.py
docker compose -f docker/docker-compose.yml run --rm --no-deps app python3 -m unittest discover -s tests
docker compose -f docker/docker-compose.yml up -d api
curl -sS -i http://localhost:8000/health
```

Resultado observado:

- La descarga directa de `python:3.12-slim` falló por timeout externo de Docker Hub/Cloudflare.
- Se añadió soporte para `PYTHON_IMAGE` como build arg y se construyó usando una imagen Python local disponible.
- La imagen Docker quedó validada con dependencias completas instaladas.
- `scripts/smoke_test.py` pasó dentro del contenedor.
- `python3 -m unittest discover -s tests` pasó dentro del contenedor.
- `/health` respondió HTTP 200 desde el servicio FastAPI levantado por Compose.
- Docker instala PyTorch CPU-only para no depender de GPU ni arrastrar ruedas CUDA innecesarias.
- La imagen final validada queda alrededor de 2.62 GB, frente a una build previa con ruedas CUDA que superaba los 12 GB.
- `scripts/run_image_inference.py --image no_existe.jpg` devuelve JSON con error controlado.
- `scripts/check_dataset_structure.py` valida el ejemplo YOLO y avisa correctamente de que el dataset CNN aún no contiene imágenes.

### Rutas absolutas en archivos del repo

Resultado: correcto.

Comando usado:

```bash
rg -n "<patrones de rutas absolutas comunes>" . -g '!*.pyc'
```

No se encontraron rutas absolutas incrustadas en archivos del repositorio.

## Correcciones realizadas

- Se añadió CLI con `argparse` a `src/fire_extinguisher_inspection/classification/predict_classifier.py`.
- Se añadió un objeto ASGI informativo en `src/fire_extinguisher_inspection/api/main.py` para que `app.title` no falle si FastAPI todavía no está instalado.
- Se añadió `tests/test_api.py` con test de importación de API y test de `/health` condicionado a que FastAPI esté instalado.
- Se actualizó `README.md` con referencia al informe de verificación y al comando de revisión de Docker Compose.

## Errores encontrados

- `predict_classifier.py` no mostraba ayuda CLI antes de la corrección.
- En este entorno local no están instaladas las dependencias de ejecución pesada. No rompe smoke tests ni imports ligeros, pero la API real, inferencia y entrenamiento requieren instalar `requirements.txt`.

## Pendiente antes de datasets

- Instalar dependencias en un entorno virtual o contenedor.
- Ejecutar opcionalmente `docker compose -f docker/docker-compose.yml build`.
- Ejecutar opcionalmente `docker compose -f docker/docker-compose.yml run --rm app python3 scripts/smoke_test.py`.

## Siguiente fase

Preparar el dataset YOLO real:

- definir `data/yolo/data.yaml`
- recopilar imágenes reales
- anotar bounding boxes de `fire_extinguisher`
- mantener separación por escena/secuencia para `train`, `val` y `test`

No se debe entrenar YOLO ni la CNN hasta que el dataset esté definido y revisado.
