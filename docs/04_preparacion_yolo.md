# Verificación previa al entrenamiento YOLO

Fecha: 2026-04-25

## Dataset usado

- YAML: `data/yolo/data.yaml`
- Clase: `fire_extinguisher`
- Estructura: `train`, `valid`, `test`

## Validación del dataset

Comando ejecutado:

```bash
PYTHONPATH=src python3 scripts/validar_dataset.py --tipo yolo --path data/yolo/data.yaml
```

Resultado:

| Split | Imágenes | Labels | Anotaciones |
| --- | ---: | ---: | ---: |
| train | 3791 | 3791 | 7355 |
| valid | 837 | 837 | 1566 |
| test | 397 | 397 | 763 |

Anotaciones por clase:

- `fire_extinguisher`: 9684

No se detectaron errores de formato YOLO, ids de clase fuera de rango ni coordenadas fuera de `[0, 1]`.

Aviso:

- `train` contiene 12 labels vacíos.

## Revisión de labels vacíos

Comando ejecutado en Docker porque el Python local no tiene Pillow instalado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app \
  python3 scripts/legacy/ver_labels_yolo_vacios.py \
  --data-yaml data/yolo/data.yaml \
  --output outputs/reports/empty_labels_contact_sheet.jpg \
  --report docs/03_labels_yolo.md
```

Salida generada:

- `outputs/reports/empty_labels_contact_sheet.jpg`
- `docs/03_labels_yolo.md`

Resultado de la revisión visual:

- Hay 12 labels vacíos, todos en `train`.
- Las 12 imágenes parecen contener extintores, grupos de extintores, ilustraciones de extintores o elementos claramente relacionados con extintores.
- No parecen negativos limpios.

Decisión inicial: estos casos deberían corregirse o excluirse antes de entrenar. Mantenerlos como labels vacíos puede introducir falsos negativos.

Decisión posterior del proyecto: se aceptan temporalmente los 12 labels vacíos porque son pocos frente al volumen total del dataset. El riesgo queda documentado.

## Contact sheet general

Archivo verificado:

```text
outputs/reports/contact_sheet_yolo_dataset.jpg
```

El archivo existe localmente y es un JPEG de 1440x1440 píxeles.

## CLI de entrenamiento

Comando ejecutado:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo --help
```

Argumentos confirmados:

- `--data`
- `--model`
- `--epochs`
- `--imgsz`
- `--batch`
- `--workers`
- `--project`
- `--name`
- `--device`

## Entrenamiento corto

Comando previsto inicialmente con YOLO26:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo26n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_test
```

Fallback previsto con YOLO11:

```bash
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo11_test
```

Estado: se intentó ejecutar el entrenamiento corto con YOLO26 después de aceptar temporalmente los 12 labels vacíos.

Modelo usado realmente: `yolo26n.pt`, usando el peso base local descargado en `models/yolo/base/yolo26n.pt`.

Fallback YOLO11: no se ejecutó porque YOLO26 sí fue reconocido por Ultralytics 8.4.41.

Primer intento:

- YOLO26 se descargó correctamente.
- El entrenamiento falló antes de empezar porque Ultralytics interpretó `path: .` relativo al directorio de ejecución y buscó `valid/images` en una ruta incorrecta.
- Corrección aplicada: `train_yolo.py` genera un YAML temporal en `outputs/logs/data_ultralytics_resuelto.yaml` con la raíz del dataset resuelta como ruta absoluta.

Segundo intento controlado en CPU:

```bash
timeout 90s docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app \
  python3 -m fire_extinguisher_inspection.detection.train_yolo \
  --data data/yolo/data.yaml \
  --model models/yolo/base/yolo26n.pt \
  --epochs 3 \
  --imgsz 640 \
  --batch 16 \
  --workers 0 \
  --project models/yolo \
  --name extinguisher_yolo_test
```

Resultado:

- El dataset se leyó correctamente.
- Ultralytics registró `3791 images, 12 backgrounds, 0 corrupt` en `train`.
- Ultralytics registró `837 images, 0 backgrounds, 0 corrupt` en `valid`.
- La salida quedó correctamente dirigida a `models/yolo/extinguisher_yolo_test`.
- El entrenamiento empezó en CPU, pero no era una vía práctica para esta comprobación. Quedó una salida parcial local; se descartó como referencia principal y se repitió el test en GPU.

Ruta local parcial:

```text
models/yolo/extinguisher_yolo_test
```

Archivos parciales generados:

- `args.yaml`
- `labels.jpg`
- `train_batch0.jpg`
- `train_batch1.jpg`
- `train_batch2.jpg`
- `weights/` con pesos parciales de la ejecución CPU

Tercer intento en GPU:

El host tenía una NVIDIA GeForce GTX 1650 visible con `nvidia-smi`, pero la imagen Docker base instalada hasta ese momento llevaba PyTorch CPU. Se confirmó que Docker sí podía exponer la GPU con `docker run --gpus all`, y después se ejecutó una sesión temporal instalando PyTorch CUDA 12.6 dentro del contenedor.

Comando ejecutado:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e CONFIG_PATH=config/default.yaml \
  -e CLASSES_PATH=config/classes.yaml \
  -e YOLO_CONFIG_DIR=/tmp \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  docker-app sh -lc '
    set -e
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 "torch==2.8.0+cu126" "torchvision==0.23.0+cu126"
    python3 -m fire_extinguisher_inspection.detection.train_yolo \
      --data data/yolo/data.yaml \
      --model models/yolo/base/yolo26n.pt \
      --epochs 3 \
      --imgsz 640 \
      --batch 16 \
      --workers 0 \
      --project models/yolo \
      --name extinguisher_yolo_test_gpu \
      --device 0
    chown -R 1000:1000 models/yolo outputs/logs data/yolo || true
  '
```

Comprobación CUDA dentro del contenedor:

- `torch`: `2.8.0+cu126`
- CUDA de PyTorch: `12.6`
- `torch.cuda.is_available()`: `True`
- Dispositivo: `NVIDIA GeForce GTX 1650`

Resultado:

- El entrenamiento corto terminó correctamente.
- Duración reportada por Ultralytics: `0.681` horas.
- Salida local: `models/yolo/extinguisher_yolo_test_gpu`
- Pesos generados: `weights/best.pt` y `weights/last.pt`
- La GPU volvió a quedar sin procesos de entrenamiento al terminar.

Aviso relevante:

- Ultralytics desactivó AMP automáticamente por una comprobación de compatibilidad en la GTX 1650. El entrenamiento funcionó igualmente, pero puede ser más lento que en GPUs con AMP plenamente soportado.

Métricas disponibles en `models/yolo/extinguisher_yolo_test_gpu/results.csv`:

| Época | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.85281 | 0.72797 | 0.83480 | 0.62632 |
| 2 | 0.82763 | 0.79105 | 0.87711 | 0.67076 |
| 3 | 0.88821 | 0.84224 | 0.92764 | 0.73846 |

La validación final de `best.pt` reportó aproximadamente:

- Precision: `0.882`
- Recall: `0.847`
- mAP50: `0.928`
- mAP50-95: `0.739`

Estas métricas corresponden solo al entrenamiento corto de prueba de 3 épocas. No deben tratarse como resultado final del proyecto.

## Problemas encontrados

- El Python local no tiene `ultralytics` ni `Pillow`; las comprobaciones que necesitan esas dependencias se ejecutaron con Docker.
- La imagen Docker CPU no ve CUDA. Para GPU se añadió un servicio `app-gpu` en `docker/docker-compose.yml` y se verificó la ejecución con `docker run --gpus all`.
- La construcción de `app-gpu` puede depender de la disponibilidad de Docker Hub para descargar `python:3.12-slim`; durante esta verificación volvió a quedarse bloqueada en la carga de metadatos de esa imagen base y se detuvo manualmente. Se puede repetir más tarde o usar una imagen Python local mediante `PYTHON_IMAGE`.
- El dataset tiene 12 falsos negativos probables en `train`, aceptados temporalmente por decisión del proyecto.

## Decisión final

El dataset queda aceptado para lanzar el entrenamiento real, asumiendo el riesgo documentado de los 12 labels vacíos.

El entrenamiento corto con YOLO26 en GPU terminó correctamente, por lo que el siguiente paso recomendado es entrenar 50 épocas con GPU:

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

Se mantiene `yolo11n.pt` como fallback si YOLO26 deja de estar disponible en la versión instalada de Ultralytics.

## Verificación final del repositorio

Comandos ejecutados:

```bash
PYTHONPATH=src python3 scripts/validar_dataset.py --tipo yolo --path data/yolo/data.yaml
PYTHONPATH=src python3 scripts/prueba_basica.py
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m fire_extinguisher_inspection.detection.train_yolo --help
docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app python3 - <<'PY'
from fastapi.testclient import TestClient
from fire_extinguisher_inspection.api.main import app
respuesta = TestClient(app).get('/health')
print(app.title)
print(respuesta.status_code)
print(respuesta.json())
PY
docker run --rm --gpus all docker-app nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
```

Resultado:

- Dataset YOLO válido, con el aviso esperado de 12 labels vacíos en `train`.
- Smoke test correcto.
- Tests mínimos correctos: 7 tests, 1 omitido en local por dependencia opcional no instalada.
- CLI de entrenamiento correcta.
- Docker Compose válido, incluyendo el servicio `app-gpu`.
- API FastAPI correcta dentro del contenedor: `/health` devolvió `200`.
- Docker puede ver la GPU del host con `--gpus all`.
- Dataset, caches, pesos base, salidas de entrenamiento y contact sheets están ignorados por Git.
