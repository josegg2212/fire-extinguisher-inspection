# Resumen del dataset YOLO de extintores

Fecha de preparación: 2026-04-25

## Origen

- ZIP usado: `Fire_Extinguisher.v3i.yolo26.zip`
- Ubicación original: `../fire-extinguisher-inspection-docs/Fire_Extinguisher.v3i.yolo26.zip`
- Tamaño aproximado del ZIP: 538963036 bytes
- El ZIP original se conserva en `fire-extinguisher-inspection-docs/`.

## Estructura detectada

El ZIP corresponde a una exportación Roboflow/YOLO con esta estructura:

```text
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
data.yaml
```

No se detectó una carpeta raíz adicional que requiriera saltarse un nivel.

## Estructura final

El dataset se organizó en:

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

El archivo `data/yolo/data.yaml` usa rutas relativas:

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  - fire_extinguisher
```

## Estadísticas

| Split | Imágenes | Labels | Anotaciones |
| --- | ---: | ---: | ---: |
| train | 3791 | 3791 | 7355 |
| valid | 837 | 837 | 1566 |
| test | 397 | 397 | 763 |
| total | 5025 | 5025 | 9684 |

Anotaciones por clase:

| Clase | train | valid | test | total |
| --- | ---: | ---: | ---: | ---: |
| `fire_extinguisher` | 7355 | 1566 | 763 | 9684 |

## Clases

- Clases detectadas originalmente: `fire_extinguisher`
- Clases finales: `fire_extinguisher`
- No se realizó conversión de clases porque el dataset ya estaba en formato monoclase coherente con el proyecto.

## Cambios realizados

- Se extrajo el ZIP en `data/raw/extracted_yolo_dataset_tmp/` para inspeccionarlo antes de moverlo.
- Se normalizó la estructura final a `data/yolo/train`, `data/yolo/valid` y `data/yolo/test`.
- Se corrigió `data/yolo/data.yaml` para usar rutas relativas desde `data/yolo/`.
- Se conservaron imágenes y anotaciones sin modificaciones.
- Se añadieron `.gitkeep` en las carpetas de estructura para que el repositorio mantenga los directorios sin versionar el dataset real.

## Problemas encontrados

- Hay 12 labels vacíos en `train`. YOLO admite labels vacíos para imágenes sin objetos, pero conviene revisarlos visualmente antes de entrenar.
- No se encontraron labels fuera de rango ni coordenadas YOLO inválidas durante la validación.
- No se encontraron labels sin imagen correspondiente.

## Verificación realizada

Comando:

```bash
PYTHONPATH=src python3 scripts/check_dataset_structure.py --tipo yolo --path data/yolo/data.yaml
```

Resultado:

- Dataset YOLO válido.
- `train`: 3791 imágenes, 3791 labels, 7355 anotaciones.
- `valid`: 837 imágenes, 837 labels, 1566 anotaciones.
- `test`: 397 imágenes, 397 labels, 763 anotaciones.
- Aviso controlado por 12 labels vacíos en `train`.

Contact sheet de control:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app \
  python3 scripts/visualize_yolo_dataset_samples.py \
  --data-yaml data/yolo/data.yaml \
  --output outputs/reports/contact_sheet_yolo_dataset.jpg \
  --num-samples 16
```

```text
outputs/reports/contact_sheet_yolo_dataset.jpg
```

La imagen generada es un JPEG de 1440x1440 píxeles con 16 muestras de `train`, `valid` y `test`.

Comprobaciones adicionales:

```bash
PYTHONPATH=src python3 scripts/smoke_test.py
PYTHONPATH=src python3 -m unittest discover -s tests
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app python3 scripts/smoke_test.py
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app python3 -m unittest discover -s tests
```

Resultado:

- Smoke test local: correcto.
- Tests locales: correctos, con 1 test omitido por dependencia no instalada en el Python local.
- Smoke test en Docker: correcto.
- Tests en Docker: correctos, 7 tests ejecutados.

## Estado

El dataset queda preparado para revisión previa al entrenamiento YOLO. Tras revisar los 12 labels vacíos, se recomendó corregirlos o excluirlos antes de entrenar. Posteriormente se decidió aceptarlos temporalmente por ser pocos frente al volumen total del dataset; los detalles están en `docs/yolo_pretraining_check.md` y `docs/empty_labels_review.md`.

Durante la verificación posterior se descargó localmente el peso base `yolo26n.pt` dentro de `models/yolo/base/`, ignorado por Git. También se completó un entrenamiento corto de prueba de 3 épocas en `models/yolo/extinguisher_yolo_test_gpu/`, igualmente ignorado por Git. Todavía no hay un modelo final entrenado de 50 épocas.
