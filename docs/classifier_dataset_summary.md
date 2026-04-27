# Resumen del dataset CNN de estado

Fecha: 2026-04-26

> Nota 2026-04-26: este documento describe la v1 del dataset CNN en `data/classifier/`. La v1 fue util para validar el flujo de generacion, entrenamiento corto e integracion preliminar, pero queda superada para el siguiente entrenamiento porque sus crops eran demasiado ajustados al bbox YOLO y perdian contexto. La version recomendada para la proxima CNN baseline es `data/classifier_context_v2/`, documentada en `docs/classifier_context_v2_summary.md`.

## Origen

El dataset inicial de clasificacion de estado se genera localmente a partir del dataset YOLO ya validado en:

```text
data/yolo/data.yaml
```

No se versionan imagenes, crops generados ni contact sheets. Solo quedan versionados los scripts, documentacion y `.gitkeep`.

## Clases

La CNN usara tres clases:

- `visible`: crop real del extintor a partir de la bbox YOLO.
- `partially_occluded`: variante semi-sintetica del crop con una oclusion parcial aproximada.
- `blocked`: variante semi-sintetica del crop con una oclusion fuerte aproximada.

Las clases `partially_occluded` y `blocked` no proceden de labels manuales reales en esta primera fase. Son una base inicial para validar el flujo y arrancar la revision visual.

## Separacion de splits

La generacion mantiene la separacion de origen:

- `data/yolo/train` -> `data/classifier/train`
- `data/yolo/valid` o `data/yolo/val` -> `data/classifier/val`
- `data/yolo/test` -> `data/classifier/test`

No se mezclan derivados entre splits.

## Parametros de la prueba limitada

Comando ejecutado:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_classifier_dataset_from_yolo.py \
  --data-yaml data/yolo/data.yaml \
  --output-dir data/classifier \
  --max-per-split 20 \
  --partial-occlusions-per-object 1 \
  --blocked-occlusions-per-object 1 \
  --image-size 224 \
  --overwrite
```

Nota del entorno: el `python3` del sistema no tenia Pillow instalado, por lo que la prueba se ejecuto con un `.venv` local ignorado por Git. En un entorno con `requirements.txt` instalado, el comando equivalente puede usarse con `python3`.

Parametros efectivos:

- Imagenes originales maximas por split: 20
- Crops visibles por objeto: 1
- Variantes parcialmente ocultas por objeto: 1
- Variantes bloqueadas por objeto: 1
- Padding alrededor de bbox: 0.10
- Tamano de salida: 224x224
- Semilla por defecto: 42
- Limpieza previa: `--overwrite`

## Resultado de la prueba limitada

Validacion ejecutada:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier
```

Conteos generados:

| Split | visible | partially_occluded | blocked | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 29 | 29 | 29 | 87 |
| val | 31 | 31 | 31 | 93 |
| test | 21 | 21 | 21 | 63 |
| total | 81 | 81 | 81 | 243 |

La estructura `train/val/test` y las tres clases existen. Las imagenes generadas fueron legibles en la validacion.

## Contact sheets generadas

Se generaron hojas de revision visual locales:

- `outputs/reports/contact_sheet_classifier_train.jpg`
- `outputs/reports/contact_sheet_classifier_val.jpg`
- `outputs/reports/contact_sheet_classifier_test.jpg`

Estos archivos estan ignorados por Git.

## Generacion completa v1

Fecha: 2026-04-26

Comando ejecutado:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_classifier_dataset_from_yolo.py \
  --data-yaml data/yolo/data.yaml \
  --output-dir data/classifier \
  --partial-occlusions-per-object 1 \
  --blocked-occlusions-per-object 1 \
  --visible-crops-per-object 1 \
  --image-size 224 \
  --overwrite
```

Parametros efectivos:

- Imagenes originales por split: todas las disponibles en el dataset YOLO.
- Crops visibles por objeto: 1
- Variantes parcialmente ocultas por objeto: 1
- Variantes bloqueadas por objeto: 1
- Padding alrededor de bbox: 0.10
- Tamano de salida: 224x224
- Semilla por defecto: 42
- Tamano minimo de crop aceptado: 32 pixeles
- Limpieza previa: `--overwrite`

Resumen de generacion:

- Imagenes YOLO procesadas: train=3791, val=837, test=397
- Objetos aceptados para crops: train=7303, val=1552, test=744
- Labels vacios detectados: 12
- Anotaciones descartadas por crop menor que `--min-crop-size`: 85

Conteos finales:

| Split | visible | partially_occluded | blocked | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 7303 | 7303 | 7303 | 21909 |
| val | 1552 | 1552 | 1552 | 4656 |
| test | 744 | 744 | 744 | 2232 |
| total | 9599 | 9599 | 9599 | 28797 |

La validacion de estructura e imagenes fue correcta:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier
```

Los splits se mantienen separados:

- `data/yolo/train` -> `data/classifier/train`
- `data/yolo/valid` -> `data/classifier/val`
- `data/yolo/test` -> `data/classifier/test`

Contact sheets completas generadas:

- `outputs/reports/contact_sheet_classifier_train_full.jpg`
- `outputs/reports/contact_sheet_classifier_val_full.jpg`
- `outputs/reports/contact_sheet_classifier_test_full.jpg`

Advertencia: `partially_occluded` y `blocked` siguen siendo clases semi-sinteticas. Las oclusiones son suficientes para la v1, pero conviene revisar las contact sheets completas antes de entrenar.

Recomendacion antes de entrenar:

1. Revisar las contact sheets completas.
2. Si la revision visual es aceptable, entrenar la CNN baseline.
3. Documentar metricas reales solo despues del entrenamiento.

## Limitaciones

- Las oclusiones son rectangulos o poligonos simples con colores neutros o tipo carton.
- Algunas oclusiones pueden parecer poco realistas o tapar mas fondo que extintor si el crop contiene mucho contexto.
- La clase `blocked` deja una parte del crop visible, pero no representa todavia todos los casos reales posibles.
- Este dataset no sustituye una revision manual ni ejemplos reales de extintores bloqueados.
- No hay metricas de clasificacion todavia porque la CNN no se ha entrenado.

## Revision visual recomendada

Antes de entrenar la CNN, revisar:

- Que `visible` conserve extintores completos o razonablemente centrados.
- Que `partially_occluded` tape una parte util del extintor sin convertirlo en `blocked`.
- Que `blocked` no sea una imagen uniforme sin contexto.
- Que no haya crops erroneos derivados de labels YOLO defectuosos.
- Que los splits mantengan el origen correcto.

## Siguiente paso

Si la revision visual de la generacion completa v1 es aceptable, entrenar la CNN baseline de estado. No entrenar la CNN hasta aprobar visualmente estos crops.
