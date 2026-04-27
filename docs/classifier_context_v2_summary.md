# Resumen del dataset CNN contextual v2

Fecha: 2026-04-26

## Motivo

La v1 del dataset CNN (`data/classifier/`) sirvio para validar el flujo completo YOLO + crop + CNN, pero la evaluacion preliminar mostro un problema conceptual: al pasar a la CNN un crop muy ajustado al bbox YOLO, se pierde contexto visual. Esto dificulta distinguir entre:

- `visible`
- `partially_occluded`
- `blocked`

La v2 corrige ese punto generando crops ampliados alrededor del extintor. La CNN ve el extintor, parte del entorno y obstaculos semi-sinteticos dentro de una escena mas parecida a la inferencia real.

## Diferencia frente a v1

v1:

```text
imagen completa -> bbox YOLO -> crop ajustado -> CNN
```

v2:

```text
imagen completa -> bbox YOLO -> crop contextual ampliado -> CNN
```

En inferencia el pipeline mantiene la bbox original de YOLO para visualizacion, pero usa el crop contextual para clasificar el estado.

## Salida

Dataset generado localmente en:

```text
data/classifier_context_v2/
```

Estructura:

```text
data/classifier_context_v2/
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

Este dataset no se versiona en Git.

## Parametros

Parametros usados:

- `--context-padding 0.75`
- `--image-size 224`
- `--square-crop`
- `--visible-crops-per-object 1`
- `--partial-occlusions-per-object 1`
- `--blocked-occlusions-per-object 1`
- `--seed 42`
- `--min-crop-size 32`

Las oclusiones `partially_occluded` y `blocked` siguen siendo semi-sinteticas. En v2 se aplican sobre el crop contextual y se intenta que entren desde bordes o zonas cercanas al extintor, con colores neutros y algo de textura.

## Prueba limitada

Comando ejecutado:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_classifier_context_dataset_from_yolo.py \
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

Conteos de la prueba limitada:

| Split | visible | partially_occluded | blocked | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 29 | 29 | 29 | 87 |
| val | 31 | 31 | 31 | 93 |
| test | 21 | 21 | 21 | 63 |
| total | 81 | 81 | 81 | 243 |

La validacion de estructura e imagenes fue correcta.

## Generacion completa v2

Comando ejecutado:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_classifier_context_dataset_from_yolo.py \
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

Resumen de generacion:

- Imagenes YOLO procesadas: train=3791, val=837, test=397
- Objetos aceptados: train=7355, val=1566, test=763
- Labels vacios detectados: 12
- Crops descartados: 0

Conteos finales:

| Split | visible | partially_occluded | blocked | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 7355 | 7355 | 7355 | 22065 |
| val | 1566 | 1566 | 1566 | 4698 |
| test | 763 | 763 | 763 | 2289 |
| total | 9684 | 9684 | 9684 | 29052 |

La validacion completa fue correcta:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier_context_v2
```

Los splits se mantienen separados:

- `data/yolo/train` -> `data/classifier_context_v2/train`
- `data/yolo/valid` -> `data/classifier_context_v2/val`
- `data/yolo/test` -> `data/classifier_context_v2/test`

## Contact sheets

Contact sheets generadas localmente:

- `outputs/reports/contact_sheet_classifier_context_v2_train.jpg`
- `outputs/reports/contact_sheet_classifier_context_v2_val.jpg`
- `outputs/reports/contact_sheet_classifier_context_v2_test.jpg`

Comando usado por split:

```bash
PYTHONPATH=src .venv/bin/python scripts/visualize_classifier_dataset_samples.py \
  --dataset-dir data/classifier_context_v2 \
  --split train \
  --output outputs/reports/contact_sheet_classifier_context_v2_train.jpg \
  --num-samples-per-class 10
```

## Limitaciones

- Las clases `partially_occluded` y `blocked` siguen siendo semi-sinteticas.
- Algunas oclusiones pueden parecer artificiales; se aceptan como v2 inicial para entrenar una baseline y revisar errores.
- El dataset aun no contiene una curacion manual de casos reales bloqueados.
- El padding contextual mejora el contexto, pero puede incluir varios extintores si la imagen original los contiene.

## Siguiente paso

Entrenar una CNN baseline usando:

```text
data/classifier_context_v2
```

Comando recomendado cuando se apruebe la revision visual:

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

No se ha entrenado la CNN en esta fase.
