# Entrenamiento CNN contextual v2

Fecha: 2026-04-26

## Objetivo

Entrenar una baseline seria de la CNN de clasificacion de estado usando el dataset contextual v2:

```text
data/classifier_context_v2
```

Este modelo no se considera el modelo final absoluto del sistema completo. Es la baseline CNN recomendada para continuar con la integracion YOLO + crop contextual + CNN.

## Motivo de usar v2

La v1 (`data/classifier/`) usaba crops mas ajustados al bbox YOLO. Sirvio para validar el flujo y el entrenamiento corto, pero perdia contexto visual. La v2 amplia el crop alrededor del extintor para que la CNN vea entorno, obstaculos y espacio alrededor.

## Dataset

Clases:

- `visible`
- `partially_occluded`
- `blocked`

Conteos validados:

| Split | visible | partially_occluded | blocked | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 7355 | 7355 | 7355 | 22065 |
| val | 1566 | 1566 | 1566 | 4698 |
| test | 763 | 763 | 763 | 2289 |
| total | 9684 | 9684 | 9684 | 29052 |

Validacion ejecutada:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier_context_v2
```

Resultado: estructura correcta, clases equilibradas e imagenes legibles.

## Entorno

Entrenamiento ejecutado en Docker con el servicio `app-gpu`.

- Dispositivo usado: `cuda`
- GPU: NVIDIA GeForce GTX 1650
- PyTorch: `2.8.0+cu126`
- Dataset: `data/classifier_context_v2`

Se hizo un primer intento en sesion interactiva de Docker que se interrumpio con codigo 130 despues de la epoca 7. No fue un error de datos ni de metricas no finitas. Para evitar perder progreso si volviese a ocurrir, se actualizo el script para guardar metricas por epoca de forma incremental y se relanzo el entrenamiento en un contenedor Docker desacoplado. La ejecucion documentada abajo es la ejecucion completa de 30 epocas.

## Comando de entrenamiento

```bash
docker compose -f docker/docker-compose.yml run -d --name cnn_context_v2_train --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m fire_extinguisher_inspection.classification.train_classifier \
  --dataset-path data/classifier_context_v2 \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --image-size 224 \
  --output-model-path models/classifier/state_classifier_context_v2.pt
```

Hiperparametros:

- Epocas: 30
- Batch size: 32
- Learning rate: 0.001
- Image size: 224
- Arquitectura: `simple_cnn`
- `num_workers`: 0

Modelo local generado:

```text
models/classifier/state_classifier_context_v2.pt
```

Metricas locales generadas:

```text
models/classifier/state_classifier_context_v2.metrics.json
```

Ambos archivos estan ignorados por Git.

## Metricas por epoca

| Epoca | train_loss | train_accuracy | val_loss | val_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8618 | 0.5828 | 0.7382 | 0.6577 |
| 2 | 0.6959 | 0.6736 | 0.5796 | 0.7373 |
| 3 | 0.5836 | 0.7229 | 0.5049 | 0.7656 |
| 4 | 0.5115 | 0.7596 | 0.4297 | 0.7963 |
| 5 | 0.4687 | 0.7745 | 0.4597 | 0.7859 |
| 6 | 0.4307 | 0.7924 | 0.3807 | 0.8218 |
| 7 | 0.4135 | 0.7996 | 0.3579 | 0.8314 |
| 8 | 0.3937 | 0.8070 | 0.3921 | 0.8227 |
| 9 | 0.3897 | 0.8114 | 0.3381 | 0.8380 |
| 10 | 0.3736 | 0.8210 | 0.3426 | 0.8342 |
| 11 | 0.3659 | 0.8220 | 0.3587 | 0.8329 |
| 12 | 0.3547 | 0.8265 | 0.3308 | 0.8465 |
| 13 | 0.3507 | 0.8301 | 0.3201 | 0.8525 |
| 14 | 0.3479 | 0.8334 | 0.3599 | 0.8255 |
| 15 | 0.3368 | 0.8375 | 0.2994 | 0.8631 |
| 16 | 0.3343 | 0.8392 | 0.2980 | 0.8525 |
| 17 | 0.3279 | 0.8405 | 0.3359 | 0.8463 |
| 18 | 0.3275 | 0.8435 | 0.2894 | 0.8689 |
| 19 | 0.3199 | 0.8451 | 0.2843 | 0.8670 |
| 20 | 0.3157 | 0.8484 | 0.2934 | 0.8633 |
| 21 | 0.3139 | 0.8516 | 0.2790 | 0.8689 |
| 22 | 0.3068 | 0.8546 | 0.2844 | 0.8659 |
| 23 | 0.3031 | 0.8553 | 0.2745 | 0.8704 |
| 24 | 0.2989 | 0.8595 | 0.2820 | 0.8702 |
| 25 | 0.2977 | 0.8586 | 0.2831 | 0.8689 |
| 26 | 0.2958 | 0.8608 | 0.2724 | 0.8723 |
| 27 | 0.2884 | 0.8653 | 0.2559 | 0.8842 |
| 28 | 0.2870 | 0.8660 | 0.2687 | 0.8814 |
| 29 | 0.2826 | 0.8686 | 0.2683 | 0.8808 |
| 30 | 0.2842 | 0.8646 | 0.3038 | 0.8563 |

Mejor epoca segun validacion:

```text
epoca 27, val_accuracy=0.8842060451255853
```

No se detectaron valores NaN ni infinitos.

## Evaluacion en test

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluate_classifier_on_test.py \
  --dataset-dir data/classifier_context_v2/test \
  --model-path models/classifier/state_classifier_context_v2.pt \
  --image-size 224 \
  --output-dir outputs/reports/classifier_context_v2_test
```

Resultados:

- Accuracy global: 0.8754914809960681
- Muestras evaluadas: 2289
- Dispositivo: `cuda`

Metricas por clase:

| Clase | precision | recall | F1 | muestras |
| --- | ---: | ---: | ---: | ---: |
| visible | 0.993455 | 0.994758 | 0.994106 | 763 |
| partially_occluded | 0.829201 | 0.788991 | 0.808596 | 763 |
| blocked | 0.804756 | 0.842726 | 0.823303 | 763 |

Matriz de confusion, filas = clase real, columnas = prediccion:

| real \ pred | visible | partially_occluded | blocked |
| --- | ---: | ---: | ---: |
| visible | 759 | 4 | 0 |
| partially_occluded | 5 | 602 | 156 |
| blocked | 0 | 120 | 643 |

Salidas locales:

- `outputs/reports/classifier_context_v2_test/metrics.json`
- `outputs/reports/classifier_context_v2_test/confusion_matrix.csv`
- `outputs/reports/classifier_context_v2_test/misclassified_examples.json`

Estas salidas estan ignoradas por Git.

## Inferencia individual

Comando ejecutado dentro de Docker para tres ejemplos:

```bash
python3 -m fire_extinguisher_inspection.classification.predict_classifier \
  --model-path models/classifier/state_classifier_context_v2.pt \
  --image data/classifier_context_v2/test/visible/test_00014_jpg_rf_9abb87c3c98effee89d928be825797b6_obj0_visible.jpg \
  --image-size 224

python3 -m fire_extinguisher_inspection.classification.predict_classifier \
  --model-path models/classifier/state_classifier_context_v2.pt \
  --image data/classifier_context_v2/test/partially_occluded/test_00014_jpg_rf_9abb87c3c98effee89d928be825797b6_obj0_partial_0.jpg \
  --image-size 224

python3 -m fire_extinguisher_inspection.classification.predict_classifier \
  --model-path models/classifier/state_classifier_context_v2.pt \
  --image data/classifier_context_v2/test/blocked/test_00014_jpg_rf_9abb87c3c98effee89d928be825797b6_obj0_blocked_0.jpg \
  --image-size 224
```

Resultados:

| Imagen real | Prediccion | Confianza |
| --- | --- | ---: |
| visible | visible | 0.9997567534446716 |
| partially_occluded | partially_occluded | 0.892981767654419 |
| blocked | partially_occluded | 0.5840994715690613 |

El ejemplo individual `blocked` se confundio con `partially_occluded`, coherente con la matriz de confusion: el mayor error del modelo esta entre esas dos clases.

## Comparacion con v1

La prueba anterior de v1 (`docs/cnn_training_test.md`) uso crops ajustados y 5 epocas. Sus metricas no son comparables directamente con v2 porque:

- v1 ve menos contexto y el problema es mas facil/artificial.
- v2 usa crops contextuales y es la version alineada con el pipeline de inferencia.
- `partially_occluded` y `blocked` siguen siendo semi-sinteticas en ambas versiones.

Aunque la v1 tuvo una accuracy mayor en su test preliminar, la v2 es la base recomendada para el pipeline final porque corrige la perdida de contexto detectada durante la integracion preliminar.

## Problemas encontrados

- Primer intento de entrenamiento interrumpido con codigo 130 tras la epoca 7 en una sesion Docker interactiva.
- Se relanzo en un contenedor Docker desacoplado y completo las 30 epocas.
- Se actualizo `train_classifier.py` para guardar metricas incrementales y detectar metricas no finitas.
- No hubo NaN, infinitos, errores de dataset ni errores de GPU en la ejecucion completa.

## Conclusion

El modelo queda aceptado como baseline CNN contextual v2.

Limitacion importante: estas metricas pertenecen al clasificador CNN sobre crops contextuales del dataset v2. No son metricas finales del sistema completo YOLO + CNN sobre imagenes reales, y las clases `partially_occluded` y `blocked` siguen siendo semi-sinteticas.

Siguiente paso recomendado: probar el pipeline completo con YOLO + CNN contextual v2, usando este checkpoint como modelo CNN.
