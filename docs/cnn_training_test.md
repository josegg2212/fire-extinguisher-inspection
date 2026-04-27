# Entrenamiento corto de prueba CNN

Fecha: 2026-04-26

> Nota 2026-04-26: este entrenamiento corto uso la v1 del dataset CNN (`data/classifier/`). Sirvio para validar que el entrenamiento funciona en Docker GPU, pero no debe considerarse base definitiva. La siguiente CNN baseline deberia entrenarse con `data/classifier_context_v2/`, que corrige la falta de contexto en los crops.

## Objetivo

Validar que el pipeline de entrenamiento de la CNN de estado funciona con el dataset completo generado en `data/classifier/`.

Este entrenamiento no es el modelo definitivo. Es una prueba corta de 5 epocas.

## Dataset usado

Ruta:

```text
data/classifier
```

Clases:

- `visible`
- `partially_occluded`
- `blocked`

Conteos validados:

| Split | visible | partially_occluded | blocked | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 7303 | 7303 | 7303 | 21909 |
| val | 1552 | 1552 | 1552 | 4656 |
| test | 744 | 744 | 744 | 2232 |
| total | 9599 | 9599 | 9599 | 28797 |

Validacion ejecutada en Docker:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/check_classifier_dataset_structure.py \
  --dataset-dir data/classifier
```

Resultado: estructura correcta, clases equilibradas e imagenes legibles.

## Entorno

Entrenamiento ejecutado en Docker con el servicio `app-gpu`.

- Dispositivo usado: `cuda`
- GPU: NVIDIA GeForce GTX 1650
- PyTorch: `2.8.0+cu126`
- Torchvision: `0.23.0+cu126`

Nota: el entorno local `.venv` no tenia `torch`/`torchvision`; se uso Docker como entorno de entrenamiento.

## Comando de entrenamiento

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

Parametros:

- Epocas: 5
- Batch size: 32
- Learning rate: 0.001
- Image size: 224
- Modelo local generado: `models/classifier/extinguisher_status_cnn_test.pth`
- Metricas locales generadas: `models/classifier/extinguisher_status_cnn_test.metrics.json`

Los dos archivos estan ignorados por Git.

## Metricas obtenidas

| Epoca | train_loss | train_accuracy | val_loss | val_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4245 | 0.8271 | 0.1977 | 0.9298 |
| 2 | 0.2661 | 0.8984 | 0.2475 | 0.9008 |
| 3 | 0.2078 | 0.9200 | 0.1675 | 0.9330 |
| 4 | 0.1730 | 0.9356 | 0.1246 | 0.9521 |
| 5 | 0.1604 | 0.9406 | 0.1520 | 0.9409 |

Mejor accuracy de validacion:

```text
0.9521048109965635 en epoca 4
```

No se detectaron valores NaN ni infinitos en las metricas.

## Prueba de inferencia

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m fire_extinguisher_inspection.classification.predict_classifier \
  --model-path models/classifier/extinguisher_status_cnn_test.pth \
  --image data/classifier/test/visible/test_00014_jpg_rf_9abb87c3c98effee89d928be825797b6_obj0_visible.jpg \
  --image-size 224
```

Resultado:

```json
{
  "status_prediction": "visible",
  "status_confidence": 0.9993841648101807,
  "probabilities": {
    "visible": 0.9993841648101807,
    "partially_occluded": 0.0006156823947094381,
    "blocked": 0.00000012807726079699933
  }
}
```

La CLI real usa `--image` para indicar el crop de entrada.

## Problemas encontrados

- El entorno local `.venv` no tenia `torch` ni `torchvision`.
- Se construyo y uso la imagen Docker `app-gpu`.
- La construccion de la imagen fue lenta por las dependencias CUDA, pero finalizo correctamente.
- No hubo errores durante el entrenamiento ni durante la inferencia de prueba.

## Decision final

El pipeline de entrenamiento CNN esta listo para lanzar un entrenamiento baseline real cuando se apruebe. Antes del entrenamiento largo conviene revisar de nuevo que los pesos de prueba no se confundan con el modelo definitivo y elegir nombre/ruta final para el checkpoint real.
