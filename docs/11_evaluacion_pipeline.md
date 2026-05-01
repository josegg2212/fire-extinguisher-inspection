# Evaluacion del pipeline integrado v1

Fecha: 2026-04-27

## Objetivo

Probar el pipeline completo con los modelos nuevos:

- Detector YOLO v1: `models/yolo/extinguisher_yolo_v1/weights/best.pt`
- CNN contextual v2: `models/classifier/state_classifier_context_v2.pt`

No se entreno ningun modelo en esta fase y no se modificaron datasets ni pesos.

## Tecnica usada

El flujo evaluado fue:

```text
imagen completa
-> YOLO v1 detecta bbox original
-> la bbox original se usa para pintar la deteccion y guardarla en JSON
-> se calcula un crop contextual ampliado alrededor de esa bbox
-> la CNN contextual v2 clasifica el estado sobre el crop contextual
```

Parametros:

- `confidence_threshold`: 0.25
- `classifier_context_padding`: 0.75
- `classifier_square_crop`: true
- `image_size` de la CNN: 224

Se verifico sobre los JSON generados que todas las detecciones guardaron `classifier_crop_bbox`, que ese crop contiene la bbox YOLO original y que no coincide exactamente con la bbox ajustada. Por tanto, la CNN recibio crop contextual ampliado.

## Modelos de referencia

Metricas YOLO v1 en `data/yolo` test, ya documentadas en `docs/05_entrenamiento_yolo.md`:

| Metrica | Valor |
| --- | ---: |
| precision(B) | 0.9455517560255178 |
| recall(B) | 0.9187418086500655 |
| mAP50(B) | 0.9737560421353847 |
| mAP75(B) | 0.8939457956072638 |
| mAP50-95(B) | 0.838757968330097 |

Metricas CNN contextual v2 en `data/classifier_context_v2/test`, ya documentadas en `docs/09_entrenamiento_cnn_contextual.md`:

| Metrica | Valor |
| --- | ---: |
| accuracy global | 0.8754914809960681 |
| muestras | 2289 |

| Clase | precision | recall | F1 | muestras |
| --- | ---: | ---: | ---: | ---: |
| visible | 0.993455 | 0.994758 | 0.994106 | 763 |
| partially_occluded | 0.829201 | 0.788991 | 0.808596 | 763 |
| blocked | 0.804756 | 0.842726 | 0.823303 | 763 |

## Evaluacion sobre imagenes completas

Dataset usado: `data/yolo/test/images`

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluar_pipeline.py \
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

Resultados:

| Campo | Valor |
| --- | ---: |
| imagenes procesadas | 50 |
| imagenes con deteccion | 50 |
| imagenes sin deteccion | 0 |
| detecciones totales | 54 |
| errores | 0 |

Distribucion de estados predichos por deteccion:

| Estado | Detecciones |
| --- | ---: |
| visible | 54 |

En este conjunto no se calculo accuracy de estado porque las imagenes completas de `data/yolo/test/images` no tienen etiqueta de estado final en esta evaluacion.

Salidas locales:

- `outputs/detections/integrated_pipeline_v1/annotated/`: 50 imagenes anotadas
- `outputs/detections/integrated_pipeline_v1/crops/`: 54 crops contextuales
- `outputs/detections/integrated_pipeline_v1/json/`: 50 JSON por imagen
- `outputs/detections/integrated_pipeline_v1/integrated_pipeline_v1_summary.json`
- `outputs/reports/integrated_pipeline_v1_contact_sheet.jpg`

## Prueba balanceada de estados

Dataset usado: `data/classifier_context_v2/test`

Esta prueba usa crops contextuales derivados/semi-sinteticos por clase. No sustituye una evaluacion real con imagenes completas de extintores bloqueados u ocultos, pero comprueba el comportamiento integrado ante ejemplos `visible`, `partially_occluded` y `blocked`.

Comando ejecutado:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluar_pipeline.py \
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

Resultados globales:

| Campo | Valor |
| --- | ---: |
| imagenes procesadas | 60 |
| imagenes con deteccion | 49 |
| imagenes sin deteccion | 11 |
| detecciones totales | 58 |
| errores | 0 |
| accuracy de estado sobre imagenes con deteccion | 0.8979591836734694 |

Imagenes por clase esperada:

| Clase esperada | Imagenes |
| --- | ---: |
| visible | 20 |
| partially_occluded | 20 |
| blocked | 20 |

Cobertura de deteccion por clase esperada:

| Clase esperada | Con deteccion | Sin deteccion | Cobertura |
| --- | ---: | ---: | ---: |
| visible | 20 | 0 | 1.00 |
| partially_occluded | 19 | 1 | 0.95 |
| blocked | 10 | 10 | 0.50 |

Predicciones por clase, contando todas las detecciones:

| Estado predicho | Detecciones |
| --- | ---: |
| visible | 23 |
| partially_occluded | 29 |
| blocked | 6 |

Matriz preliminar de estado usando la deteccion de mayor confianza por imagen detectada:

| real \ pred | visible | partially_occluded | blocked |
| --- | ---: | ---: | ---: |
| visible | 20 | 0 | 0 |
| partially_occluded | 0 | 19 | 0 |
| blocked | 0 | 5 | 5 |

Las 10 imagenes `blocked` sin deteccion y la imagen `partially_occluded` sin deteccion no entran en esta matriz porque la CNN no se ejecuta si YOLO no detecta extintor.

Salidas locales:

- `outputs/detections/integrated_pipeline_v1_balanced/annotated/`: 49 imagenes anotadas
- `outputs/detections/integrated_pipeline_v1_balanced/crops/`: 58 crops contextuales
- `outputs/detections/integrated_pipeline_v1_balanced/json/`: 60 JSON por imagen
- `outputs/detections/integrated_pipeline_v1_balanced/integrated_pipeline_v1_summary.json`
- `outputs/reports/integrated_pipeline_v1_balanced_contact_sheet.jpg`

## Observaciones visuales

- En las imagenes completas, las cajas YOLO se dibujan sobre la bbox original y las clasificaciones aparecen como `visible`, coherente con el contenido revisado.
- En las imagenes completas hay casos con mas de un extintor, por eso 50 imagenes produjeron 54 detecciones.
- En la prueba balanceada, `visible` y `partially_occluded` mantienen buena cobertura de deteccion.
- En `blocked`, la cobertura baja a 10/20. Es el comportamiento esperado cuando la oclusion elimina demasiada evidencia visual para que YOLO localice el extintor.
- En los `blocked` detectados, la CNN separa parte de los casos como `blocked`, pero tambien confunde algunos como `partially_occluded`.

## Limitaciones

- `partially_occluded` y `blocked` siguen siendo clases semi-sinteticas en el dataset contextual v2.
- La prueba balanceada no es una evaluacion final sobre imagenes reales completas bloqueadas.
- Si YOLO no detecta el extintor, la CNN no actua.
- Las imagenes totalmente ocultas no pueden resolverse solo con vision si no hay evidencia visual suficiente.
- Esta evaluacion integrada v1 no debe presentarse como resultado final absoluto del proyecto completo.

## Conclusion

El pipeline integrado YOLO v1 + CNN contextual v2 funciona de extremo a extremo con crop contextual ampliado. La evaluacion sobre imagenes completas procesa correctamente el split YOLO test usado y la prueba balanceada confirma que la clasificacion de estado se ejecuta sobre crops contextuales.

El principal punto pendiente es reunir o validar casos reales completos con extintores parcialmente bloqueados y bloqueados, porque la limitacion actual mas clara aparece antes de la CNN: cuando YOLO no detecta por falta de evidencia visual, no existe crop que clasificar.
