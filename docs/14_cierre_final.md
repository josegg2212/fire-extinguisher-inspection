# Resumen final del proyecto

Fecha de cierre operativo: 2026-04-30.

## Decision final

Se deja como configuracion final:

- Detector YOLO: `models/yolo/extinguisher_yolo_v1/weights/best.pt`
- Clasificador CNN: `models/classifier/state_classifier_context_v7_real_tuned_with_manual_tests.pt`
- `visibility_verifier`: desactivado
- `yolo_imgsz`: `1920`
- `detection_confidence_threshold`: `0.05`
- `classifier_context_padding`: `0.75`

Motivo: en las validaciones manuales limpias, el YOLO era razonablemente estable y el cuello de botella estaba en la CNN. La combinacion final sacrifica algo de precision en el lote A, pero mejora el lote B y el agregado frente a la opcion con verificador.

## Resultados conservados

Validacion manual A:

- Carpeta: `outputs/final_validation/batch_a_20260430`
- 36 crops totales
- 4 excluidos sin extintor
- 32 evaluados
- Mejor historico de ese lote: v7 + verifier, `imgsz=1792`
- Accuracy: `23/32 = 0.7188`

Validacion manual B:

- Carpeta: `outputs/final_validation/batch_b_20260430`
- 51 crops totales
- 1 excluido sin extintor
- 50 evaluados
- Configuracion final: v7 sin verifier, `imgsz=1920`, `conf=0.05`
- Accuracy sobre detectados: `32/45 = 0.7111`
- Accuracy contando no detecciones: `32/50 = 0.6400`

Agregado de los dos ultimos lotes limpios:

- Configuracion final: `53/77 = 0.6883` sobre detectados
- Configuracion final contando no detecciones: `53/82 = 0.6463`
- Configuracion anterior con verifier: `51/75 = 0.6800` sobre detectados
- Configuracion anterior contando no detecciones: `51/82 = 0.6220`

Test interno de la CNN:

- Carpeta: `outputs/final_validation/classifier_context_v7_test`
- Accuracy: `0.7280`
- Muestras: `783`
- F1 visible: `0.8376`
- F1 partially_occluded: `0.6430`
- F1 blocked: `0.6963`

## Diagnostico

El problema principal no era el YOLO, sino la frontera entre `visible` y `partially_occluded` en la CNN. En imagenes reales o realistas, muchos visibles con fondo dificil, baja resolucion, cristal, sombras o objetos cercanos caen como `partially_occluded`. `blocked` y muchos `partially_occluded` quedaron aceptables para el objetivo practico.

Entrenar mas epocas no era la palanca mas clara: la CNN v7 ya venia de ajuste fino y el patron de errores apunta mas a calidad/distribucion de datos y criterio de etiqueta que a falta bruta de entrenamiento.

## Estructura final

Se ha limpiado el proyecto dejando:

- `data/yolo`: dataset YOLO final.
- `data/classifier_context_final_v7`: dataset CNN final, materializado sin symlinks.
- `models/yolo/extinguisher_yolo_v1`: entrenamiento final YOLO y `weights/best.pt`.
- `models/classifier`: checkpoint final CNN y metricas.
- `outputs/final_validation`: evidencia final de validacion.
- `docs/`: documentos numerados con la evolución del proyecto.
- `scripts`: comandos finales reducidos para demo, inferencia, validacion y prueba basica.

Tamanio antes de limpieza: aprox. `4.8G`.

Tamanio tras limpieza: aprox. `707M`.
