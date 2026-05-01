# Pruebas manuales

## Objetivo

Después de validar el pipeline con datos de test, se probaron imágenes manuales más parecidas a un caso real. Estas imágenes incluían extintores con fondos distintos, distintos colores, reflejos, baja resolución, varios extintores en la misma imagen y casos con obstáculos.

## Procedimiento

1. Se ejecutó YOLO sobre las imágenes completas.
2. Se revisaron las cajas detectadas.
3. Se pasó cada recorte contextual por la CNN.
4. Se comparó la predicción con una etiqueta manual.
5. En imágenes con varios extintores se evaluó cada extintor por separado.

## Hallazgo principal

El detector YOLO funcionó de forma razonable en la mayoría de casos. El problema principal apareció en la CNN, especialmente en la separación entre `visible` y `partially_occluded`.

Los errores más frecuentes fueron:

- extintores visibles clasificados como parcialmente ocultos;
- extintores con cristal, sombras o elementos cercanos tratados como oclusiones;
- casos con poca resolución donde la CNN no veía suficiente detalle.

## Decisión

Se decidió no seguir ajustando únicamente el umbral de decisión. El problema no era solo de umbral, sino de distribución de datos y de criterio visual. Por eso se incorporaron ejemplos manuales y se revisó el dataset de entrenamiento de la CNN.
