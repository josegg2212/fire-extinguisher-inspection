# Datos sinteticos y ajuste final

## Motivo

El dataset inicial ayudaba a entrenar el modelo, pero varias oclusiones eran demasiado simples. La CNN aprendía bien esos casos, aunque después fallaba con imágenes manuales más realistas.

Por ese motivo se generaron ejemplos con más variación visual:

- distintos colores de extintor;
- obstáculos con textura;
- fondos más variados;
- recortes con más contexto alrededor del extintor;
- casos visibles, parcialmente ocultos y bloqueados.

## Ajuste

La versión final de la CNN se ajustó con el dataset contextual y con ejemplos manuales revisados. El objetivo no fue obtener una puntuación perfecta, sino reducir los fallos más graves del pipeline en imágenes cercanas al uso real.

## Resultado

La versión final conservada es:

```text
models/classifier/state_classifier_context_v7_real_tuned_with_manual_tests.pt
```

El detector final conservado es:

```text
models/yolo/extinguisher_yolo_v1/weights/best.pt
```

La validación final se guarda en:

```text
outputs/final_validation/
```
