# Indice de desarrollo

Esta carpeta resume el desarrollo del proyecto en orden cronologico. La idea es que pueda leerse como una memoria breve: primero se revisa la base del repositorio, despues se prepara el detector, luego se construye el clasificador y finalmente se valida el pipeline completo.

## Orden recomendado

1. `01_revision_inicial.md`: comprobacion inicial del repositorio.
2. `02_dataset_yolo.md`: preparacion del dataset de deteccion.
3. `03_labels_yolo.md`: revision de etiquetas vacias.
4. `04_preparacion_yolo.md`: validaciones previas al entrenamiento.
5. `05_entrenamiento_yolo.md`: entrenamiento del detector.
6. `06_dataset_cnn_inicial.md`: primer dataset para la CNN.
7. `07_prueba_cnn_inicial.md`: primera prueba de entrenamiento de la CNN.
8. `08_dataset_cnn_contextual.md`: correccion del dataset para conservar contexto.
9. `09_entrenamiento_cnn_contextual.md`: entrenamiento de la CNN contextual.
10. `10_evaluacion_preliminar.md`: primeras pruebas del pipeline.
11. `11_evaluacion_pipeline.md`: evaluacion integrada YOLO + CNN.
12. `12_pruebas_manuales.md`: diagnostico con imagenes manuales.
13. `13_datos_sinteticos.md`: mejora de datos sinteticos y ajuste fino.
14. `14_cierre_final.md`: configuracion final y resultados conservados.
15. `15_fundamento_tecnico.md`: justificacion tecnica de YOLO, YOLO26n, CNN y datasets.

Los comandos actuales para usar el proyecto estan en el `README.md` principal y en `scripts/README.md`. Algunos documentos historicos mencionan scripts de desarrollo que ya no se conservan en la carpeta principal, porque el repositorio final deja solo los comandos necesarios para demo, inferencia y validacion.
