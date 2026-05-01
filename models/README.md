# Modelos

Esta carpeta guarda los resultados ligeros de entrenamiento y define las rutas locales de los pesos finales.

Pesos esperados para ejecutar el proyecto:

- YOLO: `models/yolo/extinguisher_yolo_v1/weights/best.pt`
- CNN: `models/classifier/state_classifier_context_v7_real_tuned_with_manual_tests.pt`

Tambien se usa `models/yolo/base/yolo26n.pt` como peso base local para entrenamientos YOLO futuros.

Los archivos de pesos (`.pt`, `.pth`, `.onnx`, `.engine`) no se versionan en Git. Si se clona el repositorio en otro equipo, estos archivos deben copiarse localmente en las rutas indicadas.

Se conservan metricas, graficas y configuraciones ligeras cuando ayudan a explicar el entrenamiento final. Se eliminaron checkpoints antiguos de la CNN y entrenamientos YOLO de prueba.
