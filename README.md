# Inspección visual de extintores

Este proyecto implementa un sistema de inspección visual de extintores. El objetivo es localizar extintores en imágenes y clasificar su estado de accesibilidad en tres clases:

- `visible`
- `partially_occluded`
- `blocked`

La solución final combina un detector YOLO para encontrar el extintor y una CNN para clasificar el estado del recorte contextual. El proyecto no se plantea como una prueba aislada, sino como un flujo completo: preparación de datos, entrenamiento, validación, pruebas manuales y uso final mediante consola o API.

## Importancia práctica

Un extintor visible y accesible es un elemento crítico de seguridad. En edificios, naves industriales, aparcamientos, colegios, hospitales, comercios o almacenes, una inspección automática puede ayudar a detectar situaciones que normalmente se revisarían de forma manual y repetitiva.

Este sistema puede servir como base para:

- inspección preventiva de zonas con muchos extintores;
- revisión periódica mediante cámaras fijas;
- integración en un robot inspector que recorra instalaciones;
- generación de evidencias visuales para mantenimiento;
- detección temprana de extintores tapados por cajas, muebles u otros objetos;
- apoyo a auditorías internas de seguridad.

El sistema no sustituye una revisión normativa completa, pero puede reducir trabajo manual y señalar casos que requieren atención.

## Funcionamiento general

El pipeline sigue estos pasos:

1. Recibe una imagen.
2. YOLO detecta las cajas donde aparecen extintores.
3. Cada caja se amplía para conservar contexto alrededor del extintor.
4. La CNN clasifica el recorte contextual.
5. Se devuelve un JSON con las detecciones, las probabilidades y, si se solicita, una imagen anotada.

La configuración final está en `config/default.yaml`.

## Fundamento técnico

El sistema se separó en dos modelos porque son problemas distintos. YOLO localiza cada extintor en la imagen y la CNN clasifica el estado del recorte resultante. Esta división permite trabajar con imágenes que contienen varios extintores y evita clasificar la escena completa como si solo hubiera un objeto.

Se usó `yolo26n.pt` como base del detector porque es una variante ligera, adecuada para una sola clase y viable en la GPU local. El dataset de detección se preparó en formato YOLO/Roboflow con cajas para la clase `fire_extinguisher`.

La CNN de estado recibe crops RGB de `224x224`. Está formada por bloques `Conv2d + BatchNorm + ReLU + MaxPool`, una agregación final con `AdaptiveAvgPool` y capas densas con `Dropout`. La salida tiene tres clases: `visible`, `partially_occluded` y `blocked`.

El entorno se ejecuta con Docker para mantener dependencias reproducibles y separar el sistema local del proyecto. Para entrenamiento e inferencia completa se usa el servicio `app-gpu`, que aprovecha CUDA cuando hay GPU disponible.

El detalle técnico completo está en `docs/15_fundamento_tecnico.md`.

## Estructura del repositorio

```text
fire-extinguisher-inspection/
├── config/
├── data/
├── demo/
├── docs/
├── docker/
├── models/
├── outputs/
├── scripts/
├── src/fire_extinguisher_inspection/
└── tests/
```

## Qué contiene cada carpeta

`config/`

Contiene la configuración principal del proyecto. `default.yaml` fija las rutas de modelos, datasets, umbrales y parámetros de inferencia. `classes.yaml` define las clases de detección y clasificación.

`data/`

Contiene los datasets locales. No se suben a Git porque son pesados.

- `data/yolo/`: dataset usado para entrenar y evaluar el detector YOLO.
- `data/classifier_context_final_v7/`: dataset final de la CNN, materializado sin enlaces simbólicos.
- `data/raw/`: carpeta sencilla para dejar imágenes nuevas de prueba.

`demo/`

Contiene las imágenes seleccionadas para enseñar el funcionamiento final del sistema. La demo está organizada por clase:

```text
demo/imagenes/
├── blocked/
├── partially_occluded/
└── visible/
```

Las imágenes de esta carpeta deben ser ejemplos revisados manualmente para enseñar el funcionamiento del sistema.

La selección actual contiene 27 imágenes verificadas con la configuración final:

- 7 `visible`;
- 12 `partially_occluded`;
- 8 `blocked`.

`docs/`

Contiene la memoria del desarrollo. Los documentos están numerados para poder leer el proyecto como una secuencia:

- preparación inicial;
- dataset YOLO;
- entrenamiento YOLO;
- dataset CNN;
- entrenamiento CNN;
- evaluación del pipeline;
- pruebas manuales;
- fundamento técnico;
- cierre final.

El índice está en `docs/00_indice.md`.

`models/`

Contiene las rutas locales de los modelos y los resultados ligeros de entrenamiento. Los pesos `.pt` no se versionan en Git, pero estas son las rutas esperadas para ejecutar el proyecto:

- YOLO final: `models/yolo/extinguisher_yolo_v1/weights/best.pt`
- CNN final: `models/classifier/state_classifier_context_v7_real_tuned_with_manual_tests.pt`
- Peso base YOLO local: `models/yolo/base/yolo26n.pt`

`outputs/`

Contiene salidas generadas. La parte importante que se conserva es:

- `outputs/final_validation/`: validaciones finales, labels manuales, resúmenes y contact sheets.

Las carpetas `outputs/detections`, `outputs/reports`, `outputs/crops` y `outputs/logs` quedan preparadas para nuevas ejecuciones.

`scripts/`

Contiene solo los comandos necesarios para usar y comprobar el proyecto final. El detalle está en `scripts/README.md`.

`src/fire_extinguisher_inspection/`

Contiene el código principal del sistema:

- `detection/`: entrenamiento y uso de YOLO;
- `classification/`: modelo CNN, dataset y predicción;
- `preprocessing/`: recortes y preparación de imágenes;
- `pipeline/`: unión de YOLO y CNN;
- `visualization/`: dibujo de resultados;
- `api/`: servicio FastAPI.

`tests/`

Contiene pruebas automáticas de configuración, API, dataset y comportamiento básico del pipeline.

## Desarrollo del proyecto

El proyecto avanzó en varias fases.

Primero se preparó el repositorio, la configuración y la estructura de datos. Después se revisó el dataset YOLO, se corrigieron rutas y se entrenó un detector de extintores. Ese detector funcionó razonablemente bien en las pruebas posteriores.

La primera CNN usaba recortes demasiado ajustados al extintor. Esto permitía entrenar, pero no daba suficiente contexto para distinguir entre un extintor visible y uno parcialmente tapado. Por eso se generó un dataset contextual, donde cada recorte conserva parte del entorno.

Más adelante se hicieron pruebas manuales con imágenes más cercanas a un caso real. Ahí se observó que el problema principal no era el detector, sino la separación entre `visible` y `partially_occluded`. Se ajustó la CNN con datos más variados y se conservaron las validaciones finales.

La decisión final fue dejar una configuración estable, sin seguir aumentando complejidad:

- YOLO final;
- CNN contextual v7;
- sin verificador auxiliar;
- inferencia con `yolo_imgsz=1920`;
- umbral YOLO `0.05`.

## Modelos finales

Rutas por defecto:

```yaml
modelos:
  yolo: models/yolo/extinguisher_yolo_v1/weights/best.pt
  yolo_base: models/yolo/base/yolo26n.pt
  cnn: models/classifier/state_classifier_context_v7_real_tuned_with_manual_tests.pt
  visibility_verifier: null
```

Parámetros principales:

```yaml
inferencia:
  detection_confidence_threshold: 0.05
  yolo_imgsz: 1920
  cnn_image_size: 224
  classifier_context_padding: 0.75
```

## Resultados conservados

La evidencia final está en `outputs/final_validation/`.

Resumen:

- test interno CNN: accuracy `0.7280` sobre `783` muestras;
- validación manual A: `23/32 = 0.7188`;
- validación manual B con configuración final: `32/45 = 0.7111` sobre detecciones;
- agregado de los dos últimos lotes limpios: `53/77 = 0.6883` sobre detecciones.

El detalle está en `docs/14_cierre_final.md`.

## Ejecutar la demo

La demo se ejecuta con las imágenes de:

```text
demo/imagenes/
├── blocked/
├── partially_occluded/
└── visible/
```

Ejecutarla desde terminal:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluar_pipeline.py \
  --images-dir demo/imagenes \
  --output-dir outputs/detections/demo \
  --max-images-per-class 10 \
  --save-json
```

Los resultados se guardan en:

```text
outputs/detections/demo/
├── annotated/
├── json/
└── resumen_pipeline.json
```

Con la selección actual, la demo queda verificada con `27/27` imágenes clasificadas correctamente.

Ejecutar una imagen concreta de la demo:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/inferir_imagen.py \
  --image demo/imagenes/visible/visible_01.jpg \
  --output-dir outputs/detections/demo_imagen \
  --save-json
```

## Probar una imagen

Coloca una imagen en `data/raw/`, por ejemplo:

```text
data/raw/imagen.jpg
```

Ejecuta:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/inferir_imagen.py \
  --image data/raw/imagen.jpg \
  --output-dir outputs/detections/prueba_imagen \
  --save-json
```

El comando devuelve un JSON en consola y guarda resultados en:

```text
outputs/detections/prueba_imagen/
├── annotated/
└── json/
```

En el JSON, cada detección incluye:

- `bbox`: caja del extintor;
- `detection_confidence`: confianza de YOLO;
- `status_prediction`: estado estimado;
- `status_confidence`: confianza de la CNN;
- `status_probabilities`: probabilidades por clase.

## Probar una carpeta de imágenes

Deja varias imágenes en `data/raw/` y ejecuta:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/evaluar_pipeline.py \
  --images-dir data/raw \
  --output-dir outputs/detections/prueba_lote \
  --save-json
```

Este modo es útil para revisar muchas imágenes seguidas y generar una carpeta con resultados.

## Usar la API

Para levantar el servicio:

```bash
docker compose -f docker/docker-compose.yml up api
```

Probar desde Swagger:

1. Abre `http://localhost:8000/docs` en el navegador.
2. Usa `GET /health` para comprobar que la API está levantada.
3. Usa `POST /inspect/image` para subir una imagen.
4. Usa `POST /inspect/images` para subir varias imágenes de la demo en una sola petición.
5. Usa `POST /inspect/folder` para subir la carpeta de la demo comprimida en un ZIP.
6. Usa `POST /inspect/folder/zip` si quieres descargar un ZIP con el JSON y las imágenes anotadas.

En Swagger hay que pulsar `Try it out`, seleccionar los archivos y ejecutar la petición. Para la demo completa se pueden seleccionar todas las imágenes de `demo/imagenes/` desde el campo `files` del endpoint `/inspect/images`.

Si se prefiere subir la demo como una carpeta completa, primero se comprime:

```bash
python3 -m zipfile -c outputs/demo_imagenes.zip demo/imagenes
```

Después se abre `POST /inspect/folder` en Swagger y se selecciona `outputs/demo_imagenes.zip` en el campo `archivo_zip`.

Si se quiere recibir también las imágenes con las detecciones dibujadas, se usa `POST /inspect/folder/zip`. Swagger descargará un archivo `resultados_inspeccion.zip` con:

- `resultados.json`;
- `annotated/`, con las imágenes procesadas y sus cajas dibujadas.

Comprobar estado:

```bash
curl http://localhost:8000/health
```

Enviar una imagen:

```bash
curl -X POST \
  -F "file=@demo/imagenes/visible/visible_01.jpg" \
  "http://localhost:8000/inspect/image"
```

Enviar la demo completa por API:

```bash
args=()
while IFS= read -r imagen; do
  args+=(-F "files=@${imagen}")
done < <(find demo/imagenes -type f \( -name "*.jpg" -o -name "*.png" \) | sort)

curl -X POST "${args[@]}" \
  "http://localhost:8000/inspect/images"
```

Enviar la demo como carpeta comprimida:

```bash
python3 -m zipfile -c outputs/demo_imagenes.zip demo/imagenes

curl -X POST \
  -F "archivo_zip=@outputs/demo_imagenes.zip" \
  "http://localhost:8000/inspect/folder"
```

Enviar la demo y descargar resultados anotados:

```bash
python3 -m zipfile -c outputs/demo_imagenes.zip demo/imagenes

curl -X POST \
  -F "archivo_zip=@outputs/demo_imagenes.zip" \
  "http://localhost:8000/inspect/folder/zip" \
  --output outputs/resultados_inspeccion.zip
```

La API no recibe una carpeta como ruta local del cliente; recibe archivos. Para una carpeta completa hay dos opciones: subir todas las imágenes con `/inspect/images` o comprimir la carpeta y subir el ZIP con `/inspect/folder`. Si además se quieren descargar las imágenes anotadas desde Swagger, se usa `/inspect/folder/zip`. Un robot inspector normalmente usaría `/inspect/image` para cada captura, aunque también puede usar los endpoints de lote si agrupa varias imágenes.

## Validaciones útiles

Validar dataset YOLO:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/validar_dataset.py --tipo yolo --path data/yolo/data.yaml
```

Validar dataset CNN:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 scripts/validar_dataset.py --tipo classifier --path data/classifier_context_final_v7 --require-images
```

Ejecutar tests:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps --user "$(id -u):$(id -g)" app-gpu \
  python3 -m unittest discover -s tests
```

## Estado final

El proyecto queda preparado para demostración, revisión académica y pruebas controladas. La estructura conserva lo necesario para explicar el proceso completo sin mantener todos los artefactos intermedios. Las carpetas antiguas de experimentos se han eliminado o resumido en la documentación.
