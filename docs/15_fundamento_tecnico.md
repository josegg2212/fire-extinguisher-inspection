# Fundamento tecnico del sistema

Este documento resume las decisiones tecnicas principales del proyecto. La idea no era solo hacer una demo, sino construir un flujo de vision por computador razonable para detectar extintores y estimar su estado de accesibilidad.

## Planteamiento general

El problema se dividio en dos partes:

1. Detectar donde esta el extintor en la imagen.
2. Clasificar el estado visual del extintor detectado.

Esta separacion es importante. Una imagen puede tener varios extintores y cada uno puede tener un estado distinto. Por eso no bastaba con clasificar la imagen completa. Primero habia que localizar cada extintor y despues analizar cada recorte por separado.

El pipeline final queda asi:

```text
imagen completa -> YOLO -> bbox del extintor -> crop contextual -> CNN -> estado
```

## Por que YOLO

YOLO se eligio para la fase de deteccion porque es un detector de objetos de una sola etapa. En una unica pasada predice cajas, clase y confianza. Esto encaja bien con el caso de uso del proyecto:

- puede detectar varios extintores en la misma imagen;
- devuelve directamente la posicion del objeto;
- es rapido para un posible uso en camaras o robot inspector;
- esta bien integrado en Ultralytics y permite entrenar con datasets en formato YOLO.

En este proyecto la deteccion solo tiene una clase:

```text
fire_extinguisher
```

La clasificacion fina del estado no se dejo en YOLO porque el estado depende de detalles de contexto: cajas delante, muebles, sombras, cristal, distancia al objeto o partes del extintor ocultas. Para eso era mas claro usar el detector solo para localizar y una CNN posterior para decidir el estado.

## Por que YOLO26n

El modelo base usado fue `yolo26n.pt`. Se eligio la variante `n` porque es ligera y suficiente para un problema de una sola clase. El objetivo era tener un detector estable sin cargar demasiado la GPU local.

La eleccion fue practica:

- el dataset de deteccion tenia una clase clara y muchas cajas anotadas;
- un modelo grande no era necesario para empezar;
- el entrenamiento debia poder ejecutarse en una GTX 1650;
- Ultralytics permitia entrenar, validar y exportar resultados de forma directa;
- `yolo26n.pt` entreno correctamente, por lo que no hizo falta usar el fallback `yolo11n.pt`.

El detector final fue entrenado 50 epocas con imagen de entrenamiento `640`. En test obtuvo:

| Metrica | Valor |
| --- | ---: |
| precision | 0.9456 |
| recall | 0.9187 |
| mAP50 | 0.9738 |
| mAP50-95 | 0.8388 |

Estos resultados indican que el detector no era el principal cuello de botella. En las pruebas manuales, los errores mas relevantes aparecieron en la clasificacion del estado.

## Dataset de deteccion

Para YOLO se uso un dataset preparado en formato Roboflow/YOLO. Esta decision tuvo sentido porque el detector necesita imagenes con cajas delimitadoras. El dataset ya venia organizado con:

- imagenes;
- labels YOLO;
- splits `train`, `valid` y `test`;
- clase `fire_extinguisher`.

Antes de entrenar se valido la estructura y se revisaron los labels. Los conteos finales fueron:

| Split | Imagenes | Anotaciones |
| --- | ---: | ---: |
| train | 3791 | 7355 |
| valid | 837 | 1566 |
| test | 397 | 763 |

Habia algunos labels vacios en `train`, pero no bloquearon el entrenamiento. Se trataron como posibles ejemplos negativos o de fondo.

## Por que una CNN para el estado

Una vez detectado el extintor, el problema ya no es localizar, sino clasificar una imagen pequeña en tres clases:

- `visible`;
- `partially_occluded`;
- `blocked`.

Para este tipo de tarea una CNN es adecuada porque aprende patrones espaciales locales: bordes, colores, texturas, zonas tapadas y relaciones entre el extintor y los objetos cercanos. Ademas, al trabajar con crops, la entrada queda normalizada y el modelo no tiene que buscar el extintor en toda la escena.

Se probo primero con crops demasiado ajustados. Esa version funcionaba en datos controlados, pero perdia contexto. Despues se paso a crops contextuales, ampliando la caja de YOLO antes de clasificar. Esta decision fue clave, porque para saber si un extintor esta bloqueado no basta con ver el cilindro: tambien hay que ver que hay alrededor.

## Arquitectura de la CNN

La CNN final usa la arquitectura `simple_cnn`, definida en `src/fire_extinguisher_inspection/classification/cnn_model.py`. Es una red pequena y explicable, suficiente para el tamano del problema.

Entrada:

```text
imagen RGB de 224 x 224
```

Bloque convolucional:

| Bloque | Operaciones |
| --- | --- |
| 1 | Conv2d 3->32, BatchNorm, ReLU, MaxPool |
| 2 | Conv2d 32->64, BatchNorm, ReLU, MaxPool |
| 3 | Conv2d 64->128, BatchNorm, ReLU, MaxPool |
| 4 | Conv2d 128->256, BatchNorm, ReLU, AdaptiveAvgPool |

Clasificador:

| Capa | Funcion |
| --- | --- |
| Flatten | convierte el mapa final en vector |
| Dropout 0.25 | reduce sobreajuste |
| Linear 256->128 | capa densa intermedia |
| ReLU | no linealidad |
| Dropout 0.25 | regularizacion |
| Linear 128->3 | salida para las tres clases |

Durante entrenamiento se uso `CrossEntropyLoss`, optimizador `Adam`, `batch_size=32`, entrada `224x224` y normalizacion RGB. En entrenamiento se aplicaron aumentos sencillos: cambios leves de brillo/contraste y volteo horizontal.

No se uso una red excesivamente grande porque el objetivo era mantener el modelo comprensible y evitar sobreajuste. Esta arquitectura tambien permite explicar con claridad que hace cada parte de la red.

## Dataset de clasificacion

El dataset de la CNN no era el mismo que el de YOLO. A partir de las cajas de deteccion se generaron crops por clase. La evolucion fue:

1. Dataset inicial con crops ajustados.
2. Dataset contextual con recortes ampliados.
3. Ajuste final con ejemplos sinteticos mas realistas y pruebas manuales revisadas.

La version final conservada es:

```text
data/classifier_context_final_v7
```

La idea de las clases fue:

- `visible`: extintor accesible y reconocible;
- `partially_occluded`: extintor visible, pero con parte tapada o con acceso dudoso;
- `blocked`: extintor claramente bloqueado por un obstaculo.

La frontera mas dificil fue `visible` frente a `partially_occluded`. En imagenes reales aparecen sombras, baja resolucion, fondos complicados, cristales o elementos cercanos que hacen que esa separacion no sea perfecta. Por eso se mantuvo una validacion manual final, no solo metricas internas.

## Decision final

La configuracion final combina:

- YOLO26n entrenado para detectar extintores;
- crop contextual alrededor de cada deteccion;
- CNN `simple_cnn` para clasificar el estado;
- inferencia con `yolo_imgsz=1920` y umbral YOLO `0.05`.

La conclusion tecnica es que el detector quedo suficientemente estable para el objetivo del proyecto. La parte mas dificil fue la clasificacion del estado, especialmente distinguir extintores realmente visibles de casos parcialmente tapados. Esa dificultad es coherente con el problema real, porque la etiqueta depende del contexto y no solo del objeto.
