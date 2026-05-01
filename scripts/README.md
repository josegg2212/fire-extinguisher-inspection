# Scripts del proyecto

Estos scripts son los puntos de entrada principales del repositorio. Los nombres se han dejado en español para que su uso sea directo.

## Scripts conservados

- `inferir_imagen.py`: ejecuta el pipeline completo sobre una imagen.
- `evaluar_pipeline.py`: procesa un directorio de imágenes y genera resultados agregados.
- `validar_dataset.py`: revisa la estructura del dataset YOLO o del dataset CNN.
- `prueba_basica.py`: comprueba que la configuración, clases y rutas básicas cargan correctamente.

## Criterio

Los scripts de generación, pruebas intermedias, vídeo y evaluaciones internas se retiraron de esta carpeta para no mezclar el uso final con el proceso de desarrollo. La trazabilidad queda descrita en `docs/`.
