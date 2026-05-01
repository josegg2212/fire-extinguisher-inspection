# Imagen base parametrizable. Por defecto se usa una imagen ligera de Python.
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

# Ajustes básicos para que Python escriba logs en tiempo real y no genere bytecode.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# El código fuente del paquete vive en /workspace/src dentro del contenedor.
ENV PYTHONPATH=/workspace/src

# Directorios temporales para evitar escrituras de configuración fuera del proyecto.
ENV YOLO_CONFIG_DIR=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

# Dependencias del sistema necesarias para OpenCV, Ultralytics y lectura de imágenes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt

# PyTorch se instala aparte para poder elegir CPU o CUDA desde docker-compose.
ARG TORCH_VERSION=2.8.0+cpu
ARG TORCHVISION_VERSION=0.23.0+cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --extra-index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Por defecto queda una shell abierta; los comandos reales se indican desde docker compose.
CMD ["bash"]
