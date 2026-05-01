# Revisión de labels YOLO vacíos

Fecha: 2026-04-25

Total de labels vacíos: 12

## Distribución por split

- `train`: 12

## Imágenes asociadas

- `train`: imagen `data/yolo/train/images/00456_jpg.rf.45e4b7a3d2082447a963fe3acf24818f.jpg` | label `data/yolo/train/labels/00456_jpg.rf.45e4b7a3d2082447a963fe3acf24818f.txt`
- `train`: imagen `data/yolo/train/images/00469_jpg.rf.1a121de0dc123393d02bfae11db05cbc.jpg` | label `data/yolo/train/labels/00469_jpg.rf.1a121de0dc123393d02bfae11db05cbc.txt`
- `train`: imagen `data/yolo/train/images/00475_jpg.rf.5d6e8798f7e43182e477724b32f07071.jpg` | label `data/yolo/train/labels/00475_jpg.rf.5d6e8798f7e43182e477724b32f07071.txt`
- `train`: imagen `data/yolo/train/images/125_jpg.rf.1a5415f1ee2287405dcb968d138eeaeb.jpg` | label `data/yolo/train/labels/125_jpg.rf.1a5415f1ee2287405dcb968d138eeaeb.txt`
- `train`: imagen `data/yolo/train/images/1538_jpg.rf.181d6518e210fe2377f4b099571db070.jpg` | label `data/yolo/train/labels/1538_jpg.rf.181d6518e210fe2377f4b099571db070.txt`
- `train`: imagen `data/yolo/train/images/192_jpg.rf.b2453a067c6425de20af348db479fa2f.jpg` | label `data/yolo/train/labels/192_jpg.rf.b2453a067c6425de20af348db479fa2f.txt`
- `train`: imagen `data/yolo/train/images/2016_jpg.rf.9968dc7f78337894ccf89af97bb101b3.jpg` | label `data/yolo/train/labels/2016_jpg.rf.9968dc7f78337894ccf89af97bb101b3.txt`
- `train`: imagen `data/yolo/train/images/2020_jpg.rf.6d1d119cbfa16036ea26eee95132e598.jpg` | label `data/yolo/train/labels/2020_jpg.rf.6d1d119cbfa16036ea26eee95132e598.txt`
- `train`: imagen `data/yolo/train/images/204_jpg.rf.20dc20e65160d59ab0a0e15991597df0.jpg` | label `data/yolo/train/labels/204_jpg.rf.20dc20e65160d59ab0a0e15991597df0.txt`
- `train`: imagen `data/yolo/train/images/2338_jpg.rf.0f0af9f7d0fa8fae4a876e360e52c9e6.jpg` | label `data/yolo/train/labels/2338_jpg.rf.0f0af9f7d0fa8fae4a876e360e52c9e6.txt`
- `train`: imagen `data/yolo/train/images/2432_jpg.rf.f37631a6d1a62641306b21fbc1e981a0.jpg` | label `data/yolo/train/labels/2432_jpg.rf.f37631a6d1a62641306b21fbc1e981a0.txt`
- `train`: imagen `data/yolo/train/images/88_jpg.rf.b56785b3b29b51fc8367c80e99c8e767.jpg` | label `data/yolo/train/labels/88_jpg.rf.b56785b3b29b51fc8367c80e99c8e767.txt`

## Interpretación

El script solo detecta labels vacíos y genera una contact sheet; no decide de forma automática si la imagen contiene o no un extintor.

- Si las imágenes parecen negativas sin extintor visible, los labels vacíos son aceptables para YOLO.
- Si alguna imagen contiene un extintor sin caja, hay que corregir la anotación antes de entrenar.

## Revisión visual manual

Se revisó `outputs/reports/empty_labels_contact_sheet.jpg`. Las 12 imágenes con label vacío no parecen negativos limpios: visualmente contienen extintores, grupos de extintores, ilustraciones de extintores o elementos claramente relacionados con extintores.

Decisión inicial: lo ideal sería corregir estos 12 casos o excluirlos de `train`.

Decisión posterior del proyecto: se aceptan temporalmente los 12 labels vacíos porque representan una proporción muy pequeña frente al volumen total del dataset. Se puede lanzar el entrenamiento corto de prueba, dejando este riesgo documentado para revisarlo si aparecen falsos negativos o métricas anómalas.

Contact sheet generada:

```text
outputs/reports/empty_labels_contact_sheet.jpg
```
