"""Tests de estructura básica del dataset."""

from __future__ import annotations

import unittest

from fire_extinguisher_inspection.classification.dataset import validar_dataset_clasificador
from fire_extinguisher_inspection.config import cargar_configuracion


class TestDatasetStructure(unittest.TestCase):
    """Comprueba la estructura base del dataset de clasificación."""

    def test_estructura_dataset_clasificador_base(self) -> None:
        config = cargar_configuracion()
        resultado = validar_dataset_clasificador(config.datasets.classifier_root)
        self.assertTrue(resultado.es_valido)
        self.assertEqual(set(resultado.conteos.keys()), {"train", "val", "test"})
