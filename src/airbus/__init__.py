"""Detección y segmentación de barcos en imágenes satelitales (Airbus / Kaggle).

El reto esconde dos tareas de granularidad distinta —«¿hay barco?» es una
respuesta por imagen; «¿dónde está?» son 65 536 respuestas por imagen— y aquí se
resuelven por separado con dos redes encadenadas en cascada:

    imagen → ResNet-50 (puerta binaria) → U-Net (segmentación) → máscara

Ver :mod:`airbus.pipeline` para la inferencia extremo a extremo.
"""

from .config import ClassifierConfig, Config, PathsConfig, SegmenterConfig
from .losses import BCEDiceLoss
from .metrics import ConfusionAccumulator, SegmentationMetrics
from .models import AirbusClassifier, UNet
from .pipeline import Prediction, ShipSegmentationPipeline

__version__ = "0.1.0"

__all__ = [
    "AirbusClassifier",
    "BCEDiceLoss",
    "ClassifierConfig",
    "Config",
    "ConfusionAccumulator",
    "PathsConfig",
    "Prediction",
    "SegmentationMetrics",
    "SegmenterConfig",
    "ShipSegmentationPipeline",
    "UNet",
    "__version__",
]
