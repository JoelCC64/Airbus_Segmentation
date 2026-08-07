"""Inferencia en cascada: clasificador como puerta, U-Net como segmentador.

Las dos redes están entrenadas sobre poblaciones distintas y por eso el sistema
solo funciona encadenado:

- El clasificador vio océano vacío y océano con barco, al 50 %.
- La U-Net vio **únicamente** imágenes que contienen barco.

Sin la puerta delante, la U-Net alucinaría cascos sobre agua limpia: nunca ha
tenido que aprender a devolver una máscara vacía. Como contrapartida, el 77,9 %
de las imágenes sale por la rama barata sin llegar a tocar los 31 M de
parámetros del segmentador.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data.rle import DEFAULT_SHAPE, rle_encode
from .data.transforms import build_classifier_transforms, build_segmentation_transforms

__all__ = ["Prediction", "ShipSegmentationPipeline"]


@dataclass(frozen=True)
class Prediction:
    """Salida del pipeline para una imagen."""

    has_ship: bool
    ship_probability: float
    mask: np.ndarray
    """Máscara binaria ``uint8`` a resolución nativa (768 × 768 por defecto)."""

    def to_rle(self) -> str:
        """RLE en el formato de envío de Kaggle. Cadena vacía si no hay barco."""
        return rle_encode(self.mask)


class ShipSegmentationPipeline:
    """Encadena clasificador y segmentador en una sola llamada.

    Args:
        classifier: :class:`~airbus.models.classifier.AirbusClassifier` entrenado.
        segmenter: :class:`~airbus.models.unet.UNet` entrenada.
        device: dispositivo sobre el que ejecutar la inferencia.
        classifier_size: lado de entrada del clasificador.
        segmenter_size: lado de entrada del segmentador.
        ship_threshold: umbral de probabilidad para la puerta binaria.
        mask_threshold: umbral de probabilidad por píxel.
        output_shape: resolución a la que se devuelve la máscara. Se reescala
            desde ``segmenter_size``, porque la métrica oficial se evalúa a
            resolución completa y no a la de entrenamiento.

    Example:
        >>> pipeline = ShipSegmentationPipeline(clf, unet, device)   # doctest: +SKIP
        >>> pred = pipeline.predict(Image.open("000155de5.jpg"))     # doctest: +SKIP
        >>> pred.has_ship, pred.mask.sum()                           # doctest: +SKIP
    """

    def __init__(
        self,
        classifier: nn.Module,
        segmenter: nn.Module,
        device: torch.device,
        classifier_size: int = 224,
        segmenter_size: int = 256,
        ship_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        output_shape: tuple[int, int] = DEFAULT_SHAPE,
    ) -> None:
        self.classifier = classifier.to(device).eval()
        self.segmenter = segmenter.to(device).eval()
        self.device = device
        self.ship_threshold = ship_threshold
        self.mask_threshold = mask_threshold
        self.output_shape = output_shape

        self.classifier_tf = build_classifier_transforms(classifier_size, train=False)
        self.segmenter_tf = build_segmentation_transforms(segmenter_size, train=False)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        """Ejecuta la cascada completa sobre una imagen PIL."""
        image = image.convert("RGB")

        gate = self.classifier(self.classifier_tf(image).unsqueeze(0).to(self.device))
        probability = torch.sigmoid(gate).item()

        if probability <= self.ship_threshold:
            # Salida temprana: máscara vacía, sin pasar por la U-Net.
            return Prediction(False, probability, np.zeros(self.output_shape, np.uint8))

        tensor = self.segmenter_tf(image).unsqueeze(0).to(self.device)
        logits = self.segmenter(tensor)
        # Se reescalan los logits y se umbraliza después: interpolar una máscara
        # ya binarizada introduce escalones en el contorno.
        logits = F.interpolate(
            logits, size=self.output_shape, mode="bilinear", align_corners=False
        )
        mask = (torch.sigmoid(logits) > self.mask_threshold).squeeze().cpu().numpy()

        return Prediction(True, probability, mask.astype(np.uint8))

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        """Versión por lotes. Hoy itera; el punto de extensión para agrupar."""
        return [self.predict(image) for image in images]
