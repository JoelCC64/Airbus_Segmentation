"""Arquitecturas: ResNet-50 para clasificar, U-Net para segmentar."""

from .classifier import AirbusClassifier
from .unet import DoubleConv, UNet

__all__ = ["AirbusClassifier", "DoubleConv", "UNet"]
