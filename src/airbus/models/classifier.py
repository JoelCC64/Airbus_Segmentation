"""ResNet-50 adaptada a clasificación binaria «¿hay barco?»."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

__all__ = ["AirbusClassifier"]


class AirbusClassifier(nn.Module):
    """ResNet-50 preentrenada con la capa final sustituida por un único logit.

    **Una salida, no dos.** Con ``nn.Linear(2048, 1)`` y ``BCEWithLogitsLoss`` el
    problema binario queda resuelto con un solo número. La alternativa —dos
    salidas y ``CrossEntropyLoss``— es equivalente pero gasta el doble de
    parámetros en la última capa para representar la misma información.

    **El backbone no se congela por defecto.** ImageNet enseña texturas y
    estructuras genéricas de fotografía terrestre a nivel de suelo; aquí las
    imágenes son cenitales, casi monocromas, y el objeto ocupa el 0,5 % del
    encuadre. Los pesos preentrenados se conservan como *inicialización*, no como
    extractor fijo.

    Args:
        pretrained: si es ``True`` descarga los pesos de ImageNet.
        freeze_backbone: congela todo salvo la capa final. Disponible para poder
            comparar los dos regímenes, pero desactivado por defecto.

    Shape:
        - Entrada: ``[N, 3, H, W]`` (entrenado a 224 × 224).
        - Salida: ``[N, 1]`` con logits crudos, **sin** sigmoide.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Linear(in_features, 1)  # siempre entrenable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
