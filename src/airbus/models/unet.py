"""U-Net para segmentación semántica binaria, entrenada desde cero."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DoubleConv", "UNet"]


class DoubleConv(nn.Module):
    """Bloque ``(conv 3×3 → BN → ReLU) × 2``, repetido en encoder y decoder.

    ``bias=False`` en las convoluciones porque el ``BatchNorm2d`` que viene
    detrás ya aporta su propio parámetro de desplazamiento; el bias de la
    convolución sería redundante y se cancelaría en la normalización.

    Ojo con los canales: la **segunda** convolución entra con ``out_channels``,
    no con ``in_channels`` — la primera ya transformó el tensor.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Encoder–decoder simétrico con skip connections.

    Cada ``MaxPool2d`` tira tres de cada cuatro píxeles: destruye el *dónde* para
    ganar semántica. Las skip connections lo devuelven, entregando al decoder el
    *qué* (del cuello de botella) y el *dónde* (bordes finos del encoder) a la
    misma resolución.

    **La capa final es una conv 1×1 desnuda**, sin BatchNorm y sin activación.
    Una ReLU ahí recortaría los valores negativos a cero y, como
    ``sigmoid(0) = 0,5``, ningún píxel podría bajar del 50 % de probabilidad: con
    el umbral en 0,5 la red predeciría barco en todas partes y no habría manera
    de expresar «aquí seguro que no hay nada». El error no lanza excepción; solo
    produce máscaras saturadas de blanco.

    Args:
        in_channels: canales de entrada (3 para RGB).
        out_channels: canales de salida (1 para máscara binaria).
        features: anchura de cada nivel del encoder. El cuello de botella duplica
            el último valor.

    Shape:
        - Entrada: ``[N, in_channels, H, W]``.
        - Salida: ``[N, out_channels, H, W]`` con logits crudos por píxel.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        features = list(features)
        self.features = features

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder. La concatenación de la skip connection duplica los canales,
        # por eso la DoubleConv siguiente entra con feature*2 y los reduce.
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # de la más pequeña a la más grande

        # self.ups alterna convT, DoubleConv, convT, DoubleConv, ...
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # Con entradas cuyo lado no es múltiplo de 2**len(features), el
            # max-pool redondea hacia abajo y la convolución traspuesta no
            # recupera el tamaño exacto. Reajustamos antes de concatenar para que
            # la red acepte cualquier resolución, no solo 256 × 256.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )

            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)
