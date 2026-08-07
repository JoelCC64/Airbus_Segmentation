"""Funciones de pérdida.

El desbalance de este problema aparece dos veces y en dos escalas distintas:

- **A nivel de imagen**: 150 000 sin barco frente a 42 556 con barco. Se corrige
  submuestreando el dataset (ver :mod:`airbus.data.splits`).
- **A nivel de píxel**: dentro de una imagen que *sí* tiene barco, apenas el
  0,51 % de los píxeles es casco. Aquí no se pueden tirar píxeles, así que se
  corrige cambiando la pérdida.

La BCE promedia el error sobre los 65 536 píxeles: los ~337 de barco quedan
ahogados por los ~65 200 de fondo, que ya están bien clasificados desde la
primera época. El Dice mide *solapamiento* y el fondo no entra ni en el
numerador ni en el denominador, así que la solución degenerada («todo es
océano») deja de ser rentable: intersección 0 → Dice 0 → pérdida 1.

Se combinan al 50 %: la BCE da gradiente estable desde el principio, cuando las
predicciones son aleatorias y el gradiente del Dice es ruidoso; el Dice fuerza
después la forma del casco.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BCEDiceLoss", "dice_coefficient"]


def dice_coefficient(
    probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Coeficiente Dice sobre probabilidades, agregado en todo el lote.

    Args:
        probs: probabilidades en ``[0, 1]`` (ya pasadas por sigmoide).
        target: máscara de referencia con valores 0/1 y la misma forma.
        eps: término de estabilidad. Evita el ``0/0`` cuando un lote no contiene
            ningún píxel de barco.
    """
    probs = probs.reshape(-1)
    target = target.reshape(-1)
    intersection = (probs * target).sum()
    return (2 * intersection + eps) / (probs.sum() + target.sum() + eps)


class BCEDiceLoss(nn.Module):
    """Combinación ponderada de BCE y (1 − Dice).

    Args:
        bce_weight: peso de la parte BCE. El resto va al término Dice, de modo
            que ``bce_weight=0.5`` reparte al 50 %.

    Shape:
        - ``inputs``: logits crudos, ``[N, 1, H, W]``.
        - ``target``: máscara 0/1 en float, misma forma.
    """

    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError(f"bce_weight debe estar en [0, 1], recibido {bce_weight}")
        self.bce_weight = bce_weight
        self.eps = eps
        # BCEWithLogitsLoss aplica la sigmoide internamente, por estabilidad
        # numérica. Por eso la red nunca la aplica y sus salidas son logits.
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(inputs, target)
        dice = 1.0 - dice_coefficient(torch.sigmoid(inputs), target, self.eps)
        return (1.0 - self.bce_weight) * dice + self.bce_weight * bce
