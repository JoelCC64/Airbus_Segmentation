"""Métricas de segmentación y de clasificación.

Las métricas de segmentación se **acumulan** sobre todo el conjunto y las razones
se calculan al final, en lugar de promediar el valor de cada imagen. Promediar
imagen a imagen daría el mismo peso a un carguero de 600 px que a una lancha de
20, y una sola imagen difícil hundiría la media.

La métrica oficial del reto es **F2**, que pondera el recall cuatro veces más que
la precisión: en vigilancia marítima omitir un barco es peor que dar una falsa
alarma.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ConfusionAccumulator", "SegmentationMetrics", "binary_accuracy"]

EPS = 1e-7


@dataclass(frozen=True)
class SegmentationMetrics:
    """Resultado de una evaluación completa, en píxeles."""

    f2: float
    iou: float
    dice: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, float]:
        return {
            "F2": self.f2,
            "IoU": self.iou,
            "Dice": self.dice,
            "Precision": self.precision,
            "Recall": self.recall,
        }

    def __str__(self) -> str:  # pragma: no cover - formato de presentación
        return (
            f"F2={self.f2:.4f}  IoU={self.iou:.4f}  Dice={self.dice:.4f}  "
            f"P={self.precision:.4f}  R={self.recall:.4f}\n"
            f"TP={self.tp:,}  FN={self.fn:,}  FP={self.fp:,}"
        )


class ConfusionAccumulator:
    """Suma la matriz de confusión píxel a píxel a lo largo de varios lotes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Acumula un lote ya binarizado.

        Args:
            preds: máscara predicha con valores 0/1.
            target: máscara de referencia con valores 0/1.
        """
        preds = preds.reshape(-1).float()
        target = target.reshape(-1).float()
        self.tp += (preds * target).sum().item()
        self.fp += (preds * (1.0 - target)).sum().item()
        self.fn += ((1.0 - preds) * target).sum().item()
        self.tn += ((1.0 - preds) * (1.0 - target)).sum().item()

    def compute(self) -> SegmentationMetrics:
        """Convierte los contadores acumulados en métricas."""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return SegmentationMetrics(
            # Forma directa desde la matriz de confusión, equivalente a
            # 5·P·R / (4·P + R) pero sin calcular P y R por separado.
            f2=(5 * tp) / (5 * tp + 4 * fn + fp + EPS),
            iou=tp / (tp + fp + fn + EPS),
            dice=(2 * tp) / (2 * tp + fp + fn + EPS),
            precision=tp / (tp + fp + EPS),
            recall=tp / (tp + fn + EPS),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            tn=int(tn),
        )


@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    """Aciertos y total de un lote de clasificación binaria.

    El umbral se aplica **sobre los logits, en 0,0**, no sobre probabilidades en
    0,5. Como ``BCEWithLogitsLoss`` lleva la sigmoide dentro, la red nunca la
    aplica; y ``sigmoid(0) = 0,5``, así que 0,0 en el espacio de logits es el
    equivalente exacto. Umbralizar los logits en 0,5 sería, en realidad,
    umbralizar la probabilidad en 0,62.
    """
    preds = (logits > 0.0).float()
    target = target.view_as(preds)
    correct = int((preds == target).sum().item())
    return correct, int(target.numel())
