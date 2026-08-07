"""Bucles de entrenamiento y evaluación, compartidos por las dos tareas."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import ConfusionAccumulator, SegmentationMetrics, binary_accuracy

__all__ = [
    "EpochResult",
    "train_classifier_epoch",
    "validate_classifier",
    "train_segmenter_epoch",
    "evaluate_segmenter",
]


@dataclass(frozen=True)
class EpochResult:
    """Resumen de una época."""

    loss: float
    accuracy: float | None = None

    def __str__(self) -> str:  # pragma: no cover - formato de presentación
        if self.accuracy is None:
            return f"loss={self.loss:.4f}"
        return f"loss={self.loss:.4f}  accuracy={self.accuracy:.4f}"


def _mean_per_image(running_loss: float, loader: DataLoader) -> float:
    """Normaliza la pérdida acumulada por **imagen**, no por lote.

    Los bucles acumulan ``loss.item() * batch_size``, es decir pérdida total por
    imagen. Dividir eso entre ``len(loader)`` —el número de lotes— mezcla
    unidades e infla el resultado por un factor igual al tamaño del lote. El
    divisor correcto es ``len(loader.dataset)``.
    """
    return running_loss / len(loader.dataset)


def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str = "train",
) -> EpochResult:
    """Una época de entrenamiento del clasificador binario."""
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False, disable=None):
        images = images.to(device, non_blocking=True)
        # [N] -> [N, 1] para calzar con la salida de la red. Sin este unsqueeze
        # el broadcasting de PyTorch emparejaría en silencio cada predicción con
        # cada etiqueta y la pérdida dejaría de significar nada.
        labels = labels.unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        batch_correct, batch_total = binary_accuracy(logits.detach(), labels)
        correct += batch_correct
        total += batch_total

    return EpochResult(_mean_per_image(running_loss, loader), correct / max(total, 1))


@torch.no_grad()
def validate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> EpochResult:
    """Evaluación del clasificador sobre datos retenidos.

    El notebook original creaba la partición de validación pero nunca la usaba,
    así que su accuracy de 0,9429 era **de entrenamiento**. Esta función es lo
    que faltaba para poder distinguir aprendizaje de memorización.
    """
    model.eval()
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False, disable=None):
        images = images.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, non_blocking=True)

        logits = model(images)
        running_loss += criterion(logits, labels).item() * images.size(0)
        batch_correct, batch_total = binary_accuracy(logits, labels)
        correct += batch_correct
        total += batch_total

    return EpochResult(_mean_per_image(running_loss, loader), correct / max(total, 1))


def _as_nchw(masks: torch.Tensor) -> torch.Tensor:
    """``[N, H, W]`` → ``[N, 1, H, W]`` para calzar con la salida de la U-Net."""
    return masks.unsqueeze(1) if masks.dim() == 3 else masks


def train_segmenter_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str = "train",
) -> EpochResult:
    """Una época de entrenamiento de la U-Net."""
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc=desc, leave=False, disable=None):
        images = images.to(device, non_blocking=True)
        masks = _as_nchw(masks.float().to(device, non_blocking=True))

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return EpochResult(_mean_per_image(running_loss, loader))


@torch.no_grad()
def evaluate_segmenter(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    criterion: nn.Module | None = None,
    desc: str = "eval",
) -> tuple[SegmentationMetrics, float | None]:
    """Métricas de segmentación acumuladas sobre todo el ``loader``.

    Args:
        threshold: umbral sobre la **probabilidad** (tras la sigmoide). En
            inferencia sí se aplica sigmoide explícitamente, a diferencia del
            bucle de entrenamiento, que umbraliza logits en 0,0.

    Returns:
        Las métricas y, si se pasó ``criterion``, la pérdida media por imagen.
    """
    model.eval()
    accumulator = ConfusionAccumulator()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc=desc, leave=False, disable=None):
        images = images.to(device, non_blocking=True)
        masks = _as_nchw(masks.float().to(device, non_blocking=True))

        logits = model(images)
        if criterion is not None:
            running_loss += criterion(logits, masks).item() * images.size(0)

        preds = (torch.sigmoid(logits) > threshold).float()
        accumulator.update(preds, masks)

    loss = _mean_per_image(running_loss, loader) if criterion is not None else None
    return accumulator.compute(), loss
