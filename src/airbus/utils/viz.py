"""Visualización de predicciones de segmentación."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.transforms import denormalize

__all__ = ["plot_segmentation_batch"]


@torch.no_grad()
def plot_segmentation_batch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 4,
    threshold: float = 0.5,
    save_path: str | Path | None = None,
):
    """Rejilla imagen / máscara real / predicción / superposición.

    Args:
        num_samples: filas a dibujar, recortado al tamaño del lote.
        save_path: si se indica, guarda la figura en disco.

    Returns:
        La figura de matplotlib.
    """
    import matplotlib.pyplot as plt  # import perezoso: no hace falta para entrenar

    model.eval()
    images, masks = next(iter(loader))
    logits = model(images.to(device))
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().cpu()

    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        image = denormalize(images[i]).permute(1, 2, 0).numpy()
        truth = masks[i].squeeze().cpu().numpy()
        pred = preds[i].squeeze().numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Muestra {i + 1}: imagen original")

        axes[i, 1].imshow(truth, cmap="gray")
        axes[i, 1].set_title("Máscara real")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Predicción (umbral > {threshold})")

        axes[i, 3].imshow(image)
        overlay = np.zeros((*pred.shape, 4))
        overlay[..., 1] = 1.0          # canal verde
        overlay[..., 3] = pred * 0.5   # alfa al 50 % solo donde hay predicción
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Superposición")

        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
