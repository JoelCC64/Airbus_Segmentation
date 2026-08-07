"""Guardado y carga de pesos y de checkpoints reanudables.

Son dos cosas distintas y conviene no confundirlas:

- **Pesos** (``state_dict``): lo mínimo para hacer inferencia.
- **Checkpoint**: pesos **más el estado del optimizador**. Adam mantiene dos
  momentos por parámetro (media y varianza móviles del gradiente); perderlos
  equivale a volver a la fase de calentamiento con pesos ya entrenados, una
  combinación que puede desestabilizar lo aprendido.

De ahí que un checkpoint pese ~3× el modelo: 31 M parámetros × 4 bytes × 3
(pesos + momento 1 + momento 2) ≈ 355 MB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

__all__ = ["save_weights", "load_weights", "save_checkpoint", "load_checkpoint"]


def save_weights(model: nn.Module, path: str | Path) -> Path:
    """Guarda solo el ``state_dict`` del modelo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_weights(
    model: nn.Module, path: str | Path, device: torch.device | str = "cpu"
) -> nn.Module:
    """Carga pesos en un modelo ya construido.

    Usa ``weights_only=True``: el fichero se deserializa como tensores puros, sin
    ejecutar código arbitrario incrustado en el pickle.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    **extra: Any,
) -> Path:
    """Guarda un checkpoint reanudable.

    Args:
        extra: campos adicionales a conservar (pérdida, métricas, configuración…).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **extra,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restaura modelo y, opcionalmente, optimizador.

    Returns:
        El checkpoint completo, para poder leer ``epoch`` y reanudar el bucle
        desde donde se quedó.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
