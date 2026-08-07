"""Selección de dispositivo y semillas."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["get_device", "set_seed"]


def get_device(prefer: str | None = None) -> torch.device:
    """Elige el mejor dispositivo disponible.

    Orden de preferencia: CUDA → MPS → CPU. El notebook original solo miraba
    CUDA, así que en un Mac con Apple Silicon caía a CPU pudiendo usar MPS.

    Args:
        prefer: fuerza un dispositivo concreto (``'cuda'``, ``'mps'``, ``'cpu'``).
            Si no está disponible se avisa y se cae al automático.
    """
    if prefer:
        prefer = prefer.lower()
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }
        if available.get(prefer):
            return torch.device(prefer)
        print(f"[aviso] dispositivo '{prefer}' no disponible; se elige automáticamente.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fija las semillas de ``random``, NumPy y PyTorch.

    Args:
        deterministic: además de las semillas, exige algoritmos deterministas en
            cuDNN. Reproducibilidad exacta a cambio de velocidad.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
