"""Transformaciones de imagen para cada una de las dos tareas.

Las dos tareas usan **APIs distintas de torchvision a propósito**:

- Clasificación → ``torchvision.transforms`` (v1). La etiqueta es un escalar, no
  hay nada que sincronizar con la imagen.
- Segmentación → ``torchvision.transforms.v2``. La etiqueta *es* una imagen y
  tiene que sufrir exactamente la misma geometría que la entrada.

Con la API v1 no hay forma correcta de hacer lo segundo: cada llamada transforma
un solo objeto, así que aplicar ``RandomHorizontalFlip`` a la imagen y a la
máscara son dos sorteos aleatorios independientes y la etiqueta deja de
corresponder al píxel que etiqueta. La v2 acepta ambas en la misma llamada.

Además, envolver la máscara en ``tv_tensors.Mask`` la marca por tipo:
``ToDtype(scale=True)`` escala la **imagen** a ``[0, 1]`` pero deja la máscara
intacta —si la dividiera entre 255 los unos pasarían a 0,0039— y ``Normalize``
ni la toca, porque una máscara normalizada con la media de ImageNet no
significaría nada.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T2

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_classifier_transforms",
    "build_segmentation_transforms",
    "denormalize",
]

# Estadísticas de ImageNet: la ResNet-50 viene preentrenada con ellas, y la U-Net
# las reutiliza por coherencia entre las dos ramas del pipeline.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_classifier_transforms(image_size: int = 224, train: bool = True) -> T.Compose:
    """Transformaciones para la ResNet-50 (API v1, una sola entrada).

    Args:
        image_size: lado del cuadrado de salida.
        train: si es ``True`` añade el volteo horizontal aleatorio. En validación
            no se aplica ninguna aleatoriedad, para que dos evaluaciones del
            mismo modelo den exactamente el mismo número.
    """
    steps: list = [T.Resize((image_size, image_size))]
    if train:
        steps.append(T.RandomHorizontalFlip(p=0.5))
    steps += [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return T.Compose(steps)


def build_segmentation_transforms(image_size: int = 256, train: bool = True) -> T2.Compose:
    """Transformaciones para la U-Net (API v2, imagen y máscara sincronizadas).

    El único aumento es el volteo horizontal: en imagen satelital cenital un
    barco espejado sigue siendo un barco plausible. Se evitan deliberadamente los
    recortes aleatorios, que podrían dejar fuera el único barco de la imagen.

    Nota: el aumento **no crea filas nuevas**. ``__getitem__`` sortea una
    transformación distinta cada vez que se pide la muestra, así que la red ve
    una versión diferente en cada época; el tamaño nominal del dataset no cambia.
    """
    steps: list = [T2.Resize((image_size, image_size))]
    if train:
        steps.append(T2.RandomHorizontalFlip(p=0.5))
    steps += [
        T2.ToImage(),
        T2.ToDtype(torch.float32, scale=True),  # escala la imagen, no la máscara
        T2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T2.Compose(steps)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Deshace :func:`Normalize` para poder visualizar un tensor como imagen.

    Args:
        tensor: tensor ``[3, H, W]`` normalizado con las estadísticas de ImageNet.

    Returns:
        Tensor ``[3, H, W]`` recortado al rango ``[0, 1]``.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
