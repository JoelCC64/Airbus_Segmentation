"""Datasets de PyTorch para las dos tareas del reto."""

from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .rle import DEFAULT_SHAPE, rle_decode

__all__ = ["AirbusClassificationDataset", "AirbusSegmentationDataset"]


class AirbusClassificationDataset(Dataset):
    """Imagen → etiqueta binaria «¿contiene algún barco?».

    Args:
        frame: dataframe con columnas ``ImageId`` y ``has_ship``.
        image_dir: carpeta con los ``.jpg``.
        transform: pipeline de :mod:`torchvision.transforms` (API v1).
    """

    def __init__(self, frame: pd.DataFrame, image_dir: str, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        path = os.path.join(self.image_dir, row["ImageId"])
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # float32 porque la pérdida es BCEWithLogitsLoss, no CrossEntropy.
        label = torch.tensor(float(row["has_ship"]), dtype=torch.float32)
        return image, label


class AirbusSegmentationDataset(Dataset):
    """Imagen → máscara binaria por píxel.

    Solo debe alimentarse con imágenes que contengan barco (ver
    :func:`airbus.data.splits.segmentation_dataframe`).

    Args:
        frame: dataframe con ``ImageId`` y ``EncodedPixels`` como lista de RLE.
        image_dir: carpeta con los ``.jpg``.
        transform: pipeline de :mod:`torchvision.transforms.v2`, que recibe
            imagen y máscara juntas.
        mask_shape: resolución nativa de la máscara antes de redimensionar.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str,
        transform=None,
        mask_shape: tuple[int, int] = DEFAULT_SHAPE,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mask_shape = mask_shape

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        path = os.path.join(self.image_dir, row["ImageId"])

        image = Image.open(path).convert("RGB")
        decoded = rle_decode(row["EncodedPixels"], shape=self.mask_shape)
        # tv_tensors.Mask es lo que permite a la API v2 distinguir geometría de
        # fotometría: la voltea con la imagen, pero no la escala ni la normaliza.
        mask = tv_tensors.Mask(decoded)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
