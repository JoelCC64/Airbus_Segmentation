"""Construcción de los dataframes y particiones de entrenamiento.

El CSV viene normalizado por *instancia*: una fila por barco. Los modelos
necesitan una fila por *imagen*, y cada tarea necesita un subconjunto distinto:

- **Clasificación**: todas las imágenes, con una etiqueta 0/1, rebalanceadas a
  50/50 porque el 77,9 % del dataset es océano vacío.
- **Segmentación**: solo las imágenes que contienen algún barco, con la lista
  completa de sus RLE.

Esa asimetría es deliberada y es lo que obliga a que la inferencia sea en
cascada: la U-Net nunca ve océano vacío durante el entrenamiento, así que
necesita al clasificador delante como puerta.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "load_dataframe",
    "image_level_dataframe",
    "balance_by_undersampling",
    "segmentation_dataframe",
    "split_dataframe",
]


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Lee el CSV del reto y añade la columna binaria ``has_ship``."""
    df = pd.read_csv(csv_path)
    if "EncodedPixels" not in df.columns:
        raise ValueError(
            f"El CSV {csv_path!r} no tiene columna 'EncodedPixels'. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    df["has_ship"] = df["EncodedPixels"].notnull().astype(int)
    return df


def image_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa el CSV a una fila por imagen.

    ``max`` sobre ``has_ship`` marca la imagen como positiva si *alguna* de sus
    filas tiene máscara; ``count`` sobre ``EncodedPixels`` ignora los nulos, de
    modo que las imágenes vacías quedan con cero barcos.
    """
    images = (
        df.groupby("ImageId")
        .agg({"has_ship": "max", "EncodedPixels": "count"})
        .reset_index()
        .rename(columns={"EncodedPixels": "total_ships"})
    )
    return images


def balance_by_undersampling(
    images: pd.DataFrame, seed: int = 42, ratio: float = 1.0
) -> pd.DataFrame:
    """Submuestrea la clase mayoritaria (sin barco) y mezcla el resultado.

    Con ``ratio=1.0`` el reparto queda 50/50 y el baseline trivial de responder
    siempre «no hay barco» cae del 77,9 % al 50 %, devolviéndole significado a la
    accuracy.

    Args:
        images: salida de :func:`image_level_dataframe`.
        seed: semilla del muestreo y de la mezcla.
        ratio: cuántas imágenes sin barco tomar por cada imagen con barco.
            ``ratio=None`` desactiva el submuestreo y devuelve el dataset entero
            (el experimento «sin balancear» que quedó pendiente en el notebook).
    """
    with_ship = images[images["has_ship"] == 1]
    without_ship = images[images["has_ship"] == 0]

    if ratio is None:
        return images.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_negatives = min(int(len(with_ship) * ratio), len(without_ship))
    sampled = without_ship.sample(n=n_negatives, random_state=seed)
    balanced = pd.concat([with_ship, sampled])
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def segmentation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por imagen con barco y la lista de todos sus RLE.

    Las imágenes sin barco se descartan: la U-Net solo se entrena sobre positivos.
    """
    with_ships = df[df["EncodedPixels"].notnull()]
    grouped = (
        with_ships.groupby("ImageId")["EncodedPixels"].apply(list).reset_index()
    )
    return grouped


def split_dataframe(
    frame: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
    stratify_on: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición train/validación reproducible.

    Args:
        stratify_on: nombre de columna para estratificar. Se usa ``'has_ship'``
            en clasificación para que el equilibrio 50/50 sobreviva al corte;
            en segmentación no aplica, porque todas las filas son positivas.
    """
    stratify = frame[stratify_on] if stratify_on else None
    return train_test_split(
        frame, test_size=val_fraction, random_state=seed, stratify=stratify
    )
