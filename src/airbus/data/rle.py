"""Codificación y decodificación Run-Length Encoding (RLE) de Airbus.

El CSV del reto no guarda máscaras como imágenes: guarda texto plano con pares
``(inicio, longitud)`` sobre la imagen **aplanada**. Tres detalles determinan que
la reconstrucción sea correcta, y los tres costaron un error en el notebook:

1. El RLE indexa un **vector plano**, no una matriz fila a fila.
2. Los índices empiezan en **1**, no en 0 (de ahí el ``- 1``).
3. El aplanado es en **orden Fortran** (columna a columna), no en orden C.
   Con ``order='C'`` la máscara sale transpuesta y —esto es lo traicionero— el
   número de píxeles encendidos es idéntico, así que ninguna comprobación de
   tamaño lo detecta.

Una imagen con varios barcos ocupa varias filas del CSV. Todos sus runs se
acumulan sobre el mismo vector, de modo que la máscara resultante es
**semántica** (casco frente a agua) y no de instancias: dos barcos adyacentes se
funden en una sola región.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["rle_decode", "rle_encode", "DEFAULT_SHAPE"]

DEFAULT_SHAPE: tuple[int, int] = (768, 768)


def _is_missing(value: object) -> bool:
    """True si el valor representa «esta imagen no tiene barco».

    Cubre los tres nulos que aparecen al leer el CSV con pandas: ``None``,
    ``float('nan')`` y la cadena vacía.
    """
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def rle_decode(
    mask_rle: str | float | None | Sequence[str | float | None],
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> np.ndarray:
    """Reconstruye una máscara binaria a partir de uno o varios RLE.

    Args:
        mask_rle: un RLE suelto, un valor nulo, o una secuencia de ellos
            (todos los barcos de una misma imagen).
        shape: alto y ancho de la máscara de salida.

    Returns:
        ``np.ndarray`` de forma ``shape`` y dtype ``uint8`` con valores 0/1.

    Examples:
        >>> rle_decode("4 3", shape=(6, 6))[:, 0]
        array([0, 0, 0, 1, 1, 1], dtype=uint8)
    """
    height, width = shape
    flat = np.zeros(height * width, dtype=np.uint8)

    # Un solo string y una lista de strings se tratan igual.
    runs: Iterable[str | float | None]
    runs = [mask_rle] if isinstance(mask_rle, str) or _is_missing(mask_rle) else mask_rle

    for run in runs:
        if _is_missing(run):
            continue
        tokens = str(run).split()
        starts = np.asarray(tokens[0::2], dtype=np.int64) - 1  # el RLE cuenta desde 1
        lengths = np.asarray(tokens[1::2], dtype=np.int64)
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            flat[lo:hi] = 1

    # El dataset de Airbus está aplanado en orden Fortran (columna a columna).
    return flat.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Operación inversa de :func:`rle_decode`, en el formato que espera Kaggle.

    Necesaria para producir un envío: la métrica oficial se evalúa sobre el RLE
    a resolución completa, no sobre el tensor con el que se entrena.

    Args:
        mask: array 2D; cualquier valor distinto de cero cuenta como barco.

    Returns:
        El RLE como cadena de pares ``"inicio longitud"``, o cadena vacía si la
        máscara no contiene ningún píxel positivo.
    """
    flat = np.asarray(mask).flatten(order="F")
    # El centinela en los extremos evita casos especiales cuando un run empieza
    # en el primer píxel o termina en el último.
    padded = np.concatenate([[0], flat, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts, ends = changes[0::2], changes[1::2]
    if starts.size == 0:
        return ""
    return " ".join(f"{s} {e - s}" for s, e in zip(starts, ends))
