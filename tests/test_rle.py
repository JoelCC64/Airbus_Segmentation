"""Tests de la codificación RLE — la parte donde se concentraron los errores."""

from __future__ import annotations

import numpy as np
import pytest

from airbus.data.rle import rle_decode, rle_encode


def test_orden_fortran_produce_segmento_vertical():
    """El run '4 3' debe caer en la primera columna, filas 3-5.

    Con orden C (fila a fila) caería horizontalmente en la primera fila: mismo
    número de píxeles, máscara transpuesta. Este es exactamente el error que no
    detecta ninguna comprobación de tamaño.
    """
    mask = rle_decode("4 3", shape=(6, 6))
    esperado = np.zeros((6, 6), dtype=np.uint8)
    esperado[3:6, 0] = 1
    np.testing.assert_array_equal(mask, esperado)


def test_segundo_run_cae_en_la_cuarta_columna():
    """El run '20 2' → índices 20 y 21 → columna 3, filas 1-2."""
    mask = rle_decode("20 2", shape=(6, 6))
    esperado = np.zeros((6, 6), dtype=np.uint8)
    esperado[1:3, 3] = 1
    np.testing.assert_array_equal(mask, esperado)


def test_el_orden_c_daria_una_mascara_distinta():
    """Prueba explícita de que order='F' y order='C' no son intercambiables."""
    fortran = rle_decode("4 3", shape=(6, 6))
    plano = np.zeros(36, dtype=np.uint8)
    plano[3:6] = 1
    c_order = plano.reshape((6, 6))  # lo que saldría con el reshape por defecto

    assert fortran.sum() == c_order.sum() == 3  # el conteo no delata el error
    assert not np.array_equal(fortran, c_order)


def test_indices_empiezan_en_uno():
    """El primer píxel de la imagen es el índice 1, no el 0."""
    mask = rle_decode("1 1", shape=(4, 4))
    assert mask[0, 0] == 1
    assert mask.sum() == 1


def test_varios_barcos_se_acumulan_en_una_mascara():
    """Una lista de RLE es una imagen con varios barcos: se funden en una máscara."""
    mask = rle_decode(["1 2", "7 2"], shape=(3, 3))
    assert mask.sum() == 4


def test_runs_solapados_no_cuentan_dos_veces():
    """La máscara es binaria: solapar dos runs no produce el valor 2."""
    mask = rle_decode(["1 3", "2 3"], shape=(3, 3))
    assert set(np.unique(mask)) <= {0, 1}
    assert mask.sum() == 4


@pytest.mark.parametrize("vacio", [None, float("nan"), "", "   "])
def test_valores_nulos_dan_mascara_vacia(vacio):
    """Una imagen sin barco produce una máscara de ceros, no un error."""
    mask = rle_decode(vacio, shape=(8, 8))
    assert mask.shape == (8, 8)
    assert mask.sum() == 0


def test_lista_con_nulos_intercalados():
    """Los nulos dentro de una lista se ignoran sin romper el resto."""
    mask = rle_decode(["4 3", None, float("nan")], shape=(6, 6))
    assert mask.sum() == 3


def test_dtype_y_forma():
    mask = rle_decode("100 5")
    assert mask.shape == (768, 768)
    assert mask.dtype == np.uint8


@pytest.mark.parametrize(
    "rle",
    ["4 3", "1 1", "20 2", "1 2 7 2", "264661 17 265429 33"],
)
def test_encode_decode_ida_y_vuelta(rle):
    """rle_encode debe ser la inversa exacta de rle_decode."""
    shape = (768, 768)
    mask = rle_decode(rle, shape=shape)
    assert rle_decode(rle_encode(mask), shape=shape).sum() == mask.sum()
    np.testing.assert_array_equal(rle_decode(rle_encode(mask), shape=shape), mask)


def test_encode_de_mascara_vacia_es_cadena_vacia():
    assert rle_encode(np.zeros((8, 8), dtype=np.uint8)) == ""


def test_encode_de_mascara_llena():
    """Una máscara completamente encendida es un único run que cubre todo."""
    mask = np.ones((4, 4), dtype=np.uint8)
    assert rle_encode(mask) == "1 16"
