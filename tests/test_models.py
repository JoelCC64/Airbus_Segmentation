"""Tests de forma y de rango de salida de las dos redes.

Reproduce la comprobación con tensor falso que el notebook hacía antes de cada
entrenamiento: cuesta un segundo y detecta los desajustes de dimensión que de
otro modo aparecen una hora después.
"""

from __future__ import annotations

import pytest
import torch

from airbus.models import UNet
from airbus.models.unet import DoubleConv


def test_unet_conserva_la_resolucion_de_entrada():
    model = UNet(in_channels=3, out_channels=1).eval()
    with torch.no_grad():
        salida = model(torch.randn(2, 3, 256, 256))
    assert salida.shape == (2, 1, 256, 256)


@pytest.mark.parametrize("lado", [64, 128, 160])
def test_unet_acepta_otras_resoluciones(lado):
    """La red no debe estar atada a 256 × 256."""
    model = UNet(features=(8, 16)).eval()
    with torch.no_grad():
        salida = model(torch.randn(1, 3, lado, lado))
    assert salida.shape == (1, 1, lado, lado)


def test_unet_acepta_lados_no_multiplos_de_la_profundidad():
    """Con lados que no son potencia de 2 el decoder tiene que reajustar."""
    model = UNet(features=(8, 16)).eval()
    with torch.no_grad():
        salida = model(torch.randn(1, 3, 70, 54))
    assert salida.shape == (1, 1, 70, 54)


def test_nada_recorta_la_salida_de_la_unet_a_valores_no_negativos():
    """La capa final no lleva activación: produce logits, no probabilidades.

    Si terminase en ReLU, ningún valor podría ser negativo y, como
    sigmoid(0) = 0,5, ningún píxel podría bajar del 50 % de probabilidad: con el
    umbral en 0,5 la red predeciría barco en todas partes.

    Se comprueba de forma determinista en lugar de confiar en la inicialización
    aleatoria: anulando los pesos de la conv final, la salida debe ser
    exactamente su sesgo. Si hubiera una ReLU detrás, saldría 0.
    """
    model = UNet(features=(8, 16)).eval()
    with torch.no_grad():
        model.final_conv.weight.zero_()
        model.final_conv.bias.fill_(-5.0)
        salida = model(torch.randn(2, 3, 64, 64))

    assert torch.allclose(salida, torch.full_like(salida, -5.0)), (
        "la salida quedó recortada: hay una activación después de final_conv"
    )


def test_numero_de_parametros_de_la_unet():
    """Cifra de referencia del proyecto: 31 037 633 parámetros."""
    assert sum(p.numel() for p in UNet().parameters()) == 31_037_633


def test_el_cuello_de_botella_domina_el_tamano():
    """El bloque de 512→1024 concentra ~46 % de los pesos de la red."""
    model = UNet()
    total = sum(p.numel() for p in model.parameters())
    bottleneck = sum(p.numel() for p in model.bottleneck.parameters())
    assert 0.44 < bottleneck / total < 0.48


def test_doubleconv_no_usa_bias_en_las_convoluciones():
    """El BatchNorm que viene detrás ya aporta el desplazamiento."""
    bloque = DoubleConv(3, 8)
    convoluciones = [m for m in bloque.conv if isinstance(m, torch.nn.Conv2d)]
    assert len(convoluciones) == 2
    assert all(conv.bias is None for conv in convoluciones)


def test_doubleconv_encadena_bien_los_canales():
    """La segunda convolución entra con out_channels, no con in_channels."""
    bloque = DoubleConv(3, 8)
    primera, segunda = [m for m in bloque.conv if isinstance(m, torch.nn.Conv2d)]
    assert primera.in_channels == 3 and primera.out_channels == 8
    assert segunda.in_channels == 8 and segunda.out_channels == 8


def test_la_capa_final_es_una_conv_1x1_desnuda():
    model = UNet(out_channels=1)
    assert isinstance(model.final_conv, torch.nn.Conv2d)
    assert model.final_conv.kernel_size == (1, 1)
    assert model.final_conv.out_channels == 1
