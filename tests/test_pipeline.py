"""Tests de la inferencia en cascada y de la persistencia."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from airbus.data.rle import rle_decode
from airbus.models import AirbusClassifier, UNet
from airbus.pipeline import ShipSegmentationPipeline
from airbus.utils import load_checkpoint, load_weights, save_checkpoint, save_weights


class _PuertaFija(torch.nn.Module):
    """Clasificador de mentira que siempre devuelve el mismo logit."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit
        self.parametro = torch.nn.Parameter(torch.zeros(1))  # para tener .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.logit)


@pytest.fixture
def imagen() -> Image.Image:
    array = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.fixture
def unet() -> UNet:
    return UNet(features=(8, 16))


def test_sin_barco_devuelve_mascara_vacia_sin_invocar_la_unet(imagen, unet):
    """La rama barata: el 77,9 % de las imágenes no debe tocar el segmentador."""

    class _Explota(torch.nn.Module):
        def forward(self, x):  # pragma: no cover - no debe ejecutarse
            raise AssertionError("la U-Net no debería ejecutarse sin barco")

    pipeline = ShipSegmentationPipeline(
        _PuertaFija(-10.0), _Explota(), torch.device("cpu"), output_shape=(64, 64)
    )
    prediccion = pipeline.predict(imagen)

    assert prediccion.has_ship is False
    assert prediccion.mask.sum() == 0
    assert prediccion.to_rle() == ""
    assert prediccion.ship_probability < 0.5


def test_con_barco_devuelve_mascara_a_resolucion_de_salida(imagen, unet):
    pipeline = ShipSegmentationPipeline(
        _PuertaFija(10.0), unet, torch.device("cpu"), output_shape=(128, 128)
    )
    prediccion = pipeline.predict(imagen)

    assert prediccion.has_ship is True
    assert prediccion.mask.shape == (128, 128)
    assert prediccion.mask.dtype == np.uint8
    assert set(np.unique(prediccion.mask)) <= {0, 1}
    assert prediccion.ship_probability > 0.5


def test_el_rle_de_la_prediccion_se_puede_volver_a_decodificar(imagen, unet):
    """Cierre del círculo: lo que se envía a Kaggle debe poder releerse."""
    pipeline = ShipSegmentationPipeline(
        _PuertaFija(10.0), unet, torch.device("cpu"), output_shape=(64, 64)
    )
    prediccion = pipeline.predict(imagen)
    reconstruida = rle_decode(prediccion.to_rle(), shape=(64, 64))
    np.testing.assert_array_equal(reconstruida, prediccion.mask)


def test_umbral_de_mascara_mas_alto_no_amplia_la_prediccion(imagen, unet):
    """Subir el umbral solo puede quitar píxeles, nunca añadirlos."""
    kwargs = dict(device=torch.device("cpu"), output_shape=(64, 64))
    laxo = ShipSegmentationPipeline(_PuertaFija(10.0), unet, mask_threshold=0.2, **kwargs)
    estricto = ShipSegmentationPipeline(_PuertaFija(10.0), unet, mask_threshold=0.8, **kwargs)

    assert estricto.predict(imagen).mask.sum() <= laxo.predict(imagen).mask.sum()


def test_predict_batch_devuelve_una_prediccion_por_imagen(imagen, unet):
    pipeline = ShipSegmentationPipeline(
        _PuertaFija(10.0), unet, torch.device("cpu"), output_shape=(64, 64)
    )
    assert len(pipeline.predict_batch([imagen, imagen, imagen])) == 3


# ── persistencia ────────────────────────────────────────────────────────────

def test_los_pesos_sobreviven_al_viaje_de_ida_y_vuelta(tmp_path):
    original = UNet(features=(8, 16))
    save_weights(original, tmp_path / "u.pth")

    restaurado = load_weights(UNet(features=(8, 16)), tmp_path / "u.pth")

    entrada = torch.randn(1, 3, 32, 32)
    original.eval(), restaurado.eval()
    with torch.no_grad():
        assert torch.allclose(original(entrada), restaurado(entrada))


def test_el_checkpoint_conserva_el_estado_del_optimizador(tmp_path):
    """Sin los dos momentos de Adam, reanudar equivale a reiniciar el calentamiento."""
    model = UNet(features=(8, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Un paso real, para que Adam tenga estado que guardar.
    model(torch.randn(1, 3, 32, 32)).sum().backward()
    optimizer.step()

    save_checkpoint(tmp_path / "c.pth", model, optimizer, epoch=3, loss=0.42)

    nuevo_modelo = UNet(features=(8, 16))
    nuevo_opt = torch.optim.Adam(nuevo_modelo.parameters(), lr=1e-4)
    checkpoint = load_checkpoint(tmp_path / "c.pth", nuevo_modelo, nuevo_opt)

    assert checkpoint["epoch"] == 3
    assert checkpoint["loss"] == 0.42
    assert nuevo_opt.state_dict()["state"], "el estado de Adam llegó vacío"


def test_el_clasificador_expone_un_solo_logit():
    model = AirbusClassifier(pretrained=False).eval()
    with torch.no_grad():
        salida = model(torch.randn(2, 3, 224, 224))
    assert salida.shape == (2, 1)


def test_congelar_el_backbone_deja_entrenable_solo_la_cabeza():
    model = AirbusClassifier(pretrained=False, freeze_backbone=True)
    entrenables = [n for n, p in model.named_parameters() if p.requires_grad]
    assert entrenables == ["backbone.fc.weight", "backbone.fc.bias"]


def test_el_clasificador_tiene_23_510_081_parametros():
    """Cifra de referencia del proyecto tras sustituir fc por Linear(2048, 1)."""
    model = AirbusClassifier(pretrained=False)
    assert sum(p.numel() for p in model.parameters()) == 23_510_081
