"""Tests de particiones, transformaciones y datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from airbus.config import Config
from airbus.data import (
    AirbusSegmentationDataset,
    balance_by_undersampling,
    build_segmentation_transforms,
    image_level_dataframe,
    segmentation_dataframe,
    split_dataframe,
)


@pytest.fixture
def csv_falso() -> pd.DataFrame:
    """CSV sintético: 3 imágenes con barco (una con dos) y 7 vacías."""
    filas = [
        {"ImageId": "a.jpg", "EncodedPixels": "1 5"},
        {"ImageId": "a.jpg", "EncodedPixels": "50 5"},   # segundo barco
        {"ImageId": "b.jpg", "EncodedPixels": "10 3"},
        {"ImageId": "c.jpg", "EncodedPixels": "20 4"},
    ]
    filas += [{"ImageId": f"v{i}.jpg", "EncodedPixels": None} for i in range(7)]
    df = pd.DataFrame(filas)
    df["has_ship"] = df["EncodedPixels"].notnull().astype(int)
    return df


# ── agregación a nivel de imagen ────────────────────────────────────────────

def test_una_fila_por_imagen(csv_falso):
    images = image_level_dataframe(csv_falso)
    assert len(images) == 10          # 3 con barco + 7 vacías, no 11 filas
    assert images["has_ship"].sum() == 3


def test_el_max_marca_la_imagen_completa_como_positiva(csv_falso):
    """'a.jpg' tiene dos filas; debe contar como UNA imagen con barco."""
    images = image_level_dataframe(csv_falso).set_index("ImageId")
    assert images.loc["a.jpg", "has_ship"] == 1
    assert images.loc["a.jpg", "total_ships"] == 2


def test_el_count_ignora_los_nulos(csv_falso):
    images = image_level_dataframe(csv_falso).set_index("ImageId")
    assert images.loc["v0.jpg", "total_ships"] == 0


# ── balanceo ────────────────────────────────────────────────────────────────

def test_el_submuestreo_deja_el_reparto_a_la_mitad(csv_falso):
    images = image_level_dataframe(csv_falso)
    balanced = balance_by_undersampling(images, seed=42, ratio=1.0)
    assert len(balanced) == 6
    assert balanced["has_ship"].sum() == 3


def test_el_submuestreo_es_reproducible(csv_falso):
    images = image_level_dataframe(csv_falso)
    a = balance_by_undersampling(images, seed=42)
    b = balance_by_undersampling(images, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_ratio_none_conserva_el_dataset_entero(csv_falso):
    """El experimento «sin balancear» que quedó pendiente en el notebook."""
    images = image_level_dataframe(csv_falso)
    completo = balance_by_undersampling(images, ratio=None)
    assert len(completo) == len(images)


# ── dataframe de segmentación ───────────────────────────────────────────────

def test_segmentacion_solo_ve_imagenes_con_barco(csv_falso):
    frame = segmentation_dataframe(csv_falso)
    assert len(frame) == 3
    assert not any(name.startswith("v") for name in frame["ImageId"])


def test_los_rle_de_una_imagen_se_agrupan_en_lista(csv_falso):
    frame = segmentation_dataframe(csv_falso).set_index("ImageId")
    assert frame.loc["a.jpg", "EncodedPixels"] == ["1 5", "50 5"]


# ── particiones ─────────────────────────────────────────────────────────────

def test_la_particion_estratificada_conserva_el_equilibrio():
    frame = pd.DataFrame(
        {"ImageId": [f"{i}.jpg" for i in range(100)], "has_ship": [1] * 50 + [0] * 50}
    )
    train, val = split_dataframe(frame, 0.2, seed=42, stratify_on="has_ship")
    assert len(train) == 80 and len(val) == 20
    assert train["has_ship"].mean() == pytest.approx(0.5)
    assert val["has_ship"].mean() == pytest.approx(0.5)


# ── transformaciones v2 ─────────────────────────────────────────────────────

def test_v2_escala_la_imagen_pero_no_la_mascara():
    """El punto clave de usar la API v2 con tv_tensors.Mask.

    ToDtype(scale=True) lleva la imagen a float32 en [0, 1]; si tocara la
    máscara, sus unos pasarían a 0,0039 y dejarían de ser etiquetas.
    """
    imagen = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
    mascara = np.zeros((16, 16), dtype=np.uint8)
    mascara[4:8, 4:8] = 1

    transform = build_segmentation_transforms(image_size=8, train=False)
    img_t, msk_t = transform(imagen, tv_tensors.Mask(mascara))

    assert img_t.dtype == torch.float32
    assert msk_t.dtype == torch.uint8
    assert set(torch.unique(msk_t).tolist()) <= {0, 1}


def test_v2_aplica_la_misma_geometria_a_imagen_y_mascara():
    """Con la API v1 esto serían dos sorteos aleatorios independientes."""
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[:, :16] = 255                       # mitad izquierda blanca
    mascara = np.zeros((32, 32), dtype=np.uint8)
    mascara[:, :16] = 1                       # la máscara marca esa misma mitad

    transform = build_segmentation_transforms(image_size=32, train=True)
    for _ in range(12):                       # varios sorteos del flip
        img_t, msk_t = transform(Image.fromarray(array), tv_tensors.Mask(mascara))
        brillante = img_t[0] > img_t[0].mean()
        # Imagen y máscara deben seguir señalando el mismo lado tras el volteo.
        assert torch.equal(brillante, msk_t.bool())


def test_validacion_sin_aleatoriedad():
    """Dos pasadas de validación sobre la misma entrada deben dar lo mismo."""
    imagen = Image.fromarray(np.random.RandomState(0).randint(0, 255, (32, 32, 3), dtype=np.uint8))
    mascara = tv_tensors.Mask(np.zeros((32, 32), dtype=np.uint8))
    transform = build_segmentation_transforms(image_size=16, train=False)

    a, _ = transform(imagen, mascara)
    b, _ = transform(imagen, mascara)
    assert torch.equal(a, b)


# ── dataset ─────────────────────────────────────────────────────────────────

def test_el_dataset_devuelve_imagen_y_mascara_alineadas(tmp_path):
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(tmp_path / "x.jpg")
    frame = pd.DataFrame({"ImageId": ["x.jpg"], "EncodedPixels": [["1 4"]]})

    dataset = AirbusSegmentationDataset(
        frame, str(tmp_path), build_segmentation_transforms(16, train=False),
        mask_shape=(32, 32),
    )
    imagen, mascara = dataset[0]

    assert imagen.shape == (3, 16, 16)
    assert mascara.shape == (16, 16)
    assert len(dataset) == 1


# ── configuración ───────────────────────────────────────────────────────────

def test_config_se_carga_desde_yaml(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        "paths:\n  csv: /d/t.csv\n  train_images: /d/imgs\n"
        "segmenter:\n  epochs: 9\n  threshold: 0.3\n",
        encoding="utf-8",
    )
    config = Config.from_yaml(yaml_path)
    assert config.paths.csv == "/d/t.csv"
    assert config.segmenter.epochs == 9
    assert config.segmenter.threshold == 0.3
    assert config.classifier.batch_size == 32       # valor por defecto
    assert config.segmenter.features == [64, 128, 256, 512]


def test_config_rechaza_claves_desconocidas(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        "paths:\n  csv: /d/t.csv\n  train_images: /d/i\nsegmenter:\n  epocas: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="Claves desconocidas"):
        Config.from_yaml(yaml_path)


def test_config_exige_la_seccion_paths(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="paths"):
        Config.from_yaml(yaml_path)
