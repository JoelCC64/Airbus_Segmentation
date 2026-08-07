"""Tests de la pérdida combinada y de las métricas acumuladas."""

from __future__ import annotations

import math

import torch

from airbus.losses import BCEDiceLoss, dice_coefficient
from airbus.metrics import ConfusionAccumulator, binary_accuracy


# ── Dice ────────────────────────────────────────────────────────────────────

def test_dice_es_uno_con_prediccion_perfecta():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:4, 2:6] = 1.0
    assert dice_coefficient(target.clone(), target).item() == 1.0


def test_dice_es_cero_sin_solapamiento():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 0, 0] = 1.0
    probs = torch.zeros(1, 1, 8, 8)
    probs[..., 7, 7] = 1.0
    assert dice_coefficient(probs, target).item() < 1e-3


# ── BCEDiceLoss ─────────────────────────────────────────────────────────────

def test_la_prediccion_degenerada_recibe_maxima_penalizacion_del_dice():
    """Predecir «todo es océano» debe dar término Dice = 1.

    Es justo el atajo que la BCE sola premiaba: con 0,51 % de píxeles de barco,
    una máscara vacía acierta el 99,5 % de los píxeles.
    """
    target = torch.zeros(1, 1, 64, 64)
    target[..., 30:34, 30:38] = 1.0          # ~0,8 % de píxeles positivos
    logits_vacios = torch.full((1, 1, 64, 64), -20.0)  # sigmoide ≈ 0

    solo_dice = BCEDiceLoss(bce_weight=0.0)
    assert math.isclose(solo_dice(logits_vacios, target).item(), 1.0, abs_tol=1e-3)


def test_la_perdida_baja_cuando_la_prediccion_acierta():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 10:20, 10:20] = 1.0

    buena = torch.where(target > 0, 6.0, -6.0)
    mala = torch.full_like(target, -6.0)

    criterion = BCEDiceLoss(bce_weight=0.5)
    assert criterion(buena, target).item() < criterion(mala, target).item()


def test_bce_weight_reparte_los_dos_terminos():
    """La pérdida combinada es la media exacta de sus dos componentes puras."""
    torch.manual_seed(0)
    target = torch.randint(0, 2, (1, 1, 16, 16)).float()
    logits = torch.randn(1, 1, 16, 16)

    solo_bce = BCEDiceLoss(bce_weight=1.0)(logits, target)
    solo_dice = BCEDiceLoss(bce_weight=0.0)(logits, target)
    mezcla = BCEDiceLoss(bce_weight=0.5)(logits, target)

    assert math.isclose(mezcla.item(), 0.5 * (solo_bce + solo_dice).item(), rel_tol=1e-5)


def test_la_perdida_es_derivable():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:8, 4:8] = 1.0
    logits = torch.randn(1, 1, 16, 16, requires_grad=True)

    BCEDiceLoss()(logits, target).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ── Métricas ────────────────────────────────────────────────────────────────

def test_metricas_con_prediccion_perfecta():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:8, 4:8] = 1.0

    acc = ConfusionAccumulator()
    acc.update(target.clone(), target)
    m = acc.compute()

    assert math.isclose(m.f2, 1.0, abs_tol=1e-5)
    assert math.isclose(m.iou, 1.0, abs_tol=1e-5)
    assert math.isclose(m.dice, 1.0, abs_tol=1e-5)
    assert m.fp == 0 and m.fn == 0


def test_f2_castiga_los_falsos_negativos_mas_que_los_falsos_positivos():
    """El corazón de la métrica oficial: omitir pesa ×4 más que la falsa alarma."""
    target = torch.zeros(100)
    target[:20] = 1.0

    omite = torch.zeros(100)
    omite[:10] = 1.0                 # 10 TP, 10 FN, 0 FP

    falsa_alarma = torch.zeros(100)
    falsa_alarma[:30] = 1.0          # 20 TP, 0 FN, 10 FP

    a, b = ConfusionAccumulator(), ConfusionAccumulator()
    a.update(omite, target)
    b.update(falsa_alarma, target)

    assert b.compute().f2 > a.compute().f2


def test_los_contadores_se_acumulan_entre_lotes():
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    preds = torch.tensor([1.0, 0.0, 1.0, 0.0])

    acc = ConfusionAccumulator()
    acc.update(preds, target)
    acc.update(preds, target)
    m = acc.compute()

    assert (m.tp, m.fn, m.fp, m.tn) == (2, 2, 2, 2)


def test_f2_coincide_con_la_formula_clasica():
    """5·TP/(5·TP+4·FN+FP) debe ser igual a 5·P·R/(4·P+R)."""
    target = torch.tensor([1.0] * 20 + [0.0] * 80)
    preds = torch.tensor([1.0] * 15 + [0.0] * 5 + [1.0] * 8 + [0.0] * 72)

    acc = ConfusionAccumulator()
    acc.update(preds, target)
    m = acc.compute()

    clasica = 5 * m.precision * m.recall / (4 * m.precision + m.recall)
    assert math.isclose(m.f2, clasica, rel_tol=1e-6)


# ── Umbral en el espacio de logits ──────────────────────────────────────────

def test_el_umbral_del_clasificador_esta_en_cero_no_en_medio():
    """sigmoid(0) = 0,5, así que 0,0 sobre logits equivale a 0,5 sobre probabilidad.

    Un logit de 0,3 corresponde a una probabilidad de 0,57: es positivo. Si se
    umbralizara en 0,5 sobre los logits quedaría mal clasificado.
    """
    logits = torch.tensor([[0.3], [-0.3]])
    etiquetas = torch.tensor([[1.0], [0.0]])
    aciertos, total = binary_accuracy(logits, etiquetas)
    assert (aciertos, total) == (2, 2)
    assert torch.sigmoid(torch.tensor(0.3)).item() > 0.5
