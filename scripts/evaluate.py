#!/usr/bin/env python3
"""Evalúa la U-Net entrenada y, opcionalmente, barre el umbral de decisión.

El umbral por defecto (0,5) es el valor neutro, no el óptimo: como F2 pondera el
recall cuatro veces más que la precisión y la red resulta ser conservadora
(precisión > recall), bajarlo suele subir la puntuación. Este barrido es la
mejora más barata disponible, porque no requiere reentrenar nada.

Uso:
    python scripts/evaluate.py --weights artifacts/unet.pth
    python scripts/evaluate.py --weights artifacts/unet.pth --sweep
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from _common import base_parser, bootstrap

from airbus.data import (
    AirbusSegmentationDataset,
    build_segmentation_transforms,
    load_dataframe,
    segmentation_dataframe,
    split_dataframe,
)
from airbus.engine import evaluate_segmenter
from airbus.metrics import ConfusionAccumulator
from airbus.models import UNet
from airbus.utils import load_weights
from airbus.utils.viz import plot_segmentation_batch


@torch.no_grad()
def sweep_threshold(model, loader, device, thresholds) -> list[tuple[float, float, float]]:
    """Calcula F2, precisión y recall para cada umbral en una sola pasada.

    Se acumulan las probabilidades una vez y se umbralizan N veces, en lugar de
    releer el dataset por cada umbral.
    """
    accumulators = {t: ConfusionAccumulator() for t in thresholds}
    model.eval()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.float().to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        probs = torch.sigmoid(model(images))
        for t in thresholds:
            accumulators[t].update((probs > t).float(), masks)

    rows = []
    for threshold, accumulator in accumulators.items():
        metrics = accumulator.compute()
        rows.append((float(threshold), metrics.f2, metrics.recall))
    return rows


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--weights", required=True, help="pesos de la U-Net (.pth)")
    parser.add_argument("--sweep", action="store_true", help="barre el umbral de 0,1 a 0,9")
    parser.add_argument("--figure", default=None, help="ruta donde guardar una rejilla de ejemplos")
    args = parser.parse_args()
    config, device = bootstrap(args)
    cfg = config.segmenter

    frame = segmentation_dataframe(load_dataframe(config.paths.csv))
    if args.limit:
        frame = frame.head(args.limit)
    _, val_df = split_dataframe(frame, cfg.val_fraction, config.seed)

    dataset = AirbusSegmentationDataset(
        val_df,
        config.paths.train_images,
        build_segmentation_transforms(cfg.image_size, train=False),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    print(f"imágenes de validación: {len(dataset):,}")

    model = UNet(in_channels=3, out_channels=1, features=cfg.features)
    load_weights(model, args.weights, device)
    model.to(device)

    metrics, _ = evaluate_segmenter(model, loader, device, cfg.threshold)
    print(f"\numbral {cfg.threshold}\n{metrics}")

    if args.sweep:
        print("\numbral    F2       recall")
        results = sweep_threshold(model, loader, device, np.round(np.arange(0.1, 0.95, 0.1), 2))
        for threshold, f2, recall in sorted(results):
            print(f"  {threshold:<6.2f}  {f2:.4f}   {recall:.4f}")
        best = max(results, key=lambda row: row[1])
        print(f"\nmejor F2 = {best[1]:.4f} con umbral {best[0]:.2f}")

    if args.figure:
        plot_segmentation_batch(
            model, loader, device, num_samples=4, threshold=cfg.threshold,
            save_path=args.figure,
        )
        print(f"figura guardada en {args.figure}")


if __name__ == "__main__":
    main()
