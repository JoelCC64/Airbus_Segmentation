#!/usr/bin/env python3
"""Entrena la U-Net que delimita el casco píxel a píxel.

Solo se entrena con imágenes que contienen barco: por eso el sistema necesita el
clasificador delante como puerta.

Uso:
    python scripts/train_segmenter.py --config configs/default.yaml
    python scripts/train_segmenter.py --resume artifacts/unet_checkpoint.pth
"""

from __future__ import annotations

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
from airbus.engine import evaluate_segmenter, train_segmenter_epoch
from airbus.losses import BCEDiceLoss
from airbus.models import UNet
from airbus.utils import load_checkpoint, save_checkpoint, save_weights


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--resume", type=str, default=None, help="checkpoint desde el que continuar"
    )
    args = parser.parse_args()
    config, device = bootstrap(args)
    cfg = config.segmenter

    # ── datos ──────────────────────────────────────────────────────────────
    df = load_dataframe(config.paths.csv)
    frame = segmentation_dataframe(df)
    if args.limit:
        frame = frame.head(args.limit)

    train_df, val_df = split_dataframe(frame, cfg.val_fraction, config.seed)
    print(f"imágenes con barco: {len(frame):,}")
    print(f"train: {len(train_df):,} | validación: {len(val_df):,}")

    def make_loader(subset, train: bool) -> DataLoader:
        dataset = AirbusSegmentationDataset(
            subset,
            config.paths.train_images,
            build_segmentation_transforms(cfg.image_size, train=train),
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_df, train=True)
    val_loader = make_loader(val_df, train=False)

    # ── modelo ─────────────────────────────────────────────────────────────
    model = UNet(in_channels=3, out_channels=1, features=cfg.features).to(device)
    criterion = BCEDiceLoss(bce_weight=cfg.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"parámetros: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"reanudado desde {args.resume} · siguiente época: {start_epoch}")

    # ── entrenamiento ──────────────────────────────────────────────────────
    best_f2 = float("-inf")  # guarda siempre la primera época evaluada
    out = config.artifacts_dir

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_result = train_segmenter_epoch(
            model, train_loader, criterion, optimizer, device, desc=f"época {epoch}"
        )
        metrics, val_loss = evaluate_segmenter(
            model, val_loader, device, cfg.threshold, criterion
        )
        print(
            f"época {epoch}/{cfg.epochs}  train[{train_result}]  "
            f"val_loss={val_loss:.4f}\n  {metrics}"
        )

        if metrics.f2 > best_f2:
            best_f2 = metrics.f2
            save_weights(model, out / "unet.pth")
            save_checkpoint(
                out / "unet_checkpoint.pth",
                model,
                optimizer,
                epoch,
                loss=train_result.loss,
                metrics=metrics.as_dict(),
            )
            print(f"  ↳ mejor F2 hasta ahora, guardado en {out}/")

    if best_f2 == float("-inf"):
        print("\nno se ejecutó ninguna época (¿el checkpoint ya alcanzó epochs?).")
    else:
        print(f"\nmejor F2 de validación: {best_f2:.4f}")


if __name__ == "__main__":
    main()
