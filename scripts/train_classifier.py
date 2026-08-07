#!/usr/bin/env python3
"""Entrena la ResNet-50 que decide si una imagen contiene algún barco.

Uso:
    python scripts/train_classifier.py --config configs/default.yaml
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _common import base_parser, bootstrap

from airbus.data import (
    AirbusClassificationDataset,
    balance_by_undersampling,
    build_classifier_transforms,
    image_level_dataframe,
    load_dataframe,
    split_dataframe,
)
from airbus.engine import train_classifier_epoch, validate_classifier
from airbus.models import AirbusClassifier
from airbus.utils import save_checkpoint, save_weights


def main() -> None:
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    config, device = bootstrap(args)
    cfg = config.classifier

    # ── datos ──────────────────────────────────────────────────────────────
    df = load_dataframe(config.paths.csv)
    images = image_level_dataframe(df)
    n_pos = int(images["has_ship"].sum())
    print(f"imágenes con barco: {n_pos:,} | sin barco: {len(images) - n_pos:,}")

    balanced = balance_by_undersampling(images, seed=config.seed, ratio=cfg.negative_ratio)
    if args.limit:
        balanced = balanced.head(args.limit)

    train_df, val_df = split_dataframe(
        balanced, cfg.val_fraction, config.seed, stratify_on="has_ship"
    )
    print(f"train: {len(train_df):,} | validación: {len(val_df):,}")

    def make_loader(frame, train: bool) -> DataLoader:
        dataset = AirbusClassificationDataset(
            frame,
            config.paths.train_images,
            build_classifier_transforms(cfg.image_size, train=train),
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
    model = AirbusClassifier(
        pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parámetros entrenables: {n_params:,}")

    # ── entrenamiento ──────────────────────────────────────────────────────
    best_accuracy = float("-inf")  # guarda siempre la primera época evaluada
    out = config.artifacts_dir

    for epoch in range(1, cfg.epochs + 1):
        train_result = train_classifier_epoch(
            model, train_loader, criterion, optimizer, device, desc=f"época {epoch}"
        )
        # A diferencia del notebook, aquí sí se evalúa sobre datos retenidos:
        # sin esto la accuracy que se reporta es de entrenamiento.
        val_result = validate_classifier(model, val_loader, criterion, device)
        print(f"época {epoch}/{cfg.epochs}  train[{train_result}]  val[{val_result}]")

        if val_result.accuracy is not None and val_result.accuracy > best_accuracy:
            best_accuracy = val_result.accuracy
            save_weights(model, out / "classifier_resnet50.pth")
            save_checkpoint(
                out / "classifier_checkpoint.pth",
                model,
                optimizer,
                epoch,
                val_accuracy=val_result.accuracy,
                val_loss=val_result.loss,
            )
            print(f"  ↳ mejor hasta ahora, guardado en {out}/")

    if best_accuracy == float("-inf"):
        print("\nno se ejecutó ninguna época.")
    else:
        print(f"\nmejor accuracy de validación: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
