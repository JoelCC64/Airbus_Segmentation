#!/usr/bin/env python3
"""Inferencia extremo a extremo: imagen → máscara, pasando por la cascada.

Es el punto de entrada que faltaba en el notebook, donde las dos redes existían
entrenadas pero nada las encadenaba.

Uso:
    python scripts/predict.py --classifier artifacts/classifier_resnet50.pth \\
                              --segmenter  artifacts/unet.pth \\
                              --images ruta/a/imagenes/*.jpg --csv salida.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from _common import base_parser, bootstrap

from airbus.models import AirbusClassifier, UNet
from airbus.pipeline import ShipSegmentationPipeline
from airbus.utils import load_weights


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--classifier", required=True, help="pesos de la ResNet-50 (.pth)")
    parser.add_argument("--segmenter", required=True, help="pesos de la U-Net (.pth)")
    parser.add_argument("--images", nargs="+", required=True, help="imágenes o carpetas")
    parser.add_argument("--csv", default=None, help="escribe un CSV en formato de envío")
    args = parser.parse_args()
    config, device = bootstrap(args)

    # ── resolver la lista de imágenes ──────────────────────────────────────
    paths: list[Path] = []
    for entry in args.images:
        path = Path(entry)
        if path.is_dir():
            paths += sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        elif path.is_file():
            paths.append(path)
        else:
            print(f"[aviso] no existe: {entry}")
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No se encontró ninguna imagen.")
    print(f"imágenes a procesar: {len(paths):,}")

    # ── construir la cascada ───────────────────────────────────────────────
    classifier = AirbusClassifier(pretrained=False)
    load_weights(classifier, args.classifier, device)

    segmenter = UNet(in_channels=3, out_channels=1, features=config.segmenter.features)
    load_weights(segmenter, args.segmenter, device)

    pipeline = ShipSegmentationPipeline(
        classifier,
        segmenter,
        device,
        classifier_size=config.classifier.image_size,
        segmenter_size=config.segmenter.image_size,
        mask_threshold=config.segmenter.threshold,
    )

    # ── inferencia ─────────────────────────────────────────────────────────
    rows, with_ship = [], 0
    for path in tqdm(paths, desc="inferencia"):
        prediction = pipeline.predict(Image.open(path))
        with_ship += int(prediction.has_ship)
        rows.append((path.name, prediction.to_rle(), prediction.ship_probability))

    print(f"\ncon barco: {with_ship:,} | vacías: {len(rows) - with_ship:,}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ImageId", "EncodedPixels"])
            writer.writerows((name, rle) for name, rle, _ in rows)
        print(f"CSV escrito en {out}")


if __name__ == "__main__":
    main()
