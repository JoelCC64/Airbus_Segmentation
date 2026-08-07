"""Piezas compartidas por los scripts de línea de comandos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar los scripts sin haber instalado el paquete (`pip install -e .`).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from airbus.config import Config  # noqa: E402
from airbus.utils import get_device, set_seed  # noqa: E402

__all__ = ["base_parser", "bootstrap"]

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def base_parser(description: str) -> argparse.ArgumentParser:
    """Parser con los argumentos comunes a todos los scripts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="fichero YAML de configuración"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="fuerza un dispositivo (por defecto: cuda → mps → cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="usa solo las N primeras imágenes; útil para una prueba rápida",
    )
    return parser


def bootstrap(args: argparse.Namespace) -> tuple[Config, "object"]:
    """Carga la configuración, fija la semilla y elige dispositivo."""
    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    device = get_device(args.device)
    print(f"config    : {args.config}")
    print(f"dispositivo: {device}")
    return config, device
