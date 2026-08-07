"""Configuración tipada, cargada desde YAML.

Los hiperparámetros viven en ``configs/*.yaml`` y no incrustados en el código:
lanzar una variante es copiar el YAML y cambiar un valor, sin tocar los scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "PathsConfig",
    "ClassifierConfig",
    "SegmenterConfig",
    "build_config",
]


@dataclass
class PathsConfig:
    """Rutas de datos y de salida."""

    csv: str
    train_images: str
    artifacts: str = "artifacts"


@dataclass
class ClassifierConfig:
    """Régimen de entrenamiento de la ResNet-50."""

    image_size: int = 224
    batch_size: int = 32
    epochs: int = 2
    lr: float = 1e-3
    num_workers: int = 2
    val_fraction: float = 0.2
    pretrained: bool = True
    freeze_backbone: bool = False
    # ratio de imágenes sin barco por cada imagen con barco. 1.0 → 50/50.
    # null en el YAML desactiva el submuestreo y usa el dataset completo.
    negative_ratio: float | None = 1.0


@dataclass
class SegmenterConfig:
    """Régimen de entrenamiento de la U-Net."""

    image_size: int = 256
    batch_size: int = 16
    epochs: int = 3
    lr: float = 1e-4
    num_workers: int = 2
    val_fraction: float = 0.2
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bce_weight: float = 0.5
    threshold: float = 0.5


@dataclass
class Config:
    """Configuración raíz del proyecto."""

    paths: PathsConfig
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Construye la configuración desde un fichero YAML.

        Raises:
            FileNotFoundError: si el fichero no existe.
            TypeError: si el YAML trae una clave que no existe en el esquema, en
                lugar de ignorarla en silencio.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el fichero de configuración: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return build_config(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def artifacts_dir(self) -> Path:
        """Carpeta de salida, creada si hace falta."""
        directory = Path(self.paths.artifacts)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


# Secciones anidadas del YAML. Se declara el mapeo de forma explícita en lugar
# de inspeccionar `field.type`: con `from __future__ import annotations` los
# tipos llegan como cadenas y la introspección no distinguiría un dataclass.
_SECTIONS: dict[str, type] = {
    "paths": PathsConfig,
    "classifier": ClassifierConfig,
    "segmenter": SegmenterConfig,
}


def _instantiate(target: type, raw: dict[str, Any]) -> Any:
    """Instancia un dataclass validando que no lleguen claves desconocidas."""
    known = {f.name for f in fields(target)}
    unknown = set(raw) - known
    if unknown:
        raise TypeError(
            f"Claves desconocidas para {target.__name__}: {sorted(unknown)}. "
            f"Válidas: {sorted(known)}"
        )
    return target(**raw)


def build_config(raw: dict[str, Any]) -> Config:
    """Construye una :class:`Config` a partir del diccionario del YAML."""
    if "paths" not in raw:
        raise TypeError("La configuración necesita una sección 'paths'.")

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTIONS:
            if not isinstance(value, dict):
                raise TypeError(f"La sección '{key}' debe ser un mapeo, no {type(value).__name__}.")
            kwargs[key] = _instantiate(_SECTIONS[key], value)
        else:
            kwargs[key] = value

    return _instantiate(Config, kwargs)
