"""Configuration centrale du projet OC-P12-Agritech.

Ce module regroupe les constantes partagées par le projet : chemins locaux et
emplacement des deux jeux de données fournis. Il ne contient pas de logique
métier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "OC-P12-Agritech"
SEED = 42

AGRICULTURE_CROP_YIELD_FILENAME = "crop_yield.csv"


@dataclass(frozen=True)
class Paths:
    """Chemins principaux du projet."""

    root: Path
    data: Path
    data_agriculture_crop_yield: Path
    data_crop_yield_prediction: Path
    docs: Path
    notebooks: Path
    src: Path


def get_paths() -> Paths:
    """Construit les chemins du projet à partir de l'emplacement de ce fichier."""
    here = Path(__file__).resolve()
    root = here.parents[2]
    data_dir = root / "data"

    return Paths(
        root=root,
        data=data_dir,
        data_agriculture_crop_yield=data_dir / "agriculture-crop-yield",
        data_crop_yield_prediction=data_dir / "crop-yield-prediction",
        docs=root / "docs",
        notebooks=root / "notebooks",
        src=root / "src",
    )


PATHS = get_paths()

# Raccourci pratique pour les notebooks.
DATA_DIR = PATHS.data
