"""Chargement des modèles servis par l'API (`/predict`, `/recommend`).

Module commun aux deux services. Il ne connaît pas HTTP : il lit les artefacts
sauvegardés dans `models/` et renvoie un objet simple, réutilisable par
n'importe quelle couche (API, script, notebook).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from agritech.config import PATHS


# Clés que le metadata d'un modèle servi par l'API doit obligatoirement porter.
# `training_domain` alimente le drapeau `out_of_training_domain` de la réponse ;
# `model_version` alimente `/health` et servira à tracer l'artefact côté client.
_REQUIRED_METADATA_KEYS = (
    "service",
    "features",
    "training_domain",
    "model_version",
    "created_on",
)


@dataclass(frozen=True, slots=True)
class Bundle:
    """Pipeline scikit-learn chargé en mémoire et ses métadonnées JSON."""

    name: str
    pipeline: Any
    metadata: dict


def load_bundle(name: str, models_dir: Path | None = None) -> Bundle:
    """Charge le bundle du service `name` depuis `models/{name}_model.*`.

    Lit `models/{name}_model.joblib` (pipeline scikit-learn) et
    `models/{name}_model_metadata.json` (dict), vérifie que les clés
    obligatoires sont présentes, puis renvoie un `Bundle` immuable.

    Lève une exception explicite plutôt qu'un chargement silencieux :
    fichier absent, JSON invalide ou clé obligatoire manquante.
    """
    root = models_dir if models_dir is not None else PATHS.root / "models"
    joblib_path = root / f"{name}_model.joblib"
    metadata_path = root / f"{name}_model_metadata.json"

    if not joblib_path.is_file():
        raise FileNotFoundError(f"artefact modèle introuvable : {joblib_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"métadonnées introuvables : {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata {metadata_path.name} : clés manquantes {missing}")

    pipeline = joblib.load(joblib_path)
    return Bundle(name=name, pipeline=pipeline, metadata=metadata)
