"""Tests du module `agritech.serving` : chargement des bundles servis par l'API.

Les cas d'erreur (fichier absent, metadata incomplète) sont testés sur un
dossier temporaire, sans jamais modifier les artefacts versionnés dans
`models/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agritech.config import PATHS
from agritech.serving import Bundle, load_bundle


MODELS_DIR = PATHS.root / "models"


def test_load_bundle_predict_reads_versioned_artifacts():
    """`load_bundle('predict')` lit l'artefact et le metadata versionnés."""
    bundle = load_bundle("predict")

    assert isinstance(bundle, Bundle)
    assert bundle.name == "predict"
    assert bundle.metadata["service"] == "predict"
    assert bundle.metadata["model_version"] == "1.0.0"

    training_domain = bundle.metadata["training_domain"]
    assert set(training_domain) == {"Rainfall_mm", "Temperature_Celsius"}
    assert training_domain["Rainfall_mm"] == {"min": 100, "max": 1000, "unit": "mm"}
    assert training_domain["Temperature_Celsius"] == {"min": 15, "max": 40, "unit": "°C"}

    assert hasattr(bundle.pipeline, "predict")


def test_load_bundle_missing_joblib(tmp_path: Path):
    """Fichier `.joblib` absent : `FileNotFoundError` explicite."""
    shutil.copy(MODELS_DIR / "predict_model_metadata.json", tmp_path / "predict_model_metadata.json")

    with pytest.raises(FileNotFoundError, match="artefact modèle introuvable"):
        load_bundle("predict", models_dir=tmp_path)


def test_load_bundle_missing_metadata(tmp_path: Path):
    """Fichier metadata absent : `FileNotFoundError` explicite."""
    shutil.copy(MODELS_DIR / "predict_model.joblib", tmp_path / "predict_model.joblib")

    with pytest.raises(FileNotFoundError, match="métadonnées introuvables"):
        load_bundle("predict", models_dir=tmp_path)


def test_load_bundle_metadata_missing_training_domain(tmp_path: Path):
    """Metadata sans le bloc `training_domain` : `ValueError` explicite."""
    shutil.copy(MODELS_DIR / "predict_model.joblib", tmp_path / "predict_model.joblib")
    metadata = json.loads((MODELS_DIR / "predict_model_metadata.json").read_text(encoding="utf-8"))
    metadata.pop("training_domain", None)
    (tmp_path / "predict_model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="training_domain"):
        load_bundle("predict", models_dir=tmp_path)


def test_load_bundle_metadata_missing_model_version(tmp_path: Path):
    """Metadata sans la clé `model_version` : `ValueError` explicite."""
    shutil.copy(MODELS_DIR / "predict_model.joblib", tmp_path / "predict_model.joblib")
    metadata = json.loads((MODELS_DIR / "predict_model_metadata.json").read_text(encoding="utf-8"))
    metadata.pop("model_version", None)
    (tmp_path / "predict_model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model_version"):
        load_bundle("predict", models_dir=tmp_path)
