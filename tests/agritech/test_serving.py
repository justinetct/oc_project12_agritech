"""Tests du module `agritech.serving` : chargement et inférence des bundles servis par l'API.

Les cas d'erreur (fichier absent, metadata incomplète) sont testés sur un
dossier temporaire, sans jamais modifier les artefacts versionnés dans
`models/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from agritech.config import PATHS
from agritech.serving import (
    PREDICT_PUBLIC_TO_MODEL,
    Bundle,
    check_training_domain,
    load_bundle,
    predict,
    public_training_domain,
)


MODELS_DIR = PATHS.root / "models"


# Payload de reference : toutes les valeurs sont dans les bornes physiques
# ET dans le domaine d'apprentissage du modele /predict.
VALID_VALUES = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}


# Copie compacte du domaine d'apprentissage tel que serialise en JSON. Reproduit la forme
# reelle : cles = noms modele, valeurs = dict avec min/max/unit. Utilise dans les tests
# unitaires de `check_training_domain` sans dependre du fichier metadata reel.
TRAINING_DOMAIN_FIXTURE = {
    "Rainfall_mm": {"min": 100, "max": 1000, "unit": "mm"},
    "Temperature_Celsius": {"min": 15, "max": 40, "unit": "°C"},
}


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


# --- Mapping public -> modele ---


def test_public_to_model_mapping_matches_metadata_features():
    """`PREDICT_PUBLIC_TO_MODEL` couvre exactement les 4 features du modèle final."""
    bundle = load_bundle("predict")
    assert set(PREDICT_PUBLIC_TO_MODEL.values()) == set(bundle.metadata["features"])


# --- check_training_domain : logique du domaine, sans HTTP ---


def test_check_training_domain_all_in_domain():
    """Toutes les valeurs sont dans les bornes du modèle → `(False, [])`."""
    out_of_domain, notes = check_training_domain(
        TRAINING_DOMAIN_FIXTURE, VALID_VALUES, PREDICT_PUBLIC_TO_MODEL
    )
    assert out_of_domain is False
    assert notes == []


def test_check_training_domain_temperature_only_out():
    """`temperature_celsius = 5` seule hors domaine → une note pour la température."""
    values = VALID_VALUES | {"temperature_celsius": 5.0}
    out_of_domain, notes = check_training_domain(
        TRAINING_DOMAIN_FIXTURE, values, PREDICT_PUBLIC_TO_MODEL
    )
    assert out_of_domain is True
    assert notes == ["temperature_celsius is out of training domain"]


def test_check_training_domain_rainfall_only_out():
    """`rainfall_mm = 50` seule hors domaine → une note pour la pluie."""
    values = VALID_VALUES | {"rainfall_mm": 50.0}
    out_of_domain, notes = check_training_domain(
        TRAINING_DOMAIN_FIXTURE, values, PREDICT_PUBLIC_TO_MODEL
    )
    assert out_of_domain is True
    assert notes == ["rainfall_mm is out of training domain"]


def test_check_training_domain_both_out_deterministic_order():
    """Les deux variables hors domaine → deux notes, dans l'ordre du metadata."""
    values = VALID_VALUES | {"rainfall_mm": 50.0, "temperature_celsius": 5.0}
    out_of_domain, notes = check_training_domain(
        TRAINING_DOMAIN_FIXTURE, values, PREDICT_PUBLIC_TO_MODEL
    )
    assert out_of_domain is True
    # `TRAINING_DOMAIN_FIXTURE` liste `Rainfall_mm` avant `Temperature_Celsius` :
    # l'ordre des notes suit celui du metadata.
    assert notes == [
        "rainfall_mm is out of training domain",
        "temperature_celsius is out of training domain",
    ]


# --- predict : orchestration complete sur le modele reel ---


def test_predict_returns_expected_shape_and_types():
    """`predict` renvoie un dict avec les 5 clés du contrat et les bons types."""
    bundle = load_bundle("predict")
    result = predict(bundle, VALID_VALUES)

    assert set(result) == {
        "yield_tons_per_hectare",
        "unit",
        "model_version",
        "out_of_training_domain",
        "notes",
    }
    assert isinstance(result["yield_tons_per_hectare"], float)
    assert result["unit"] == "t/ha"
    assert result["model_version"] == bundle.metadata["model_version"]
    assert isinstance(result["out_of_training_domain"], bool)
    assert isinstance(result["notes"], list)


def test_predict_matches_direct_pipeline_call():
    """`predict(...)` donne strictement la même valeur qu'un appel direct au pipeline."""
    bundle = load_bundle("predict")
    result = predict(bundle, VALID_VALUES)

    row = {PREDICT_PUBLIC_TO_MODEL[public]: value for public, value in VALID_VALUES.items()}
    direct = bundle.pipeline.predict(pd.DataFrame([row]))[0]

    assert result["yield_tons_per_hectare"] == float(direct)


# --- public_training_domain : sérialisation publique du domaine ---


def test_public_training_domain_uses_snake_case_from_metadata():
    """Le domaine renvoyé est reformaté avec les noms publics snake_case."""
    bundle = load_bundle("predict")
    reformatted = public_training_domain(bundle)

    assert set(reformatted) == {"rainfall_mm", "temperature_celsius"}
    assert reformatted["rainfall_mm"] == {"min": 100, "max": 1000, "unit": "mm"}
    assert reformatted["temperature_celsius"] == {"min": 15, "max": 40, "unit": "°C"}


def test_public_training_domain_raises_on_unmapped_feature():
    """Une feature de `training_domain` sans correspondance publique lève `ValueError`."""
    bundle = load_bundle("predict")
    bogus_bundle = Bundle(
        name=bundle.name,
        pipeline=bundle.pipeline,
        metadata={
            **bundle.metadata,
            "training_domain": {
                **bundle.metadata["training_domain"],
                "Unknown_Feature": {"min": 0, "max": 1, "unit": ""},
            },
        },
    )

    with pytest.raises(ValueError, match="Unknown_Feature"):
        public_training_domain(bogus_bundle)
