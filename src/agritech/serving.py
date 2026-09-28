"""Chargement et service des modèles servis par l'API (`/predict`, `/recommend`).

Module commun aux deux services. Il ne connaît pas HTTP : il lit les artefacts
sauvegardés dans `models/`, expose le pipeline chargé, et fournit la logique
d'inférence réutilisable par n'importe quelle couche (API, script, notebook).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

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


# Mapping des noms publics snake_case (contrat API) vers les noms de features attendus par
# le pipeline scikit-learn sauvegardé. Source unique : toute inversion est calculée depuis
# ce dictionnaire, jamais recopiée. Le test `test_public_to_model_mapping_matches_metadata_features`
# vérifie la cohérence avec `metadata["features"]`.
PREDICT_PUBLIC_TO_MODEL: dict[str, str] = {
    "rainfall_mm": "Rainfall_mm",
    "temperature_celsius": "Temperature_Celsius",
    "fertilizer_used": "Fertilizer_Used",
    "irrigation_used": "Irrigation_Used",
}


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


def _public_by_model(public_to_model: dict[str, str]) -> dict[str, str]:
    """Inverse un mapping public → modèle, sans recopie manuelle."""
    return {model: public for public, model in public_to_model.items()}


def check_training_domain(
    training_domain: dict[str, dict[str, float]],
    values_by_public_name: dict[str, float | bool],
    public_to_model: dict[str, str],
) -> tuple[bool, list[str]]:
    """Compare les valeurs reçues aux bornes numériques du domaine d'apprentissage.

    Itère sur `training_domain` dans son ordre d'insertion pour un résultat déterministe.
    Une variable dont le nom modèle ne figure pas dans `public_to_model`, ou dont le nom
    public n'est pas fourni, est ignorée : la fonction ne concerne que les variables
    numériques dont on peut effectivement vérifier les bornes.

    Retourne `(True, notes)` dès qu'au moins une variable est hors bornes, sinon
    `(False, [])`. Chaque note utilise le nom public sous la forme
    `"<nom_public> is out of training domain"`.
    """
    public_by_model = _public_by_model(public_to_model)
    notes: list[str] = []
    for model_name, bounds in training_domain.items():
        public_name = public_by_model.get(model_name)
        if public_name is None or public_name not in values_by_public_name:
            continue
        value = values_by_public_name[public_name]
        if value < bounds["min"] or value > bounds["max"]:
            notes.append(f"{public_name} is out of training domain")
    return (bool(notes), notes)


def predict(bundle: Bundle, values_by_public_name: dict[str, float | bool]) -> dict:
    """Chaîne d'inférence `/predict` : construit le DataFrame, appelle le pipeline, compose la réponse.

    `values_by_public_name` utilise les noms publics snake_case du contrat API. Ils sont
    remappés vers les noms attendus par le pipeline via `PREDICT_PUBLIC_TO_MODEL`. Le
    résultat est un dict directement prêt à être passé à `PredictResponse(...)`.
    """
    row = {PREDICT_PUBLIC_TO_MODEL[public]: value for public, value in values_by_public_name.items()}
    X = pd.DataFrame([row])
    y = bundle.pipeline.predict(X)
    out_of_domain, notes = check_training_domain(
        bundle.metadata["training_domain"],
        values_by_public_name,
        PREDICT_PUBLIC_TO_MODEL,
    )
    return {
        "yield_tons_per_hectare": float(y[0]),
        "unit": "t/ha",
        "model_version": bundle.metadata["model_version"],
        "out_of_training_domain": out_of_domain,
        "notes": notes,
    }


def public_training_domain(
    bundle: Bundle,
    public_to_model: dict[str, str] = PREDICT_PUBLIC_TO_MODEL,
) -> dict[str, dict[str, float | str]]:
    """Reformate `bundle.metadata["training_domain"]` avec les noms publics snake_case.

    Utilisé par `GET /predict/schema` pour exposer le domaine d'apprentissage sous des
    noms cohérents avec le reste du contrat API. Une feature du `training_domain` sans
    correspondance dans `public_to_model` révèle une incohérence entre les metadata et
    le contrat API : on lève `ValueError` plutôt que d'exposer silencieusement le nom
    interne du modèle.
    """
    public_by_model = _public_by_model(public_to_model)
    result: dict[str, dict[str, float | str]] = {}
    for model_name, bounds in bundle.metadata["training_domain"].items():
        if model_name not in public_by_model:
            raise ValueError(
                f"training_domain contient la feature {model_name!r} sans correspondance "
                "dans public_to_model : incohérence entre les metadata et le contrat API."
            )
        result[public_by_model[model_name]] = dict(bounds)
    return result
