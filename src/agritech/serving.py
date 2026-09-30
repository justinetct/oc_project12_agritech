"""Chargement et service des modèles servis par l'API (`/predict`, `/recommend`).

Module commun aux deux services. Il ne connaît pas HTTP : il lit les artefacts
sauvegardés dans `models/`, expose le pipeline chargé, et fournit la logique
d'inférence réutilisable par n'importe quelle couche (API, script, notebook).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from agritech.config import PATHS
from agritech.recommend_config import RECOMMEND_CROPS


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

    Utilisé par `GET /predict/context` pour exposer le domaine d'apprentissage sous des
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


# ---------------------------------------------------------------------------
# Serving /recommend
# ---------------------------------------------------------------------------

# Année cible fixée par le service : la recommandation est toujours 2014, à partir
# de l'historique 2011-2013 connu. Pas de champ `year` dans la requête publique.
RECOMMEND_TARGET_YEAR = 2014

# Mapping public → modèle pour /recommend. Chaque entrée porte le nom de feature
# interne du modèle ET la transformation à appliquer avant de nourrir le pipeline.
# Les pesticides sont exposés en tonnes côté API ; le modèle voit `log1p(tonnes)`.
RECOMMEND_PUBLIC_TO_MODEL: dict[str, tuple[str, Callable[[float], float]]] = {
    "average_temperature_celsius":    ("temp_hist",     lambda x: x),
    "annual_rainfall_mm":             ("rain_mm",       lambda x: x),
    "average_annual_pesticides_tons": ("log_pest_hist", math.log1p),
}

# Conversion inverse pour publier `training_domain` en unités publiques via
# `GET /recommend/context`. `expm1` remet `log_pest_hist` en tonnes.
RECOMMEND_MODEL_TO_PUBLIC: dict[str, tuple[str, Callable[[float], float]]] = {
    "temp_hist":     ("average_temperature_celsius",    lambda x: x),
    "rain_mm":       ("annual_rainfall_mm",             lambda x: x),
    "log_pest_hist": ("average_annual_pesticides_tons", math.expm1),
}

# Unités publiques exposées par `/recommend/context`, indexées par nom public.
RECOMMEND_PUBLIC_UNITS: dict[str, str] = {
    "average_temperature_celsius":    "°C",
    "annual_rainfall_mm":             "mm",
    "average_annual_pesticides_tons": "t",
}


@dataclass(frozen=True, slots=True)
class RecommendContext:
    """État runtime du contexte pays de /recommend, chargé une fois au démarrage.

    Immutable : lu par le service à chaque requête, jamais modifié.

    - `countries` : mapping `iso3→ entrée du JSON versionné` (tel quel).
    - `observed_couples` : ensemble des couples `(iso3, crop)` réellement observés
      dans l'historique, dérivé de `observed_crops` de chaque pays. Sert au flag
      `observed_in_country` retourné par chaque recommandation.
    - `country_entries` : liste `[(iso3, country_name)]` triée par nom de pays,
      prête pour `/recommend/context`.
    """

    countries: dict[str, dict]
    observed_couples: frozenset[tuple[str, str]]
    country_entries: list[tuple[str, str]]


def _validate_recommend_context_payload(payload: dict) -> None:
    """Vérifie les clés obligatoires du contexte pays chargé depuis le JSON.

    Lève `ValueError` explicite plutôt qu'un `KeyError` opaque plus loin dans le
    chemin d'inférence.
    """
    for key in ("history_years", "target_year", "countries"):
        if key not in payload:
            raise ValueError(f"contexte /recommend : clé manquante « {key} »")
    if payload["target_year"] != RECOMMEND_TARGET_YEAR:
        raise ValueError(
            f"contexte /recommend : target_year={payload['target_year']} attendu {RECOMMEND_TARGET_YEAR}"
        )
    for iso3, entry in payload["countries"].items():
        for key in ("country", "rain_mm", "geography", "history_2011_2013", "observed_crops"):
            if key not in entry:
                raise ValueError(f"contexte /recommend : {iso3} clé manquante « {key} »")
        for key in ("lat_abs", "geo_x", "geo_y", "geo_z"):
            if key not in entry["geography"]:
                raise ValueError(
                    f"contexte /recommend : {iso3} géographie clé manquante « {key} »"
                )


def load_recommend_context(path: Path) -> RecommendContext:
    """Charge et valide `models/recommend_context.json`.

    Lève `FileNotFoundError` si le fichier est absent, `ValueError` si une clé
    obligatoire manque. Le contexte retourné est immuable et prêt à l'usage :
    l'ensemble `observed_couples` et la liste triée `country_entries` sont
    calculés une fois pour toutes.
    """
    if not path.is_file():
        raise FileNotFoundError(f"contexte /recommend introuvable : {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_recommend_context_payload(payload)

    countries: dict[str, dict] = payload["countries"]
    observed_couples = frozenset(
        (iso3, crop)
        for iso3, entry in countries.items()
        for crop in entry["observed_crops"]
    )
    country_entries = sorted(
        [(iso3, entry["country"]) for iso3, entry in countries.items()],
        key=lambda pair: pair[1],
    )
    return RecommendContext(
        countries=countries,
        observed_couples=observed_couples,
        country_entries=country_entries,
    )


def _country_defaults(entry: dict) -> dict[str, float]:
    """Defaults 2014 depuis l'historique 2011-2013 du pays.

    Renvoie les trois conditions **en unités publiques** : température moyenne
    des 3 années précédentes (°C), pluie annuelle du pays (mm), tonnage annuel
    moyen de pesticides (t). La transformation `log1p` n'intervient qu'ensuite,
    au moment d'assembler les 10 lignes candidates.
    """
    temps = [float(item["avg_temp"]) for item in entry["history_2011_2013"]]
    pests = [float(item["pesticides_t"]) for item in entry["history_2011_2013"]]
    return {
        "average_temperature_celsius":    float(np.mean(temps)),
        "annual_rainfall_mm":             float(entry["rain_mm"]),
        "average_annual_pesticides_tons": float(np.mean(pests)),
    }


def _effective_conditions(
    defaults: dict[str, float], overrides: dict[str, float | None]
) -> dict[str, float]:
    """Fusionne defaults ⊕ overrides.

    Un override manquant OU `None` laisse le default en place. Les clés inconnues
    dans `overrides` sont ignorées : le contrat public est fixé côté Pydantic,
    cette fonction n'a pas à en refaire la validation.
    """
    return {
        name: (overrides[name] if overrides.get(name) is not None else defaults[name])
        for name in defaults
    }


def _assemble_candidates(
    context: RecommendContext, iso3: str, effective: dict[str, float]
) -> pd.DataFrame:
    """Construit les 10 lignes candidates prêtes à être passées au pipeline.

    Les 3 conditions numériques viennent des `effective_conditions` (defaults ⊕
    overrides). La géographie vient du contexte : les 4 features `lat_abs`,
    `geo_x`, `geo_y`, `geo_z` sont pré-calculées par le script de génération, ce
    qui évite toute dépendance du serving au GeoJSON.
    """
    entry = context.countries[iso3]
    geo = entry["geography"]
    base_row = {
        "year": RECOMMEND_TARGET_YEAR,
        "temp_hist":     effective["average_temperature_celsius"],
        "rain_mm":       effective["annual_rainfall_mm"],
        "log_pest_hist": math.log1p(effective["average_annual_pesticides_tons"]),
        "lat_abs": geo["lat_abs"],
        "geo_x":   geo["geo_x"],
        "geo_y":   geo["geo_y"],
        "geo_z":   geo["geo_z"],
    }
    return pd.DataFrame([{**base_row, "crop": crop} for crop in RECOMMEND_CROPS])


def check_recommend_training_domain(
    training_domain: dict[str, dict[str, float]],
    effective_public: dict[str, float],
) -> tuple[bool, list[str]]:
    """Compare les conditions effectives aux bornes apprises du modèle.

    Le `training_domain` est stocké en unités internes ; les bornes sont
    ramenées en unités publiques via `RECOMMEND_MODEL_TO_PUBLIC` (notamment
    `expm1` pour les pesticides). Itère dans l'ordre du metadata pour un
    résultat déterministe.
    """
    notes: list[str] = []
    for model_name, bounds in training_domain.items():
        mapping = RECOMMEND_MODEL_TO_PUBLIC.get(model_name)
        if mapping is None:
            continue
        public_name, to_public = mapping
        if public_name not in effective_public:
            continue
        lo_public, hi_public = to_public(bounds["min"]), to_public(bounds["max"])
        value = effective_public[public_name]
        if value < lo_public or value > hi_public:
            notes.append(f"{public_name} is out of training domain")
    return (bool(notes), notes)


def public_training_domain_recommend(bundle: Bundle) -> dict[str, dict[str, float | str]]:
    """Reformate `training_domain` en unités publiques snake_case.

    Utilisé par `GET /recommend/context`. Une feature du `training_domain` sans
    correspondance publique révèle une incohérence entre metadata et contrat
    API : on lève `ValueError` plutôt que d'exposer silencieusement le nom
    interne du modèle.
    """
    result: dict[str, dict[str, float | str]] = {}
    for model_name, bounds in bundle.metadata["training_domain"].items():
        mapping = RECOMMEND_MODEL_TO_PUBLIC.get(model_name)
        if mapping is None:
            raise ValueError(
                f"training_domain contient la feature {model_name!r} sans correspondance "
                "publique : incohérence entre les metadata et le contrat API."
            )
        public_name, to_public = mapping
        result[public_name] = {
            "min": float(to_public(bounds["min"])),
            "max": float(to_public(bounds["max"])),
            "unit": RECOMMEND_PUBLIC_UNITS[public_name],
        }
    return result


def recommend(
    bundle: Bundle,
    context: RecommendContext,
    iso3: str,
    conditions: dict[str, float | None] | None = None,
) -> dict:
    """Chaîne d'inférence /recommend : assemble les 10 candidats, prédit, classe, compose la réponse.

    `iso3` doit être présent dans `context.countries` : sinon `ValueError` que la
    couche API (lot 5) traduit en 422 `unknown_country`.

    `conditions` est un dict Python à trois champs publics optionnels ; un champ
    absent ou `None` prend la valeur préremplie du pays. Le service n'a aucune
    dépendance Pydantic ; la validation stricte du payload est faite côté router.

    Retourne un dict directement passable à `RecommendResponse(...)` (lot 4).
    """
    if iso3 not in context.countries:
        raise ValueError(f"unknown country: {iso3}")

    entry = context.countries[iso3]
    defaults = _country_defaults(entry)
    effective = _effective_conditions(defaults, conditions or {})
    candidates = _assemble_candidates(context, iso3, effective)

    features = bundle.metadata["features"]
    missing = [name for name in features if name not in candidates.columns]
    if missing:
        raise RuntimeError(
            f"features attendues du modèle absentes des candidats : {missing}"
        )
    yields = bundle.pipeline.predict(candidates[features])

    order = np.argsort(-yields, kind="stable")
    ranked_crops = candidates["crop"].to_numpy()[order]
    ranked_yields = yields[order]
    recommendations = [
        {
            "rank": rank,
            "crop": str(crop),
            "predicted_yield_tons_per_hectare": float(yield_value),
            "observed_in_country": (iso3, str(crop)) in context.observed_couples,
        }
        for rank, (crop, yield_value) in enumerate(
            zip(ranked_crops, ranked_yields), start=1
        )
    ]

    out_of_domain, notes = check_recommend_training_domain(
        bundle.metadata["training_domain"], effective
    )

    return {
        "iso3": iso3,
        "country": entry["country"],
        "year": RECOMMEND_TARGET_YEAR,
        "unit": "t/ha",
        "model_version": bundle.metadata["model_version"],
        "recommendations": recommendations,
        "context": {
            "country_defaults": defaults,
            "effective_conditions": effective,
        },
        "out_of_training_domain": out_of_domain,
        "notes": notes,
    }
