"""Tests du module `agritech.serving` : chargement et inférence des bundles servis par l'API.

Les cas d'erreur (fichier absent, metadata incomplète) sont testés sur un
dossier temporaire, sans jamais modifier les artefacts versionnés dans
`models/`.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agritech.config import PATHS
from agritech.predict_config import PREDICT_MODEL_VERSION
from agritech.recommend_config import RECOMMEND_CROPS
from agritech.serving import (
    MODEL_RAINFALL_FLOAT32_CEILING,
    PREDICT_PUBLIC_TO_MODEL,
    RECOMMEND_TARGET_YEAR,
    Bundle,
    RecommendContext,
    _assemble_candidates,
    _country_defaults,
    _effective_conditions,
    check_recommend_training_domain,
    check_training_domain,
    load_bundle,
    load_recommend_context,
    predict,
    public_training_domain,
    public_training_domain_recommend,
    recommend,
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
    assert bundle.metadata["model_version"] == PREDICT_MODEL_VERSION

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


# ===========================================================================
# Serving /recommend
# ===========================================================================

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CONTEXT_MIN_PATH = FIXTURES_DIR / "recommend_context_min.json"


# --- Fixtures partagées ---


@pytest.fixture(scope="module")
def bundle_recommend() -> Bundle:
    """Bundle réel /recommend chargé une fois pour le module."""
    return load_bundle("recommend")


@pytest.fixture(scope="module")
def context_min() -> RecommendContext:
    """Contexte pays synthétique in-domain (3 pays : AAA, BBB, CCC)."""
    return load_recommend_context(CONTEXT_MIN_PATH)


def _entry_aaa(context: RecommendContext) -> dict:
    return context.countries["AAA"]


# --- load_recommend_context ---


def test_load_recommend_context_reads_fixture(context_min: RecommendContext):
    """La fixture est bien chargée : 3 pays, structure attendue."""
    assert set(context_min.countries) == {"AAA", "BBB", "CCC"}
    assert context_min.countries["AAA"]["country"] == "Alpha"
    assert context_min.countries["AAA"]["rain_mm"] == 500.0
    assert set(context_min.countries["AAA"]["geography"]) == {
        "lat_abs", "geo_x", "geo_y", "geo_z"
    }


def test_load_recommend_context_missing_file_raises(tmp_path: Path):
    """Fichier absent → `FileNotFoundError` explicite."""
    with pytest.raises(FileNotFoundError, match="contexte /recommend introuvable"):
        load_recommend_context(tmp_path / "does_not_exist.json")


def _write_bad_context(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "ctx.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_recommend_context_missing_top_level_key_raises(tmp_path: Path):
    """Clé top-level manquante → `ValueError` nommant la clé."""
    payload = {"history_years": [2011, 2012, 2013], "countries": {}}
    path = _write_bad_context(tmp_path, payload)
    with pytest.raises(ValueError, match="target_year"):
        load_recommend_context(path)


def test_load_recommend_context_wrong_target_year_raises(tmp_path: Path):
    """`target_year` différent de 2014 → `ValueError`."""
    payload = {
        "history_years": [2011, 2012, 2013],
        "target_year": 2013,
        "countries": {},
    }
    path = _write_bad_context(tmp_path, payload)
    with pytest.raises(ValueError, match="target_year"):
        load_recommend_context(path)


def test_load_recommend_context_missing_country_key_raises(tmp_path: Path):
    """Une clé pays manquante → `ValueError` nommant le pays et la clé."""
    payload = {
        "history_years": [2011, 2012, 2013],
        "target_year": RECOMMEND_TARGET_YEAR,
        "countries": {
            "AAA": {
                "country": "Alpha",
                "rain_mm": 500.0,
                # geography absent
                "history_2011_2013": [],
                "observed_crops": ["Wheat"],
            }
        },
    }
    path = _write_bad_context(tmp_path, payload)
    with pytest.raises(ValueError, match="AAA.*geography"):
        load_recommend_context(path)


def test_load_recommend_context_missing_geography_key_raises(tmp_path: Path):
    """Une feature géographique manquante → `ValueError`."""
    payload = {
        "history_years": [2011, 2012, 2013],
        "target_year": RECOMMEND_TARGET_YEAR,
        "countries": {
            "AAA": {
                "country": "Alpha",
                "rain_mm": 500.0,
                "geography": {"lat_abs": 30.0, "geo_x": 0.7, "geo_y": 0.2},  # geo_z absent
                "history_2011_2013": [],
                "observed_crops": ["Wheat"],
            }
        },
    }
    path = _write_bad_context(tmp_path, payload)
    with pytest.raises(ValueError, match="geo_z"):
        load_recommend_context(path)


def test_load_recommend_context_country_entries_sorted_by_country_name(
    context_min: RecommendContext,
):
    """`country_entries` est trié par nom de pays."""
    names = [name for _, name in context_min.country_entries]
    assert names == ["Alpha", "Bravo", "Charlie"]


def test_load_recommend_context_builds_observed_couples(
    context_min: RecommendContext,
):
    """`observed_couples` reprend exactement les couples (iso3, crop) de la fixture."""
    assert ("AAA", "Wheat") in context_min.observed_couples
    assert ("AAA", "Cassava") not in context_min.observed_couples  # AAA n'observe pas Cassava
    assert ("CCC", "Yams") in context_min.observed_couples
    # CCC observe les 10 cultures
    assert sum(1 for iso3, _ in context_min.observed_couples if iso3 == "CCC") == 10


# --- _country_defaults et _effective_conditions ---


def test_country_defaults_from_history_means(context_min: RecommendContext):
    """Defaults = moyennes sur l'historique 2011-2013 + `rain_mm` du pays."""
    defaults = _country_defaults(_entry_aaa(context_min))
    assert defaults == {
        "average_temperature_celsius": 16.0,      # mean(15, 16, 17)
        "annual_rainfall_mm": 500.0,               # constante pays
        "average_annual_pesticides_tons": 1050.0,  # mean(1000, 1050, 1100)
    }


def test_effective_conditions_uses_defaults_when_no_override():
    """`overrides` vide ou `None` : les defaults sont conservés à l'identique."""
    defaults = {
        "average_temperature_celsius": 12.0,
        "annual_rainfall_mm": 700.0,
        "average_annual_pesticides_tons": 2000.0,
    }
    assert _effective_conditions(defaults, {}) == defaults
    assert _effective_conditions(
        defaults,
        {
            "average_temperature_celsius": None,
            "annual_rainfall_mm": None,
            "average_annual_pesticides_tons": None,
        },
    ) == defaults


def test_effective_conditions_applies_partial_override():
    """Un override partiel remplace seulement les champs fournis non-None."""
    defaults = {
        "average_temperature_celsius": 12.0,
        "annual_rainfall_mm": 700.0,
        "average_annual_pesticides_tons": 2000.0,
    }
    overrides = {"annual_rainfall_mm": 1200.0}
    assert _effective_conditions(defaults, overrides) == {
        "average_temperature_celsius": 12.0,   # default
        "annual_rainfall_mm": 1200.0,           # override
        "average_annual_pesticides_tons": 2000.0,  # default
    }


def test_effective_conditions_all_overrides_replace_defaults():
    """Trois overrides fournis : les trois defaults sont remplacés."""
    defaults = {
        "average_temperature_celsius": 12.0,
        "annual_rainfall_mm": 700.0,
        "average_annual_pesticides_tons": 2000.0,
    }
    overrides = {
        "average_temperature_celsius": 25.0,
        "annual_rainfall_mm": 300.0,
        "average_annual_pesticides_tons": 5000.0,
    }
    assert _effective_conditions(defaults, overrides) == overrides


# --- _assemble_candidates ---


def test_assemble_candidates_returns_10_rows_all_crops(context_min: RecommendContext):
    """Assemblage : 10 lignes, une par culture, dans l'ordre de `RECOMMEND_CROPS`."""
    defaults = _country_defaults(_entry_aaa(context_min))
    df = _assemble_candidates(context_min, "AAA", defaults)
    assert len(df) == 10
    assert df["crop"].tolist() == list(RECOMMEND_CROPS)
    # Colonnes attendues par le modèle
    for col in ("year", "temp_hist", "rain_mm", "log_pest_hist",
                "lat_abs", "geo_x", "geo_y", "geo_z", "crop"):
        assert col in df.columns


def test_assemble_candidates_applies_log1p_to_pesticides(context_min: RecommendContext):
    """Le tonnage public → `log_pest_hist` via `log1p` côté service."""
    effective = _country_defaults(_entry_aaa(context_min))
    df = _assemble_candidates(context_min, "AAA", effective)
    expected = math.log1p(effective["average_annual_pesticides_tons"])
    assert df["log_pest_hist"].iloc[0] == pytest.approx(expected)
    # Les 10 lignes partagent la même valeur (broadcast)
    assert (df["log_pest_hist"] == df["log_pest_hist"].iloc[0]).all()


def test_assemble_candidates_uses_geography_from_context(context_min: RecommendContext):
    """La géographie vient du contexte, pas d'un recalcul GeoJSON."""
    effective = _country_defaults(_entry_aaa(context_min))
    df = _assemble_candidates(context_min, "AAA", effective)
    geo = context_min.countries["AAA"]["geography"]
    assert (df["lat_abs"] == geo["lat_abs"]).all()
    assert (df["geo_x"] == geo["geo_x"]).all()
    assert (df["geo_y"] == geo["geo_y"]).all()
    assert (df["geo_z"] == geo["geo_z"]).all()


# --- Plafond technique float32 de la pluie ---


def _max_learned_rain_threshold(bundle: Bundle) -> float:
    """Plus grand seuil sur `rain_mm` parmi tous les arbres du modèle /recommend."""
    preprocess, forest = bundle.pipeline.steps[0][1], bundle.pipeline.steps[-1][1]
    rain_index = list(preprocess.get_feature_names_out()).index("numeric__rain_mm")
    thresholds = np.concatenate(
        [tree.tree_.threshold[tree.tree_.feature == rain_index] for tree in forest.estimators_]
    )
    return float(thresholds.max())


def test_rainfall_ceiling_survives_float32_conversion():
    """Le plafond converti en float32 reste fini et inchangé (jamais `inf`)."""
    as_float32 = np.float32(MODEL_RAINFALL_FLOAT32_CEILING)

    assert np.isfinite(as_float32)
    assert as_float32 == np.finfo(np.float32).max
    assert float(as_float32) == MODEL_RAINFALL_FLOAT32_CEILING


def test_assemble_candidates_caps_only_the_rainfall_sent_to_the_model(
    context_min: RecommendContext,
):
    """Pluie énorme : plafonnée dans les lignes du modèle, intacte dans `effective`.

    Une pluie normale n'est pas touchée par le plafond.
    """
    effective = _country_defaults(_entry_aaa(context_min)) | {"annual_rainfall_mm": 1e300}

    df = _assemble_candidates(context_min, "AAA", effective)

    assert (df["rain_mm"] == MODEL_RAINFALL_FLOAT32_CEILING).all()
    assert effective["annual_rainfall_mm"] == 1e300

    normal = _country_defaults(_entry_aaa(context_min))
    df_normal = _assemble_candidates(context_min, "AAA", normal)
    assert (df_normal["rain_mm"] == normal["annual_rainfall_mm"]).all()


def test_recommend_huge_rainfall_behaves_like_any_rainfall_beyond_learned_thresholds(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Pluie 1e300 : mêmes classement et rendements qu'au-delà de tous les seuils appris.

    Les arbres ne comparent la pluie qu'à leurs seuils appris : toute valeur
    supérieure au plus grand seuil donne la même prédiction. Le plafond technique
    est lui-même au-delà de ces seuils, il ne change donc pas le résultat.
    """
    max_threshold = _max_learned_rain_threshold(bundle_recommend)
    assert max_threshold < MODEL_RAINFALL_FLOAT32_CEILING

    def ranking(rainfall_mm: float) -> list[tuple[str, float]]:
        response = recommend(
            bundle_recommend, context_min, "AAA", conditions={"annual_rainfall_mm": rainfall_mm}
        )
        return [
            (item["crop"], item["predicted_yield_tons_per_hectare"])
            for item in response["recommendations"]
        ]

    huge = ranking(1e300)
    assert huge == ranking(MODEL_RAINFALL_FLOAT32_CEILING)
    assert huge == ranking(max_threshold + 1.0)


# --- public_training_domain_recommend ---


def test_public_training_domain_recommend_uses_expm1_on_pesticides(
    bundle_recommend: Bundle,
):
    """Les bornes exposées sont en unités publiques : °C, mm, t (via expm1 sur log_pest_hist)."""
    public = public_training_domain_recommend(bundle_recommend)
    assert set(public) == {
        "average_temperature_celsius",
        "annual_rainfall_mm",
        "average_annual_pesticides_tons",
    }
    assert public["average_temperature_celsius"]["unit"] == "°C"
    assert public["annual_rainfall_mm"]["unit"] == "mm"
    assert public["average_annual_pesticides_tons"]["unit"] == "t"

    # Conversion inverse alignée sur le metadata versionné
    log_bounds = bundle_recommend.metadata["training_domain"]["log_pest_hist"]
    tons = public["average_annual_pesticides_tons"]
    assert tons["min"] == pytest.approx(math.expm1(log_bounds["min"]))
    assert tons["max"] == pytest.approx(math.expm1(log_bounds["max"]))


def test_public_training_domain_recommend_raises_on_unmapped_feature(
    bundle_recommend: Bundle,
):
    """Une feature de `training_domain` sans mapping public lève `ValueError`."""
    bogus = Bundle(
        name=bundle_recommend.name,
        pipeline=bundle_recommend.pipeline,
        metadata={
            **bundle_recommend.metadata,
            "training_domain": {
                **bundle_recommend.metadata["training_domain"],
                "unknown_feature": {"min": 0.0, "max": 1.0, "unit": ""},
            },
        },
    )
    with pytest.raises(ValueError, match="unknown_feature"):
        public_training_domain_recommend(bogus)


# --- check_recommend_training_domain ---


# Domaine modèle synthétique, calqué sur l'ordre réel (temp, rain, log_pest).
_TRAINING_DOMAIN_FAKE = {
    "temp_hist":     {"min": 2.5,  "max": 30.0,  "unit": "°C"},
    "rain_mm":       {"min": 50.0, "max": 3000.0, "unit": "mm"},
    "log_pest_hist": {"min": 0.0,  "max": 14.0,  "unit": "log(t+1)"},
}


def _in_domain_conditions() -> dict[str, float]:
    return {
        "average_temperature_celsius": 15.0,
        "annual_rainfall_mm": 500.0,
        "average_annual_pesticides_tons": 1000.0,  # log1p ≈ 6.9, dans [0, 14]
    }


def test_check_recommend_training_domain_all_in_domain_no_notes():
    """Toutes les conditions publiques dans les bornes → `(False, [])`."""
    out, notes = check_recommend_training_domain(
        _TRAINING_DOMAIN_FAKE, _in_domain_conditions()
    )
    assert out is False
    assert notes == []


def test_check_recommend_training_domain_temperature_out_note():
    """Température seule hors bornes → une note pour la température."""
    conditions = _in_domain_conditions() | {"average_temperature_celsius": 40.0}
    out, notes = check_recommend_training_domain(_TRAINING_DOMAIN_FAKE, conditions)
    assert out is True
    assert notes == ["average_temperature_celsius is out of training domain"]


def test_check_recommend_training_domain_pesticides_out_via_public_units():
    """Pesticides en tonnes comparés aux bornes converties par `expm1`."""
    # expm1(14) ≈ 1_202_603 t : au-dessus → OOD
    conditions = _in_domain_conditions() | {
        "average_annual_pesticides_tons": 2_000_000.0
    }
    out, notes = check_recommend_training_domain(_TRAINING_DOMAIN_FAKE, conditions)
    assert out is True
    assert notes == ["average_annual_pesticides_tons is out of training domain"]


def test_check_recommend_training_domain_multiple_out_deterministic_order():
    """Ordre des notes = ordre des clés du `training_domain`."""
    conditions = _in_domain_conditions() | {
        "average_temperature_celsius": 40.0,
        "average_annual_pesticides_tons": 2_000_000.0,
    }
    out, notes = check_recommend_training_domain(_TRAINING_DOMAIN_FAKE, conditions)
    assert out is True
    # temp_hist est déclaré avant log_pest_hist dans _TRAINING_DOMAIN_FAKE
    assert notes == [
        "average_temperature_celsius is out of training domain",
        "average_annual_pesticides_tons is out of training domain",
    ]


# --- recommend : orchestration complète avec vrai bundle + fixture ---


def test_recommend_returns_expected_response_shape(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """La réponse porte toutes les clés attendues par la couche API."""
    response = recommend(bundle_recommend, context_min, "AAA")
    assert set(response) == {
        "iso3", "country", "year", "unit", "model_version",
        "recommendations", "context", "out_of_training_domain", "notes",
    }
    assert response["iso3"] == "AAA"
    assert response["country"] == "Alpha"
    assert response["year"] == RECOMMEND_TARGET_YEAR
    assert response["unit"] == "t/ha"
    assert response["model_version"] == bundle_recommend.metadata["model_version"]


def test_recommend_returns_exactly_10_recommendations(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Toujours exactement 10 recommandations, une par culture du modèle."""
    response = recommend(bundle_recommend, context_min, "AAA")
    assert len(response["recommendations"]) == 10
    crops = {item["crop"] for item in response["recommendations"]}
    assert crops == set(RECOMMEND_CROPS)


def test_recommend_recommendations_sorted_desc_by_predicted_yield(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Le classement est trié décroissant sur `predicted_yield_tons_per_hectare`."""
    response = recommend(bundle_recommend, context_min, "AAA")
    yields = [item["predicted_yield_tons_per_hectare"] for item in response["recommendations"]]
    assert yields == sorted(yields, reverse=True)


def test_recommend_ranks_are_1_to_10(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Les rangs vont de 1 à 10 dans l'ordre du classement."""
    response = recommend(bundle_recommend, context_min, "AAA")
    ranks = [item["rank"] for item in response["recommendations"]]
    assert ranks == list(range(1, 11))


def test_recommend_defaults_match_history_means_of_fixture_country(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """`context.country_defaults` reflète les moyennes 2011-2013 du contexte."""
    response = recommend(bundle_recommend, context_min, "AAA")
    assert response["context"]["country_defaults"] == {
        "average_temperature_celsius": 16.0,
        "annual_rainfall_mm": 500.0,
        "average_annual_pesticides_tons": 1050.0,
    }


def test_recommend_override_replaces_only_that_field(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Un override partiel remplace ce champ dans `effective_conditions` et rien d'autre."""
    response = recommend(
        bundle_recommend, context_min, "AAA",
        conditions={"annual_rainfall_mm": 1200.0},
    )
    assert response["context"]["country_defaults"]["annual_rainfall_mm"] == 500.0
    assert response["context"]["effective_conditions"] == {
        "average_temperature_celsius": 16.0,          # default
        "annual_rainfall_mm": 1200.0,                  # override
        "average_annual_pesticides_tons": 1050.0,      # default
    }


def test_recommend_matches_direct_pipeline_call_on_same_10_rows(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Les rendements retournés sont ceux du pipeline appelé directement sur les mêmes 10 lignes."""
    response = recommend(bundle_recommend, context_min, "AAA")
    # Rebuild the same 10 rows and call pipeline directly
    defaults = _country_defaults(_entry_aaa(context_min))
    direct_df = _assemble_candidates(context_min, "AAA", defaults)
    features = bundle_recommend.metadata["features"]
    direct_yields = bundle_recommend.pipeline.predict(direct_df[features])

    api_by_crop = {item["crop"]: item["predicted_yield_tons_per_hectare"]
                   for item in response["recommendations"]}
    direct_by_crop = dict(zip(direct_df["crop"], direct_yields))
    for crop, direct_yield in direct_by_crop.items():
        assert api_by_crop[crop] == float(direct_yield)


def test_recommend_observed_in_country_reflects_fixture_observed_crops(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """`observed_in_country` correspond exactement à `observed_crops` de la fixture."""
    response = recommend(bundle_recommend, context_min, "AAA")
    observed_aaa = set(context_min.countries["AAA"]["observed_crops"])
    for item in response["recommendations"]:
        assert item["observed_in_country"] is (item["crop"] in observed_aaa)


def test_recommend_observed_in_country_all_true_when_all_crops_observed(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """CCC observe les 10 cultures : tous les `observed_in_country` valent True."""
    response = recommend(bundle_recommend, context_min, "CCC")
    assert all(item["observed_in_country"] for item in response["recommendations"])


def test_recommend_unknown_country_raises_valueerror(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Un `iso3` inconnu du contexte lève `ValueError` explicite (traduite en 422 au lot 5)."""
    with pytest.raises(ValueError, match="unknown country: ZZZ"):
        recommend(bundle_recommend, context_min, "ZZZ")


def test_recommend_default_out_of_domain_sets_flag_and_note(bundle_recommend: Bundle):
    """Contexte synthétique aux pesticides très élevés : default hors domaine → flag + note."""
    # Historique dont la moyenne dépasse expm1(14.394) ≈ 1_783_762 t
    huge_pest = {
        "country": "Extreme",
        "rain_mm": 500.0,
        "geography": {"lat_abs": 40.0, "geo_x": 0.5, "geo_y": 0.5, "geo_z": 0.7},
        "history_2011_2013": [
            {"year": 2011, "avg_temp": 15.0, "pesticides_t": 2_000_000.0},
            {"year": 2012, "avg_temp": 16.0, "pesticides_t": 2_000_000.0},
            {"year": 2013, "avg_temp": 17.0, "pesticides_t": 2_000_000.0},
        ],
        "observed_crops": ["Wheat"],
    }
    ctx = RecommendContext(
        countries={"XXX": huge_pest},
        observed_couples=frozenset({("XXX", "Wheat")}),
        country_entries=[("XXX", "Extreme")],
    )
    response = recommend(bundle_recommend, ctx, "XXX")
    assert response["out_of_training_domain"] is True
    assert response["notes"] == ["average_annual_pesticides_tons is out of training domain"]


def test_recommend_user_override_out_of_domain_sets_flag_and_note(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Override utilisateur volontairement hors bornes apprises → flag + note."""
    response = recommend(
        bundle_recommend, context_min, "AAA",
        conditions={"average_temperature_celsius": 50.0},   # physique valide, hors [2.57, 30.25]
    )
    assert response["out_of_training_domain"] is True
    assert "average_temperature_celsius is out of training domain" in response["notes"]


def test_recommend_in_domain_defaults_no_flag_no_notes(
    bundle_recommend: Bundle, context_min: RecommendContext
):
    """Fixture in-domain sans override : ni flag ni note."""
    response = recommend(bundle_recommend, context_min, "BBB")
    assert response["out_of_training_domain"] is False
    assert response["notes"] == []
