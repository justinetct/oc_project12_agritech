"""Tests HTTP de `POST /recommend` et `GET /recommend/context`.

Chaque test s'exécute dans le cycle de vie normal de l'application, via
`with TestClient(app) as client:` : le lifespan charge bundle et contexte
avant la première requête, comme en production.

Les tests couvrent uniquement le comportement HTTP nouveau, sans dupliquer
les validations Pydantic déjà exhaustivement testées dans
`tests/api/schemas/test_recommend.py`.
"""

from __future__ import annotations

import math
from dataclasses import replace

from fastapi.testclient import TestClient

from agritech.api.core import runtime
from agritech.api.main import app
from agritech.serving import recommend as serving_recommend


VALID_ISO3 = "FRA"
VALID_MINIMAL_PAYLOAD = {"iso3": VALID_ISO3}


# ===========================================================================
# POST /recommend
# ===========================================================================


def test_post_recommend_iso3_only_returns_200_and_10_recommendations():
    """Payload minimal : 200, réponse au contrat, 10 recommandations triées décroissant."""
    with TestClient(app) as client:
        response = client.post("/recommend", json=VALID_MINIMAL_PAYLOAD)

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "iso3", "country", "year", "unit", "model_version",
        "recommendations", "context", "out_of_training_domain", "notes",
    }
    assert body["iso3"] == VALID_ISO3
    assert body["country"] == "France"
    assert body["year"] == 2014
    assert body["unit"] == "t/ha"
    assert isinstance(body["model_version"], str) and body["model_version"]

    recs = body["recommendations"]
    assert len(recs) == 10
    yields = [item["predicted_yield_tons_per_hectare"] for item in recs]
    assert yields == sorted(yields, reverse=True)
    ranks = [item["rank"] for item in recs]
    assert ranks == list(range(1, 11))


def test_post_recommend_response_matches_serving_direct_call():
    """La réponse HTTP est strictement égale à un appel direct à `serving.recommend`."""
    with TestClient(app) as client:
        http_body = client.post("/recommend", json=VALID_MINIMAL_PAYLOAD).json()
        # Snapshot du bundle/contexte chargés par le lifespan
        bundle = runtime.bundle_recommend
        context = runtime.recommend_context
        direct = serving_recommend(bundle, context, VALID_ISO3, {})

    assert http_body == direct


def test_post_recommend_partial_conditions_reflects_mix_in_effective():
    """Override partiel (rain_mm) : `country_defaults` intact, `effective_conditions` reflète le mix."""
    payload = {"iso3": VALID_ISO3, "conditions": {"annual_rainfall_mm": 1234.0}}

    with TestClient(app) as client:
        body = client.post("/recommend", json=payload).json()

    defaults = body["context"]["country_defaults"]
    effective = body["context"]["effective_conditions"]

    assert effective["annual_rainfall_mm"] == 1234.0
    assert effective["average_temperature_celsius"] == defaults["average_temperature_celsius"]
    assert effective["average_annual_pesticides_tons"] == defaults["average_annual_pesticides_tons"]
    # rain_mm surchargé ≠ default du pays
    assert defaults["annual_rainfall_mm"] != 1234.0


def test_post_recommend_unknown_country_returns_422_validation_error():
    """`iso3` syntaxiquement valide mais absent du contexte → 422 `unknown_country` sur `body.iso3`."""
    payload = {"iso3": "ZZZ"}

    with TestClient(app) as client:
        response = client.post("/recommend", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["message"] == "Request payload is invalid."
    assert isinstance(body["details"], list) and len(body["details"]) == 1
    detail = body["details"][0]
    assert detail["field"] == "body.iso3"
    assert detail["type"] == "unknown_country"
    assert "ZZZ" in detail["message"]


def test_post_recommend_iso3_lowercase_returns_422_pydantic():
    """Erreur Pydantic (pattern iso3) → 422 traversant la couche HTTP normalement."""
    payload = {"iso3": "fra"}

    with TestClient(app) as client:
        response = client.post("/recommend", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    detail = body["details"][0]
    assert detail["field"] == "body.iso3"
    assert detail["type"] == "string_pattern_mismatch"


def test_post_recommend_bundle_none_returns_503():
    """`runtime.bundle_recommend = None` → 503 `model_unavailable`."""
    with TestClient(app) as client:
        saved = runtime.bundle_recommend
        runtime.bundle_recommend = None
        try:
            response = client.post("/recommend", json=VALID_MINIMAL_PAYLOAD)
        finally:
            runtime.bundle_recommend = saved

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "model_unavailable"
    assert body["message"] == "Model is unavailable."
    assert body["details"] is None


def test_post_recommend_context_none_returns_503():
    """`runtime.recommend_context = None` → 503 également."""
    with TestClient(app) as client:
        saved = runtime.recommend_context
        runtime.recommend_context = None
        try:
            response = client.post("/recommend", json=VALID_MINIMAL_PAYLOAD)
        finally:
            runtime.recommend_context = saved

    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_post_recommend_pipeline_exception_returns_500():
    """Un pipeline qui lève → 500 `internal_error` sans fuite d'info."""
    secret = "recommend_pipeline_secret_54321"

    class BrokenPipeline:
        def predict(self, X):
            raise RuntimeError(secret)

    with TestClient(app, raise_server_exceptions=False) as client:
        saved = runtime.bundle_recommend
        runtime.bundle_recommend = replace(saved, pipeline=BrokenPipeline())
        try:
            response = client.post("/recommend", json=VALID_MINIMAL_PAYLOAD)
        finally:
            runtime.bundle_recommend = saved

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert body["message"] == "Internal server error."
    assert body["details"] is None
    assert secret not in response.text


# ===========================================================================
# GET /recommend/context
# ===========================================================================


def test_get_recommend_context_returns_200_and_expected_keys():
    """`GET /recommend/context` : 200 + 6 clés attendues + `year=2014`."""
    with TestClient(app) as client:
        response = client.get("/recommend/context")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "year", "target_year_note", "crops", "countries",
        "physical_bounds", "training_domain",
    }
    assert body["year"] == 2014
    assert isinstance(body["target_year_note"], str) and "2013" in body["target_year_note"]


def test_get_recommend_context_crops_come_from_bundle_metadata():
    """Les 10 cultures exposées sont exactement celles du metadata du modèle chargé."""
    with TestClient(app) as client:
        body = client.get("/recommend/context").json()
        expected_crops = runtime.bundle_recommend.metadata["categorical_values"]["crop"]

    assert body["crops"] == expected_crops
    assert len(body["crops"]) == 10


def test_get_recommend_context_countries_sorted_by_country_name():
    """Les 115 pays sont triés par nom (ordre du contexte)."""
    with TestClient(app) as client:
        body = client.get("/recommend/context").json()

    countries = body["countries"]
    assert len(countries) == 115
    names = [entry["country"] for entry in countries]
    assert names == sorted(names)
    for entry in countries:
        assert set(entry) == {"iso3", "country"}


def test_get_recommend_context_physical_bounds_expose_three_conditions():
    """`physical_bounds` : 3 conditions avec noms publics et bornes attendues."""
    with TestClient(app) as client:
        body = client.get("/recommend/context").json()

    bounds = body["physical_bounds"]
    assert set(bounds) == {
        "average_temperature_celsius",
        "annual_rainfall_mm",
        "average_annual_pesticides_tons",
    }
    assert bounds["average_temperature_celsius"] == {"min": -50.0, "max": 60.0, "unit": "°C"}
    assert bounds["annual_rainfall_mm"] == {"min": 0.0, "max": None, "unit": "mm"}
    assert bounds["average_annual_pesticides_tons"] == {"min": 0.0, "max": None, "unit": "t"}


def test_get_recommend_context_training_domain_pesticides_in_public_tons():
    """`training_domain` : bornes en unités publiques (tonnes pour pesticides via `expm1`)."""
    with TestClient(app) as client:
        body = client.get("/recommend/context").json()
        log_bounds = runtime.bundle_recommend.metadata["training_domain"]["log_pest_hist"]

    td = body["training_domain"]
    assert set(td) == {
        "average_temperature_celsius",
        "annual_rainfall_mm",
        "average_annual_pesticides_tons",
    }
    tons = td["average_annual_pesticides_tons"]
    assert tons["unit"] == "t"
    assert math.isclose(tons["min"], math.expm1(log_bounds["min"]))
    assert math.isclose(tons["max"], math.expm1(log_bounds["max"]))
    # températures et pluie : identité
    assert td["average_temperature_celsius"]["unit"] == "°C"
    assert td["annual_rainfall_mm"]["unit"] == "mm"


def test_get_recommend_context_bundle_none_returns_503():
    """`runtime.bundle_recommend = None` → 503 pour le context également."""
    with TestClient(app) as client:
        saved = runtime.bundle_recommend
        runtime.bundle_recommend = None
        try:
            response = client.get("/recommend/context")
        finally:
            runtime.bundle_recommend = saved

    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_get_recommend_context_context_none_returns_503():
    """`runtime.recommend_context = None` → 503 pour le context également."""
    with TestClient(app) as client:
        saved = runtime.recommend_context
        runtime.recommend_context = None
        try:
            response = client.get("/recommend/context")
        finally:
            runtime.recommend_context = saved

    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"
