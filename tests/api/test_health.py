"""Tests de l'endpoint `GET /health` et du document OpenAPI.

Chaque test s'exécute dans le cycle de vie normal de l'application, via
`with TestClient(app) as client:` : le lifespan charge les deux modèles et le
contexte `/recommend` avant la première requête, comme en production. Le
démarrage refusé quand un de ces artefacts manque est testé dans
`test_monitoring_startup.py`.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from agritech.api import main as api_main
from agritech.api.core import runtime
from agritech.api.main import app
from agritech.api.version import API_VERSION
from agritech.config import PATHS
from agritech.predict_config import PREDICT_MODEL_VERSION
from agritech.recommend_config import RECOMMEND_MODEL_VERSION


MODELS_DIR = PATHS.root / "models"
SERVICES = ("predict", "recommend")


def _health() -> dict:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    return response.json()


def test_health_returns_200_after_startup():
    """`/health` répond 200, avec la version de l'API et le contexte `/recommend` chargé."""
    body = _health()

    assert body["status"] == "ok"
    assert body["api_version"] == API_VERSION
    assert body["recommend_context_loaded"] is True


@pytest.mark.parametrize(
    ("service", "expected_version"),
    [("predict", PREDICT_MODEL_VERSION), ("recommend", RECOMMEND_MODEL_VERSION)],
)
def test_health_reports_each_model_loaded_with_its_version(service: str, expected_version: str):
    """Chaque modèle est chargé et expose sa propre version, distincte de celle de l'API."""
    model = _health()["models"][service]

    assert model["loaded"] is True
    assert model["version"] == expected_version


@pytest.mark.parametrize("service", SERVICES)
def test_health_refit_date_comes_from_model_metadata(service: str):
    """`refit_on` reprend la date `created_on` des metadata du modèle servi."""
    metadata = json.loads((MODELS_DIR / f"{service}_model_metadata.json").read_text(encoding="utf-8"))

    refit_on = _health()["models"][service]["refit_on"]

    assert refit_on == metadata["created_on"]
    assert date.fromisoformat(refit_on)


@pytest.mark.parametrize("service", SERVICES)
def test_health_artifact_size_is_the_served_joblib_file(service: str):
    """`artifact_size_bytes` est la taille du `.joblib` servi, pas celle du metadata JSON."""
    size = _health()["models"][service]["artifact_size_bytes"]

    assert size == (MODELS_DIR / f"{service}_model.joblib").stat().st_size
    assert size != (MODELS_DIR / f"{service}_model_metadata.json").stat().st_size


def test_health_reports_the_configured_environment(monkeypatch: pytest.MonkeyPatch):
    """`environment` reprend la variable `ENVIRONMENT`, comme le monitoring."""
    monkeypatch.setenv("ENVIRONMENT", "staging")

    assert _health()["environment"] == "staging"


def test_health_contract_has_no_legacy_or_monitoring_fields():
    """Contrat fermé : ni anciens champs, ni métriques, ni chemins d'artefacts."""
    body = _health()

    assert set(body) == {"status", "api_version", "environment", "models", "recommend_context_loaded"}
    assert set(body["models"]) == set(SERVICES)
    for model in body["models"].values():
        assert set(model) == {"loaded", "version", "refit_on", "artifact_size_bytes"}


def test_health_without_loaded_resources_reports_them_as_not_loaded(monkeypatch: pytest.MonkeyPatch):
    """Sans bundles ni contexte (avant le lifespan), `/health` les signale non chargés."""
    for attribute in ("bundle_predict", "bundle_recommend", "recommend_context"):
        monkeypatch.setattr(runtime, attribute, None)

    response = api_main.health()

    assert response.recommend_context_loaded is False
    for model in (response.models.predict, response.models.recommend):
        assert model.loaded is False
        assert model.version is None
        assert model.refit_on is None
        assert model.artifact_size_bytes is None


def test_openapi_document_lists_health():
    """`/openapi.json` répond 200 et documente le schéma typé de `/health`."""
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["title"] == "Agritech Answers API"
    schema_ref = document["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema_ref == {"$ref": "#/components/schemas/HealthResponse"}
    schemas = document["components"]["schemas"]
    assert "api_version" in schemas["HealthResponse"]["properties"]
    assert "version" in schemas["ModelStatus"]["properties"]
