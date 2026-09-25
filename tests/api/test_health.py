"""Tests de l'endpoint `GET /health` et du document OpenAPI.

Chaque test s'exécute dans le cycle de vie normal de l'application, via
`with TestClient(app) as client:` : le lifespan charge le modèle avant la
première requête, comme en production.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agritech.api.main import app


def test_health_returns_200_after_startup():
    """`/health` répond 200 avec le modèle chargé et la bonne version."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["api_version"], str) and body["api_version"]
    assert body["model_loaded"] is True
    assert body["model_version"] == "1.0.0"


def test_openapi_document_lists_health():
    """`/openapi.json` répond 200 et inclut l'endpoint `/health`."""
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["title"] == "Agritech Answers API"
    assert "/health" in document["paths"]
