"""Tests de l'endpoint `GET /health`.

Premier test de vie du socle FastAPI : il ne charge aucun modèle et ne dépend
d'aucune ressource externe.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_returns_200_and_expected_keys():
    """`GET /health` répond 200 et contient les quatre clés attendues."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert isinstance(body["api_version"], str) and body["api_version"]
    assert body["model_loaded"] is False
    assert body["model_version"] is None


def test_openapi_document_is_served():
    """`GET /openapi.json` renvoie un document OpenAPI exploitable, incluant `/health`."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["title"] == "Agritech Answers API"
    assert "/health" in document["paths"]
