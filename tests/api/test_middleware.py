"""Tests d'intégration du middleware `RequestLoggerMiddleware`.

Vérifie sur une base SQLite jetable :

- les endpoints métier (`POST /predict`, `POST /recommend`) sont persistés
  en une seule ligne, avec tous les champs attendus ;
- les 422 (Pydantic et `unknown_country`), 500 et 503 sont persistées avec
  l'`error_type` extrait de l'`ErrorResponse` unifiée existante ;
- les endpoints non métier (`/health`, `*/context`, `/docs`, `/openapi.json`,
  `/redoc`) ne sont jamais persistés ;
- une panne du repository ne casse pas la réponse HTTP ;
- aucun header sensible (`Authorization`) ne fuite dans les payloads
  archivés.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agritech.api.core import runtime
from agritech.api.main import app
from agritech.api.middleware import request_logger
from agritech.monitoring.models import ApiRequest


VALID_PREDICT_PAYLOAD = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}
VALID_RECOMMEND_PAYLOAD = {"iso3": "FRA"}


def _read_rows() -> list[ApiRequest]:
    """Lit toutes les lignes de la base monitoring courante (post-startup)."""
    factory = runtime.monitoring_session_factory
    assert factory is not None, "monitoring session factory not initialised"
    with factory() as sess:
        return list(sess.execute(select(ApiRequest)).scalars())


# ===========================================================================
# 200 — Réponses réussies persistées
# ===========================================================================


def test_post_predict_success_persists_full_row():
    """POST `/predict` 200 → exactement 1 ligne, tous les champs métier renseignés."""
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 200
        rows = _read_rows()

    assert len(rows) == 1
    row = rows[0]
    assert row.service == "predict"
    assert row.endpoint == "/predict"
    assert row.method == "POST"
    assert row.status_code == 200
    assert row.success is True
    assert row.duration_ms > 0
    assert row.api_version == "1.0.0"
    assert row.model_version == "1.0.0"
    assert row.environment == "test"
    assert row.request_payload == VALID_PREDICT_PAYLOAD
    assert isinstance(row.response_payload, dict)
    assert row.response_payload["yield_tons_per_hectare"] == response.json()[
        "yield_tons_per_hectare"
    ]
    assert row.error_type is None
    assert row.error_message is None
    assert row.logfire_trace_id is None
    assert row.timestamp.tzinfo is not None


def test_post_recommend_success_persists_full_row():
    """POST `/recommend` 200 → exactement 1 ligne avec le service et la bonne version."""
    with TestClient(app) as client:
        response = client.post("/recommend", json=VALID_RECOMMEND_PAYLOAD)
        assert response.status_code == 200
        rows = _read_rows()

    assert len(rows) == 1
    row = rows[0]
    assert row.service == "recommend"
    assert row.endpoint == "/recommend"
    assert row.status_code == 200
    assert row.success is True
    assert row.model_version == "2.0.0"
    assert row.request_payload == VALID_RECOMMEND_PAYLOAD
    assert row.response_payload["iso3"] == "FRA"


# ===========================================================================
# 422 — Erreurs de validation persistées
# ===========================================================================


def test_post_predict_422_pydantic_is_persisted():
    """Un champ invalide côté Pydantic → 422 persistée avec `error_type=validation_error`."""
    payload = VALID_PREDICT_PAYLOAD | {"rainfall_mm": -1.0}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        rows = _read_rows()

    assert len(rows) == 1
    row = rows[0]
    assert row.service == "predict"
    assert row.status_code == 422
    assert row.success is False
    assert row.error_type == "validation_error"
    assert row.error_message == "rainfall_mm: Input should be greater than or equal to 0"
    assert row.request_payload == payload  # payload conservé (JSON exploitable)
    assert row.response_payload["error"] == "validation_error"
    # Même en 422, on archive la version du bundle déployé.
    assert row.model_version == "1.0.0"


def test_post_recommend_422_unknown_country_is_persisted():
    """Un `iso3` syntaxiquement valide mais non servi → 422 persistée."""
    with TestClient(app) as client:
        response = client.post("/recommend", json={"iso3": "ZZZ"})
        assert response.status_code == 422
        rows = _read_rows()

    assert len(rows) == 1
    row = rows[0]
    assert row.service == "recommend"
    assert row.status_code == 422
    assert row.success is False
    assert row.error_type == "validation_error"
    assert row.error_message == "iso3: Country not served: ZZZ"
    assert row.model_version == "2.0.0"


@pytest.mark.parametrize(
    ("path", "payload", "expected"),
    [
        (
            "/predict",
            VALID_PREDICT_PAYLOAD | {"rainfall_mm": -99427, "temperature_celsius": -92370947320927},
            "rainfall_mm: Input should be greater than or equal to 0 · "
            "temperature_celsius: Input should be greater than or equal to -50",
        ),
        (
            "/recommend",
            {"iso3": "FRA", "conditions": {"average_temperature_celsius": -1000.0,
                                           "average_annual_pesticides_tons": -1.0}},
            "average_temperature_celsius: Input should be greater than or equal to -50 · "
            "average_annual_pesticides_tons: Input should be greater than or equal to 0",
        ),
    ],
    ids=["predict-deux-champs", "recommend-deux-conditions"],
)
def test_422_error_message_identifies_every_refused_field(path, payload, expected):
    """Le message archivé nomme chaque champ refusé et sa règle, sans dump Pydantic."""
    with TestClient(app) as client:
        assert client.post(path, json=payload).status_code == 422
        rows = _read_rows()

    assert rows[0].error_type == "validation_error"
    assert rows[0].error_message == expected


def test_error_message_is_truncated_to_2000_chars():
    """`error_message` ne dépasse jamais 2000 caractères (garde-fou)."""
    # Le message par défaut de `ErrorResponse` est court, on vérifie surtout la
    # limite formelle : la longueur du champ persisté ≤ 2000.
    with TestClient(app) as client:
        client.post("/predict", json={"rainfall_mm": -1.0})
        rows = _read_rows()

    assert rows[0].error_message is not None
    assert len(rows[0].error_message) <= 2000


# ===========================================================================
# 500 / 503 — Erreurs serveur persistées
# ===========================================================================


def test_post_predict_503_when_bundle_missing_is_persisted():
    """`bundle_predict = None` → 503 archivé avec `error_type=model_unavailable`."""
    with TestClient(app) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = None
        try:
            response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        finally:
            runtime.bundle_predict = saved
        rows = _read_rows()

    assert response.status_code == 503
    assert len(rows) == 1
    row = rows[0]
    assert row.status_code == 503
    assert row.success is False
    assert row.error_type == "model_unavailable"
    assert row.error_message == "Model is unavailable."
    # bundle None au moment de la requête → model_version non déterminable
    assert row.model_version is None


def test_post_predict_500_is_persisted():
    """Un pipeline qui lève → 500 archivé avec `error_type=internal_error`."""

    class BrokenPipeline:
        def predict(self, X):
            raise RuntimeError("boom-should-not-leak")

    with TestClient(app, raise_server_exceptions=False) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = replace(saved, pipeline=BrokenPipeline())
        try:
            response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        finally:
            runtime.bundle_predict = saved
        rows = _read_rows()

    assert response.status_code == 500
    assert len(rows) == 1
    row = rows[0]
    assert row.status_code == 500
    assert row.success is False
    assert row.error_type == "internal_error"
    assert row.error_message == "Internal server error."


# ===========================================================================
# Endpoints non persistés
# ===========================================================================


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/health"),
        ("GET", "/predict/context"),
        ("GET", "/recommend/context"),
        ("GET", "/docs"),
        ("GET", "/openapi.json"),
        ("GET", "/redoc"),
    ],
)
def test_non_business_endpoints_are_not_persisted(method: str, path: str):
    """`/health`, les contextes et la doc ne créent aucune ligne monitoring."""
    with TestClient(app) as client:
        response = client.request(method, path)
        # Toutes ces routes doivent au moins répondre — mais l'important est
        # qu'aucune ligne ne soit écrite. `/redoc` peut renvoyer 200 ou 404
        # selon la configuration ; le contrat porte sur la non-persistance.
        assert response.status_code in (200, 404)
        rows = _read_rows()

    assert rows == []


def test_get_predict_is_not_persisted_even_though_path_matches():
    """Seul `POST /predict` est persisté ; un `GET /predict` (405) ne l'est pas."""
    with TestClient(app) as client:
        response = client.get("/predict")
        assert response.status_code == 405
        rows = _read_rows()

    assert rows == []


# ===========================================================================
# Non-blocage : panne de persistance
# ===========================================================================


def test_persistence_failure_does_not_break_response(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Un repository qui lève → réponse HTTP inchangée + warning loggé."""

    def _boom(session, values):
        raise RuntimeError("simulated sqlite failure")

    with TestClient(app) as client:
        monkeypatch.setattr(request_logger, "insert_api_request", _boom)
        with caplog.at_level("WARNING", logger=request_logger.__name__):
            response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    # Contrat HTTP intact : la prédiction est renvoyée normalement.
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "yield_tons_per_hectare",
        "unit",
        "model_version",
        "out_of_training_domain",
        "notes",
    }
    assert isinstance(body["yield_tons_per_hectare"], float)

    # Un warning a bien été loggé, sans traceback exposée au client.
    assert any(
        "monitoring persistence failed" in record.message for record in caplog.records
    )


def test_persistence_failure_when_session_factory_is_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """Session factory `None` → aucune écriture mais aucune erreur non plus."""
    with TestClient(app) as client:
        monkeypatch.setattr(runtime, "monitoring_session_factory", None)
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    assert response.status_code == 200


# ===========================================================================
# Sécurité : pas de fuite d'en-têtes sensibles
# ===========================================================================


def test_authorization_header_is_not_captured():
    """Un `Authorization: Bearer …` ne se retrouve nulle part dans la ligne archivée."""
    secret = "Bearer very-secret-token-XYZ-123"

    with TestClient(app) as client:
        client.post(
            "/predict",
            json=VALID_PREDICT_PAYLOAD,
            headers={"Authorization": secret, "Cookie": "session=confidential"},
        )
        rows = _read_rows()

    row = rows[0]
    # On sérialise toutes les colonnes texte / JSON pour un scan robuste.
    haystack = " ".join(
        str(value)
        for value in (
            row.endpoint,
            row.method,
            row.request_payload,
            row.response_payload,
            row.error_type,
            row.error_message,
            row.environment,
        )
    )
    assert "very-secret-token-XYZ-123" not in haystack
    assert "confidential" not in haystack
