"""Tests HTTP de `GET /monitoring/summary` et `GET /monitoring/requests`.

Les tests passent par l'application réelle, avec son lifespan : la base de
monitoring est redirigée vers `tmp_path` par la fixture racine
`_isolate_monitoring_database`. Les lignes de test sont insérées directement
avec `insert_api_request`, à des dates fixes.

Le détail de l'authentification (schéma, token vide, logs) est déjà couvert
par `tests/api/test_dependencies.py` ; ici, on vérifie seulement que les
vrais endpoints sont bien protégés.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agritech.api.core import runtime
from agritech.api.main import app
from agritech.monitoring import repository
from agritech.monitoring.models import ApiRequest
from agritech.monitoring.repository import insert_api_request


TOKEN = "test-monitoring-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# Horloge figée du repository pendant les tests : 2 octobre 2026, 15 h UTC.
NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)

PUBLIC_FIELDS = {
    "id",
    "timestamp",
    "service",
    "status_code",
    "success",
    "duration_ms",
    "model_version",
    "request_payload",
    "error_type",
    "error_message",
}


class _FrozenDatetime(datetime):
    """`datetime` dont `now()` renvoie toujours `NOW`."""

    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Application démarrée avec un token de monitoring et une horloge figée."""
    monkeypatch.setenv("MONITORING_API_TOKEN", TOKEN)
    monkeypatch.setattr(repository, "datetime", _FrozenDatetime)
    with TestClient(app) as test_client:
        yield test_client


def _insert(
    timestamp: datetime,
    service: str = "predict",
    success: bool = True,
    duration_ms: int = 10,
) -> int:
    """Insère une ligne `api_requests` dans la base de l'application et renvoie son id."""
    values = {
        "timestamp": timestamp,
        "service": service,
        "endpoint": f"/{service}",
        "method": "POST",
        "status_code": 200 if success else 422,
        "success": success,
        "duration_ms": duration_ms,
        "api_version": "1.0.0",
        "model_version": "1.0.0" if service == "predict" else "2.0.0",
        "request_payload": {"iso3": "FRA"} if service == "recommend" else {"rainfall_mm": 500.0},
        "response_payload": {"archived": "response"},
        "error_type": None if success else "validation_error",
        "error_message": None if success else "Request payload is invalid.",
        "logfire_trace_id": "0123456789abcdef0123456789abcdef",
        "environment": "test",
    }
    with runtime.monitoring_session_factory() as session:
        return insert_api_request(session, values)


def _snapshot() -> list[tuple]:
    """Toutes les colonnes de toutes les lignes `api_requests`, triées par id."""
    with runtime.monitoring_session_factory() as session:
        rows = session.execute(select(ApiRequest).order_by(ApiRequest.id)).scalars().all()
        columns = ApiRequest.__table__.columns.keys()
        return [tuple(getattr(row, column) for column in columns) for row in rows]


def _at(days_ago: int, hour: int = 12) -> datetime:
    """Timestamp UTC `days_ago` jours avant `NOW`, à l'heure donnée."""
    return (NOW - timedelta(days=days_ago)).replace(hour=hour, minute=0)


# ===========================================================================
# GET /monitoring/summary
# ===========================================================================


def test_summary_without_data_returns_valid_empty_summary(client: TestClient):
    """Base vide : 200, compteurs à 0, valeurs `null`, 30 jours à zéro par défaut."""
    response = client.get("/monitoring/summary", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] == 0
    assert body["error_count"] == 0
    assert body["success_rate"] is None
    assert body["last_request_at"] is None
    assert body["errors_by_type"] == {}
    assert [entry["service"] for entry in body["services"]] == ["predict", "recommend"]
    assert body["services"][0]["latency_ms"] == {"mean": None, "median": None, "p95": None, "max": None}
    assert len(body["requests_per_day"]) == 30
    assert body["requests_per_day"][-1] == {"date": "2026-10-02", "predict": 0, "recommend": 0}


def test_summary_aggregates_real_rows(client: TestClient):
    """Les lignes de la période sont agrégées ; une ligne trop ancienne est ignorée."""
    _insert(_at(0, hour=9), duration_ms=10)
    _insert(_at(0, hour=10), duration_ms=20)
    _insert(_at(1), success=False, duration_ms=1)
    _insert(_at(2), service="recommend", duration_ms=30)
    _insert(_at(40), duration_ms=999)

    response = client.get("/monitoring/summary", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] == 4
    assert body["error_count"] == 1
    assert body["success_rate"] == pytest.approx(0.75)
    assert body["last_request_at"] == "2026-10-02T10:00:00Z"
    assert body["services"] == [
        {
            "service": "predict",
            "total_requests": 3,
            "error_count": 1,
            "success_rate": pytest.approx(2 / 3),
            "latency_ms": {"mean": 15.0, "median": 15.0, "p95": 19.5, "max": 20},
        },
        {
            "service": "recommend",
            "total_requests": 1,
            "error_count": 0,
            "success_rate": 1.0,
            "latency_ms": {"mean": 30.0, "median": 30.0, "p95": 30.0, "max": 30},
        },
    ]
    assert body["errors_by_type"] == {"validation_error": 1}
    assert body["requests_per_day"][-3:] == [
        {"date": "2026-09-30", "predict": 0, "recommend": 1},
        {"date": "2026-10-01", "predict": 1, "recommend": 0},
        {"date": "2026-10-02", "predict": 2, "recommend": 0},
    ]


def test_summary_days_one_covers_only_today(client: TestClient):
    """`days=1` ne renvoie que le jour courant."""
    _insert(_at(0))
    _insert(_at(1))

    body = client.get("/monitoring/summary", params={"days": 1}, headers=AUTH).json()

    assert body["total_requests"] == 1
    assert body["requests_per_day"] == [{"date": "2026-10-02", "predict": 1, "recommend": 0}]


def test_summary_days_365_is_accepted(client: TestClient):
    """`days=365` est la borne haute : 365 jours, du 03/10/2025 au 02/10/2026."""
    response = client.get("/monitoring/summary", params={"days": 365}, headers=AUTH)

    assert response.status_code == 200
    days = response.json()["requests_per_day"]
    assert len(days) == 365
    assert days[0]["date"] == "2025-10-03"
    assert days[-1]["date"] == "2026-10-02"


@pytest.mark.parametrize("days", [0, 366, -1])
def test_summary_rejects_days_out_of_bounds(client: TestClient, days: int):
    """`days` hors de [1, 365] → 422 `validation_error` sur `query.days`."""
    response = client.get("/monitoring/summary", params={"days": days}, headers=AUTH)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["details"][0]["field"] == "query.days"


# ===========================================================================
# GET /monitoring/requests
# ===========================================================================


@pytest.fixture
def seeded_ids(client: TestClient) -> dict[str, int]:
    """Cinq appels à des dates différentes, du plus ancien au plus récent."""
    return {
        "predict_ok_old": _insert(_at(4)),
        "recommend_error": _insert(_at(3), service="recommend", success=False),
        "predict_error": _insert(_at(2), success=False),
        "recommend_ok": _insert(_at(1), service="recommend"),
        "predict_ok_new": _insert(_at(0)),
    }


def _ids(response) -> list[int]:
    assert response.status_code == 200
    return [item["id"] for item in response.json()["items"]]


def test_requests_without_data_returns_empty_list(client: TestClient):
    """Base vide : 200 et `items` vide."""
    response = client.get("/monitoring/requests", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_requests_are_ordered_from_most_recent(client: TestClient, seeded_ids: dict):
    """Sans filtre, tous les appels du plus récent au plus ancien."""
    response = client.get("/monitoring/requests", headers=AUTH)

    assert _ids(response) == [
        seeded_ids["predict_ok_new"],
        seeded_ids["recommend_ok"],
        seeded_ids["predict_error"],
        seeded_ids["recommend_error"],
        seeded_ids["predict_ok_old"],
    ]


def test_requests_respects_limit(client: TestClient, seeded_ids: dict):
    """`limit=2` garde les deux appels les plus récents."""
    response = client.get("/monitoring/requests", params={"limit": 2}, headers=AUTH)

    assert _ids(response) == [seeded_ids["predict_ok_new"], seeded_ids["recommend_ok"]]


@pytest.mark.parametrize(
    ("params", "expected_keys"),
    [
        ({"service": "predict"}, ["predict_ok_new", "predict_error", "predict_ok_old"]),
        ({"service": "recommend"}, ["recommend_ok", "recommend_error"]),
        ({"success": "true"}, ["predict_ok_new", "recommend_ok", "predict_ok_old"]),
        ({"success": "false"}, ["predict_error", "recommend_error"]),
        ({"service": "recommend", "success": "false"}, ["recommend_error"]),
        ({"service": "predict", "success": "true", "limit": 1}, ["predict_ok_new"]),
    ],
)
def test_requests_filters(client: TestClient, seeded_ids: dict, params: dict, expected_keys):
    """Les filtres `service` et `success`, seuls ou combinés avec `limit`."""
    response = client.get("/monitoring/requests", params=params, headers=AUTH)

    assert _ids(response) == [seeded_ids[key] for key in expected_keys]


def test_requests_expose_only_public_fields(client: TestClient, seeded_ids: dict):
    """Chaque appel n'expose que les 10 champs publics : rien sur la réponse, la trace ou le déploiement."""
    items = client.get("/monitoring/requests", headers=AUTH).json()["items"]

    for item in items:
        assert set(item) == PUBLIC_FIELDS
    error_item = next(item for item in items if item["id"] == seeded_ids["predict_error"])
    assert error_item["status_code"] == 422
    assert error_item["error_type"] == "validation_error"
    assert error_item["error_message"] == "Request payload is invalid."
    assert error_item["request_payload"] == {"rainfall_mm": 500.0}


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"limit": 0}, "query.limit"),
        ({"limit": 101}, "query.limit"),
        ({"service": "health"}, "query.service"),
        ({"success": "maybe"}, "query.success"),
    ],
)
def test_requests_rejects_invalid_parameters(client: TestClient, params: dict, field: str):
    """Paramètre hors contrat → 422 `validation_error` sur le bon champ."""
    response = client.get("/monitoring/requests", params=params, headers=AUTH)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["details"][0]["field"] == field


# ===========================================================================
# Authentification et indisponibilité
# ===========================================================================

MONITORING_PATHS = ["/monitoring/summary", "/monitoring/requests"]


@pytest.mark.parametrize("path", MONITORING_PATHS)
def test_missing_authorization_header_returns_401(client: TestClient, path: str):
    """Sans en-tête `Authorization` → 401 + `WWW-Authenticate: Bearer`."""
    response = client.get(path)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"] == "unauthorized"


@pytest.mark.parametrize("path", MONITORING_PATHS)
def test_wrong_token_returns_401(client: TestClient, path: str):
    """Mauvais token → 401."""
    response = client.get(path, headers={"Authorization": "Bearer wrong-token"})

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


@pytest.mark.parametrize("path", MONITORING_PATHS)
def test_missing_server_token_returns_503(path: str):
    """Sans `MONITORING_API_TOKEN` côté serveur (valeur par défaut des tests) → 503."""
    with TestClient(app) as client:
        response = client.get(path, headers=AUTH)

    assert response.status_code == 503
    assert response.json()["error"] == "monitoring_unavailable"


@pytest.mark.parametrize("path", MONITORING_PATHS)
def test_missing_session_factory_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
):
    """Auth valide mais base indisponible → 503."""
    monkeypatch.setattr(runtime, "monitoring_session_factory", None)

    response = client.get(path, headers=AUTH)

    assert response.status_code == 503
    assert response.json()["error"] == "monitoring_unavailable"


@pytest.mark.parametrize("path", MONITORING_PATHS)
def test_rejected_calls_do_not_open_a_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
):
    """Un appel refusé (401, puis 503 sans token serveur) n'ouvre jamais de session."""
    real_factory = runtime.monitoring_session_factory
    opened = []

    def counting_factory():
        opened.append(path)
        return real_factory()

    monkeypatch.setattr(runtime, "monitoring_session_factory", counting_factory)

    assert client.get(path).status_code == 401
    monkeypatch.setattr(runtime, "monitoring_api_token", None)
    assert client.get(path, headers=AUTH).status_code == 503
    assert opened == []


# ===========================================================================
# Lecture seule
# ===========================================================================


def test_monitoring_endpoints_never_modify_api_requests(client: TestClient, seeded_ids: dict):
    """Appels réussis, refusés ou invalides : la table `api_requests` reste identique.

    Vérifie aussi que le middleware n'archive pas ces GET : sinon le
    dashboard mesurerait ses propres lectures.
    """
    before = _snapshot()

    client.get("/monitoring/summary", headers=AUTH)
    client.get("/monitoring/summary", params={"days": 365}, headers=AUTH)
    client.get("/monitoring/requests", headers=AUTH)
    client.get("/monitoring/requests", params={"service": "predict", "success": "false"}, headers=AUTH)
    client.get("/monitoring/requests", params={"limit": 0}, headers=AUTH)
    client.get("/monitoring/summary")

    after = _snapshot()
    assert len(before) == 5
    assert after == before


# ===========================================================================
# OpenAPI
# ===========================================================================


def _parameters(operation: dict) -> dict[str, dict]:
    return {parameter["name"]: parameter for parameter in operation["parameters"]}


def test_openapi_documents_monitoring_routes(client: TestClient):
    """Routes, paramètres, bornes, sécurité Bearer et réponses 200/401/422/503 documentés."""
    document = client.get("/openapi.json").json()

    assert document["components"]["securitySchemes"]["HTTPBearer"] == {
        "type": "http",
        "scheme": "bearer",
    }

    summary = document["paths"]["/monitoring/summary"]["get"]
    requests_ = document["paths"]["/monitoring/requests"]["get"]
    for operation in (summary, requests_):
        assert operation["security"] == [{"HTTPBearer": []}]
        assert {"200", "401", "422", "503"} <= set(operation["responses"])

    days = _parameters(summary)["days"]["schema"]
    assert (days["minimum"], days["maximum"], days["default"]) == (1, 365, 30)

    parameters = _parameters(requests_)
    limit = parameters["limit"]["schema"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 100, 20)
    assert {"enum": ["predict", "recommend"], "type": "string"} in parameters["service"]["schema"]["anyOf"]
    assert {"type": "boolean"} in parameters["success"]["schema"]["anyOf"]


def test_openapi_has_no_monitoring_errors_route(client: TestClient):
    """V1 : seuls `/monitoring/summary` et `/monitoring/requests` existent."""
    paths = client.get("/openapi.json").json()["paths"]

    assert sorted(path for path in paths if path.startswith("/monitoring")) == [
        "/monitoring/requests",
        "/monitoring/summary",
    ]
