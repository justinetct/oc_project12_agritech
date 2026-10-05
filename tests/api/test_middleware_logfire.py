"""Tests d'intégration de la corrélation Logfire dans `RequestLoggerMiddleware`.

Aucun test ne fait d'appel réseau réel : `logfire.info` est mocké pour
capturer les événements en mémoire, et `_current_trace_id` est mocké pour
simuler un contexte OTel actif ou absent selon les besoins.
"""

from __future__ import annotations

from dataclasses import replace

import logfire
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


def _read_row() -> ApiRequest:
    """Lit l'unique ligne persistée sur la base monitoring courante."""
    factory = runtime.monitoring_session_factory
    assert factory is not None
    with factory() as sess:
        rows = list(sess.execute(select(ApiRequest)).scalars())
    assert len(rows) == 1
    return rows[0]


# ===========================================================================
# Trace ID capturé et persisté
# ===========================================================================


def test_trace_id_is_none_when_no_span_is_active():
    """Sans Logfire configuré, aucun span n'est actif → `logfire_trace_id is None`."""
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 200
        row = _read_row()

    assert row.logfire_trace_id is None


def test_trace_id_is_persisted_when_span_is_active(monkeypatch: pytest.MonkeyPatch):
    """Un span OTel actif au moment de `http.response.start` est persisté en base."""
    fake_trace_id = "0123456789abcdef0123456789abcdef"

    monkeypatch.setattr(
        request_logger, "_current_trace_id", lambda: fake_trace_id
    )

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 200
        row = _read_row()

    assert row.logfire_trace_id == fake_trace_id
    assert len(row.logfire_trace_id) == 32  # 128 bits en hex


# ===========================================================================
# Événement `api_request_persisted`
# ===========================================================================


def test_persisted_event_is_emitted_after_successful_insert(
    monkeypatch: pytest.MonkeyPatch,
):
    """`logfire.info("api_request_persisted", ...)` est appelé après un INSERT réussi."""
    events: list[tuple[str, dict]] = []

    def _capture_info(message: str, **kwargs) -> None:
        events.append((message, kwargs))

    monkeypatch.setattr(logfire, "info", _capture_info)

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 200
        row = _read_row()

    api_request_events = [
        attrs for name, attrs in events if name == "api_request_persisted"
    ]
    assert len(api_request_events) == 1
    attrs = api_request_events[0]

    # Attributs attendus, alignés sur la ligne SQLite.
    assert attrs["api_request_id"] == row.id
    assert attrs["service"] == "predict"
    assert attrs["endpoint"] == "/predict"
    assert attrs["status_code"] == 200
    assert attrs["success"] is True
    assert attrs["duration_ms"] > 0
    assert attrs["api_version"] == "1.0.0"
    assert attrs["model_version"] == "1.0.0"
    assert attrs["environment"] == "test"


def test_persisted_event_carries_no_payload_and_no_secret(
    monkeypatch: pytest.MonkeyPatch,
):
    """L'événement Logfire ne transporte ni request/response payload, ni en-têtes."""
    events: list[tuple[str, dict]] = []

    def _capture_info(message: str, **kwargs) -> None:
        events.append((message, kwargs))

    monkeypatch.setattr(logfire, "info", _capture_info)

    with TestClient(app) as client:
        client.post(
            "/predict",
            json=VALID_PREDICT_PAYLOAD,
            headers={"Authorization": "Bearer super-secret-XYZ-999"},
        )

    api_request_events = [
        attrs for name, attrs in events if name == "api_request_persisted"
    ]
    assert len(api_request_events) == 1
    attrs = api_request_events[0]

    # Rien du payload utilisateur, rien d'un header sensible.
    forbidden_keys = {
        "request_payload",
        "response_payload",
        "error_message",
        "authorization",
        "cookies",
        "headers",
        "logfire_token",
    }
    assert forbidden_keys.isdisjoint(attrs)

    # Aucune valeur d'attribut ne contient le secret.
    haystack = " ".join(str(value) for value in attrs.values())
    assert "super-secret-XYZ-999" not in haystack


def test_no_persisted_event_when_insert_fails(monkeypatch: pytest.MonkeyPatch):
    """Si l'INSERT SQLite échoue, aucun événement Logfire n'est émis."""
    events: list[tuple[str, dict]] = []

    def _capture_info(message: str, **kwargs) -> None:
        events.append((message, kwargs))

    def _broken_insert(session, values):
        raise RuntimeError("simulated sqlite failure")

    monkeypatch.setattr(logfire, "info", _capture_info)
    monkeypatch.setattr(request_logger, "insert_api_request", _broken_insert)

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    # Contrat HTTP intact.
    assert response.status_code == 200
    # Événement final non émis.
    assert not any(name == "api_request_persisted" for name, _ in events)


def test_logfire_info_error_does_not_break_response(monkeypatch: pytest.MonkeyPatch):
    """Un `logfire.info` qui lève ne casse pas la réponse HTTP."""

    def _broken_info(message, **kwargs):
        raise RuntimeError("logfire down")

    monkeypatch.setattr(logfire, "info", _broken_info)

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        # Lire la ligne AVANT la sortie du context manager : le shutdown
        # remet `monitoring_session_factory` à None.
        row = _read_row()

    assert response.status_code == 200
    # La ligne SQLite a bien été écrite malgré l'échec Logfire.
    assert row.status_code == 200


# ===========================================================================
# Nom des spans et attributs métier
# ===========================================================================


class _FakeSpan:
    """Faux `LogfireSpan` qui enregistre attributs et niveau pour inspection."""

    def __init__(self, name: str, initial_kwargs: dict) -> None:
        self.name = name
        self.initial_kwargs = initial_kwargs
        self.attributes: dict[str, object] = {}
        self.level: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_level(self, level: str) -> None:
        self.level = level


def _install_fake_span(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSpan]:
    """Remplace `logfire.span` par un mock qui capture les spans créés."""
    created: list[_FakeSpan] = []

    def _fake_span(name: str, **kwargs):
        span = _FakeSpan(name, kwargs)
        created.append(span)
        return span

    monkeypatch.setattr(logfire, "span", _fake_span)
    return created


def test_span_name_is_method_plus_path_for_predict(
    monkeypatch: pytest.MonkeyPatch,
):
    """Un POST /predict ouvre un span nommé exactement `POST /predict`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    assert any(s.name == "POST /predict" for s in spans)


def test_span_name_is_method_plus_path_for_recommend(
    monkeypatch: pytest.MonkeyPatch,
):
    """Un POST /recommend ouvre un span nommé exactement `POST /recommend`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/recommend", json={"iso3": "FRA"})

    assert any(s.name == "POST /recommend" for s in spans)


def test_span_name_is_method_plus_path_for_health(
    monkeypatch: pytest.MonkeyPatch,
):
    """Un GET /health ouvre un span nommé exactement `GET /health`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.get("/health")

    assert any(s.name == "GET /health" for s in spans)


def test_health_creates_span_but_no_sqlite_row(
    monkeypatch: pytest.MonkeyPatch,
):
    """`/health` est tracé côté Logfire mais n'écrit aucune ligne SQLite."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        factory = runtime.monitoring_session_factory
        assert factory is not None
        with factory() as sess:
            rows = list(sess.execute(select(ApiRequest)).scalars())

    assert rows == []
    health_spans = [s for s in spans if s.name == "GET /health"]
    assert len(health_spans) == 1
    # Le span porte au moins le statut et la réponse.
    span = health_spans[0]
    assert span.attributes["response.status_code"] == 200
    assert "response.body" in span.attributes


@pytest.mark.parametrize(
    "path, path_not_in_routes",
    [
        ("/predict/context", "GET"),
        ("/recommend/context", "GET"),
        ("/openapi.json", "GET"),
    ],
)
def test_untraced_routes_do_not_open_a_span(
    monkeypatch: pytest.MonkeyPatch, path: str, path_not_in_routes: str
):
    """Les endpoints hors table de routage ne créent pas de span."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.request(path_not_in_routes, path)

    assert spans == []


def test_predict_span_carries_request_and_response_bodies(
    monkeypatch: pytest.MonkeyPatch,
):
    """Le span `POST /predict` porte `request.body`, `response.body`, `response.status_code`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 200
        expected_response = response.json()

    predict_spans = [s for s in spans if s.name == "POST /predict"]
    assert len(predict_spans) == 1
    attrs = predict_spans[0].attributes

    # Requête et réponse : dicts métier sérialisés en JSON string.
    import json

    assert attrs["response.status_code"] == 200
    assert json.loads(attrs["request.body"]) == VALID_PREDICT_PAYLOAD
    assert json.loads(attrs["response.body"]) == expected_response
    # Réponse à succès : aucun `error_type`.
    assert "error_type" not in attrs


def test_recommend_span_carries_business_response(monkeypatch: pytest.MonkeyPatch):
    """Le span `POST /recommend` inclut la réponse métier structurée."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/recommend", json={"iso3": "FRA"})
        assert response.status_code == 200

    recommend_spans = [s for s in spans if s.name == "POST /recommend"]
    assert len(recommend_spans) == 1
    attrs = recommend_spans[0].attributes

    import json

    body = json.loads(attrs["response.body"])
    assert body["iso3"] == "FRA"
    assert len(body["recommendations"]) == 10


def test_predict_422_span_carries_error_type(monkeypatch: pytest.MonkeyPatch):
    """Une 422 ajoute `error_type=validation_error` en attribut du span."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/predict", json={"rainfall_mm": -1.0})

    predict_spans = [s for s in spans if s.name == "POST /predict"]
    assert len(predict_spans) == 1
    attrs = predict_spans[0].attributes

    assert attrs["response.status_code"] == 422
    assert attrs["error_type"] == "validation_error"


def test_spans_never_carry_headers_or_secrets(monkeypatch: pytest.MonkeyPatch):
    """Aucun attribut de span ne contient un header sensible (`Authorization`, cookie)."""
    spans = _install_fake_span(monkeypatch)
    secret = "Bearer very-secret-xyz-42"

    with TestClient(app) as client:
        client.post(
            "/predict",
            json=VALID_PREDICT_PAYLOAD,
            headers={"Authorization": secret, "Cookie": "session=confidential"},
        )

    for span in spans:
        haystack = " ".join(str(v) for v in span.attributes.values())
        assert secret not in haystack
        assert "confidential" not in haystack
        # Contrôle par nom d'attribut : aucun nom relatif aux en-têtes.
        forbidden = {"authorization", "cookie", "cookies", "headers", "ip"}
        assert forbidden.isdisjoint(k.lower() for k in span.attributes)


def test_health_span_carries_no_request_body(monkeypatch: pytest.MonkeyPatch):
    """`/health` est un GET : pas d'attribut `request.body` sur son span."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.get("/health")

    health_spans = [s for s in spans if s.name == "GET /health"]
    assert len(health_spans) == 1
    assert "request.body" not in health_spans[0].attributes


# ===========================================================================
# Niveau du span selon le status HTTP
# ===========================================================================


def test_span_level_is_info_for_predict_200(monkeypatch: pytest.MonkeyPatch):
    """Un 200 laisse le span au niveau `info` (parcours nominal)."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    predict_spans = [s for s in spans if s.name == "POST /predict"]
    assert predict_spans[0].level == "info"


def test_span_level_is_info_for_health_200(monkeypatch: pytest.MonkeyPatch):
    """`GET /health` en 200 : niveau `info` — pas de bruit visuel côté Logfire."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.get("/health")

    health_spans = [s for s in spans if s.name == "GET /health"]
    assert health_spans[0].level == "info"


def test_span_level_is_warning_for_predict_422(monkeypatch: pytest.MonkeyPatch):
    """Une 422 Pydantic passe le span au niveau `warning`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/predict", json={"rainfall_mm": -1.0})

    predict_spans = [s for s in spans if s.name == "POST /predict"]
    assert predict_spans[0].level == "warning"


def test_span_level_is_warning_for_recommend_422(monkeypatch: pytest.MonkeyPatch):
    """Un `iso3` non servi (422 métier) passe le span au niveau `warning`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/recommend", json={"iso3": "ZZZ"})

    recommend_spans = [s for s in spans if s.name == "POST /recommend"]
    assert recommend_spans[0].level == "warning"


def test_422_span_exposes_error_type_and_error_message(
    monkeypatch: pytest.MonkeyPatch,
):
    """Le span 422 porte `error_type`, `error_message` et `response.status_code`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/predict", json={"rainfall_mm": -1.0})

    attrs = [s for s in spans if s.name == "POST /predict"][0].attributes
    assert attrs["response.status_code"] == 422
    assert attrs["error_type"] == "validation_error"
    # Corps partiel : la pluie négative et les trois champs manquants sont tous nommés.
    assert attrs["error_message"] == (
        "rainfall_mm: Input should be greater than or equal to 0 · "
        "temperature_celsius: Field required · "
        "fertilizer_used: Field required · "
        "irrigation_used: Field required"
    )


def test_span_level_is_error_for_503(monkeypatch: pytest.MonkeyPatch):
    """Un 503 `model_unavailable` fait passer le span au niveau `error`."""
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = None
        try:
            client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        finally:
            runtime.bundle_predict = saved

    predict_spans = [s for s in spans if s.name == "POST /predict"]
    span = predict_spans[0]
    assert span.attributes["response.status_code"] == 503
    assert span.level == "error"
    assert span.attributes["error_type"] == "model_unavailable"
    assert span.attributes["error_message"] == "Model is unavailable."


def test_span_level_is_error_for_500(monkeypatch: pytest.MonkeyPatch):
    """Un 500 `internal_error` (exception non gérée) fait passer le span au niveau `error`."""
    spans = _install_fake_span(monkeypatch)

    class _Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = replace(saved, pipeline=_Broken())
        try:
            client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        finally:
            runtime.bundle_predict = saved

    predict_spans = [s for s in spans if s.name == "POST /predict"]
    span = predict_spans[0]
    assert span.attributes["response.status_code"] == 500
    assert span.level == "error"
    assert span.attributes["error_type"] == "internal_error"
    assert span.attributes["error_message"] == "Internal server error."


def test_error_message_is_truncated_at_2000_chars_on_span(
    monkeypatch: pytest.MonkeyPatch,
):
    """L'attribut `error_message` du span ne dépasse jamais 2000 caractères."""
    from agritech.api.middleware import request_logger

    long_message = "X" * 3000
    original = request_logger._parse_json_or_none

    def _inject(raw):
        parsed = original(raw)
        if isinstance(parsed, dict) and parsed.get("error") == "validation_error":
            parsed["details"][0]["message"] = long_message
        return parsed

    monkeypatch.setattr(request_logger, "_parse_json_or_none", _inject)
    spans = _install_fake_span(monkeypatch)

    with TestClient(app) as client:
        client.post("/predict", json={"rainfall_mm": -1.0})

    attrs = [s for s in spans if s.name == "POST /predict"][0].attributes
    assert len(attrs["error_message"]) == 2000
