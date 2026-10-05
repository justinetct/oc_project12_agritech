"""Tests de ``agritech.ui.api_client``.

Les appels réseau sont interceptés en remplaçant ``requests.request``.
Aucune API réelle n'est démarrée pendant les tests.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from agritech.api.schemas.monitoring import (
    MonitoringRequestsResponse,
    MonitoringSummaryResponse,
)
from agritech.ui import api_client
from agritech.ui.errors import (
    ApiConfigurationError,
    ApiConnectionError,
    ApiError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
    format_monitoring_error,
)


class _FakeResponse:
    """Réponse minimale imitant ``requests.Response`` pour les tests."""

    def __init__(
        self, status_code: int, payload: Any = None, raise_json: bool = False
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._raise_json = raise_json

    def json(self) -> Any:
        if self._raise_json:
            raise ValueError("bad json")
        return self._payload


@pytest.fixture
def base_url(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("AGRITECH_API_URL", "http://api.test")
    return "http://api.test"


def _record_call(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> dict:
    """Remplace ``requests.request`` et enregistre les arguments reçus."""
    calls: dict = {}

    def fake_request(
        method: str,
        url: str,
        json: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        calls["method"] = method
        calls["url"] = url
        calls["json"] = json
        calls["params"] = params
        calls["headers"] = headers
        calls["timeout"] = timeout
        return response

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    return calls


# --- succès ---------------------------------------------------------------

def test_get_predict_context_returns_json(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"physical_bounds": {}, "training_domain": {}}
    calls = _record_call(monkeypatch, _FakeResponse(200, payload))
    result = api_client.get_predict_context()
    assert result == payload
    assert calls["method"] == "GET"
    assert calls["url"] == "http://api.test/predict/context"
    assert calls["json"] is None
    assert calls["params"] is None
    assert calls["headers"] is None
    assert calls["timeout"] == 5.0


def test_post_predict_sends_json_body(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_in = {
        "rainfall_mm": 500.0,
        "temperature_celsius": 25.0,
        "fertilizer_used": True,
        "irrigation_used": False,
    }
    payload_out = {
        "yield_tons_per_hectare": 4.82,
        "unit": "t/ha",
        "model_version": "1.0.0",
        "out_of_training_domain": False,
        "notes": [],
    }
    calls = _record_call(monkeypatch, _FakeResponse(200, payload_out))
    result = api_client.post_predict(payload_in)
    assert result == payload_out
    assert calls["method"] == "POST"
    assert calls["url"] == "http://api.test/predict"
    assert calls["json"] == payload_in
    assert calls["params"] is None
    assert calls["headers"] is None
    assert calls["timeout"] == 5.0


def test_get_recommend_context_for_a_country(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"country": {"iso3": "FRA", "country": "France", "country_defaults": {}}}
    calls = _record_call(monkeypatch, _FakeResponse(200, payload))
    assert api_client.get_recommend_context("FRA") == payload
    assert calls["method"] == "GET"
    assert calls["url"] == "http://api.test/recommend/context?iso3=FRA"
    assert calls["json"] is None


def test_post_recommend_sends_json_body(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_in = {"iso3": "FRA", "conditions": {"annual_rainfall_mm": 867.0}}
    payload_out = {"iso3": "FRA", "recommendations": []}
    calls = _record_call(monkeypatch, _FakeResponse(200, payload_out))
    assert api_client.post_recommend(payload_in) == payload_out
    assert calls["method"] == "POST"
    assert calls["url"] == "http://api.test/recommend"
    assert calls["json"] == payload_in


def test_base_url_trailing_slash_is_normalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGRITECH_API_URL", "http://api.test/")
    calls = _record_call(monkeypatch, _FakeResponse(200, {}))
    api_client.get_predict_context()
    assert calls["url"] == "http://api.test/predict/context"


# --- erreurs réseau -------------------------------------------------------

def test_timeout_raises_api_timeout_error(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_request(*args: Any, **kwargs: Any) -> Any:
        raise requests.Timeout("boom")

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    with pytest.raises(ApiTimeoutError):
        api_client.get_predict_context()


def test_connection_error_raises_api_connection_error(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_request(*args: Any, **kwargs: Any) -> Any:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    with pytest.raises(ApiConnectionError):
        api_client.get_predict_context()


def test_generic_request_exception_raises_api_connection_error(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_request(*args: Any, **kwargs: Any) -> Any:
        raise requests.RequestException("other")

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    with pytest.raises(ApiConnectionError):
        api_client.get_predict_context()


# --- erreurs HTTP ---------------------------------------------------------

def test_http_422_parses_validation_error(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {
        "error": "validation_error",
        "message": "Request payload is invalid.",
        "details": [
            {
                "field": "body.rainfall_mm",
                "type": "greater_than_equal",
                "message": "must be >= 0",
            }
        ],
    }
    _record_call(monkeypatch, _FakeResponse(422, body))
    with pytest.raises(ApiHttpError) as info:
        api_client.post_predict({})
    exc = info.value
    assert exc.status_code == 422
    assert exc.code == "validation_error"
    assert exc.message == "Request payload is invalid."
    assert exc.details == body["details"]


def test_http_503_parses_model_unavailable(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {
        "error": "model_unavailable",
        "message": "Model is unavailable.",
        "details": None,
    }
    _record_call(monkeypatch, _FakeResponse(503, body))
    with pytest.raises(ApiHttpError) as info:
        api_client.get_predict_context()
    exc = info.value
    assert exc.status_code == 503
    assert exc.code == "model_unavailable"
    assert exc.details == []


def test_http_500_parses_internal_error(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {
        "error": "internal_error",
        "message": "Internal server error.",
        "details": None,
    }
    _record_call(monkeypatch, _FakeResponse(500, body))
    with pytest.raises(ApiHttpError) as info:
        api_client.get_predict_context()
    assert info.value.code == "internal_error"


def test_http_error_with_unreadable_body_falls_back(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_call(monkeypatch, _FakeResponse(500, payload=None, raise_json=True))
    with pytest.raises(ApiHttpError) as info:
        api_client.get_predict_context()
    exc = info.value
    assert exc.status_code == 500
    assert exc.code == "unknown_error"
    assert exc.message == "Erreur inconnue."
    assert exc.details == []


# --- réponses JSON non conformes ----------------------------------------

def test_success_with_invalid_json_raises_api_invalid_response(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_call(monkeypatch, _FakeResponse(200, payload=None, raise_json=True))
    with pytest.raises(ApiInvalidResponseError):
        api_client.get_predict_context()


def test_success_with_non_dict_json_raises_api_invalid_response(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Un JSON valide mais qui n'est pas un objet (ex. `null`) ne correspond
    # à aucun contrat côté API : le client doit le rejeter.
    _record_call(monkeypatch, _FakeResponse(200, payload=None))
    with pytest.raises(ApiInvalidResponseError):
        api_client.get_predict_context()


# --- _request : query string et en-têtes -----------------------------------

def test_request_passes_query_params_and_headers(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_call(monkeypatch, _FakeResponse(200, {"ok": True}))
    result = api_client._request(
        "GET", "/some/path", params={"a": 1}, headers={"X-Test": "value"}
    )
    assert result == {"ok": True}
    assert calls["url"] == "http://api.test/some/path"
    assert calls["params"] == {"a": 1}
    assert calls["headers"] == {"X-Test": "value"}
    assert calls["json"] is None


# --- monitoring -------------------------------------------------------------

MONITORING_TOKEN = "test-monitoring-secret"


@pytest.fixture
def monitoring_token(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MONITORING_API_TOKEN", MONITORING_TOKEN)
    return MONITORING_TOKEN


def _summary_json() -> dict[str, Any]:
    """Réponse JSON de `GET /monitoring/summary`, telle que l'API la sérialise."""
    return {
        "total_requests": 3,
        "error_count": 1,
        "success_rate": 2 / 3,
        "last_request_at": "2026-10-02T14:00:00Z",
        "services": [
            {
                "service": "predict",
                "total_requests": 2,
                "error_count": 1,
                "success_rate": 0.5,
                "latency_ms": {"mean": 12.0, "median": 12.0, "max": 12},
            },
            {
                "service": "recommend",
                "total_requests": 1,
                "error_count": 0,
                "success_rate": 1.0,
                "latency_ms": {"mean": 8.0, "median": 8.0, "max": 8},
            },
        ],
        "errors_by_type": {"validation_error": 1},
        "requests_per_day": [
            {"date": "2026-10-01", "predict": 0, "recommend": 1},
            {"date": "2026-10-02", "predict": 2, "recommend": 0},
        ],
    }


def _requests_json() -> dict[str, Any]:
    """Réponse JSON de `GET /monitoring/requests`."""
    return {
        "items": [
            {
                "id": 3,
                "timestamp": "2026-10-02T14:00:00Z",
                "service": "predict",
                "status_code": 422,
                "success": False,
                "duration_ms": 1,
                "model_version": "1.0.0",
                "request_payload": {"rainfall_mm": -1},
                "error_type": "validation_error",
                "error_message": "Request payload is invalid.",
            }
        ]
    }


def test_get_monitoring_summary_sends_days_and_bearer_token(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_call(monkeypatch, _FakeResponse(200, _summary_json()))
    api_client.get_monitoring_summary(7)
    assert calls["method"] == "GET"
    assert calls["url"] == "http://api.test/monitoring/summary"
    assert calls["params"] == {"days": 7}
    assert calls["headers"] == {"Authorization": f"Bearer {monitoring_token}"}
    assert calls["json"] is None
    assert calls["timeout"] == 5.0


def test_get_monitoring_summary_returns_validated_model(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_call(monkeypatch, _FakeResponse(200, _summary_json()))
    summary = api_client.get_monitoring_summary(30)
    assert isinstance(summary, MonitoringSummaryResponse)
    assert summary.total_requests == 3
    assert summary.last_request_at == datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc)
    assert summary.services[0].latency_ms.max == 12
    assert summary.requests_per_day[0].date == date(2026, 10, 1)


def test_get_monitoring_summary_invalid_json_raises_api_invalid_response(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_call(monkeypatch, _FakeResponse(200, payload=None, raise_json=True))
    with pytest.raises(ApiInvalidResponseError):
        api_client.get_monitoring_summary(30)


def test_get_monitoring_summary_off_contract_json_raises_api_invalid_response(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON valide mais hors contrat : la `ValidationError` est convertie."""
    payload = _summary_json()
    del payload["requests_per_day"]
    payload["total_requests"] = "beaucoup"
    _record_call(monkeypatch, _FakeResponse(200, payload))
    with pytest.raises(ApiInvalidResponseError) as info:
        api_client.get_monitoring_summary(30)
    assert isinstance(info.value.__cause__, ValidationError)
    assert "beaucoup" not in str(info.value)


def test_get_monitoring_requests_sends_only_limit_by_default(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_call(monkeypatch, _FakeResponse(200, _requests_json()))
    api_client.get_monitoring_requests(limit=20)
    assert calls["method"] == "GET"
    assert calls["url"] == "http://api.test/monitoring/requests"
    assert calls["params"] == {"limit": 20}
    assert calls["headers"] == {"Authorization": f"Bearer {monitoring_token}"}
    assert calls["timeout"] == 5.0


@pytest.mark.parametrize(
    ("kwargs", "expected_params"),
    [
        ({"service": "predict"}, {"limit": 50, "service": "predict"}),
        ({"service": "recommend"}, {"limit": 50, "service": "recommend"}),
        ({"success": True}, {"limit": 50, "success": "true"}),
        ({"success": False}, {"limit": 50, "success": "false"}),
        (
            {"service": "recommend", "success": False},
            {"limit": 50, "service": "recommend", "success": "false"},
        ),
        ({"service": None, "success": None}, {"limit": 50}),
    ],
)
def test_get_monitoring_requests_query_params(
    base_url: str,
    monitoring_token: str,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    expected_params: dict[str, Any],
) -> None:
    """`service` et `success` ne sont envoyés que s'ils sont renseignés ; `False` est envoyé."""
    calls = _record_call(monkeypatch, _FakeResponse(200, _requests_json()))
    api_client.get_monitoring_requests(limit=50, **kwargs)
    assert calls["params"] == expected_params


def test_get_monitoring_requests_returns_validated_model(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_call(monkeypatch, _FakeResponse(200, _requests_json()))
    response = api_client.get_monitoring_requests(limit=20)
    assert isinstance(response, MonitoringRequestsResponse)
    item = response.items[0]
    assert item.timestamp == datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc)
    assert item.success is False
    assert item.request_payload == {"rainfall_mm": -1}


def test_get_monitoring_requests_off_contract_json_raises_api_invalid_response(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _requests_json()
    payload["items"][0]["service"] = "health"
    _record_call(monkeypatch, _FakeResponse(200, payload))
    with pytest.raises(ApiInvalidResponseError):
        api_client.get_monitoring_requests(limit=20)


@pytest.mark.parametrize("token_value", [None, "", "   "])
@pytest.mark.parametrize(
    "call",
    [
        lambda: api_client.get_monitoring_summary(30),
        lambda: api_client.get_monitoring_requests(limit=20),
    ],
    ids=["summary", "requests"],
)
def test_missing_monitoring_token_raises_before_any_http_call(
    base_url: str, monkeypatch: pytest.MonkeyPatch, token_value: str | None, call
) -> None:
    if token_value is None:
        monkeypatch.delenv("MONITORING_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MONITORING_API_TOKEN", token_value)
    calls = _record_call(monkeypatch, _FakeResponse(200, _summary_json()))
    with pytest.raises(ApiConfigurationError):
        call()
    assert calls == {}


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (401, {"error": "unauthorized", "message": "Authentication required.", "details": None}),
        (503, {"error": "monitoring_unavailable", "message": "Monitoring is unavailable.", "details": None}),
    ],
)
def test_monitoring_http_errors_keep_their_status(
    base_url: str,
    monitoring_token: str,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    body: dict[str, Any],
) -> None:
    _record_call(monkeypatch, _FakeResponse(status_code, body))
    with pytest.raises(ApiHttpError) as info:
        api_client.get_monitoring_summary(30)
    assert info.value.status_code == status_code
    assert info.value.code == body["error"]


def _raise(exc: Exception):
    def fake_request(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return fake_request


@pytest.mark.parametrize(
    "fake_request",
    [
        _raise(requests.ConnectionError("connection refused")),
        _raise(requests.Timeout("read timed out")),
        lambda *a, **k: _FakeResponse(401, {"error": "unauthorized", "message": "Authentication required."}),
        lambda *a, **k: _FakeResponse(503, payload=None, raise_json=True),
        lambda *a, **k: _FakeResponse(500, {"error": "internal_error", "message": "Internal server error."}),
        lambda *a, **k: _FakeResponse(200, payload=None, raise_json=True),
        lambda *a, **k: _FakeResponse(200, {"items": "not a list"}),
    ],
    ids=["connection", "timeout", "401", "503", "500", "invalid-json", "off-contract"],
)
def test_monitoring_token_never_appears_in_errors(
    base_url: str, monitoring_token: str, monkeypatch: pytest.MonkeyPatch, fake_request
) -> None:
    """Quel que soit l'échec, ni l'exception ni le message affiché ne contiennent le token."""
    monkeypatch.setattr(api_client.requests, "request", fake_request)
    with pytest.raises(ApiError) as info:
        api_client.get_monitoring_requests(limit=20)
    exc = info.value
    assert monitoring_token not in str(exc)
    assert monitoring_token not in repr(exc)
    assert monitoring_token not in format_monitoring_error(exc)
