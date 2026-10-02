"""Tests de ``agritech.ui.api_client``.

Les appels réseau sont interceptés en remplaçant ``requests.request``.
Aucune API réelle n'est démarrée pendant les tests.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from agritech.ui import api_client
from agritech.ui.errors import (
    ApiConnectionError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
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
        timeout: float | None = None,
    ) -> _FakeResponse:
        calls["method"] = method
        calls["url"] = url
        calls["json"] = json
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
