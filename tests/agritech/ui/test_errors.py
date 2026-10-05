"""Tests de ``agritech.ui.errors``."""

from __future__ import annotations

import pytest

from agritech.ui.errors import (
    ApiConfigurationError,
    ApiConnectionError,
    ApiError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
    format_api_error,
    format_monitoring_error,
)


def test_format_timeout() -> None:
    assert format_api_error(ApiTimeoutError("x")) == (
        "Le service de calcul n'a pas répondu à temps. Réessayez dans un instant."
    )


def test_format_connection() -> None:
    assert format_api_error(ApiConnectionError("x")) == (
        "Impossible de joindre le service de calcul. Réessayez dans un instant."
    )


def test_format_invalid_response() -> None:
    assert "inattendue" in format_api_error(ApiInvalidResponseError("x"))


def test_format_model_unavailable_in_french() -> None:
    exc = ApiHttpError(503, "model_unavailable", "Model is unavailable.")
    assert format_api_error(exc) == "Le modèle est momentanément indisponible. Réessayez dans un instant."


def test_format_internal_error_in_french() -> None:
    exc = ApiHttpError(500, "internal_error", "Internal server error.")
    assert format_api_error(exc) == "Le service a rencontré une erreur. Réessayez dans un instant."


def test_format_validation_error_without_technical_details() -> None:
    exc = ApiHttpError(
        422,
        "validation_error",
        "Request payload is invalid.",
        details=[
            {
                "field": "body.rainfall_mm",
                "type": "greater_than_equal",
                "message": "must be >= 0",
            }
        ],
    )
    out = format_api_error(exc)
    assert out == "Les valeurs envoyées n’ont pas été acceptées."
    assert "body.rainfall_mm" not in out and "must be >= 0" not in out


def test_format_unknown_http_error_shows_its_code() -> None:
    exc = ApiHttpError(404, "unknown_error", "Erreur inconnue.")
    assert format_api_error(exc) == "Le service a renvoyé une erreur inattendue (code 404)."


def test_api_http_error_default_details_is_empty_list() -> None:
    exc = ApiHttpError(500, "internal_error", "Internal server error.")
    assert exc.details == []
    assert exc.status_code == 500
    assert exc.code == "internal_error"
    assert exc.message == "Internal server error."


# --- format_monitoring_error ------------------------------------------------

def test_monitoring_missing_token_message() -> None:
    assert format_monitoring_error(ApiConfigurationError("x")) == (
        "Monitoring non configuré : le token d'accès n'est pas défini pour cette interface."
    )


def test_monitoring_configuration_error_is_an_api_error() -> None:
    """Une seule clause `except ApiError` suffit côté interface."""
    assert issubclass(ApiConfigurationError, ApiError)


def test_monitoring_connection_message() -> None:
    assert format_monitoring_error(ApiConnectionError("x")) == (
        "Impossible de joindre l'API de monitoring. Vérifiez qu'elle est démarrée."
    )


def test_monitoring_timeout_message() -> None:
    assert format_monitoring_error(ApiTimeoutError("x")) == (
        "L'API de monitoring n'a pas répondu à temps. Réessayez dans un instant."
    )


def test_monitoring_invalid_response_message() -> None:
    assert format_monitoring_error(ApiInvalidResponseError("x")) == (
        "L'API de monitoring a renvoyé une réponse inattendue."
    )


def test_monitoring_401_message() -> None:
    exc = ApiHttpError(401, "unauthorized", "Authentication required.")
    assert format_monitoring_error(exc) == (
        "Accès refusé : le token de monitoring n'est pas accepté par l'API."
    )


@pytest.mark.parametrize("code", ["monitoring_unavailable", "unknown_error"])
def test_monitoring_503_message_depends_on_status_only(code: str) -> None:
    """Un 503 sans corps `ErrorResponse` (proxy, hébergeur) a le même message."""
    exc = ApiHttpError(503, code, "Monitoring is unavailable.")
    assert format_monitoring_error(exc) == "Le monitoring est momentanément indisponible côté API."


@pytest.mark.parametrize("status_code", [422, 500, 404])
def test_monitoring_unexpected_http_error_shows_its_code(status_code: int) -> None:
    exc = ApiHttpError(status_code, "internal_error", "Internal server error.")
    assert format_monitoring_error(exc) == (
        f"L'API de monitoring a renvoyé une erreur inattendue (code {status_code})."
    )


def test_monitoring_unknown_api_error_has_generic_message() -> None:
    assert format_monitoring_error(ApiError("x")) == (
        "Erreur inattendue lors de l'appel à l'API de monitoring."
    )


def test_monitoring_messages_never_repeat_api_details() -> None:
    """Le message et les détails renvoyés par l'API ne sont jamais réaffichés."""
    exc = ApiHttpError(
        500,
        "internal_error",
        "secret internal detail",
        details=[{"field": "query.days", "type": "x", "message": "secret field detail"}],
    )
    out = format_monitoring_error(exc)
    assert "secret" not in out
