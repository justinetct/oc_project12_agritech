"""Tests de ``agritech.ui.errors``."""

from __future__ import annotations

from agritech.ui.errors import (
    ApiConnectionError,
    ApiHttpError,
    ApiInvalidResponseError,
    ApiTimeoutError,
    format_api_error,
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
