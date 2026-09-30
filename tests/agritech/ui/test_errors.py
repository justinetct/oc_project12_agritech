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
    assert "n'a pas répondu" in format_api_error(ApiTimeoutError("x"))


def test_format_connection() -> None:
    assert "joindre" in format_api_error(ApiConnectionError("x"))


def test_format_invalid_response() -> None:
    assert "inattendue" in format_api_error(ApiInvalidResponseError("x"))


def test_format_http_error_without_details() -> None:
    exc = ApiHttpError(503, "model_unavailable", "Model is unavailable.")
    out = format_api_error(exc)
    assert "503" in out
    assert "Model is unavailable." in out
    assert "\n" not in out


def test_format_http_error_with_details() -> None:
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
    assert "422" in out
    assert "body.rainfall_mm" in out
    assert "must be >= 0" in out


def test_api_http_error_default_details_is_empty_list() -> None:
    exc = ApiHttpError(500, "internal_error", "Internal server error.")
    assert exc.details == []
    assert exc.status_code == 500
    assert exc.code == "internal_error"
    assert exc.message == "Internal server error."
