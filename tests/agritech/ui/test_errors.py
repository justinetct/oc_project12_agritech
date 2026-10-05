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


# --- 422 : un message par champ refusé, bornes lues dans /context ---

# Libellés et bornes physiques tels que les pages les reçoivent de /predict/context
# et /recommend/context.
_PREDICT_LABELS = {"rainfall_mm": "Pluie", "temperature_celsius": "Température"}
_PREDICT_BOUNDS = {
    "rainfall_mm": {"min": 0.0, "max": None, "unit": "mm"},
    "temperature_celsius": {"min": -50.0, "max": 60.0, "unit": "°C"},
}
_RECOMMEND_LABELS = {
    "average_temperature_celsius": "Température",
    "annual_rainfall_mm": "Pluie",
    "average_annual_pesticides_tons": "Pesticides",
}
_RECOMMEND_BOUNDS = {
    "average_temperature_celsius": {"min": -50.0, "max": 60.0, "unit": "°C"},
    "annual_rainfall_mm": {"min": 0.0, "max": None, "unit": "mm"},
    "average_annual_pesticides_tons": {"min": 0.0, "max": None, "unit": "t"},
}

_RAIN = "Pluie : la valeur doit être supérieure ou égale à 0\u00a0mm."
_TEMPERATURE = "Température : la valeur doit être comprise entre −50 et 60\u00a0°C."


def _validation_error(*details: tuple[str, str]) -> ApiHttpError:
    """422 au format de l'API : un détail ``(field, type)`` par champ refusé."""
    return ApiHttpError(
        422,
        "validation_error",
        "Request payload is invalid.",
        details=[
            {"field": field, "type": kind, "message": "Input should be ..."} for field, kind in details
        ],
    )


def _lines(*lines: str) -> str:
    return "Certaines valeurs sont invalides :\n" + "\n".join(f"- {line}" for line in lines)


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (("body.rainfall_mm", "greater_than_equal"), _RAIN),
        (("body.temperature_celsius", "greater_than_equal"), _TEMPERATURE),
        (("body.temperature_celsius", "less_than_equal"), _TEMPERATURE),
    ],
    ids=["pluie-negative", "temperature-sous-50", "temperature-au-dessus-60"],
)
def test_predict_validation_error_names_the_field_and_accepted_values(detail, expected) -> None:
    out = format_api_error(_validation_error(detail), _PREDICT_LABELS, _PREDICT_BOUNDS)
    assert out == _lines(expected)
    assert "body." not in out and "Input should" not in out


def test_predict_validation_error_lists_every_refused_field_in_api_order() -> None:
    exc = _validation_error(
        ("body.rainfall_mm", "greater_than_equal"),
        ("body.temperature_celsius", "less_than_equal"),
    )
    assert format_api_error(exc, _PREDICT_LABELS, _PREDICT_BOUNDS) == _lines(_RAIN, _TEMPERATURE)


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (("body.conditions.average_temperature_celsius", "less_than_equal"), _TEMPERATURE),
        (("body.conditions.annual_rainfall_mm", "greater_than_equal"), _RAIN),
        (
            ("body.conditions.average_annual_pesticides_tons", "greater_than_equal"),
            "Pesticides : la valeur doit être supérieure ou égale à 0\u00a0t.",
        ),
    ],
    ids=["temperature", "pluie-negative", "pesticides-negatifs"],
)
def test_recommend_validation_error_names_the_condition_and_accepted_values(detail, expected) -> None:
    out = format_api_error(_validation_error(detail), _RECOMMEND_LABELS, _RECOMMEND_BOUNDS)
    assert out == _lines(expected)


def test_non_finite_value_shows_the_accepted_values() -> None:
    exc = _validation_error(("body.rainfall_mm", "finite_number"))
    assert format_api_error(exc, _PREDICT_LABELS, _PREDICT_BOUNDS) == _lines(_RAIN)


@pytest.mark.parametrize(
    ("kind", "bounds"),
    [("float_parsing", _PREDICT_BOUNDS), ("greater_than_equal", {})],
    ids=["autre-type-erreur", "bornes-absentes"],
)
def test_known_field_without_usable_rule_stays_generic_for_that_field(kind, bounds) -> None:
    exc = _validation_error(("body.rainfall_mm", kind))
    assert format_api_error(exc, _PREDICT_LABELS, bounds) == _lines("Pluie : la valeur n’a pas été acceptée.")


@pytest.mark.parametrize(
    "details",
    [
        None,
        [],
        [{"field": "body.iso3", "type": "unknown_country", "message": "Country not served: ZZZ"}],
        ["pas un objet", {"type": "greater_than_equal"}],
    ],
    ids=["sans-details", "details-vides", "champ-inconnu", "details-mal-formes"],
)
def test_validation_error_without_usable_details_falls_back_to_generic(details) -> None:
    exc = ApiHttpError(422, "validation_error", "Request payload is invalid.", details=details)
    assert format_api_error(exc, _PREDICT_LABELS, _PREDICT_BOUNDS) == (
        "Les valeurs envoyées n’ont pas été acceptées."
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ApiHttpError(503, "model_unavailable", "Model is unavailable."),
            "Le modèle est momentanément indisponible. Réessayez dans un instant.",
        ),
        (
            ApiHttpError(500, "internal_error", "Internal server error."),
            "Le service a rencontré une erreur. Réessayez dans un instant.",
        ),
        (
            ApiTimeoutError("x"),
            "Le service de calcul n'a pas répondu à temps. Réessayez dans un instant.",
        ),
    ],
    ids=["503", "500", "timeout"],
)
def test_other_errors_are_unchanged_when_labels_are_given(exc, expected) -> None:
    assert format_api_error(exc, _PREDICT_LABELS, _PREDICT_BOUNDS) == expected


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
