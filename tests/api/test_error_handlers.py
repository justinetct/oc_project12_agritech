"""Tests des handlers d'erreur 422 / 503 / 500 et de `format_validation_errors`.

Vérifie que les trois catégories d'erreur produisent une `ErrorResponse` unifiée,
qu'aucune information interne ne fuite au client, et que la fonction pure
`format_validation_errors` reformate correctement les erreurs Pydantic.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, ValidationError

from agritech.api.core import runtime
from agritech.api.error_handlers import (
    format_validation_errors,
    validation_error_body,
    validation_exception_handler,
    validation_summary,
)
from agritech.api.main import app


VALID_PAYLOAD = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}


# --- 422 : erreurs de validation Pydantic ---


def test_422_missing_field_returns_error_response():
    """Champ obligatoire absent → 422 + `validation_error` + détail sur `body.rainfall_mm`."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "rainfall_mm"}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["message"] == "Request payload is invalid."
    assert isinstance(body["details"], list) and len(body["details"]) == 1
    detail = body["details"][0]
    assert detail["field"] == "body.rainfall_mm"
    assert detail["type"] == "missing"
    assert "required" in detail["message"].lower()


def test_422_extra_field_crop_returns_error_response():
    """`crop` en trop → 422 + détail `body.crop` / `extra_forbidden`."""
    payload = VALID_PAYLOAD | {"crop": "Wheat"}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["details"][0]
    assert detail["field"] == "body.crop"
    assert detail["type"] == "extra_forbidden"


def test_422_strict_bool_int_returns_error_response():
    """`fertilizer_used = 1` → 422 + détail `bool_type`."""
    payload = VALID_PAYLOAD | {"fertilizer_used": 1}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["details"][0]
    assert detail["field"] == "body.fertilizer_used"
    assert detail["type"] == "bool_type"


def test_422_rainfall_negative_returns_error_response():
    """Pluie négative → 422 + détail `greater_than_equal`."""
    payload = VALID_PAYLOAD | {"rainfall_mm": -1.0}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["details"][0]
    assert detail["field"] == "body.rainfall_mm"
    assert detail["type"] == "greater_than_equal"


def test_422_temperature_out_of_physical_range_returns_error_response():
    """Température physiquement invalide → 422 + détail `less_than_equal`."""
    payload = VALID_PAYLOAD | {"temperature_celsius": 100.0}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["details"][0]
    assert detail["field"] == "body.temperature_celsius"
    assert detail["type"] == "less_than_equal"


def test_422_response_does_not_leak_input_payload():
    """La réponse 422 ne contient jamais le payload original (contraire à Pydantic brut)."""
    payload = VALID_PAYLOAD | {"rainfall_mm": -1.0}
    with TestClient(app) as client:
        text = client.post("/predict", json=payload).text

    # Pydantic brut aurait mis "input": -1.0 dans les détails ; nous ne le recopions pas.
    assert '"input"' not in text
    assert '"ctx"' not in text
    # Le mot "input" apparait dans les messages Pydantic ("Input should be ..."), donc
    # on verifie l'absence de la CLE JSON precisement.


# --- 503 : ModelUnavailableError ---


def test_503_post_predict_when_bundle_is_none():
    """POST `/predict` avec `runtime.bundle_predict = None` → 503 `model_unavailable`."""
    with TestClient(app) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = None
        try:
            response = client.post("/predict", json=VALID_PAYLOAD)
        finally:
            runtime.bundle_predict = saved

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "model_unavailable"
    assert body["message"] == "Model is unavailable."
    assert body["details"] is None


def test_503_get_predict_context_when_bundle_is_none():
    """GET `/predict/context` avec `bundle_predict = None` → 503 également."""
    with TestClient(app) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = None
        try:
            response = client.get("/predict/context")
        finally:
            runtime.bundle_predict = saved

    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_503_response_leaks_no_internal_information():
    """La réponse 503 ne contient ni chemin local ni détail Python interne."""
    with TestClient(app) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = None
        try:
            text = client.post("/predict", json=VALID_PAYLOAD).text
        finally:
            runtime.bundle_predict = saved

    assert "runtime" not in text.lower()
    assert "traceback" not in text.lower()
    assert "/Users/" not in text
    assert ".py" not in text


# --- 500 : catch-all Exception ---


def test_500_pipeline_exception_returns_internal_error():
    """Un pipeline qui lève → 500 `internal_error` avec message secret non exposé."""
    secret = "secret_should_not_leak_12345"

    class BrokenPipeline:
        def predict(self, X):
            raise RuntimeError(secret)

    with TestClient(app, raise_server_exceptions=False) as client:
        saved = runtime.bundle_predict
        runtime.bundle_predict = replace(saved, pipeline=BrokenPipeline())
        try:
            response = client.post("/predict", json=VALID_PAYLOAD)
        finally:
            runtime.bundle_predict = saved

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert body["message"] == "Internal server error."
    assert body["details"] is None

    text = response.text
    assert secret not in text
    assert "Traceback" not in text
    assert ".py" not in text
    assert "/Users/" not in text


# --- format_validation_errors : test unitaire pur ---


def test_format_validation_errors_transforms_loc_and_drops_input_and_ctx():
    """`format_validation_errors` produit `body.<field>`, garde `type`/`message`, jette `input`/`ctx`."""

    class Toy(BaseModel):
        rainfall_mm: float = Field(ge=0)

    # Provoque une ValidationError Pydantic identique en structure à celle de FastAPI.
    try:
        Toy(rainfall_mm=-1)
    except ValidationError as pyd_exc:
        # Emballe la ValidationError dans un RequestValidationError-like en simulant
        # `loc = ("body", "<field>")` comme FastAPI l'ajouterait.
        from fastapi.exceptions import RequestValidationError

        raw_errors = [
            {**error, "loc": ("body",) + tuple(error["loc"])} for error in pyd_exc.errors()
        ]
        req_exc = RequestValidationError(raw_errors)

    details = format_validation_errors(req_exc)

    assert len(details) == 1
    d = details[0]
    assert d.field == "body.rainfall_mm"
    assert d.type == "greater_than_equal"
    assert "greater than or equal to 0" in d.message
    # Les attributs `input` et `ctx` ne doivent PAS avoir de correspondant public.
    assert not hasattr(d, "input")
    assert not hasattr(d, "ctx")


def test_validation_error_body_is_the_body_of_the_422_handler():
    """Sans requête HTTP, `validation_error_body` donne le corps exact de la réponse 422."""
    exc = RequestValidationError(
        [{"type": "missing", "loc": ("body", "irrigation_used"), "msg": "Field required", "input": {}}]
    )

    response = asyncio.run(validation_exception_handler(None, exc))

    assert response.status_code == 422
    assert validation_error_body(exc) == json.loads(response.body)


# --- validation_summary : résumé court d'une 422 pour le monitoring ---


def test_validation_summary_names_each_refused_field_with_the_pydantic_message():
    details = [
        {"field": "body.rainfall_mm", "type": "greater_than_equal",
         "message": "Input should be greater than or equal to 0"},
        {"field": "body.conditions.average_temperature_celsius", "type": "less_than_equal",
         "message": "Input should be less than or equal to 60"},
    ]

    assert validation_summary(details) == (
        "rainfall_mm: Input should be greater than or equal to 0 · "
        "average_temperature_celsius: Input should be less than or equal to 60"
    )


@pytest.mark.parametrize(
    "details",
    [None, [], ["pas un objet"], [{"field": "body.rainfall_mm"}], [{"message": "x"}]],
    ids=["absent", "vide", "mal-forme", "sans-message", "sans-champ"],
)
def test_validation_summary_returns_none_without_usable_details(details):
    assert validation_summary(details) is None
