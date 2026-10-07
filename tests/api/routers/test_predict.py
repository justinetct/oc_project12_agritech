"""Tests HTTP de `POST /predict` et `GET /predict/context`.

Chaque test s'exécute dans le cycle de vie normal de l'application, via
`with TestClient(app) as client:` : le lifespan charge le modèle avant la
première requête, comme en production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agritech.api.main import app
from agritech.predict_config import PREDICT_MODEL_VERSION


VALID_PAYLOAD = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}


def test_post_predict_in_domain_returns_200_no_notes():
    """Payload nominal : 200, réponse au contrat complet, aucune note, `model_version` du metadata."""
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)

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
    assert body["unit"] == "t/ha"
    assert body["model_version"] == PREDICT_MODEL_VERSION
    assert body["out_of_training_domain"] is False
    assert body["notes"] == []


@pytest.mark.parametrize("field", ["rainfall_mm", "temperature_celsius"])
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_post_predict_non_finite_value_returns_422_finite_number(field, literal):
    """Valeur non finie dans le JSON brut → 422 `finite_number`, jamais 500.

    Le décodeur JSON de l'API accepte `NaN`, `Infinity` et `-Infinity`, et lit
    `1e400` comme l'infini : le corps est donc écrit à la main, `json=` de
    `TestClient` refusant d'encoder ces valeurs.
    """
    fields = [
        f'"{name}": {literal if name == field else value}'
        for name, value in [("rainfall_mm", "500.0"), ("temperature_celsius", "25.0")]
    ]
    body = "{" + ", ".join(fields) + ', "fertilizer_used": true, "irrigation_used": false}'

    with TestClient(app) as client:
        response = client.post(
            "/predict", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"
    assert [(d["field"], d["type"]) for d in payload["details"]] == [
        (f"body.{field}", "finite_number")
    ]


def test_post_predict_temperature_out_of_domain_returns_one_note():
    """`temperature_celsius=5` : 200 + une note pour la température."""
    payload = VALID_PAYLOAD | {"temperature_celsius": 5.0}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["out_of_training_domain"] is True
    assert body["notes"] == ["temperature_celsius is out of training domain"]


def test_post_predict_rainfall_out_of_domain_returns_one_note():
    """`rainfall_mm=50` : 200 + une note pour la pluie."""
    payload = VALID_PAYLOAD | {"rainfall_mm": 50.0}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["out_of_training_domain"] is True
    assert body["notes"] == ["rainfall_mm is out of training domain"]


def test_post_predict_both_out_of_domain_returns_two_notes():
    """`temperature_celsius=5` et `rainfall_mm=50` : 200 + deux notes ordre déterministe."""
    payload = VALID_PAYLOAD | {"rainfall_mm": 50.0, "temperature_celsius": 5.0}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["out_of_training_domain"] is True
    assert body["notes"] == [
        "rainfall_mm is out of training domain",
        "temperature_celsius is out of training domain",
    ]


def test_get_predict_context_exposes_physical_bounds_and_training_domain():
    """`GET /predict/context` : bornes physiques + domaine d'apprentissage, snake_case."""
    with TestClient(app) as client:
        response = client.get("/predict/context")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"physical_bounds", "training_domain"}

    assert set(body["physical_bounds"]) == {"rainfall_mm", "temperature_celsius"}
    assert body["physical_bounds"]["rainfall_mm"] == {"min": 0.0, "max": None, "unit": "mm"}
    assert body["physical_bounds"]["temperature_celsius"] == {
        "min": -50.0,
        "max": 60.0,
        "unit": "°C",
    }

    assert set(body["training_domain"]) == {"rainfall_mm", "temperature_celsius"}
    assert body["training_domain"]["rainfall_mm"] == {"min": 100.0, "max": 1000.0, "unit": "mm"}
    assert body["training_domain"]["temperature_celsius"] == {
        "min": 15.0,
        "max": 40.0,
        "unit": "°C",
    }


def test_predict_contract_never_mentions_crop():
    """`crop` n'apparaît ni dans la requête, ni dans la réponse, ni dans /predict/context."""
    with TestClient(app) as client:
        post_body = client.post("/predict", json=VALID_PAYLOAD).text
        context_body = client.get("/predict/context").text

    assert "crop" not in post_body.lower()
    assert "crop" not in context_body.lower()
