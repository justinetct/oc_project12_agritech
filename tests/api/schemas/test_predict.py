"""Tests des schémas Pydantic de /predict : `PredictRequest` et `PredictResponse`.

À cette sous-étape, la route `/predict` n'existe pas encore : ces tests
vérifient le comportement des schémas seuls (validation d'entrée et forme de
sortie), pas la couche HTTP. Les tests HTTP arriveront quand la route sera
ajoutée.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agritech.api.schemas.predict import PredictRequest, PredictResponse


# Payload nominal : toutes les valeurs sont dans les bornes physiques ET dans
# le domaine d'apprentissage du modèle /predict.
VALID_PAYLOAD = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}


# --- PredictRequest : validation d'entrée ---


def test_valid_request_accepts_typical_payload():
    """Un payload nominal est accepté sans transformation surprenante."""
    request = PredictRequest(**VALID_PAYLOAD)

    assert request.rainfall_mm == 500.0
    assert request.temperature_celsius == 25.0
    assert request.fertilizer_used is True
    assert request.irrigation_used is False


def test_valid_request_out_of_training_domain_is_still_accepted():
    """Une valeur physiquement valide mais hors domaine d'apprentissage passe."""
    payload = VALID_PAYLOAD | {"rainfall_mm": 50.0, "temperature_celsius": 5.0}

    request = PredictRequest(**payload)

    assert request.rainfall_mm == 50.0
    assert request.temperature_celsius == 5.0


def test_rainfall_zero_accepted():
    """La borne basse `rainfall_mm = 0` est acceptée (inclusion)."""
    request = PredictRequest(**(VALID_PAYLOAD | {"rainfall_mm": 0.0}))

    assert request.rainfall_mm == 0.0


def test_rainfall_negative_raises_validation_error():
    """Une pluie négative est rejetée."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"rainfall_mm": -1.0}))


def test_temperature_at_physical_min_accepted():
    """La borne basse `temperature_celsius = -50` est acceptée (inclusion)."""
    request = PredictRequest(**(VALID_PAYLOAD | {"temperature_celsius": -50.0}))

    assert request.temperature_celsius == -50.0


def test_temperature_below_physical_min_raises_validation_error():
    """`temperature_celsius = -51` est rejetée."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"temperature_celsius": -51.0}))


def test_temperature_at_physical_max_accepted():
    """La borne haute `temperature_celsius = 60` est acceptée (inclusion)."""
    request = PredictRequest(**(VALID_PAYLOAD | {"temperature_celsius": 60.0}))

    assert request.temperature_celsius == 60.0


def test_temperature_above_physical_max_raises_validation_error():
    """`temperature_celsius = 61` est rejetée."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"temperature_celsius": 61.0}))


def test_missing_field_raises_validation_error():
    """Un champ absent est rejeté."""
    incomplete = {key: value for key, value in VALID_PAYLOAD.items() if key != "rainfall_mm"}

    with pytest.raises(ValidationError):
        PredictRequest(**incomplete)


def test_wrong_type_raises_validation_error():
    """Un type non convertible est rejeté."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"rainfall_mm": "beaucoup"}))


def test_extra_field_raises_validation_error():
    """Un champ non déclaré (par exemple `crop`) est rejeté par `extra=\"forbid\"`."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"crop": "Wheat"}))


def test_fertilizer_used_int_raises_validation_error():
    """`fertilizer_used = 1` est rejeté par `StrictBool`."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"fertilizer_used": 1}))


def test_fertilizer_used_string_raises_validation_error():
    """`fertilizer_used = \"true\"` est rejeté par `StrictBool`."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"fertilizer_used": "true"}))


def test_irrigation_used_int_raises_validation_error():
    """`irrigation_used = 0` est rejeté par `StrictBool`."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"irrigation_used": 0}))


def test_irrigation_used_string_raises_validation_error():
    """`irrigation_used = \"false\"` est rejeté par `StrictBool`."""
    with pytest.raises(ValidationError):
        PredictRequest(**(VALID_PAYLOAD | {"irrigation_used": "false"}))


# --- PredictResponse : forme de sortie ---


def test_response_defaults_unit_and_notes():
    """Sans passer `unit` ni `notes`, les valeurs par défaut sont correctes."""
    response = PredictResponse(
        yield_tons_per_hectare=4.82,
        model_version="1.0.0",
        out_of_training_domain=False,
    )

    assert response.unit == "t/ha"
    assert response.notes == []


def test_response_serializes_expected_keys():
    """La sérialisation contient exactement les 5 clés du contrat public."""
    response = PredictResponse(
        yield_tons_per_hectare=4.82,
        model_version="1.0.0",
        out_of_training_domain=False,
    )

    assert set(response.model_dump()) == {
        "yield_tons_per_hectare",
        "unit",
        "model_version",
        "out_of_training_domain",
        "notes",
    }
