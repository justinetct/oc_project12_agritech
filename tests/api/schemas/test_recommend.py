"""Tests des schémas Pydantic de /recommend : `RecommendRequest`, `RecommendResponse`,
`RecommendContextResponse` et leurs sous-modèles.

À cette sous-étape, le routeur `/recommend` n'existe pas encore : ces tests
vérifient uniquement le comportement des schémas (validation d'entrée et forme
de sortie), pas la couche HTTP. Les tests HTTP arriveront quand la route sera
ajoutée.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agritech.api.schemas.common import VariableSchema
from agritech.api.schemas.recommend import (
    CountryContext,
    CountryEntry,
    Recommendation,
    RecommendConditions,
    RecommendConditionValues,
    RecommendContextResponse,
    RecommendRequest,
    RecommendResponse,
)


VALID_ISO3 = "FRA"

VALID_CONDITIONS = {
    "average_temperature_celsius": 15.2,
    "annual_rainfall_mm": 900.0,
    "average_annual_pesticides_tons": 60_000.0,
}


def _valid_condition_values() -> dict:
    """Trois conditions publiques toutes présentes, in-domain."""
    return {
        "average_temperature_celsius": 11.83,
        "annual_rainfall_mm": 867.0,
        "average_annual_pesticides_tons": 65_103.33,
    }


def _valid_recommendations(count: int = 10) -> list[dict]:
    """Liste de `count` recommandations bien formées, rangs 1..count."""
    return [
        {
            "rank": i,
            "crop": f"Crop{i}",
            "predicted_yield_tons_per_hectare": 40.0 - i,
            "observed_in_country": i % 2 == 0,
        }
        for i in range(1, count + 1)
    ]


# ===========================================================================
# RecommendRequest — validation d'entrée
# ===========================================================================


def test_valid_request_iso3_only():
    """Payload minimal `{iso3: 'FRA'}` : accepté, `conditions=None`."""
    request = RecommendRequest(iso3=VALID_ISO3)
    assert request.iso3 == "FRA"
    assert request.conditions is None


def test_valid_request_empty_conditions_object():
    """`{iso3, conditions: {}}` : accepté, tous les champs de conditions sont `None`."""
    request = RecommendRequest(iso3=VALID_ISO3, conditions={})
    assert request.conditions is not None
    assert request.conditions.average_temperature_celsius is None
    assert request.conditions.annual_rainfall_mm is None
    assert request.conditions.average_annual_pesticides_tons is None


def test_valid_request_partial_condition_temperature_only():
    """Un seul champ dans `conditions` (température) : accepté."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_temperature_celsius": 15.2},
    )
    assert request.conditions.average_temperature_celsius == 15.2
    assert request.conditions.annual_rainfall_mm is None
    assert request.conditions.average_annual_pesticides_tons is None


def test_valid_request_partial_condition_rainfall_only():
    """Un seul champ dans `conditions` (pluie) : accepté."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"annual_rainfall_mm": 900.0},
    )
    assert request.conditions.annual_rainfall_mm == 900.0
    assert request.conditions.average_temperature_celsius is None


def test_valid_request_partial_condition_pesticides_only():
    """Un seul champ dans `conditions` (pesticides) : accepté."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_annual_pesticides_tons": 60_000.0},
    )
    assert request.conditions.average_annual_pesticides_tons == 60_000.0


def test_valid_request_all_conditions_present():
    """Les trois champs de `conditions` remplis : accepté."""
    request = RecommendRequest(iso3=VALID_ISO3, conditions=VALID_CONDITIONS)
    assert request.conditions.average_temperature_celsius == 15.2
    assert request.conditions.annual_rainfall_mm == 900.0
    assert request.conditions.average_annual_pesticides_tons == 60_000.0


def test_valid_request_condition_field_null_accepted():
    """Un champ explicitement `null` est équivalent à un champ absent."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_temperature_celsius": None},
    )
    assert request.conditions.average_temperature_celsius is None


# --- iso3 : validation de format ---


def test_iso3_lowercase_raises():
    """`iso3='fra'` : rejeté par le pattern `^[A-Z]{3}$`."""
    with pytest.raises(ValidationError):
        RecommendRequest(iso3="fra")


def test_iso3_wrong_length_short_raises():
    """`iso3='FR'` : rejeté (2 lettres au lieu de 3)."""
    with pytest.raises(ValidationError):
        RecommendRequest(iso3="FR")


def test_iso3_wrong_length_long_raises():
    """`iso3='FRAN'` : rejeté (4 lettres)."""
    with pytest.raises(ValidationError):
        RecommendRequest(iso3="FRAN")


def test_iso3_with_digits_raises():
    """`iso3='F1A'` : rejeté (chiffre dans le code)."""
    with pytest.raises(ValidationError):
        RecommendRequest(iso3="F1A")


def test_iso3_empty_raises():
    """`iso3=''` : rejeté."""
    with pytest.raises(ValidationError):
        RecommendRequest(iso3="")


def test_iso3_missing_raises():
    """`iso3` absent : rejeté (champ obligatoire)."""
    with pytest.raises(ValidationError):
        RecommendRequest()  # type: ignore[call-arg]


# --- extra="forbid" ---


def test_extra_root_field_raises():
    """Champ inconnu au niveau racine (`year`) : rejeté."""
    with pytest.raises(ValidationError):
        RecommendRequest(iso3=VALID_ISO3, year=2015)  # type: ignore[call-arg]


def test_extra_conditions_field_raises():
    """Champ inconnu dans `conditions` : rejeté."""
    with pytest.raises(ValidationError):
        RecommendRequest(
            iso3=VALID_ISO3,
            conditions={"average_temperature_celsius": 15.2, "unknown": 1},
        )


# --- Bornes physiques : température ---


def test_conditions_temperature_below_physical_min_raises():
    """`average_temperature_celsius=-51` : rejeté (borne physique -50)."""
    with pytest.raises(ValidationError):
        RecommendRequest(
            iso3=VALID_ISO3,
            conditions={"average_temperature_celsius": -51.0},
        )


def test_conditions_temperature_above_physical_max_raises():
    """`average_temperature_celsius=61` : rejeté (borne physique 60)."""
    with pytest.raises(ValidationError):
        RecommendRequest(
            iso3=VALID_ISO3,
            conditions={"average_temperature_celsius": 61.0},
        )


def test_conditions_temperature_at_physical_min_accepted():
    """`average_temperature_celsius=-50` : accepté (borne inclusive)."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_temperature_celsius": -50.0},
    )
    assert request.conditions.average_temperature_celsius == -50.0


def test_conditions_temperature_at_physical_max_accepted():
    """`average_temperature_celsius=60` : accepté (borne inclusive)."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_temperature_celsius": 60.0},
    )
    assert request.conditions.average_temperature_celsius == 60.0


# --- Bornes physiques : pluie ---


def test_conditions_rainfall_negative_raises():
    """`annual_rainfall_mm=-0.1` : rejeté."""
    with pytest.raises(ValidationError):
        RecommendRequest(
            iso3=VALID_ISO3,
            conditions={"annual_rainfall_mm": -0.1},
        )


def test_conditions_rainfall_zero_accepted():
    """`annual_rainfall_mm=0` : accepté (borne inclusive)."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"annual_rainfall_mm": 0.0},
    )
    assert request.conditions.annual_rainfall_mm == 0.0


# --- Bornes physiques : pesticides ---


def test_conditions_pesticides_negative_raises():
    """`average_annual_pesticides_tons=-1` : rejeté."""
    with pytest.raises(ValidationError):
        RecommendRequest(
            iso3=VALID_ISO3,
            conditions={"average_annual_pesticides_tons": -1.0},
        )


def test_conditions_pesticides_zero_accepted():
    """`average_annual_pesticides_tons=0` : accepté (borne inclusive)."""
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_annual_pesticides_tons": 0.0},
    )
    assert request.conditions.average_annual_pesticides_tons == 0.0


# --- Hors domaine d'apprentissage mais physiquement valide ---


def test_conditions_out_of_training_domain_but_physical_valid_accepted():
    """`average_temperature_celsius=45` (hors [2.57, 30.25] train) : accepté par Pydantic.

    Le rejet éventuel se fait au niveau serving (flag `out_of_training_domain`),
    jamais dans les schémas.
    """
    request = RecommendRequest(
        iso3=VALID_ISO3,
        conditions={"average_temperature_celsius": 45.0},
    )
    assert request.conditions.average_temperature_celsius == 45.0


# ===========================================================================
# Sous-modèles de la réponse — sérialisation
# ===========================================================================


def test_recommendation_valid_serializes_expected_keys():
    """Une recommandation bien formée sérialise les 4 clés attendues."""
    rec = Recommendation(
        rank=1,
        crop="Potatoes",
        predicted_yield_tons_per_hectare=42.24,
        observed_in_country=True,
    )
    assert set(rec.model_dump()) == {
        "rank",
        "crop",
        "predicted_yield_tons_per_hectare",
        "observed_in_country",
    }


def test_recommendation_rank_below_1_raises():
    """`rank=0` : rejeté (ge=1)."""
    with pytest.raises(ValidationError):
        Recommendation(rank=0, crop="X", predicted_yield_tons_per_hectare=1.0, observed_in_country=False)


def test_recommendation_rank_above_10_raises():
    """`rank=11` : rejeté (le=10)."""
    with pytest.raises(ValidationError):
        Recommendation(rank=11, crop="X", predicted_yield_tons_per_hectare=1.0, observed_in_country=False)


def test_country_context_serializes_both_blocks():
    """`CountryContext` sérialise `country_defaults` et `effective_conditions`."""
    ctx = CountryContext(
        country_defaults=RecommendConditionValues(**_valid_condition_values()),
        effective_conditions=RecommendConditionValues(**_valid_condition_values()),
    )
    dumped = ctx.model_dump()
    assert set(dumped) == {"country_defaults", "effective_conditions"}
    assert set(dumped["country_defaults"]) == {
        "average_temperature_celsius",
        "annual_rainfall_mm",
        "average_annual_pesticides_tons",
    }


# ===========================================================================
# RecommendResponse — sérialisation
# ===========================================================================


def _valid_response_payload() -> dict:
    return {
        "iso3": "FRA",
        "country": "France",
        "year": 2014,
        "model_version": "2.0.0",
        "recommendations": _valid_recommendations(10),
        "context": {
            "country_defaults": _valid_condition_values(),
            "effective_conditions": _valid_condition_values(),
        },
        "out_of_training_domain": False,
    }


def test_response_defaults_unit_and_notes():
    """Sans passer `unit` ni `notes`, valeurs par défaut correctes."""
    response = RecommendResponse(**_valid_response_payload())
    assert response.unit == "t/ha"
    assert response.notes == []


def test_response_serializes_expected_keys():
    """La sérialisation contient exactement les 9 clés du contrat public."""
    response = RecommendResponse(**_valid_response_payload())
    assert set(response.model_dump()) == {
        "iso3",
        "country",
        "year",
        "unit",
        "model_version",
        "recommendations",
        "context",
        "out_of_training_domain",
        "notes",
    }


def test_response_recommendations_too_few_raises():
    """Moins de 10 recommandations : rejeté."""
    payload = _valid_response_payload()
    payload["recommendations"] = _valid_recommendations(9)
    with pytest.raises(ValidationError):
        RecommendResponse(**payload)


def test_response_recommendations_too_many_raises():
    """Plus de 10 recommandations : rejeté."""
    payload = _valid_response_payload()
    payload["recommendations"] = _valid_recommendations(11)
    with pytest.raises(ValidationError):
        RecommendResponse(**payload)


def test_response_recommendations_exactly_10_accepted():
    """Exactement 10 recommandations : accepté."""
    response = RecommendResponse(**_valid_response_payload())
    assert len(response.recommendations) == 10


# ===========================================================================
# RecommendContextResponse — sérialisation
# ===========================================================================


def _valid_context_payload(crops_count: int = 10) -> dict:
    return {
        "year": 2014,
        "target_year_note": "The model has been trained on years 1991-2013. Year 2014 is a technical convention.",
        "crops": [f"Crop{i}" for i in range(1, crops_count + 1)],
        "countries": [{"iso3": "FRA", "country": "France"}, {"iso3": "USA", "country": "United States"}],
        "physical_bounds": {
            "average_temperature_celsius":    {"min": -50.0, "max": 60.0, "unit": "°C"},
            "annual_rainfall_mm":             {"min":   0.0, "max": None, "unit": "mm"},
            "average_annual_pesticides_tons": {"min":   0.0, "max": None, "unit": "t"},
        },
        "training_domain": {
            "average_temperature_celsius":    {"min":  2.567, "max":    30.247, "unit": "°C"},
            "annual_rainfall_mm":             {"min": 51.0,   "max":  3240.0,   "unit": "mm"},
            "average_annual_pesticides_tons": {"min":  0.04,  "max": 1_783_762.0, "unit": "t"},
        },
    }


def test_context_response_serializes_expected_keys():
    """`RecommendContextResponse` : les 6 clés attendues."""
    schema = RecommendContextResponse(**_valid_context_payload())
    assert set(schema.model_dump()) == {
        "year",
        "target_year_note",
        "crops",
        "countries",
        "physical_bounds",
        "training_domain",
    }


def test_context_response_countries_are_country_entry_instances():
    """`countries` est bien une liste de `CountryEntry` typés."""
    schema = RecommendContextResponse(**_valid_context_payload())
    assert all(isinstance(entry, CountryEntry) for entry in schema.countries)
    assert schema.countries[0].iso3 == "FRA"


def test_context_response_crops_less_than_10_raises():
    """Moins de 10 cultures : rejeté."""
    payload = _valid_context_payload(crops_count=9)
    with pytest.raises(ValidationError):
        RecommendContextResponse(**payload)


def test_context_response_crops_more_than_10_raises():
    """Plus de 10 cultures : rejeté."""
    payload = _valid_context_payload(crops_count=11)
    with pytest.raises(ValidationError):
        RecommendContextResponse(**payload)


def test_context_response_bounds_use_shared_variable_schema():
    """`physical_bounds` et `training_domain` sont bien des `VariableSchema`."""
    schema = RecommendContextResponse(**_valid_context_payload())
    for value in schema.physical_bounds.values():
        assert isinstance(value, VariableSchema)
    for value in schema.training_domain.values():
        assert isinstance(value, VariableSchema)
    assert schema.physical_bounds["annual_rainfall_mm"].max is None
