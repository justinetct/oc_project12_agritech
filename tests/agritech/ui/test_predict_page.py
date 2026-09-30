"""Tests du parcours Streamlit /predict.

Les appels API sont interceptés en remplaçant les fonctions du module
``agritech.ui.api_client`` avant chaque exécution de la page. Aucune
API réelle n'est démarrée.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from agritech.ui import api_client
from agritech.ui.errors import ApiConnectionError, ApiHttpError


PAGE = str(
    Path(__file__).resolve().parents[3]
    / "streamlit_app"
    / "pages"
    / "predict.py"
)


_CONTEXT_OK: dict[str, Any] = {
    "physical_bounds": {
        "rainfall_mm": {"min": 0.0, "max": None, "unit": "mm"},
        "temperature_celsius": {"min": -50.0, "max": 60.0, "unit": "°C"},
    },
    "training_domain": {
        "rainfall_mm": {"min": 100.0, "max": 3000.0, "unit": "mm"},
        "temperature_celsius": {"min": 5.0, "max": 40.0, "unit": "°C"},
    },
}

# Milieux du domaine d'apprentissage, utilisés comme valeurs initiales du form.
_RAINFALL_DEFAULT = 1550.0
_TEMPERATURE_DEFAULT = 22.5

_PREDICT_OK: dict[str, Any] = {
    "yield_tons_per_hectare": 4.82,
    "unit": "t/ha",
    "model_version": "1.0.0",
    "out_of_training_domain": False,
    "notes": [],
}


def _stub_context(payload: dict[str, Any]):
    def _call() -> dict[str, Any]:
        return payload

    return _call


def _stub_predict(payload: dict[str, Any]):
    def _call(_body: dict[str, Any]) -> dict[str, Any]:
        return payload

    return _call


def test_context_loads_and_form_is_displayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    at = AppTest.from_file(PAGE).run()
    assert not at.exception
    assert any("Prédire un rendement" in t.value for t in at.title)
    assert len(at.number_input) == 2
    assert len(at.checkbox) == 2
    # valeurs initiales = milieu du training_domain
    assert at.number_input[0].value == _RAINFALL_DEFAULT
    assert at.number_input[1].value == _TEMPERATURE_DEFAULT


def test_context_load_error_shows_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise ApiConnectionError("boom")

    monkeypatch.setattr(api_client, "get_predict_context", _raise)

    at = AppTest.from_file(PAGE).run()
    assert not at.exception
    assert at.error, "expected an st.error to be displayed"
    assert "joindre" in at.error[0].value


def test_invalid_context_shows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # training_domain manquant : structure attendue non respectée.
    invalid = {"physical_bounds": _CONTEXT_OK["physical_bounds"]}
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(invalid))

    at = AppTest.from_file(PAGE).run()
    assert not at.exception
    assert at.error
    assert "invalide" in at.error[0].value.lower()
    # aucun formulaire construit lorsque le contexte est refusé.
    assert len(at.number_input) == 0


def test_context_default_outside_physical_bounds_shows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Milieu du training_domain de la température : (100 + 200) / 2 = 150 °C,
    # hors des bornes physiques [-50, 60]. Le contexte est structurellement
    # cohérent en clés/types mais incohérent en valeurs : la page doit refuser.
    incoherent = {
        "physical_bounds": {
            "rainfall_mm": {"min": 0.0, "max": None, "unit": "mm"},
            "temperature_celsius": {"min": -50.0, "max": 60.0, "unit": "°C"},
        },
        "training_domain": {
            "rainfall_mm": {"min": 100.0, "max": 3000.0, "unit": "mm"},
            "temperature_celsius": {"min": 100.0, "max": 200.0, "unit": "°C"},
        },
    }
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(incoherent))

    at = AppTest.from_file(PAGE).run()
    assert not at.exception
    assert at.error
    message = at.error[0].value.lower()
    assert "invalide" in message
    assert "temperature_celsius" in message
    assert len(at.number_input) == 0


def test_submit_success_shows_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK

    monkeypatch.setattr(api_client, "post_predict", _predict)

    at = AppTest.from_file(PAGE).run()
    at.checkbox[0].check()  # fertilizer_used
    # irrigation_used reste False ; valeurs numériques : défauts training_domain
    at.button[0].click().run()

    assert not at.exception
    assert seen["payload"] == {
        "rainfall_mm": _RAINFALL_DEFAULT,
        "temperature_celsius": _TEMPERATURE_DEFAULT,
        "fertilizer_used": True,
        "irrigation_used": False,
    }
    assert at.metric
    assert "4.82" in at.metric[0].value
    assert "t/ha" in at.metric[0].value
    assert any("1.0.0" in c.value for c in at.caption)
    assert not at.warning


def test_submit_out_of_training_domain_shows_warning_with_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    monkeypatch.setattr(
        api_client,
        "post_predict",
        _stub_predict(
            {
                "yield_tons_per_hectare": 4.82,
                "unit": "t/ha",
                "model_version": "1.0.0",
                "out_of_training_domain": True,
                "notes": ["rainfall_mm is out of training domain"],
            }
        ),
    )
    at = AppTest.from_file(PAGE).run()
    at.button[0].click().run()

    assert not at.exception
    assert at.warning, "expected an st.warning"
    body = at.warning[0].value
    assert "domaine" in body
    assert "rainfall_mm is out of training domain" in body


def test_submit_api_error_shows_formatted_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))

    def _raise(_body: dict[str, Any]) -> None:
        raise ApiHttpError(503, "model_unavailable", "Model is unavailable.")

    monkeypatch.setattr(api_client, "post_predict", _raise)

    at = AppTest.from_file(PAGE).run()
    at.button[0].click().run()

    assert not at.exception
    assert at.error
    assert "503" in at.error[0].value
    assert "Model is unavailable." in at.error[0].value
