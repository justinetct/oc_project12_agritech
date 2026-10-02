"""Tests du parcours Streamlit /predict.

Les appels API sont interceptés en remplaçant les fonctions du module
``agritech.ui.api_client`` avant chaque exécution de la page. Aucune
API réelle n'est démarrée.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from agritech.ui import api_client
from agritech.ui.errors import INVALID_CONTEXT_MESSAGE, ApiConnectionError, ApiHttpError


APP_DIR = Path(__file__).resolve().parents[3] / "streamlit_app"
PAGE = str(APP_DIR / "views" / "predict.py")
ENTRYPOINT = str(APP_DIR / "app.py")
LAST_KEY = "predict_last"


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

# Contexte renvoyé aujourd'hui par GET /predict/context (domaine plus étroit que
# _CONTEXT_OK) : sert aux tests des valeurs hors domaine d'apprentissage.
_CONTEXT_API: dict[str, Any] = {
    "physical_bounds": _CONTEXT_OK["physical_bounds"],
    "training_domain": {
        "rainfall_mm": {"min": 100.0, "max": 1000.0, "unit": "mm"},
        "temperature_celsius": {"min": 15.0, "max": 40.0, "unit": "°C"},
    },
}

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


def _markdown_containing(at: AppTest, marker: str) -> str:
    """Corps du premier bloc Markdown contenant ``marker`` (bandeau, panneau…)."""
    for block in at.markdown:
        if marker in block.value:
            return block.value
    raise AssertionError(f"aucun bloc Markdown ne contient {marker!r}")


def test_context_loads_and_page_shows_header_panel_and_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    at = AppTest.from_file(PAGE).run()
    assert not at.exception

    header = _markdown_containing(at, "ag-topbar")
    assert "Predict" in header
    assert 'href="/recommend"' in header

    panel = _markdown_containing(at, "ag-panel")
    assert "Votre parcelle." in panel
    assert "t/ha" not in panel  # aucun faux résultat avant la première estimation

    assert len(at.number_input) == 2
    # valeurs initiales = milieu du training_domain
    assert at.number_input[0].value == _RAINFALL_DEFAULT
    assert at.number_input[1].value == _TEMPERATURE_DEFAULT

    # Un seul bloc : les 4 conditions envoyées au modèle, sans contexte descriptif
    # ni sous-titre ; aucune phrase sous le bouton avant la première estimation.
    assert "Conditions de votre parcelle" in _markdown_containing(at, "ag-sec-title")
    assert "ag-sec-sub" not in _markdown_containing(at, "ag-sec-title")
    assert not any("Contexte de votre parcelle" in block.value for block in at.markdown)
    assert not any("ag-cta-note" in block.value for block in at.markdown)

    # Seuls sélecteurs de la page : les 2 choix Oui / Non, sur « Non ».
    assert len(at.button_group) == 2
    assert at.button_group(key="fertilizer_used").value == "Non"
    assert at.button_group(key="irrigation_used").value == "Non"


def test_sliders_follow_the_training_domain_and_fields_the_physical_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_API))
    at = AppTest.from_file(PAGE).run()
    assert not at.exception

    rain_field, rain_slider = at.number_input(key="rainfall_mm"), at.slider(key="rainfall_mm_slider")
    temp_field, temp_slider = at.number_input(key="temperature_celsius"), at.slider(key="temperature_celsius_slider")
    # Curseurs : domaine d'apprentissage.
    assert (rain_slider.min, rain_slider.max) == (100.0, 1000.0)
    assert (temp_slider.min, temp_slider.max) == (15.0, 40.0)
    # Champs : bornes physiques. Sans maximum, Streamlit utilise le plus grand flottant.
    assert (rain_field.min, rain_field.max) == (0.0, sys.float_info.max)
    assert (temp_field.min, temp_field.max) == (-50.0, 60.0)


def test_temperature_outside_training_domain_is_kept_and_sent_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_API))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK | {
            "out_of_training_domain": True,
            "notes": ["temperature_celsius is out of training domain"],
        }

    monkeypatch.setattr(api_client, "post_predict", _predict)

    at = AppTest.from_file(PAGE).run()
    at.number_input(key="temperature_celsius").set_value(12.0).run()
    # 12 °C reste dans le champ ; seul le curseur se met en butée à 15 °C.
    assert at.number_input(key="temperature_celsius").value == 12.0
    assert at.slider(key="temperature_celsius_slider").value == 15.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["temperature_celsius"] == 12.0

    panel = _markdown_containing(at, "ag-panel")
    assert "Hors domaine d'apprentissage" in panel
    assert "Température&nbsp;: 12&nbsp;°C" in panel
    assert "15&nbsp;–&nbsp;40&nbsp;°C" in panel


def test_rainfall_outside_training_domain_stays_editable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_API))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK

    monkeypatch.setattr(api_client, "post_predict", _predict)

    at = AppTest.from_file(PAGE).run()
    at.number_input(key="rainfall_mm").set_value(50.0).run()  # sous le domaine
    assert at.number_input(key="rainfall_mm").value == 50.0
    assert at.slider(key="rainfall_mm_slider").value == 100.0
    at.number_input(key="rainfall_mm").set_value(1500.0).run()  # au-dessus du domaine
    assert at.number_input(key="rainfall_mm").value == 1500.0
    assert at.slider(key="rainfall_mm_slider").value == 1000.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["rainfall_mm"] == 1500.0


def test_moving_the_slider_after_an_out_of_domain_value_resyncs_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_API))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK

    monkeypatch.setattr(api_client, "post_predict", _predict)

    at = AppTest.from_file(PAGE).run()
    at.number_input(key="temperature_celsius").set_value(12.0).run()
    at.slider(key="temperature_celsius_slider").set_value(20.0).run()
    # La valeur du curseur redevient celle du champ, et donc celle envoyée.
    assert at.number_input(key="temperature_celsius").value == 20.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["temperature_celsius"] == 20.0


def test_physically_invalid_values_are_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_API))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK

    monkeypatch.setattr(api_client, "post_predict", _predict)

    # Le navigateur refuse ces saisies ; côté serveur, Streamlit écarte aussi toute
    # valeur hors des bornes du champ (règle existante, inchangée).
    at = AppTest.from_file(PAGE).run()
    at.number_input(key="temperature_celsius").set_value(70.0).run()  # > 60 °C
    at.number_input(key="rainfall_mm").set_value(-5.0).run()  # < 0 mm
    assert at.number_input(key="temperature_celsius").value != 70.0
    assert at.number_input(key="rainfall_mm").value != -5.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["temperature_celsius"] != 70.0
    assert seen["payload"]["rainfall_mm"] != -5.0


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
    assert at.error[0].value == INVALID_CONTEXT_MESSAGE
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
    assert at.error[0].value == INVALID_CONTEXT_MESSAGE
    assert len(at.number_input) == 0


def test_submit_success_shows_yield_in_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK

    monkeypatch.setattr(api_client, "post_predict", _predict)

    at = AppTest.from_file(PAGE).run()
    at.button_group(key="fertilizer_used").set_value("Oui")
    # irrigation_used reste « Non » ; valeurs numériques : défauts training_domain
    at.button[0].click().run()

    assert not at.exception
    expected_payload = {
        "rainfall_mm": _RAINFALL_DEFAULT,
        "temperature_celsius": _TEMPERATURE_DEFAULT,
        "fertilizer_used": True,
        "irrigation_used": False,
    }
    assert seen["payload"] == expected_payload

    panel = _markdown_containing(at, "ag-panel")
    assert "Votre rendement estimé" in panel
    assert "4,82" in panel
    assert "t/ha" in panel
    assert "Modèle" not in panel  # version technique, absente de l'interface
    assert "Hors domaine" not in panel
    # Conditions utilisées = valeurs envoyées à POST /predict.
    assert "Conditions utilisées" in panel
    assert "<span>Fertilisation</span><b>Oui</b>" in panel
    assert "<span>Irrigation</span><b>Non</b>" in panel
    # Après une estimation, une invitation discrète apparaît sous le bouton.
    assert "Ajustez les conditions" in _markdown_containing(at, "ag-cta-note")

    # Le résultat est conservé pour les exécutions suivantes (rerun).
    assert at.session_state[LAST_KEY]["payload"] == expected_payload
    assert not at.run().exception
    assert "4,82" in _markdown_containing(at, "ag-panel")


def test_slider_and_field_share_the_value_sent_to_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))

    seen: dict[str, Any] = {}

    def _predict(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _PREDICT_OK

    monkeypatch.setattr(api_client, "post_predict", _predict)

    at = AppTest.from_file(PAGE).run()
    # curseur → champ
    at.slider(key="rainfall_mm_slider").set_value(800.0).run()
    assert at.number_input(key="rainfall_mm").value == 800.0
    # champ → curseur
    at.number_input(key="temperature_celsius").set_value(31.5).run()
    assert at.slider(key="temperature_celsius_slider").value == 31.5
    assert "payload" not in seen  # bouger un champ ou un curseur n'appelle pas l'API

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["rainfall_mm"] == 800.0
    assert seen["payload"]["temperature_celsius"] == 31.5


def _stub_predict_from_rainfall(sent: list[dict[str, Any]]):
    """Rendement qui dépend de la pluie envoyée : distingue un ancien résultat d'un nouveau."""

    def _call(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(dict(body))
        return _PREDICT_OK | {"yield_tons_per_hectare": body["rainfall_mm"] / 100}

    return _call


def test_changing_an_input_after_estimation_hides_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    monkeypatch.setattr(api_client, "post_predict", _stub_predict_from_rainfall([]))

    at = AppTest.from_file(PAGE).run()
    at.button[0].click().run()
    assert "Votre rendement estimé" in _markdown_containing(at, "ag-panel")

    at.number_input(key="rainfall_mm").set_value(800.0).run()
    assert not at.exception
    panel = _markdown_containing(at, "ag-panel")
    assert "Votre parcelle." in panel
    assert "Votre rendement estimé" not in panel and "Conditions utilisées" not in panel
    assert not any("ag-cta-note" in block.value for block in at.markdown)

    # Retour exact aux valeurs de l'estimation : son résultat redevient valable.
    at.number_input(key="rainfall_mm").set_value(_RAINFALL_DEFAULT).run()
    assert "15,50" in _markdown_containing(at, "ag-panel")


def test_new_estimation_after_a_change_shows_the_new_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    monkeypatch.setattr(api_client, "post_predict", _stub_predict_from_rainfall(sent))

    at = AppTest.from_file(PAGE).run()
    at.button[0].click().run()
    at.number_input(key="rainfall_mm").set_value(800.0).run()
    at.button_group(key="irrigation_used").set_value("Oui").run()
    at.button[0].click().run()

    assert not at.exception
    assert sent[-1] == {
        "rainfall_mm": 800.0,
        "temperature_celsius": _TEMPERATURE_DEFAULT,
        "fertilizer_used": False,
        "irrigation_used": True,
    }
    panel = _markdown_containing(at, "ag-panel")
    assert "8,00" in panel and "15,50" not in panel
    assert "<span>Irrigation</span><b>Oui</b>" in panel


def test_submit_out_of_training_domain_shows_warning_in_panel(
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
    panel = _markdown_containing(at, "ag-panel")
    assert "Hors domaine d'apprentissage" in panel
    assert "Pluie" in panel
    assert "Température" not in panel.split('class="ag-warn"')[1]  # seule la pluie est signalée
    assert "interprétez-la avec prudence" in panel


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
    assert at.error[0].value == "Le modèle est momentanément indisponible. Réessayez dans un instant."
    # aucun résultat affiché après une erreur
    assert "t/ha" not in _markdown_containing(at, "ag-panel")


def test_submit_response_without_numeric_yield_shows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    monkeypatch.setattr(
        api_client, "post_predict", _stub_predict(_PREDICT_OK | {"yield_tons_per_hectare": None})
    )

    at = AppTest.from_file(PAGE).run()
    at.button[0].click().run()

    assert not at.exception
    assert at.error
    assert "inattendue" in at.error[0].value
    assert "t/ha" not in _markdown_containing(at, "ag-panel")


def test_streamlit_app_has_no_pages_directory() -> None:
    assert not (APP_DIR / "pages").exists()


def test_entrypoint_opens_predict_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "get_predict_context", _stub_context(_CONTEXT_OK))
    at = AppTest.from_file(ENTRYPOINT).run()
    assert not at.exception
    assert "Votre parcelle." in _markdown_containing(at, "ag-panel")


def test_zero_mark_is_drawn_only_when_the_slider_range_crosses_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zero_marks(context: dict[str, Any]) -> list[str]:
        monkeypatch.setattr(api_client, "get_predict_context", _stub_context(context))
        at = AppTest.from_file(PAGE).run()
        assert not at.exception
        return [element.proto.body for element in at.get("html") if 'content:"0 °C"' in element.proto.body]

    # Domaine actuel de l'API (15 à 40 °C) : la plage du curseur ne passe pas par zéro.
    assert zero_marks(_CONTEXT_API) == []
    # Domaine qui passe par zéro (-10 à 40 °C) : repère « 0 °C » sur le curseur de température.
    crossing = _CONTEXT_API | {
        "training_domain": _CONTEXT_API["training_domain"]
        | {"temperature_celsius": {"min": -10.0, "max": 40.0, "unit": "°C"}}
    }
    marks = zero_marks(crossing)
    assert len(marks) == 1 and ".st-key-temperature_celsius_slider" in marks[0]
