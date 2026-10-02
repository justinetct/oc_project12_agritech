"""Tests du parcours Streamlit /recommend.

Les appels API sont interceptés en remplaçant les fonctions du module
``agritech.ui.api_client`` avant chaque exécution de la page. Aucune
API réelle n'est démarrée.
"""

from __future__ import annotations

import math
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from agritech.ui import api_client
from agritech.ui.errors import INVALID_CONTEXT_MESSAGE, ApiConnectionError, ApiHttpError, ApiTimeoutError


PAGE = str(Path(__file__).resolve().parents[3] / "streamlit_app" / "views" / "recommend.py")

_CONTEXT: dict[str, Any] = {
    "year": 2014,
    "crops": ["Maize", "Potatoes", "Sweet potatoes"],
    "countries": [
        {"iso3": "FRA", "country": "France"},
        {"iso3": "MLI", "country": "Mali"},
        {"iso3": "EGY", "country": "Egypt"},
    ],
    "physical_bounds": {
        "average_temperature_celsius": {"min": -50.0, "max": 60.0, "unit": "°C"},
        "annual_rainfall_mm": {"min": 0.0, "max": None, "unit": "mm"},
        "average_annual_pesticides_tons": {"min": 0.0, "max": None, "unit": "t"},
    },
    "training_domain": {
        "average_temperature_celsius": {"min": 2.57, "max": 30.25, "unit": "°C"},
        "annual_rainfall_mm": {"min": 51.0, "max": 3240.0, "unit": "mm"},
        "average_annual_pesticides_tons": {"min": 0.04, "max": 1783667.33, "unit": "t"},
    },
    "country": None,
}

_FRA_DEFAULTS = {
    "average_temperature_celsius": 11.52,
    "annual_rainfall_mm": 867.0,
    "average_annual_pesticides_tons": 63694.63,
}

_RECOMMEND_OK: dict[str, Any] = {
    "iso3": "FRA",
    "country": "France",
    "year": 2014,
    "unit": "t/ha",
    "model_version": "2.0.0",
    "recommendations": [
        {"rank": 1, "crop": "Potatoes", "predicted_yield_tons_per_hectare": 42.24, "observed_in_country": True},
        {"rank": 2, "crop": "Sweet potatoes", "predicted_yield_tons_per_hectare": 20.74, "observed_in_country": False},
        {"rank": 3, "crop": "Maize", "predicted_yield_tons_per_hectare": 8.62, "observed_in_country": True},
    ],
    "context": {"country_defaults": _FRA_DEFAULTS, "effective_conditions": _FRA_DEFAULTS},
    "out_of_training_domain": False,
    "notes": [],
}


def _stub_context(calls: list[str | None]):
    """Contexte général, ou contexte du pays quand ``iso3`` est fourni ; chaque appel est noté."""

    def _call(iso3: str | None = None) -> dict[str, Any]:
        calls.append(iso3)
        if iso3 is None:
            return _CONTEXT
        return _CONTEXT | {"country": {"iso3": iso3, "country": "France", "country_defaults": _FRA_DEFAULTS}}

    return _call


def _run_with_typed_country(monkeypatch: pytest.MonkeyPatch, typed: str) -> tuple[AppTest, list[str | None]]:
    """Page exécutée comme si ``typed`` avait été tapé puis validé par Entrée dans le sélecteur."""
    calls: list[str | None] = []
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context(calls))
    at = AppTest.from_file(PAGE)
    at.session_state["rec_country"] = typed
    return at.run(), calls


def _markdown_containing(at: AppTest, marker: str) -> str:
    """Corps du premier bloc Markdown contenant ``marker`` (panneau, classement…)."""
    for block in at.markdown:
        if marker in block.value:
            return block.value
    raise AssertionError(f"aucun bloc Markdown ne contient {marker!r}")


def test_page_loads_context_and_lists_its_countries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str | None] = []
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context(calls))

    at = AppTest.from_file(PAGE).run()

    assert not at.exception
    assert calls == [None]
    options = at.selectbox(key="rec_country").options
    # Ordre alphabétique français ; « Égypte » est affiché avec son accent, décomposé pour la recherche.
    assert [unicodedata.normalize("NFC", label) for label in options] == ["Égypte", "France", "Mali"]
    assert options[0] == unicodedata.normalize("NFD", "Égypte")
    # conditions visibles mais vides et désactivées tant qu'aucun pays n'est choisi
    assert [field.value for field in at.number_input] == [None, None, None]
    assert all(field.disabled for field in at.number_input)
    panel = _markdown_containing(at, "ag-panel")
    assert "Votre pays." in panel
    # Cultures évaluées : une ligne de texte dans le panneau, plus de pastilles dans le formulaire.
    assert "3 cultures évaluées" in panel
    assert "Maïs&nbsp;· Pomme de terre&nbsp;· Patate douce" in panel
    assert not any("ag-crop\"" in block.value or "ag-crops" in block.value for block in at.markdown)
    assert "Proposées automatiquement dès qu’un pays est choisi" in _markdown_containing(at, "Conditions du pays")
    # Domaines arrondis pour la lecture seulement.
    helps = " ".join(block.value for block in at.markdown if "ag-help" in block.value)
    assert "2,6&nbsp;–&nbsp;30,2&nbsp;°C" in helps
    assert "51&nbsp;–&nbsp;3\u202f240&nbsp;mm" in helps
    assert "≈&nbsp;0&nbsp;–&nbsp;1,8&nbsp;M&nbsp;t" in helps


def test_selected_country_fetches_its_context_and_prefills_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context(calls))

    at = AppTest.from_file(PAGE).run()
    at.selectbox(key="rec_country").select("FRA").run()

    assert not at.exception
    assert calls[-1] == "FRA"
    assert [field.value for field in at.number_input] == list(_FRA_DEFAULTS.values())
    assert "Valeurs proposées automatiquement, modifiables" in _markdown_containing(at, "Conditions du pays")
    # curseurs aux valeurs par défaut ; celui des pesticides est logarithmique : log10(1 + t)
    assert at.slider(key="average_temperature_celsius_FRA_slider").value == 11.52
    assert at.slider(key="annual_rainfall_mm_FRA_slider").value == 867.0
    assert at.slider(key="average_annual_pesticides_tons_FRA_slider").value == pytest.approx(
        math.log10(1 + 63694.63)
    )


def test_post_sends_only_iso3_and_the_three_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context([]))
    seen: dict[str, Any] = {}

    def _recommend(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        return _RECOMMEND_OK

    monkeypatch.setattr(api_client, "post_recommend", _recommend)

    at = AppTest.from_file(PAGE).run()
    at.selectbox(key="rec_country").select("FRA").run()
    at.number_input(key="annual_rainfall_mm_FRA").set_value(900.0)  # champ
    # champ → curseur logarithmique : 99 999 t → position log10(100 000) = 5
    at.number_input(key="average_annual_pesticides_tons_FRA").set_value(99999.0).run()
    assert at.slider(key="average_annual_pesticides_tons_FRA_slider").value == pytest.approx(5.0)
    # curseur logarithmique en position 4 → champ à 10**4 - 1 = 9 999 t, la vraie valeur
    at.slider(key="average_annual_pesticides_tons_FRA_slider").set_value(4.0).run()
    assert at.number_input(key="average_annual_pesticides_tons_FRA").value == 9999.0
    assert "payload" not in seen  # bouger un champ ou un curseur n'appelle pas l'API
    at.button[0].click().run()

    assert not at.exception
    assert at.slider(key="annual_rainfall_mm_FRA_slider").value == 900.0
    assert seen["payload"] == {
        "iso3": "FRA",
        "conditions": {
            "average_temperature_celsius": 11.52,
            "annual_rainfall_mm": 900.0,
            "average_annual_pesticides_tons": 9999.0,
        },
    }


def test_ranking_is_displayed_in_api_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context([]))
    monkeypatch.setattr(api_client, "post_recommend", lambda _body: _RECOMMEND_OK)

    at = AppTest.from_file(PAGE).run()
    at.selectbox(key="rec_country").select("FRA").run()
    at.button[0].click().run()

    assert not at.exception
    ranking = _markdown_containing(at, "ag-rank-row")
    positions = [ranking.index(label) for label in ("Pomme de terre", "Patate douce", "Maïs")]
    assert positions == sorted(positions)
    assert "42,24" in ranking
    assert "Cultivée dans le pays" in ranking
    assert "Jamais observée dans le pays" in ranking
    assert "Aucun rendement" not in ranking  # le badge suffit, sans légende technique

    panel = _markdown_containing(at, "ag-panel")
    assert "Pomme de terre" in panel
    assert "Modèle" not in panel  # version technique : reste dans l'API, pas dans l'interface
    assert "2014" not in panel + ranking


def test_country_typed_without_accent_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run_with_typed_country(monkeypatch, "Egypte")
    assert not at.exception
    assert calls[-1] == "EGY"  # seul le code ISO3 retrouvé est utilisé
    assert not any(field.disabled for field in at.number_input)


def test_country_typed_with_accent_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run_with_typed_country(monkeypatch, "  égypte ")
    assert not at.exception
    assert calls[-1] == "EGY"


def test_unknown_typed_country_shows_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run_with_typed_country(monkeypatch, "Atlantide")
    assert not at.exception
    assert calls == [None]  # aucun contexte pays demandé
    assert "Aucun pays ne correspond à « Atlantide »" in _markdown_containing(at, "Aucun pays")
    assert all(field.disabled for field in at.number_input)


# --- Bornes : champ physique, curseur sur le domaine d'apprentissage (comme Predict) ---

def _stub_recommend(seen: dict[str, Any], notes: list[str] | None = None):
    """Réponse de POST /recommend qui reprend les conditions reçues, comme la vraie API."""

    def _call(body: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = body
        effective = {"country_defaults": _FRA_DEFAULTS, "effective_conditions": body["conditions"]}
        return _RECOMMEND_OK | {"context": effective, "out_of_training_domain": bool(notes), "notes": notes or []}

    return _call


def _page_for_france(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], notes: list[str] | None = None) -> AppTest:
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context([]))
    monkeypatch.setattr(api_client, "post_recommend", _stub_recommend(seen, notes))
    at = AppTest.from_file(PAGE).run()
    return at.selectbox(key="rec_country").select("FRA").run()


def test_sliders_follow_the_training_domain_and_fields_the_physical_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _page_for_france(monkeypatch, {})
    assert not at.exception

    temp_field, temp_slider = at.number_input(key="average_temperature_celsius_FRA"), at.slider(
        key="average_temperature_celsius_FRA_slider"
    )
    rain_field, rain_slider = at.number_input(key="annual_rainfall_mm_FRA"), at.slider(key="annual_rainfall_mm_FRA_slider")
    pest_field, pest_slider = at.number_input(key="average_annual_pesticides_tons_FRA"), at.slider(
        key="average_annual_pesticides_tons_FRA_slider"
    )
    # Curseurs : domaine d'apprentissage, ramené vers l'intérieur sur le pas du curseur pour
    # n'offrir que des valeurs rondes (2,57-30,25 °C → 2,6-30,2 °C). Pesticides en log10(1 + t) :
    # départ à 0 t (position 0), maximum dans le domaine.
    assert (temp_slider.min, temp_slider.max) == (2.6, 30.2)
    assert (rain_slider.min, rain_slider.max) == (51.0, 3240.0)
    assert (pest_slider.min, pest_slider.max) == (0.0, 6.25)
    assert pest_slider.max <= math.log10(1 + 1783667.33)
    # Champs : bornes physiques. Sans maximum, Streamlit utilise le plus grand flottant.
    assert (temp_field.min, temp_field.max) == (-50.0, 60.0)
    assert (rain_field.min, rain_field.max) == (0.0, sys.float_info.max)
    assert (pest_field.min, pest_field.max) == (0.0, sys.float_info.max)


def test_values_outside_training_domain_are_kept_and_sent_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen, notes=["average_temperature_celsius is out of training domain"])

    at.number_input(key="average_temperature_celsius_FRA").set_value(35.0).run()
    at.number_input(key="annual_rainfall_mm_FRA").set_value(4000.0).run()
    at.number_input(key="average_annual_pesticides_tons_FRA").set_value(2000000.0).run()
    # Les champs gardent la valeur saisie ; seuls les curseurs se mettent en butée.
    assert at.number_input(key="average_temperature_celsius_FRA").value == 35.0
    assert at.slider(key="average_temperature_celsius_FRA_slider").value == 30.2
    assert at.number_input(key="annual_rainfall_mm_FRA").value == 4000.0
    assert at.slider(key="annual_rainfall_mm_FRA_slider").value == 3240.0
    assert at.number_input(key="average_annual_pesticides_tons_FRA").value == 2000000.0
    assert at.slider(key="average_annual_pesticides_tons_FRA_slider").value == 6.25

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["conditions"] == {
        "average_temperature_celsius": 35.0,
        "annual_rainfall_mm": 4000.0,
        "average_annual_pesticides_tons": 2000000.0,
    }
    panel = _markdown_containing(at, "ag-panel")
    assert "Hors domaine d'apprentissage" in panel
    assert "Température&nbsp;: 35&nbsp;°C" in panel
    assert "2,6&nbsp;–&nbsp;30,2&nbsp;°C" in panel  # même format que sous le curseur
    assert "2,57" not in panel and "30,25" not in panel


def test_moving_the_slider_after_an_out_of_domain_value_resyncs_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen)

    at.number_input(key="average_temperature_celsius_FRA").set_value(35.0).run()
    at.slider(key="average_temperature_celsius_FRA_slider").set_value(20.0).run()
    # La valeur du curseur redevient celle du champ, et donc celle envoyée.
    assert at.number_input(key="average_temperature_celsius_FRA").value == 20.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["conditions"]["average_temperature_celsius"] == 20.0


def test_physically_invalid_values_are_not_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen)

    # Le navigateur refuse ces saisies ; côté serveur, Streamlit écarte aussi toute
    # valeur hors des bornes du champ (règle existante, inchangée).
    at.number_input(key="average_temperature_celsius_FRA").set_value(70.0).run()  # > 60 °C
    at.number_input(key="annual_rainfall_mm_FRA").set_value(-5.0).run()  # < 0 mm
    assert at.number_input(key="average_temperature_celsius_FRA").value != 70.0
    assert at.number_input(key="annual_rainfall_mm_FRA").value != -5.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["conditions"]["average_temperature_celsius"] != 70.0
    assert seen["payload"]["conditions"]["annual_rainfall_mm"] != -5.0



def test_display_precision_does_not_change_the_values_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen)

    # Champs : 1 décimale pour la température, aucune pour la pluie, 2 pour les pesticides
    # (le domaine descend à 0,04 t), avec un pas de 0,01 t.
    formats = [at.number_input(key=f"{field}_FRA").proto.format for field in _FRA_DEFAULTS]
    assert formats == ["%.1f", "%.0f", "%.2f"]
    assert at.number_input(key="average_annual_pesticides_tons_FRA").proto.step == 0.01

    at.button[0].click().run()
    assert not at.exception
    # La valeur envoyée garde toute sa précision (11,52 °C ; 63 694,63 t).
    assert seen["payload"]["conditions"] == _FRA_DEFAULTS
    panel = _markdown_containing(at, "ag-panel")
    assert "<b>11,5&nbsp;°C</b>" in panel
    assert "<b>63\u202f695&nbsp;t</b>" in panel


def test_small_pesticide_value_keeps_its_decimals(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen)

    at.number_input(key="average_annual_pesticides_tons_FRA").set_value(0.87).run()
    assert at.number_input(key="average_annual_pesticides_tons_FRA").value == 0.87

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["conditions"]["average_annual_pesticides_tons"] == 0.87


def test_pesticide_slider_starts_at_zero_tons(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen, notes=["average_annual_pesticides_tons is out of training domain"])

    # Curseur complètement à gauche (position log 0) après la valeur du pays : le champ vaut 0 t.
    at.slider(key="average_annual_pesticides_tons_FRA_slider").set_value(0.0).run()
    assert at.number_input(key="average_annual_pesticides_tons_FRA").value == 0.0

    at.button[0].click().run()
    assert not at.exception
    assert seen["payload"]["conditions"]["average_annual_pesticides_tons"] == 0.0
    # 0 t est sous le minimum appris (≈ 0,04 t) : l'avertissement de l'API est affiché.
    panel = _markdown_containing(at, "ag-panel")
    assert "Hors domaine d'apprentissage" in panel
    assert "Pesticides&nbsp;: 0&nbsp;t" in panel


# --- Erreurs API -------------------------------------------------------------

def test_connection_error_at_load_shows_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(iso3: str | None = None) -> None:
        raise ApiConnectionError("boom")

    monkeypatch.setattr(api_client, "get_recommend_context", _raise)
    at = AppTest.from_file(PAGE).run()

    assert not at.exception
    assert at.error[0].value == "Impossible de joindre le service de calcul. Réessayez dans un instant."
    assert len(at.number_input) == 0


def test_invalid_context_at_load_shows_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "get_recommend_context", lambda iso3=None: {"countries": "?"})
    at = AppTest.from_file(PAGE).run()

    assert not at.exception
    assert at.error[0].value == INVALID_CONTEXT_MESSAGE
    assert len(at.number_input) == 0


def test_country_context_error_keeps_conditions_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _context(iso3: str | None = None) -> dict[str, Any]:
        if iso3 is None:
            return _CONTEXT
        raise ApiTimeoutError("lent")

    monkeypatch.setattr(api_client, "get_recommend_context", _context)
    at = AppTest.from_file(PAGE).run()
    at.selectbox(key="rec_country").select("FRA").run()

    assert not at.exception
    assert at.error[0].value == "Le service de calcul n'a pas répondu à temps. Réessayez dans un instant."
    assert all(field.disabled for field in at.number_input)
    assert at.button[0].disabled


def test_submit_api_error_shows_message_without_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_body: dict[str, Any]) -> None:
        raise ApiHttpError(500, "internal_error", "Internal server error.")

    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context([]))
    monkeypatch.setattr(api_client, "post_recommend", _raise)
    at = AppTest.from_file(PAGE).run()
    at.selectbox(key="rec_country").select("FRA").run()
    at.button[0].click().run()

    assert not at.exception
    assert at.error[0].value == "Le service a rencontré une erreur. Réessayez dans un instant."
    assert not any("ag-rank" in block.value for block in at.markdown)
    assert "Votre pays." in _markdown_containing(at, "ag-panel")


@pytest.mark.parametrize(
    "response",
    [
        _RECOMMEND_OK | {"recommendations": [{"rank": 1, "crop": "Maize"}]},
        _RECOMMEND_OK | {"context": {}},
    ],
    ids=["culture-sans-rendement", "sans-conditions-utilisees"],
)
def test_incomplete_ranking_shows_message_without_traceback(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> None:
    monkeypatch.setattr(api_client, "get_recommend_context", _stub_context([]))
    monkeypatch.setattr(api_client, "post_recommend", lambda _body: response)
    at = AppTest.from_file(PAGE).run()
    at.selectbox(key="rec_country").select("FRA").run()
    at.button[0].click().run()

    assert not at.exception
    assert at.error[0].value == "Le service a renvoyé une réponse inattendue."
    assert not any("ag-rank" in block.value for block in at.markdown)


# --- Résultat affiché seulement pour les entrées qui l'ont produit -----------

def _has_ranking(at: AppTest) -> bool:
    return any("ag-rank-row" in block.value for block in at.markdown)


def test_changing_a_condition_after_ranking_hides_it(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _page_for_france(monkeypatch, {})
    at.button[0].click().run()
    assert _has_ranking(at)

    at.number_input(key="average_temperature_celsius_FRA").set_value(18.0).run()
    assert not at.exception
    assert not _has_ranking(at)
    assert "Votre pays." in _markdown_containing(at, "ag-panel")

    # Retour exact aux conditions du classement : il redevient valable.
    at.number_input(key="average_temperature_celsius_FRA").set_value(11.52).run()
    assert _has_ranking(at)


def test_returning_to_a_previous_country_does_not_show_an_outdated_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _page_for_france(monkeypatch, {})
    at.number_input(key="average_temperature_celsius_FRA").set_value(25.0).run()
    at.button[0].click().run()
    assert _has_ranking(at)

    at.selectbox(key="rec_country").select("MLI").run()
    assert not _has_ranking(at)
    at.selectbox(key="rec_country").select("FRA").run()

    assert not at.exception
    # Valeurs par défaut du pays rechargées : le classement obtenu à 25 °C ne s'affiche pas.
    assert at.number_input(key="average_temperature_celsius_FRA").value == 11.52
    assert not _has_ranking(at)
    assert "Votre pays." in _markdown_containing(at, "ag-panel")


def test_new_ranking_after_a_change_shows_the_new_result(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    at = _page_for_france(monkeypatch, seen)
    at.button[0].click().run()
    at.number_input(key="average_temperature_celsius_FRA").set_value(20.0).run()
    at.button[0].click().run()

    assert not at.exception
    assert seen["payload"]["conditions"]["average_temperature_celsius"] == 20.0
    assert _has_ranking(at)
    assert "<b>20&nbsp;°C</b>" in _markdown_containing(at, "ag-panel")
