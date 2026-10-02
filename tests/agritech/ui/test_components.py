"""Tests de ``agritech.ui.components`` (HTML du bandeau et du panneau)."""

from __future__ import annotations

from agritech.ui.components import (
    compact_number,
    crops_line_html,
    cta_note_html,
    domain_help_html,
    fr_number,
    header_html,
    out_of_domain_fields,
    panel_html,
    ranking_html,
    recommend_panel_html,
    rounded_number,
    section_head_html,
)


NNBSP = " "  # espace fine insécable des milliers

_TRAINING_DOMAIN = {
    "rainfall_mm": {"min": 100.0, "max": 1000.0, "unit": "mm"},
    "temperature_celsius": {"min": 15.0, "max": 40.0, "unit": "°C"},
}
_PAYLOAD = {
    "rainfall_mm": 1200.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
}
_RESULT_OK = {
    "yield_tons_per_hectare": 4.7516,
    "unit": "t/ha",
    "model_version": "1.0.0",
    "out_of_training_domain": False,
    "notes": [],
}


# --- Nombres à la française -------------------------------------------

def test_fr_number_uses_comma_decimal() -> None:
    assert fr_number(4.7516, 2) == "4,75"
    assert fr_number(27.5, 1) == "27,5"


def test_fr_number_groups_thousands_with_narrow_space() -> None:
    assert fr_number(1200) == f"1{NNBSP}200"


# --- Rapprochement note ↔ champ ----------------------------------------

def test_out_of_domain_fields_uses_strict_equality() -> None:
    notes = ["rainfall_mm is out of training domain"]
    assert out_of_domain_fields(notes, ["rainfall_mm", "temperature_celsius"]) == ["rainfall_mm"]


def test_out_of_domain_fields_ignores_near_miss_and_unknown_notes() -> None:
    notes = ["rainfall_mm is out of training domain.", "soil is out of training domain"]
    assert out_of_domain_fields(notes, ["rainfall_mm", "temperature_celsius"]) == []


def test_out_of_domain_fields_keeps_field_order() -> None:
    notes = [
        "temperature_celsius is out of training domain",
        "rainfall_mm is out of training domain",
    ]
    fields = ["rainfall_mm", "temperature_celsius"]
    assert out_of_domain_fields(notes, fields) == fields


# --- Bandeau -------------------------------------------------------------

def test_header_has_logo_and_active_predict() -> None:
    html = header_html(active="predict")
    assert "data:image/png;base64," in html
    assert 'class="ag-nav-item active" href="/"' in html
    assert ">Predict</a>" in html


def test_header_links_predict_and_recommend() -> None:
    html = header_html(active="recommend")
    assert 'class="ag-nav-item" href="/"' in html
    assert 'class="ag-nav-item active" href="/recommend"' in html
    assert "À venir" not in html


def test_header_single_line_for_markdown() -> None:
    assert "\n" not in header_html()


# --- Panneau -------------------------------------------------------------

def test_panel_initial_presents_the_service_without_result() -> None:
    html = panel_html()
    assert "Votre parcelle." in html
    assert "Votre rendement estimé" not in html
    assert "t/ha" not in html
    assert "\n" not in html


def test_panel_result_shows_yield_and_unit_without_model_version() -> None:
    html = panel_html(_RESULT_OK, _PAYLOAD, _TRAINING_DOMAIN)
    assert "Votre rendement estimé" in html
    assert "4,75" in html
    assert "<small>t/ha</small>" in html
    assert "Modèle" not in html  # version technique : reste dans l'API, pas dans l'interface
    assert "Hors domaine" not in html


def test_panel_out_of_domain_lists_identified_field_with_domain() -> None:
    result = _RESULT_OK | {
        "out_of_training_domain": True,
        "notes": ["rainfall_mm is out of training domain"],
    }
    html = panel_html(result, _PAYLOAD, _TRAINING_DOMAIN)
    assert "Hors domaine d'apprentissage" in html
    assert f"Pluie&nbsp;: 1{NNBSP}200&nbsp;mm" in html
    assert f"100&nbsp;–&nbsp;1{NNBSP}000&nbsp;mm" in html
    assert "Température" not in html.split('class="ag-warn"')[1]  # seule la pluie est signalée


def test_panel_out_of_domain_with_unknown_note_keeps_general_warning() -> None:
    result = _RESULT_OK | {"out_of_training_domain": True, "notes": ["unexpected note"]}
    html = panel_html(result, _PAYLOAD, _TRAINING_DOMAIN)
    assert "Hors domaine d'apprentissage" in html
    assert "interprétez-la avec prudence" in html
    assert "<li>" not in html


def test_panel_escapes_api_strings() -> None:
    result = _RESULT_OK | {"unit": "<i>t</i>"}
    html = panel_html(result, _PAYLOAD, _TRAINING_DOMAIN)
    assert "&lt;i&gt;t&lt;/i&gt;" in html
    assert "<i>t</i>" not in html


# --- Formulaire ----------------------------------------------------------

def test_section_head_without_subtitle_has_only_the_title() -> None:
    html = section_head_html("Conditions de votre parcelle")
    assert "Conditions de votre parcelle" in html
    assert "ag-sec-sub" not in html


def test_section_head_with_subtitle_and_step() -> None:
    html = section_head_html("Choisissez", "Sous-titre", step="1")
    assert '<span class="ag-step">1</span>Choisissez' in html
    assert '<div class="ag-sec-sub">Sous-titre</div>' in html


def test_cta_note_invites_to_adjust_conditions() -> None:
    assert "Ajustez les conditions" in cta_note_html()


# --- Domaines d'apprentissage : affichage arrondi ------------------------

def test_compact_number_rounds_abbreviates_millions_and_marks_near_zero() -> None:
    assert compact_number(2.566666, 1) == "2,6"
    assert compact_number(30.246666, 1) == "30,2"  # arrondi de la vraie valeur, pas d'un arrondi
    assert compact_number(3240.0) == f"3{NNBSP}240"
    assert compact_number(1783667.33) == "1,8&nbsp;M"
    assert compact_number(0.04) == "≈&nbsp;0"
    assert compact_number(0.0) == "0"


def test_domain_help_rounds_only_when_asked() -> None:
    pesticides = {"min": 0.04, "max": 1783667.33, "unit": "t"}
    assert "≈&nbsp;0&nbsp;–&nbsp;1,8&nbsp;M&nbsp;t" in domain_help_html(pesticides, 0)
    # Sans ``decimals`` (Predict), affichage inchangé : jusqu'à deux décimales.
    assert f"0,04&nbsp;–&nbsp;1{NNBSP}783{NNBSP}667,33&nbsp;t" in domain_help_html(pesticides)


def test_domain_range_is_a_single_unbreakable_block() -> None:
    # Même plage insécable sous le curseur et dans l'avertissement hors domaine.
    expected = f'<span class="ag-range">100&nbsp;–&nbsp;1{NNBSP}000&nbsp;mm</span>'
    assert expected in domain_help_html(_TRAINING_DOMAIN["rainfall_mm"])
    result = _RESULT_OK | {"out_of_training_domain": True, "notes": ["rainfall_mm is out of training domain"]}
    assert expected in panel_html(result, _PAYLOAD, _TRAINING_DOMAIN).split('class="ag-warn"')[1]


# --- Recommend ------------------------------------------------------------

_RECOMMEND_RESULT = {
    "iso3": "FRA",
    "country": "France",
    "unit": "t/ha",
    "model_version": "2.0.0",
    "recommendations": [
        {"rank": 1, "crop": "Potatoes", "predicted_yield_tons_per_hectare": 42.24, "observed_in_country": True},
        {"rank": 2, "crop": "Yams", "predicted_yield_tons_per_hectare": 9.5, "observed_in_country": False},
    ],
    "context": {"effective_conditions": {}},
    "out_of_training_domain": False,
    "notes": [],
}


def test_crops_line_is_plain_text_in_api_order() -> None:
    html = crops_line_html(["Cassava", "Maize", "Yams"])
    assert "3 cultures évaluées" in html
    assert "Manioc&nbsp;· Maïs&nbsp;· Igname" in html
    assert "<button" not in html and "<span" not in html  # ni bouton ni pastille : du texte


def test_recommend_panel_lists_crops_before_ranking_only() -> None:
    assert "2 cultures évaluées" in recommend_panel_html(crops=["Maize", "Yams"])
    html = recommend_panel_html(_RECOMMEND_RESULT, crops=["Maize", "Yams"])
    assert "cultures évaluées" not in html  # le classement liste déjà les cultures
    assert "Pomme de terre" in html
    assert "Modèle" not in html


def test_ranking_badges_without_technical_legend() -> None:
    html = ranking_html(_RECOMMEND_RESULT)
    assert "Cultivée dans le pays" in html
    assert "Jamais observée dans le pays" in html
    assert "Non observée" not in html
    assert "1990-2013" not in html



# --- Précision affichée des conditions /recommend -------------------------

_RECOMMEND_DOMAIN = {
    "average_temperature_celsius": {"min": 2.566666666666667, "max": 30.24666666666667, "unit": "°C"},
    "annual_rainfall_mm": {"min": 51.0, "max": 3240.0, "unit": "mm"},
    "average_annual_pesticides_tons": {"min": 0.04, "max": 1783667.333333334, "unit": "t"},
}


def test_rounded_number_drops_useless_decimals() -> None:
    assert rounded_number(18.391111, 1) == "18,4"
    assert rounded_number(35.0, 1) == "35"
    assert rounded_number(1766036.82) == f"1{NNBSP}766{NNBSP}037"
    assert rounded_number(63694.63) == f"63{NNBSP}695"
    assert rounded_number(0.04) == "≈&nbsp;0"


def test_recommend_panel_shows_readable_conditions_and_warning() -> None:
    conditions = {
        "average_temperature_celsius": 35.0,
        "annual_rainfall_mm": 495.0,
        "average_annual_pesticides_tons": 1766036.82,
    }
    result = _RECOMMEND_RESULT | {
        "context": {"effective_conditions": conditions},
        "out_of_training_domain": True,
        "notes": ["average_temperature_celsius is out of training domain"],
    }
    html = recommend_panel_html(result, _RECOMMEND_DOMAIN)
    assert "<b>495&nbsp;mm</b>" in html
    assert f"<b>1{NNBSP}766{NNBSP}037&nbsp;t</b>" in html
    # Avertissement au même format que sous le curseur ; les bornes précises restent internes.
    assert "Température&nbsp;: 35&nbsp;°C" in html
    assert "2,6&nbsp;–&nbsp;30,2&nbsp;°C" in html
    assert "2,57" not in html and "30,25" not in html
