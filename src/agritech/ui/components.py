"""Blocs HTML de l'interface : bandeau, panneaux, formulaire et classement.

Fonctions pures, testables sans Streamlit. Le HTML est produit sur une seule
ligne : une ligne vide ou une indentation serait interprétée par le Markdown de
Streamlit (paragraphe, bloc de code).
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

from agritech.ui.country_labels import country_label
from agritech.ui.crop_labels import crop_label
from agritech.ui.icons import icon_svg
from agritech.ui.theme import logo_data_uri

# Format documenté des notes de POST /predict : une note par champ hors domaine.
OOD_NOTE_TEMPLATE = "{field} is out of training domain"

# Libellés d'interface des entrées numériques du modèle.
NUMERIC_LABELS = {"rainfall_mm": "Pluie", "temperature_celsius": "Température"}

# Libellés d'interface des conditions de /recommend.
RECOMMEND_LABELS = {
    "average_temperature_celsius": "Température",
    "annual_rainfall_mm": "Pluie",
    "average_annual_pesticides_tons": "Pesticides",
}

# Décimales affichées pour chaque condition de /recommend : champ, domaine, résultats et
# avertissement. Affichage seulement : les valeurs envoyées et les vraies bornes ne changent pas.
RECOMMEND_DECIMALS = {
    "average_temperature_celsius": 1,
    "annual_rainfall_mm": 0,
    "average_annual_pesticides_tons": 0,
}


def fr_number(value: float, decimals: int = 0) -> str:
    """Nombre à la française : virgule décimale, espace fine insécable entre les milliers."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def rounded_number(value: float, decimals: int = 0) -> str:
    """Nombre arrondi pour l'affichage, zéros inutiles retirés ; quasi-zéro noté « ≈ 0 ».

    18,39 → « 18,4 » (1 décimale), 35,0 → « 35 », 1 766 036,82 → « 1 766 037 », 0,04 → « ≈ 0 ».
    """
    if value != 0 and round(value, decimals) == 0:
        return "≈&nbsp;0"
    text = fr_number(value, decimals)
    return text.rstrip("0").rstrip(",") if "," in text else text


def compact_number(value: float, decimals: int = 0) -> str:
    """Comme ``rounded_number``, avec les millions abrégés : 1 783 667 → « 1,8 M »."""
    if abs(value) >= 1_000_000:
        return f"{fr_number(value / 1_000_000, 1)}&nbsp;M"
    return rounded_number(value, decimals)


def _fr_value(value: float) -> str:
    """Jusqu'à deux décimales, zéros inutiles retirés (550 → « 550 », 27.5 → « 27,5 », 0.04 → « 0,04 »)."""
    return fr_number(value, 2).rstrip("0").rstrip(",")


def out_of_domain_fields(notes: list[str], fields: list[str]) -> list[str]:
    """Champs signalés par l'API, par égalité stricte avec le format documenté des notes.

    Une note qui ne correspond à aucun champ connu est ignorée ici ; l'alerte
    générale reste affichée dès que ``out_of_training_domain`` vaut vrai.
    """
    return [field for field in fields if OOD_NOTE_TEMPLATE.format(field=field) in notes]


def _nav_item(label: str, href: str, icon: str, active: bool) -> str:
    """Lien du bandeau ; la page affichée porte la classe ``active``."""
    css = "ag-nav-item active" if active else "ag-nav-item"
    return f'<a class="{css}" href="{href}" target="_self">{icon_svg(icon)}{label}</a>'


def header_html(active: str = "predict") -> str:
    """Bandeau vert : logo à gauche, navigation Predict / Recommend à droite."""
    return (
        '<div class="ag-topbar">'
        f'<img class="ag-logo" src="{logo_data_uri()}" alt="Agritech Answers">'
        '<nav class="ag-nav" aria-label="Services">'
        f'{_nav_item("Predict", "/", "rendement", active == "predict")}'
        f'{_nav_item("Recommend", "/recommend", "pousse", active == "recommend")}'
        "</nav></div>"
    )


def panel_html(
    result: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    training_domain: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Panneau vert de gauche : présentation du service, ou rendement estimé."""
    head = '<div class="ag-eyebrow">Predict · Agritech Answers</div>'
    foot = '<div class="ag-panel-foot">La donnée au service<br>de vos décisions agricoles.</div>'
    if result is None:
        body = (
            f'<div class="ag-big-ic">{icon_svg("ble")}</div>'
            '<div class="ag-panel-title">Votre parcelle.<br>Ses conditions.<br>Votre rendement.</div>'
            '<div class="ag-panel-lead">Une estimation claire pour éclairer vos décisions de culture.</div>'
        )
    else:
        body = (
            f'<div class="ag-big-ic">{icon_svg("rendement")}</div>'
            '<div class="ag-res-label">Votre rendement estimé</div>'
            f'<div class="ag-res-value">{fr_number(result["yield_tons_per_hectare"], 2)}'
            f'<small>{escape(str(result.get("unit", "t/ha")))}</small></div>'
            f"{_predict_conditions_html(payload or {}, training_domain or {})}"
        )
        if result.get("out_of_training_domain"):
            body += _warning_html(result.get("notes") or [], payload or {}, training_domain or {})
    return f'<div class="ag-panel">{head}{body}{foot}</div>'


def _predict_conditions_html(payload: dict[str, Any], training_domain: dict[str, Any]) -> str:
    """Les 4 valeurs envoyées à ``POST /predict``, affichées telles quelles."""
    rows = ""
    for field in ("temperature_celsius", "rainfall_mm"):
        if field in payload:
            unit = escape(str(training_domain.get(field, {}).get("unit", "")))
            rows += (
                f'<div class="ag-cond-row"><span>{NUMERIC_LABELS[field]}</span>'
                f"<b>{_fr_value(payload[field])}&nbsp;{unit}</b></div>"
            )
    for field, label in (("fertilizer_used", "Fertilisation"), ("irrigation_used", "Irrigation")):
        if field in payload:
            rows += f'<div class="ag-cond-row"><span>{label}</span><b>{"Oui" if payload[field] else "Non"}</b></div>'
    if not rows:
        return ""
    return f'<div class="ag-cond"><div class="ag-cond-title">Conditions utilisées</div>{rows}</div>'


def _range_html(low: str, high: str, unit: str) -> str:
    """Plage « min – max unité » insécable : si elle ne tient pas, elle passe entière à la ligne."""
    return f'<span class="ag-range">{low}&nbsp;–&nbsp;{high}&nbsp;{unit}</span>'


def _warning_html(
    notes: list[str],
    values: dict[str, Any],
    training_domain: dict[str, Any],
    labels: dict[str, str] = NUMERIC_LABELS,
    text: str = "Cette estimation extrapole&nbsp;: interprétez-la avec prudence.",
    decimals: dict[str, int] | None = None,
) -> str:
    """Alerte hors domaine : une ligne par champ identifié, message général dans tous les cas.

    Avec ``decimals``, valeur et domaine reprennent le format affiché sous le curseur
    (« 2,6 – 30,2 °C ») ; sinon, jusqu'à deux décimales (Predict).
    """
    items = ""
    for field in out_of_domain_fields(notes, list(labels)):
        domain = training_domain[field]
        unit = escape(str(domain["unit"]))
        if decimals is None:
            value, low, high = _fr_value(values[field]), _fr_value(domain["min"]), _fr_value(domain["max"])
        else:
            value = rounded_number(values[field], decimals[field])
            low, high = compact_number(domain["min"], decimals[field]), compact_number(domain["max"], decimals[field])
        items += (
            f"<li><b>{labels[field]}&nbsp;: {value}&nbsp;{unit}</b>"
            f" · domaine&nbsp;: {_range_html(low, high, unit)}</li>"
        )
    listing = f"<ul>{items}</ul>" if items else ""
    return (
        f'<div class="ag-warn" role="status">{icon_svg("alerte")}<div>'
        "<div class=\"ag-warn-title\">Hors domaine d'apprentissage</div>"
        f"{listing}"
        f'<div class="ag-warn-text">{text}</div>'
        "</div></div>"
    )


# --- Formulaire -------------------------------------------------------------

def section_head_html(title: str, subtitle: str | None = None, step: str | None = None) -> str:
    """Titre d'un bloc du formulaire, avec sous-titre et numéro d'étape facultatifs."""
    marker = f'<span class="ag-step">{escape(step)}</span>' if step else ""
    sub = f'<div class="ag-sec-sub">{escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="ag-sec-head"><div class="ag-sec-title">{marker}{escape(title)}</div>{sub}</div>'


def field_label_html(icon: str, label: str, large: bool = False) -> str:
    """Libellé d'un champ précédé de son pictogramme dans un cercle vert pâle."""
    size = "ag-ic lg" if large else "ag-ic"
    return (
        f'<div class="ag-field-label"><span class="{size}">{icon_svg(icon)}</span>'
        f"<span>{escape(label)}</span></div>"
    )


def domain_help_html(domain: dict[str, Any], decimals: int | None = None) -> str:
    """Domaine d'apprentissage affiché sous un champ numérique (valeurs du contexte API).

    Avec ``decimals``, les bornes sont arrondies pour l'affichage (``compact_number``) ;
    les vraies bornes restent celles du contexte API.
    """
    unit = escape(str(domain["unit"]))
    if decimals is None:
        low, high = _fr_value(domain["min"]), _fr_value(domain["max"])
    else:
        low, high = compact_number(domain["min"], decimals), compact_number(domain["max"], decimals)
    return f"<div class=\"ag-help\">Domaine d'apprentissage&nbsp;: {_range_html(low, high, unit)}</div>"


def unit_suffix_css(key: str, unit: str) -> str:
    """Unité affichée à droite du champ numérique ``key`` (texte venant du contexte API)."""
    text = unit.replace("\\", "\\\\").replace('"', '\\"')
    return f'<style>.st-key-{key} [data-testid="stNumberInputContainer"]::after{{content:"{text}"}}</style>'


def cta_note_html() -> str:
    """Phrase discrète sous le bouton principal, affichée après une première estimation."""
    return '<div class="ag-cta-note">Ajustez les conditions pour explorer une nouvelle estimation.</div>'


def footer_html(label: str = "Predict · Estimation parcellaire") -> str:
    """Pied du formulaire."""
    return f'<div class="ag-foot"><span>{escape(label)}</span><span>Agritech Answers</span></div>'


# --- Recommend --------------------------------------------------------------

def crops_line_html(crops: Sequence[str]) -> str:
    """Cultures évaluées, dans l'ordre de ``GET /recommend/context`` : simple ligne de texte."""
    names = "&nbsp;· ".join(escape(crop_label(str(crop))) for crop in crops)
    return f'<div class="ag-panel-crops"><b>{len(crops)} cultures évaluées&nbsp;:</b> {names}</div>'


def recommend_panel_html(
    result: dict[str, Any] | None = None,
    training_domain: dict[str, dict[str, Any]] | None = None,
    crops: Sequence[str] = (),
) -> str:
    """Panneau vert de /recommend : présentation et cultures évaluées, ou culture classée première."""
    head = '<div class="ag-eyebrow">Recommend · Agritech Answers</div>'
    foot = '<div class="ag-panel-foot">La donnée au service<br>de vos décisions agricoles.</div>'
    if result is None:
        body = (
            f'<div class="ag-big-ic">{icon_svg("podium")}</div>'
            '<div class="ag-panel-title">Votre pays.<br>Ses conditions.<br>Vos cultures.</div>'
            '<div class="ag-panel-lead">Un classement clair des cultures pour éclairer vos choix.</div>'
            f"{crops_line_html(crops) if crops else ''}"
        )
    else:
        top = result["recommendations"][0]
        country = country_label(str(result.get("iso3", "")), str(result.get("country", "")))
        conditions = (result.get("context") or {}).get("effective_conditions") or {}
        body = (
            f'<div class="ag-big-ic">{icon_svg("podium")}</div>'
            f'<div class="ag-res-label">Culture recommandée · {escape(country)}</div>'
            f'<div class="ag-top-crop">{escape(crop_label(str(top["crop"])))}</div>'
            f'<div class="ag-top-yield">{fr_number(top["predicted_yield_tons_per_hectare"], 2)}'
            f'<small>{escape(str(result.get("unit", "t/ha")))}</small></div>'
            f'{_observed_html(bool(top["observed_in_country"]))}'
            f"{_conditions_html(conditions, training_domain or {})}"
        )
        if result.get("out_of_training_domain"):
            body += _warning_html(
                result.get("notes") or [],
                conditions,
                training_domain or {},
                RECOMMEND_LABELS,
                "Ce classement extrapole&nbsp;: interprétez-le avec prudence.",
                RECOMMEND_DECIMALS,
            )
    return f'<div class="ag-panel">{head}{body}{foot}</div>'


def ranking_html(result: dict[str, Any]) -> str:
    """Classement des cultures dans l'ordre renvoyé par l'API : rang, nom, barre, rendement, statut."""
    recommendations = result["recommendations"]
    unit = escape(str(result.get("unit", "t/ha")))
    # Longueur des barres : rendement rapporté au plus élevé du classement (échelle graphique seulement).
    top_yield = max(item["predicted_yield_tons_per_hectare"] for item in recommendations)
    rows = ""
    for item in recommendations:
        value = item["predicted_yield_tons_per_hectare"]
        width = max(100 * value / top_yield, 0) if top_yield > 0 else 0
        rows += (
            '<div class="ag-rank-row">'
            f'<span class="ag-rank-num">{escape(str(item["rank"]))}</span>'
            f'<span class="ag-rank-crop">{escape(crop_label(str(item["crop"])))}</span>'
            f'<span class="ag-rank-bar"><span style="width:{width:.1f}%"></span></span>'
            f'<span class="ag-rank-value">{fr_number(value, 2)}&nbsp;{unit}</span>'
            f'{_observed_html(bool(item["observed_in_country"]))}'
            "</div>"
        )
    return f'<div class="ag-rank">{rows}</div>'


def _observed_html(observed: bool) -> str:
    """Statut ``observed_in_country`` : fait constaté dans l'historique, sans jugement de fiabilité."""
    if observed:
        return '<span class="ag-badge observed">Cultivée dans le pays</span>'
    return '<span class="ag-badge unobserved">Jamais observée dans le pays</span>'


def _conditions_html(conditions: dict[str, Any], training_domain: dict[str, Any]) -> str:
    """Conditions effectivement utilisées par l'API (``context.effective_conditions``)."""
    rows = ""
    for field, label in RECOMMEND_LABELS.items():
        if field in conditions:
            unit = escape(str(training_domain.get(field, {}).get("unit", "")))
            rows += (
                f'<div class="ag-cond-row"><span>{label}</span>'
                f"<b>{rounded_number(conditions[field], RECOMMEND_DECIMALS[field])}&nbsp;{unit}</b></div>"
            )
    if not rows:
        return ""
    return f'<div class="ag-cond"><div class="ag-cond-title">Conditions utilisées</div>{rows}</div>'
