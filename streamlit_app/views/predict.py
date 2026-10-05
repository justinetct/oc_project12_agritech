"""Parcours Streamlit du service /predict.

Charge le contexte de l'API (bornes physiques du contrat et domaine
d'apprentissage), construit le formulaire des 4 entrées du modèle et affiche
le rendement renvoyé par ``POST /predict`` dans le panneau vert de gauche.

Le champ numérique porte la valeur envoyée, sans bornes : l'API décide si elle
est physiquement valide. Son curseur couvre le domaine d'apprentissage. Une
valeur hors domaine reste donc saisissable : l'API la signale et le panneau
affiche l'avertissement hors domaine.

Le front n'embarque aucune logique ML. Toutes les valeurs affichables
(bornes, unités, notes hors domaine) sont fournies par l'API et
consommées telles quelles. Si le contexte est incomplet ou mal formé,
la page échoue proprement plutôt que d'inventer des valeurs par défaut.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agritech.ui import api_client
from agritech.ui.components import (
    NUMERIC_LABELS,
    cta_note_html,
    domain_help_html,
    field_label_html,
    footer_html,
    header_html,
    panel_html,
    section_head_html,
    unit_suffix_css,
)
from agritech.ui.errors import INVALID_CONTEXT_MESSAGE, ApiError, ApiInvalidResponseError, format_api_error
from agritech.ui.theme import page_css, predict_form_css
from agritech.ui.widgets import number_with_slider

# Chaque clic sur « Estimer mon rendement » relance le script (rerun). La
# dernière estimation réussie (réponse de l'API et valeurs envoyées) est donc
# conservée dans ``st.session_state`` et relue à chaque exécution pour
# dessiner le panneau de résultat, tant que les valeurs affichées sont celles envoyées.
LAST_KEY = "predict_last"

st.html(page_css() + predict_form_css())  # + espacements propres au formulaire Predict
page = st.container(key="page", gap=None)
page.markdown(header_html(active="predict"), unsafe_allow_html=True)
notice = page.container(key="notice")


# --- Lecture stricte du contexte API ---------------------------------

class _InvalidContextError(RuntimeError):
    """Le contexte renvoyé par l'API ne respecte pas la structure attendue."""


def _required_number(source: Any, key: str) -> float:
    """Retourne la valeur numérique attendue, ou lève ``_InvalidContextError``."""
    if not isinstance(source, dict):
        raise _InvalidContextError(f"champ « {key} » manquant")
    value = source.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _InvalidContextError(f"champ « {key} » n'est pas un nombre")
    return float(value)


def _optional_number(source: Any, key: str) -> float | None:
    """Comme ``_required_number`` mais accepte l'absence ou ``null``.

    Utilisé pour ``max`` des bornes physiques (rainfall n'a pas de borne haute).
    """
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _InvalidContextError(f"champ « {key} » n'est pas un nombre")
    return float(value)


def _required_str(source: Any, key: str) -> str:
    if not isinstance(source, dict):
        raise _InvalidContextError(f"champ « {key} » manquant")
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise _InvalidContextError(f"champ « {key} » n'est pas une chaîne")
    return value


def _required_dict(source: Any, key: str) -> dict[str, Any]:
    if not isinstance(source, dict) or not isinstance(source.get(key), dict):
        raise _InvalidContextError(f"section « {key} » manquante")
    return source[key]


def _within_physical_bounds(
    value: float, low: float, high: float | None, label: str
) -> float:
    """Vérifie ``low <= value <= high`` (``high`` peut être ``None``).

    Si la valeur est en dehors, lève ``_InvalidContextError`` : on refuse
    de corriger silencieusement une incohérence du contexte API.
    """
    if value < low or (high is not None and value > high):
        raise _InvalidContextError(
            f"valeur initiale « {label} » hors des bornes physiques"
        )
    return value


try:
    context = api_client.get_predict_context()
except ApiError as exc:
    notice.error(format_api_error(exc))
    st.stop()

try:
    physical = _required_dict(context, "physical_bounds")
    training = _required_dict(context, "training_domain")

    rainfall_phys = _required_dict(physical, "rainfall_mm")
    temperature_phys = _required_dict(physical, "temperature_celsius")
    rainfall_train = _required_dict(training, "rainfall_mm")
    temperature_train = _required_dict(training, "temperature_celsius")

    rainfall_min = _required_number(rainfall_phys, "min")
    rainfall_max = _optional_number(rainfall_phys, "max")
    rainfall_unit = _required_str(rainfall_phys, "unit")

    temperature_min = _required_number(temperature_phys, "min")
    temperature_max = _required_number(temperature_phys, "max")
    temperature_unit = _required_str(temperature_phys, "unit")

    rainfall_train_min = _required_number(rainfall_train, "min")
    rainfall_train_max = _required_number(rainfall_train, "max")
    temperature_train_min = _required_number(temperature_train, "min")
    temperature_train_max = _required_number(temperature_train, "max")

    # Valeurs initiales : milieu du domaine d'apprentissage. On refuse toute
    # incohérence avec les bornes physiques plutôt que de la masquer par
    # une correction silencieuse.
    rainfall_default = _within_physical_bounds(
        (rainfall_train_min + rainfall_train_max) / 2.0,
        rainfall_min,
        rainfall_max,
        "rainfall_mm",
    )
    temperature_default = _within_physical_bounds(
        (temperature_train_min + temperature_train_max) / 2.0,
        temperature_min,
        temperature_max,
        "temperature_celsius",
    )
except _InvalidContextError:
    notice.error(INVALID_CONTEXT_MESSAGE)
    st.stop()


# --- Ossature : panneau de résultat à gauche, formulaire à droite -----

panel_col, form_col = page.container(key="shell", gap=None).columns([0.355, 0.645], gap=None)
shown: dict[str, Any] = {}  # estimation affichée : seulement celle des valeurs affichées

with form_col:
    # Pas de st.form : un champ et son curseur se synchronisent par callback, ce
    # qu'un formulaire interdit. Seul le clic sur le bouton appelle l'API.

    # Conditions envoyées à POST /predict. Espacement « medium » (2rem) entre le
    # titre et les deux lignes de champs.
    with st.container(key="cond_card", gap="medium"):
        st.markdown(section_head_html("Conditions de votre parcelle"), unsafe_allow_html=True)
        # Champ : valeur envoyée telle quelle, validée par l'API. Curseur : domaine
        # d'apprentissage, en butée si le champ en sort (le champ n'est jamais réécrit).
        rain_col, temp_col = st.columns(2)
        with rain_col:
            st.markdown(field_label_html("goutte", "Pluie", large=True), unsafe_allow_html=True)
            rainfall = number_with_slider(
                f"Pluie ({rainfall_unit})",
                "rainfall_mm",
                rainfall_default,
                slider_range=(rainfall_train_min, rainfall_train_max),
                step=1.0,
                number_format="%.0f",
            )
            st.markdown(domain_help_html(rainfall_train), unsafe_allow_html=True)
        with temp_col:
            st.markdown(field_label_html("thermo", "Température", large=True), unsafe_allow_html=True)
            temperature = number_with_slider(
                f"Température ({temperature_unit})",
                "temperature_celsius",
                temperature_default,
                slider_range=(temperature_train_min, temperature_train_max),
                step=0.1,
                number_format="%.1f",
                unit=temperature_unit,  # repère « 0 °C » si la plage passe par zéro
            )
            st.markdown(domain_help_html(temperature_train), unsafe_allow_html=True)
        fert_col, irr_col = st.columns(2)
        with fert_col:
            st.markdown(field_label_html("engrais", "Fertilisation", large=True), unsafe_allow_html=True)
            fertilizer = st.segmented_control(
                "Fertilisation",
                ["Oui", "Non"],
                default="Non",
                required=True,
                key="fertilizer_used",
                label_visibility="collapsed",
                width="stretch",
            )
        with irr_col:
            st.markdown(field_label_html("irrigation", "Irrigation", large=True), unsafe_allow_html=True)
            irrigation = st.segmented_control(
                "Irrigation",
                ["Oui", "Non"],
                default="Non",
                required=True,
                key="irrigation_used",
                label_visibility="collapsed",
                width="stretch",
            )

    submitted = st.button(
        "Estimer mon rendement",
        type="primary",
        width="stretch",
        icon=":material/arrow_forward:",
        icon_position="right",
    )

    st.html(unit_suffix_css("rainfall_mm", rainfall_unit) + unit_suffix_css("temperature_celsius", temperature_unit))

    # Les 4 conditions affichées forment le corps de POST /predict.
    payload = {
        "rainfall_mm": float(rainfall),
        "temperature_celsius": float(temperature),
        "fertilizer_used": fertilizer == "Oui",
        "irrigation_used": irrigation == "Oui",
    }

    # Exécution déclenchée par le clic : appel de l'API, puis mémorisation du
    # résultat dans ``st.session_state`` pour les exécutions suivantes.
    if submitted:
        st.session_state[LAST_KEY] = None
        try:
            result = api_client.post_predict(payload)
            if not isinstance(result.get("yield_tons_per_hectare"), (int, float)):
                raise ApiInvalidResponseError("rendement absent ou non numérique")
        except ApiError as exc:
            st.error(format_api_error(exc, NUMERIC_LABELS, physical))  # 422 : champs refusés et valeurs acceptées
        else:
            st.session_state[LAST_KEY] = {"result": result, "payload": payload}

    # Dernière estimation réussie, affichée seulement si elle a été obtenue avec les
    # valeurs affichées ; invitation sous le bouton dans ce seul cas.
    last = st.session_state.get(LAST_KEY)
    if last and last["payload"] == payload:
        shown = last
        st.markdown(cta_note_html(), unsafe_allow_html=True)
    st.markdown(footer_html(), unsafe_allow_html=True)


# --- Panneau : présentation du service ou dernière estimation --------

panel_col.markdown(
    panel_html(shown.get("result"), shown.get("payload"), training),
    unsafe_allow_html=True,
)
