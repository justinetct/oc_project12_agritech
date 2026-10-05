"""Parcours Streamlit du service /recommend.

Charge le contexte de l'API (pays, bornes physiques, domaine d'apprentissage),
préremplit les conditions avec les valeurs du pays choisi, puis affiche le
classement des cultures renvoyé par ``POST /recommend``.

Le front n'embarque aucune logique ML : l'ordre du classement, les rendements
et les notes hors domaine viennent de l'API et sont affichés tels quels. Seuls
les noms de cultures sont traduits, pour l'affichage (``crop_labels``).

Comme sur Predict, le champ numérique porte la valeur envoyée, sans bornes :
l'API décide si elle est physiquement valide. Son curseur couvre le domaine
d'apprentissage. Une valeur hors domaine reste saisissable : l'API la
signale et le panneau affiche l'avertissement hors domaine.
"""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from agritech.ui import api_client
from agritech.ui.components import (
    RECOMMEND_DECIMALS,
    RECOMMEND_LABELS,
    domain_help_html,
    field_label_html,
    footer_html,
    header_html,
    ranking_html,
    recommend_panel_html,
    section_head_html,
    unit_suffix_css,
)
from agritech.ui.country_labels import country_label, country_sort_key, find_country, searchable_label
from agritech.ui.errors import INVALID_CONTEXT_MESSAGE, ApiError, ApiInvalidResponseError, format_api_error
from agritech.ui.theme import page_css
from agritech.ui.widgets import number_with_slider

# Chaque interaction relance le script (rerun). Le dernier classement réussi est
# donc conservé dans ``st.session_state``, avec le pays et les conditions qui l'ont produit.
LAST_KEY = "recommend_last"

# Conditions envoyées à POST /recommend : pictogramme, format et pas du champ, échelle du
# curseur. Les pesticides couvrent plusieurs ordres de grandeur : curseur logarithmique, et
# champ à 2 décimales (le domaine descend à 0,04 t). Les décimales du domaine, des résultats
# et de l'avertissement viennent de RECOMMEND_DECIMALS.
FIELDS = {
    "average_temperature_celsius": ("thermo", "%.1f", 0.1, False),
    "annual_rainfall_mm": ("goutte", "%.0f", 1.0, False),
    "average_annual_pesticides_tons": ("pesticide", "%.2f", 0.01, True),
}

# Le curseur des pesticides part de 0 t (« aucun pesticide ») et non du minimum appris
# (≈ 0,04 t) : l'échelle log10(1 + t) place 0 t à l'extrême gauche. Son maximum reste celui
# du domaine d'apprentissage ; 0 t est signalé hors domaine par l'API, comme toute valeur sous ce minimum.
SLIDER_FROM_ZERO = {"average_annual_pesticides_tons"}

st.html(page_css())
page = st.container(key="page", gap=None)
page.markdown(header_html(active="recommend"), unsafe_allow_html=True)
notice = page.container(key="notice")


def _bounds(section: dict[str, Any]) -> dict[str, Any]:
    """Bornes d'un champ converties en nombres ; ``max`` peut valoir ``None``."""
    high = section["max"]
    return {
        "min": float(section["min"]),
        "max": None if high is None else float(high),
        "unit": str(section["unit"]),
    }


def _is_complete_ranking(result: dict[str, Any]) -> bool:
    """Vrai si la réponse contient ce que l'affichage utilise : un classement non vide où
    chaque culture a son rang, son nom, son rendement et son statut, et les conditions utilisées."""
    items = result.get("recommendations")
    context = result.get("context")
    conditions = context.get("effective_conditions") if isinstance(context, dict) else None
    return (
        isinstance(items, list)
        and bool(items)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("rank"), int)
            and isinstance(item.get("crop"), str)
            and isinstance(item.get("predicted_yield_tons_per_hectare"), (int, float))
            and isinstance(item.get("observed_in_country"), bool)
            for item in items
        )
        and isinstance(conditions, dict)
        and all(isinstance(conditions.get(field), (int, float)) for field in FIELDS)
    )


# --- Contexte général : pays, bornes physiques, domaine d'apprentissage ---

try:
    context = api_client.get_recommend_context()
except ApiError as exc:
    notice.error(format_api_error(exc))
    st.stop()

try:
    # ISO3 → nom affiché en français ; seul le code ISO3 est envoyé à l'API.
    countries = {entry["iso3"]: country_label(entry["iso3"], entry["country"]) for entry in context["countries"]}
    crops = [str(crop) for crop in context["crops"]]  # ordre de l'API conservé
    physical = {field: _bounds(context["physical_bounds"][field]) for field in FIELDS}
    training = {field: _bounds(context["training_domain"][field]) for field in FIELDS}
except (KeyError, TypeError, ValueError):
    notice.error(INVALID_CONTEXT_MESSAGE)
    st.stop()


# --- Ossature : panneau à gauche, parcours à droite ---------------------

panel_col, form_col = page.container(key="shell", gap=None).columns([0.355, 0.645], gap=None)
shown = None  # classement affiché : seulement celui du pays et des conditions affichés

with form_col:
    with st.container(key="country_card"):
        st.markdown(
            section_head_html("Choisissez d’abord votre pays", "Ses conditions sont proposées par défaut", step="1"),
            unsafe_allow_html=True,
        )
        # Recherche insensible aux accents : libellés décomposés (« Egypte » trouve « Égypte »),
        # et texte libre validé par Entrée rapproché du bon pays (« Égypte » tapé avec l'accent).
        # Seul le code ISO3 du pays retrouvé est envoyé à l'API.
        iso3 = st.selectbox(
            "Pays",
            sorted(countries, key=lambda code: country_sort_key(countries[code])),  # ordre alphabétique français
            index=None,
            format_func=lambda code: searchable_label(countries.get(code, code)),
            placeholder="Choisir un pays",
            key="rec_country",
            label_visibility="collapsed",
            accept_new_options=True,
        )
        if iso3 is not None and iso3 not in countries:  # texte libre : on cherche le pays correspondant
            typed, iso3 = iso3, find_country(iso3, countries)
            if iso3 is None:
                st.markdown(
                    f'<div class="ag-help">Aucun pays ne correspond à « {escape(typed)} ».</div>',
                    unsafe_allow_html=True,
                )

    # Valeurs par défaut du pays choisi, fournies par GET /recommend/context?iso3=…
    defaults = None
    if iso3:
        try:
            country = api_client.get_recommend_context(iso3)["country"]
            defaults = {field: float(country["country_defaults"][field]) for field in FIELDS}
        except ApiError as exc:
            st.error(format_api_error(exc))
        except (KeyError, TypeError, ValueError):
            st.error(INVALID_CONTEXT_MESSAGE)

    # Un seul bloc de conditions, toujours affiché : vide et désactivé tant qu'aucun pays
    # n'a fourni ses valeurs par défaut, puis actif et prérempli avec ces valeurs.
    # Pas de st.form : un champ et son curseur se synchronisent par callback.
    active = defaults is not None
    with st.container(key="cond_card", gap="medium"):
        st.markdown(
            section_head_html(
                "Conditions du pays",
                "Valeurs proposées automatiquement, modifiables"
                if active
                else "Proposées automatiquement dès qu’un pays est choisi",
                step="2",
            ),
            unsafe_allow_html=True,
        )
        values = {}
        for column, (field, (icon, number_format, step, log_scale)) in zip(st.columns(3), FIELDS.items()):
            decimals = RECOMMEND_DECIMALS[field]
            with column:
                st.markdown(
                    field_label_html(icon, RECOMMEND_LABELS[field], large=True),
                    unsafe_allow_html=True,
                )
                # Clé propre au pays : changer de pays recharge ses valeurs par défaut.
                # Champ : valeur envoyée telle quelle, validée par l'API. Curseur : domaine
                # d'apprentissage, en butée si le champ en sort (le champ n'est jamais réécrit).
                values[field] = number_with_slider(
                    RECOMMEND_LABELS[field],
                    f"{field}_{iso3}",
                    defaults[field] if active else None,
                    slider_range=(
                        0.0 if field in SLIDER_FROM_ZERO else training[field]["min"],
                        training[field]["max"],
                    ),
                    step=step,
                    number_format=number_format,  # affichage seulement : la valeur envoyée garde sa précision
                    log_scale=log_scale,
                    unit=physical[field]["unit"],  # repère « 0 » si la plage passe par zéro (température)
                )
                # Bornes arrondies pour la lecture ; les widgets gardent les vraies valeurs.
                st.markdown(domain_help_html(training[field], decimals), unsafe_allow_html=True)
    submitted = st.button(
        "Classer les cultures",
        type="primary",
        width="stretch",
        icon=":material/arrow_forward:",
        icon_position="right",
        disabled=not active,
    )
    st.html("".join(unit_suffix_css(f"{field}_{iso3}", physical[field]["unit"]) for field in FIELDS))
    # Étape 2 grisée (curseurs sans poignée) tant qu'aucun pays n'a fourni ses valeurs :
    # la règle n'est injectée que dans cet état et disparaît dès que le bloc devient actif.
    if not active:
        st.html('<style>.st-key-cond_card{opacity:.55}.st-key-cond_card [data-testid="stSlider"] [style*="translate"]{visibility:hidden}</style>')

    # Conditions affichées, telles qu'elles partent vers l'API.
    conditions = {field: float(value) for field, value in values.items()} if active else None
    if submitted and active:
        # Seuls le pays et ses 3 conditions partent vers l'API.
        payload = {"iso3": iso3, "conditions": conditions}
        st.session_state[LAST_KEY] = None
        try:
            result = api_client.post_recommend(payload)
            if not _is_complete_ranking(result):
                raise ApiInvalidResponseError("classement incomplet")
        except ApiError as exc:
            st.error(format_api_error(exc, RECOMMEND_LABELS, physical))  # 422 : champs refusés et valeurs acceptées
        else:
            st.session_state[LAST_KEY] = {"iso3": iso3, "conditions": conditions, "result": result}

    # Dernier classement, affiché seulement s'il a été obtenu avec le pays et les conditions affichés.
    last = st.session_state.get(LAST_KEY)
    if last and last["iso3"] == iso3 and last["conditions"] == conditions:
        shown = last["result"]
        with st.container(key="ranking_card"):
            st.markdown(
                section_head_html("Classement des cultures", "Du rendement estimé le plus élevé au plus faible"),
                unsafe_allow_html=True,
            )
            st.markdown(ranking_html(shown), unsafe_allow_html=True)

    st.markdown(footer_html("Recommend · Classement des cultures"), unsafe_allow_html=True)


# --- Panneau : présentation du service ou culture classée première ------

panel_col.markdown(recommend_panel_html(shown, training, crops), unsafe_allow_html=True)
