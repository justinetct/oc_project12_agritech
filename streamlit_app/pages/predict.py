"""Parcours Streamlit du service /predict.

Charge le contexte de l'API (bornes physiques du contrat et domaine
d'apprentissage), construit un petit formulaire avec les 4 entrées
du modèle et affiche le rendement prédit renvoyé par ``POST /predict``.

Le front n'embarque aucune logique ML. Toutes les valeurs affichables
(bornes, unités, notes hors domaine) sont fournies par l'API et
consommées telles quelles. Si le contexte est incomplet ou mal formé,
la page échoue proprement plutôt que d'inventer des valeurs par défaut.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agritech.ui import api_client
from agritech.ui.errors import ApiError, format_api_error


st.set_page_config(page_title="Predict — Agritech Answers", page_icon="🌿")

st.title("Prédire un rendement")
st.write(
    "Renseignez les conditions de votre parcelle pour obtenir une "
    "estimation du rendement attendu."
)


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
    st.error(format_api_error(exc))
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
except _InvalidContextError as exc:
    st.error(f"Contexte API invalide : {exc}.")
    st.stop()


# --- Formulaire ------------------------------------------------------

with st.form("predict_form"):
    rainfall = st.number_input(
        f"Pluie totale ({rainfall_unit})",
        key="rainfall_mm",
        min_value=rainfall_min,
        max_value=rainfall_max,
        value=rainfall_default,
        step=1.0,
    )
    temperature = st.number_input(
        f"Température moyenne ({temperature_unit})",
        key="temperature_celsius",
        min_value=temperature_min,
        max_value=temperature_max,
        value=temperature_default,
        step=0.1,
    )
    fertilizer_used = st.checkbox("Fertilisation appliquée", key="fertilizer_used")
    irrigation_used = st.checkbox("Irrigation appliquée", key="irrigation_used")
    submitted = st.form_submit_button("Prédire le rendement")


# --- Appel API et affichage du résultat ------------------------------

if submitted:
    payload = {
        "rainfall_mm": float(rainfall),
        "temperature_celsius": float(temperature),
        "fertilizer_used": bool(fertilizer_used),
        "irrigation_used": bool(irrigation_used),
    }
    try:
        result = api_client.post_predict(payload)
    except ApiError as exc:
        st.error(format_api_error(exc))
    else:
        yield_value = result.get("yield_tons_per_hectare")
        unit = result.get("unit", "t/ha")
        try:
            yield_text = f"{float(yield_value):.2f} {unit}"
        except (TypeError, ValueError):
            yield_text = f"{yield_value} {unit}"
        st.metric("Rendement estimé", yield_text)

        model_version = result.get("model_version", "?")
        st.caption(f"Modèle version {model_version}")

        if result.get("out_of_training_domain"):
            notes = result.get("notes") or []
            body = (
                "Certaines conditions sortent du domaine "
                "d'apprentissage du modèle."
            )
            if notes:
                body += "\n\n" + "\n".join(f"- {note}" for note in notes)
            st.warning(body)
