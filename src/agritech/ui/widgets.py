"""Champ numérique et curseur synchronisés : une seule valeur, deux façons de la saisir.

Le champ porte la vraie valeur, celle envoyée à l'API ; le curseur n'en est
que la représentation. Le champ n'a pas de bornes : la valeur saisie part
telle quelle et l'API décide si elle est physiquement valide. Le curseur couvre
le domaine d'apprentissage : hors de ce domaine, il se met en butée sans jamais
réécrire le champ. Streamlit interdit les callbacks dans un ``st.form`` : ces
widgets sont donc placés hors formulaire.

Échelle logarithmique (option ``log_scale``) : le curseur se déplace sur
``log10(1 + valeur)``, uniquement pour l'affichage. Cette transformation ne
sort jamais du frontend.
"""

from __future__ import annotations

import math

import streamlit as st

LOG_STEP = 0.01  # pas du curseur logarithmique, soit environ 2,3 % de variation par cran


def _to_slider(value: float, log_scale: bool) -> float:
    # Valeur négative (refusée ensuite par l'API) : curseur logarithmique en butée basse.
    return math.log10(1 + max(value, 0.0)) if log_scale else value


def _from_slider(position: float, log_scale: bool) -> float:
    return round(10**position - 1, 2) if log_scale else position


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _snap_inward(low: float, high: float, step: float) -> tuple[float, float]:
    """Bornes du curseur ramenées vers l'intérieur sur la grille de son pas.

    Le curseur avance de ``low`` en ``low + step`` : avec une borne non ronde
    (2,5667 °C), toutes ses valeurs le seraient aussi (2,8667 °C). Ramenées vers
    l'intérieur (2,6 → 30,2 °C), elles restent rondes et dans la plage demandée.
    Plage plus étroite qu'un pas : bornes inchangées.
    """
    first = round(math.ceil(low / step - 1e-9) * step, 6)
    last = round(math.floor(high / step + 1e-9) * step, 6)
    return (first, last) if first < last else (low, high)


def _number_to_slider(key: str, low: float, high: float, log_scale: bool) -> None:
    # Une valeur hors de la plage du curseur reste dans le champ ; le curseur se met en butée.
    st.session_state[f"{key}_slider"] = _clamp(_to_slider(st.session_state[key], log_scale), low, high)


def _slider_to_number(key: str, log_scale: bool) -> None:
    st.session_state[key] = _from_slider(st.session_state[f"{key}_slider"], log_scale)


def _zero_mark_css(slider_key: str, low: float, high: float, unit: str) -> str:
    """Repère vertical et libellé « 0 unité » sur la barre, à la position du zéro."""
    left = f"{100 * (0 - low) / (high - low):.2f}%"
    text = f"0 {unit}".replace("\\", "\\\\").replace('"', '\\"')
    track = f'.st-key-{slider_key} [data-orientation="horizontal"][style*="relative"]'
    return (
        "<style>"
        f'{track}::before{{content:"";position:absolute;left:{left};top:calc(50% - 7px);width:2px;height:14px;'
        "border-radius:1px;background:var(--field-line);transform:translateX(-50%)}"
        f'{track}::after{{content:"{text}";position:absolute;left:{left};top:calc(50% + 9px);'
        "transform:translateX(-50%);font-size:12px;line-height:1;color:var(--faint);white-space:nowrap}"
        "</style>"
    )


def number_with_slider(
    label: str,
    key: str,
    default: float | None,
    *,
    slider_range: tuple[float, float],
    step: float,
    number_format: str,
    log_scale: bool = False,
    unit: str | None = None,
) -> float | None:
    """Champ numérique suivi d'un curseur synchronisé ; ``default=None`` les affiche désactivés et vides.

    Avec ``unit``, un repère « 0 unité » est dessiné sur la barre quand sa plage passe par zéro
    (température, par exemple).
    """
    slider_step = LOG_STEP if log_scale else step
    low, high = _snap_inward(*(_to_slider(bound, log_scale) for bound in slider_range), slider_step)
    slider_key = f"{key}_slider"
    if unit is not None and low < 0 < high:
        st.html(_zero_mark_css(slider_key, low, high, unit))
    if default is None:
        st.number_input(label, value=None, key=key, disabled=True, label_visibility="collapsed")
        st.slider(label, low, high, value=low, key=slider_key, disabled=True, label_visibility="collapsed")
        return None
    if key not in st.session_state:  # valeur par défaut posée une seule fois, dans les deux widgets
        st.session_state[key] = default
        st.session_state[slider_key] = _clamp(_to_slider(default, log_scale), low, high)
    value = st.number_input(
        label,
        step=step,
        format=number_format,
        key=key,
        on_change=_number_to_slider,
        args=(key, low, high, log_scale),
        label_visibility="collapsed",
    )
    st.slider(
        label,
        low,
        high,
        step=slider_step,
        key=slider_key,
        on_change=_slider_to_number,
        args=(key, log_scale),
        label_visibility="collapsed",
    )
    return value
