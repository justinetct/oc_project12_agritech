"""Interface Gradio de monitoring de l'API Agritech Answers.

L'interface ne lit que l'API HTTP, jamais la base SQLite :

- appels HTTP : ``agritech.ui.api_client`` (URL ``AGRITECH_API_URL``,
  token ``MONITORING_API_TOKEN``, lus dans l'environnement ; au lancement,
  un éventuel ``.env`` local complète l'environnement, comme pour l'API) ;
- mise en forme : ``agritech.ui.monitoring_view`` ;
- messages d'erreur : ``agritech.ui.errors.format_monitoring_error``.

Ce module ne fait que construire la page et brancher ses événements.
``load_dashboard(days)`` est la seule fonction appelée par les événements :
elle renvoie, dans l'ordre des composants, tout ce que la page affiche.

Lancement depuis la racine du dépôt : ``make gradio`` (fonction ``main``).
"""

from __future__ import annotations

from html import escape
from typing import NamedTuple

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from agritech.ui import api_client
from agritech.ui.errors import ApiError, format_monitoring_error
from agritech.ui.monitoring_view import (
    DAILY_COLUMNS,
    ERRORS_COLUMNS,
    SERVICES,
    Kpi,
    daily_volume_frame,
    empty_kpis,
    empty_service_kpis,
    errors_frame,
    service_kpis,
    summary_kpis,
)
from agritech.ui.theme import COLORS, logo_data_uri


# Périodes proposées : libellé affiché, nombre de jours envoyé à l'API.
PERIODS = [("7 jours", 7), ("30 jours", 30), ("90 jours", 90)]
DEFAULT_DAYS = 30

NO_REQUEST_MESSAGE = "Aucune requête sur cette période."
NO_ERROR_MESSAGE = "Aucune erreur sur cette période."

# Couleur de chaque service, la même dans le graphique et dans ses indicateurs.
# Deux services de même niveau : bleu ardoise et bleu-vert ; jaune, orange et
# rouge restent réservés aux erreurs et alertes.
SERVICE_COLORS = {"predict": COLORS["slate"], "recommend": COLORS["teal"]}

# Nom affiché de chaque service, en tête de sa rangée d'indicateurs.
SERVICE_TITLES = {"predict": "Predict", "recommend": "Recommend"}


class DashboardView(NamedTuple):
    """Valeurs affichées par la page, dans l'ordre des composants de sortie."""

    status_html: str
    kpis_html: str
    daily: pd.DataFrame
    daily_note_html: str
    services_html: str
    errors: pd.DataFrame
    errors_note_html: str


def load_dashboard(days: int) -> DashboardView:
    """Charge le résumé des ``days`` derniers jours et prépare tout l'affichage.

    En cas d'erreur API, la page est entièrement remise à vide (aucune donnée
    d'un chargement précédent ne reste affichée) et le message d'erreur
    s'affiche en tête.
    """
    try:
        summary = api_client.get_monitoring_summary(days)
    except ApiError as exc:
        return DashboardView(
            status_html=alert_html(format_monitoring_error(exc)),
            kpis_html=kpi_cards_html(empty_kpis()),
            daily=pd.DataFrame(columns=DAILY_COLUMNS),
            daily_note_html="",
            services_html=service_rows_html(empty_service_kpis()),
            errors=pd.DataFrame(columns=ERRORS_COLUMNS),
            errors_note_html="",
        )

    errors = errors_frame(summary)
    return DashboardView(
        status_html="",
        kpis_html=kpi_cards_html(summary_kpis(summary)),
        daily=daily_volume_frame(summary),
        daily_note_html=note_html(NO_REQUEST_MESSAGE) if summary.total_requests == 0 else "",
        services_html=service_rows_html(service_kpis(summary)),
        errors=errors,
        errors_note_html=note_html(NO_ERROR_MESSAGE) if errors.empty else "",
    )


# --- Blocs HTML ------------------------------------------------------------------


def header_html() -> str:
    """En-tête : logo, titre et sous-titre, sur le fond vert de la page."""
    return (
        '<div class="mon-header">'
        f'<img class="mon-logo" src="{logo_data_uri()}" alt="Agritech Answers">'
        '<div class="mon-heading"><div class="mon-title">Monitoring Agritech</div>'
        '<div class="mon-subtitle">Suivi des appels <code>/predict</code> et '
        "<code>/recommend</code> de l'API</div></div>"
        "</div>"
    )


def section_title_html(title: str) -> str:
    """Titre de section, collé à son contenu."""
    return f'<h3 class="mon-section-title">{escape(title)}</h3>'


def kpi_cards_html(kpis: list[Kpi]) -> str:
    """Les quatre cartes d'indicateurs globaux."""
    return f'<div class="mon-kpis">{_cards_html(kpis)}</div>'


def service_rows_html(kpis_by_service: dict[str, list[Kpi]]) -> str:
    """Une rangée de cartes par service, avec son nom et sa couleur (celle du graphique)."""
    rows = "".join(
        f'<div class="mon-service" style="--accent:{SERVICE_COLORS[service]}">'
        f'<div class="mon-service-name">{escape(SERVICE_TITLES[service])}</div>'
        f'<div class="mon-service-kpis">{_cards_html(kpis_by_service[service])}</div>'
        "</div>"
        for service in SERVICES
    )
    return f'<div class="mon-services">{rows}</div>'


def _cards_html(kpis: list[Kpi]) -> str:
    """Cartes « libellé + valeur », communes aux indicateurs globaux et par service."""
    return "".join(
        '<div class="mon-kpi">'
        f'<div class="mon-kpi-label">{escape(kpi.label)}</div>'
        f'<div class="mon-kpi-value">{escape(kpi.value)}</div>'
        "</div>"
        for kpi in kpis
    )


def alert_html(message: str) -> str:
    """Bandeau d'erreur affiché en tête de page."""
    return f'<div class="mon-alert" role="alert">{escape(message)}</div>'


def note_html(message: str) -> str:
    """Message discret (période sans requête, aucune erreur...)."""
    return f'<p class="mon-note">{escape(message)}</p>'


# --- Page ------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Construit la page et branche les événements sur ``load_dashboard``."""
    with gr.Blocks(title="Monitoring Agritech") as demo:
        # En-tête et contrôles sur une seule ligne (repliée sur petit écran).
        with gr.Row(equal_height=True, elem_classes="mon-top"):
            # Sur un écran étroit, les contrôles passent sous l'en-tête
            # au lieu de le comprimer (largeur minimale fixée dans le CSS).
            gr.HTML(header_html(), padding=False, scale=1, elem_classes="mon-flush mon-header-block")
            period = gr.Radio(
                choices=PERIODS,
                value=DEFAULT_DAYS,
                label="Période",
                show_label=False,
                container=False,
                elem_classes="mon-period",
                scale=0,
                min_width=290,
            )
            refresh = gr.Button(
                "Actualiser",
                variant="secondary",
                size="sm",
                elem_classes="mon-refresh",
                scale=0,
                min_width=120,
            )

        # Zones de message : masquées par le CSS tant qu'elles sont vides.
        status = gr.HTML(padding=False, elem_classes="mon-flush mon-slot")
        kpis = gr.HTML(kpi_cards_html(empty_kpis()), padding=False, elem_classes="mon-flush")

        # Indicateurs par service juste sous les indicateurs globaux.
        services = gr.HTML(
            service_rows_html(empty_service_kpis()), padding=False, elem_classes="mon-flush"
        )

        with gr.Column(elem_classes="mon-section"):
            gr.HTML(section_title_html("Volume quotidien"), padding=False, elem_classes="mon-flush")
            daily_note = gr.HTML(padding=False, elem_classes="mon-flush mon-slot")
            daily_plot = gr.BarPlot(
                x="date",
                y="requests",
                color="service",
                color_map=SERVICE_COLORS,
                sort="x",
                x_title="Jour (UTC)",
                y_title="Requêtes",
                color_title="Service",
                x_label_angle=-45,
                show_label=False,
                height=300,
                buttons=[],
                elem_classes="mon-chart",
            )

        with gr.Column(elem_classes="mon-section"):
            gr.HTML(section_title_html("Erreurs par type"), padding=False, elem_classes="mon-flush")
            errors_note = gr.HTML(padding=False, elem_classes="mon-flush mon-slot")
            errors_table = gr.Dataframe(
                headers=ERRORS_COLUMNS,
                interactive=False,
                show_label=False,
                buttons=[],
                wrap=True,
                elem_classes="mon-table",
            )

        outputs = [
            status,
            kpis,
            daily_plot,
            daily_note,
            services,
            errors_table,
            errors_note,
        ]
        demo.load(load_dashboard, inputs=period, outputs=outputs)
        period.change(load_dashboard, inputs=period, outputs=outputs)
        refresh.click(load_dashboard, inputs=period, outputs=outputs)
    return demo


# --- Apparence -------------------------------------------------------------------


_BASE_THEME = gr.themes.Base(primary_hue="green", neutral_hue="stone")


def _same_in_both_modes(**values: str) -> dict[str, str]:
    """Mêmes couleurs en mode clair et en mode sombre (variables ``*_dark``).

    Sans cela, Gradio suit le mode sombre du système et remplace la palette
    du projet par un fond presque noir. Seules les variables qui ont une
    variante sombre la reçoivent.
    """
    dark = {
        f"{name}_dark": value
        for name, value in values.items()
        if hasattr(_BASE_THEME, f"{name}_dark")
    }
    return {**values, **dark}


# Page sombre vert Agritech ; cartes, graphique et tableaux sur surfaces claires.
THEME = _BASE_THEME.set(
    layout_gap="14px",
    **_same_in_both_modes(
        body_background_fill=COLORS["forest-top"],
        body_text_color=COLORS["ink"],
        body_text_color_subdued=COLORS["muted"],
        background_fill_primary=COLORS["field"],
        background_fill_secondary=COLORS["card"],
        border_color_primary=COLORS["card-line"],
        block_background_fill=COLORS["field"],
        block_border_color=COLORS["card-line"],
        block_label_text_color=COLORS["muted"],
        block_title_text_color=COLORS["ink"],
        color_accent=COLORS["leaf"],
        color_accent_soft=COLORS["mint"],
        input_background_fill=COLORS["field"],
        table_border_color=COLORS["card-line"],
        table_even_background_fill=COLORS["field"],
        table_odd_background_fill=COLORS["card"],
        table_text_color=COLORS["ink"],
        table_row_focus=COLORS["mint"],
        checkbox_label_background_fill="rgba(255,255,255,0.08)",
        checkbox_label_background_fill_hover="rgba(255,255,255,0.16)",
        checkbox_label_background_fill_selected=COLORS["cream"],
        checkbox_label_border_color="rgba(255,255,255,0.18)",
        checkbox_label_border_color_hover="rgba(255,255,255,0.30)",
        checkbox_label_border_color_selected=COLORS["cream"],
        checkbox_label_text_color="#ffffff",
        checkbox_label_text_color_selected=COLORS["forest"],
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_hover="rgba(255,255,255,0.12)",
        button_secondary_border_color="rgba(255,255,255,0.45)",
        button_secondary_border_color_hover="#ffffff",
        button_secondary_text_color="#ffffff",
        button_secondary_text_color_hover="#ffffff",
    ),
)

CSS = f"""
/* Nos blocs HTML sans marge intérieure : titres alignés et collés à leur contenu */
.mon-flush .html-container{{padding:0 !important}}

/* En-tête et contrôles */
.mon-header-block{{min-width:min(440px,100%) !important}}
.mon-top{{align-items:center;padding-bottom:14px;border-bottom:1px solid rgba(255,255,255,.12)}}
.mon-header{{display:flex;align-items:center;gap:20px}}
.mon-logo{{height:46px;width:auto}}
.mon-heading{{padding-left:20px;border-left:1px solid rgba(255,255,255,.18)}}
.mon-title{{color:#fff;font-size:26px;font-weight:700;line-height:1.15;letter-spacing:-.01em}}
.mon-subtitle{{color:{COLORS["mint-soft"]};font-size:14px;margin-top:2px}}
.mon-subtitle code{{background:transparent;color:#fff;padding:0;font-size:13px}}
.mon-period .wrap{{gap:6px}}
.mon-period label{{border-radius:999px !important;padding:6px 16px !important;font-weight:600}}
.mon-period input[type=radio]{{display:none}}
.mon-refresh{{border-radius:999px !important;font-weight:600}}

/* Zones de message : masquées tant qu'elles ne contiennent rien */
.mon-slot:not(:has(.mon-alert, .mon-note)){{display:none !important}}
.mon-alert{{padding:12px 16px;border-left:4px solid {COLORS["warn-line"]};border-radius:8px;
  background:{COLORS["warn-bg"]};color:{COLORS["warn-ink"]};font-weight:600}}
.mon-note{{margin:0;color:{COLORS["mint-soft"]};font-size:14px}}

/* Indicateurs */
.mon-kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}
.mon-kpi{{background:{COLORS["cream"]};border-radius:12px;padding:14px 18px}}
.mon-kpi-label{{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{COLORS["muted"]}}}
.mon-kpi-value{{margin-top:4px;font-size:22px;font-weight:700;line-height:1.25;color:{COLORS["forest"]}}}
/* Dernière carte globale (« Dernière requête ») : date plus longue, police réduite */
.mon-kpis > .mon-kpi:last-child .mon-kpi-value{{font-size:18px;line-height:1.35}}

/* Indicateurs par service : cartes sur fond de la couleur du service (celle du graphique) */
.mon-services{{display:flex;flex-direction:column;gap:14px;margin-top:6px}}
.mon-service-name{{display:flex;align-items:center;gap:8px;margin-bottom:8px;color:#fff;font-size:17px;font-weight:700}}
.mon-service-name::before{{content:"";width:12px;height:12px;border-radius:3px;background:var(--accent);
  box-shadow:0 0 0 2px rgba(255,255,255,.85)}}
.mon-service-kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}
.mon-service .mon-kpi{{background:var(--accent);padding:12px 16px}}
.mon-service .mon-kpi-label{{color:#ffffff}}
.mon-service .mon-kpi-value{{font-size:20px;color:#ffffff}}

/* Sections : titre collé à son contenu */
.mon-section{{gap:8px !important;margin-top:6px}}
.mon-section-title{{margin:0 !important;color:#fff !important;font-size:17px;font-weight:700}}
/* Graphique : fin trait de la couleur du fond autour des barres (SVG, sans JavaScript).
   Invisible sur le contour, il ne se voit qu'à la jonction Predict / Recommend. */
.mon-chart g.mark-rect.role-mark path{{stroke:{COLORS["field"]};stroke-width:1px}}
.mon-table table, .mon-table th, .mon-table td{{font-family:inherit !important}}
/* En-têtes : retour à la ligne entre les mots, jamais au milieu d'un mot */
.mon-table th, .mon-table th *{{word-break:normal !important;overflow-wrap:normal !important}}

@media (max-width:760px){{
  .mon-heading{{padding-left:0;border-left:0}}
  .mon-title{{font-size:22px}}
  .mon-kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .mon-kpi-value{{font-size:20px}}
  .mon-service-kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}
}}
@media (min-width:761px) and (max-width:1000px){{
  .mon-service-kpis{{grid-template-columns:repeat(3,minmax(0,1fr))}}
}}
"""


def main() -> None:
    """Lance l'interface en local.

    Charge d'abord un éventuel ``.env``, comme le fait l'API au démarrage :
    sans ``.env``, ``load_dotenv`` ne fait rien ; avec ``override=False``,
    une variable déjà présente dans l'environnement (shell, Docker, secret
    d'hébergement) n'est jamais remplacée par la valeur du fichier.
    """
    load_dotenv(override=False)
    build_app().launch(theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
