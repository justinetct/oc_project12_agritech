"""Mise en forme des données de monitoring pour l'interface de suivi.

Fonctions pures : elles reçoivent les réponses déjà validées par
``api_client`` (``MonitoringSummaryResponse``, ``MonitoringRequestsResponse``)
et renvoient des valeurs prêtes à afficher. Aucun appel HTTP, aucune
dépendance à l'interface graphique : les widgets sont construits ailleurs.

- ``summary_kpis`` : les quatre indicateurs du bandeau (libellé + valeur),
  et ``empty_kpis`` pour les remettre à « — » quand le résumé est indisponible ;
- ``daily_volume_frame`` : volume quotidien au format « long » pour un
  graphique en barres empilées (une ligne par jour et par service) ;
- ``daily_errors_frame`` : nombre d'erreurs par jour, pour un graphique en barres ;
- ``date_axis_labels`` : les jours étiquetés sous ces deux graphiques ;
- ``service_kpis`` : cinq indicateurs par service (volumes, réussite,
  latences médiane et max), et ``empty_service_kpis`` pour les remettre à « — » ;
- ``recent_errors_frame`` et ``recent_successes_frame`` : les derniers appels
  en erreur et réussis, une ligne par appel.

Les tableaux sont des ``pandas.DataFrame`` aux colonnes fixes, même vides.
Les dates reçues en UTC sont affichées à l'heure de Paris (heure d'été
comprise). Une mesure absente s'affiche « — », jamais « 0 ».
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from agritech.api.schemas.monitoring import (
    MonitoringRequestsResponse,
    MonitoringSummaryResponse,
)
from agritech.ui.components import fr_number


# Valeur affichée quand une mesure n'existe pas (aucun appel, aucune latence...).
MISSING = "—"

# Ordre d'affichage des services, identique à celui de l'API.
SERVICES = ("predict", "recommend")

# Fuseau d'affichage des dates ; l'API et SQLite restent en UTC.
DISPLAY_TIMEZONE = ZoneInfo("Europe/Paris")

# Mois abrégés en français, pour une date lisible (« 05 oct. 2026 »).
FRENCH_MONTHS = (
    "janv.", "févr.", "mars", "avr.", "mai", "juin",
    "juil.", "août", "sept.", "oct.", "nov.", "déc.",
)

# Libellés des quatre indicateurs du bandeau, dans l'ordre d'affichage.
KPI_LABELS = ("Requêtes", "Requêtes réussies", "Erreurs", "Dernière requête")

# Colonnes du graphique quotidien : noms techniques, utilisés comme x / y / couleur.
DAILY_COLUMNS = ["date", "service", "requests"]

# Colonnes des tableaux affichés tels quels.
# Libellés des cinq indicateurs affichés pour chaque service.
SERVICE_KPI_LABELS = ("Requêtes", "Réussies", "Erreurs", "Latence médiane", "Latence max")

RECENT_ERRORS_COLUMNS = ["Date", "Service", "Statut", "Erreur", "Message", "Durée", "Entrées"]
RECENT_SUCCESSES_COLUMNS = ["Date", "Service", "Durée", "Modèle", "Entrées"]

# Colonnes du graphique des erreurs quotidiennes.
DAILY_ERRORS_COLUMNS = ["date", "errors"]

# Écart entre deux jours étiquetés sous les graphiques quotidiens, selon la
# longueur de la période : (nombre de jours maximal, écart). Chaque jour sur
# 7 jours, tous les 3 jours sur 30, puis chaque semaine au-delà (90 jours).
DATE_LABEL_STEPS = ((7, 1), (31, 3))
WEEKLY_LABEL_STEP = 7


@dataclass(frozen=True)
class Kpi:
    """Un indicateur du bandeau : son libellé et sa valeur déjà formatée."""

    label: str
    value: str


def summary_kpis(summary: MonitoringSummaryResponse) -> list[Kpi]:
    """Les quatre indicateurs de la période, dans l'ordre du bandeau."""
    values = (
        format_count(summary.total_requests),
        format_rate(summary.success_rate),
        format_count(summary.error_count),
        format_readable_datetime(summary.last_request_at),
    )
    return [Kpi(label, value) for label, value in zip(KPI_LABELS, values)]


def empty_kpis() -> list[Kpi]:
    """Les quatre indicateurs sans valeur (« — »), quand aucun résumé n'est disponible."""
    return [Kpi(label, MISSING) for label in KPI_LABELS]


def daily_volume_frame(summary: MonitoringSummaryResponse) -> pd.DataFrame:
    """Volume quotidien au format long : colonnes ``date``, ``service``, ``requests``.

    Deux lignes par jour (``predict`` puis ``recommend``), dans l'ordre
    chronologique fourni par l'API, jours à zéro compris. La date est une
    chaîne ISO ``YYYY-MM-DD`` : triée comme du texte, elle reste chronologique.
    """
    rows = [
        {"date": day.date.isoformat(), "service": service, "requests": getattr(day, service)}
        for day in summary.requests_per_day
        for service in SERVICES
    ]
    frame = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    return frame.astype({"requests": "int64"})


def daily_errors_frame(summary: MonitoringSummaryResponse) -> pd.DataFrame:
    """Erreurs par jour : colonnes ``date`` et ``errors``.

    Une ligne par jour de la période, jours sans erreur à 0, dans l'ordre
    chronologique.
    """
    rows = [
        {"date": day.date.isoformat(), "errors": day.errors} for day in summary.requests_per_day
    ]
    frame = pd.DataFrame(rows, columns=DAILY_ERRORS_COLUMNS)
    return frame.astype({"errors": "int64"})


def date_label_step(day_count: int) -> int:
    """Écart entre deux jours étiquetés pour une période de ``day_count`` jours."""
    for max_days, step in DATE_LABEL_STEPS:
        if day_count <= max_days:
            return step
    return WEEKLY_LABEL_STEP


def date_axis_labels(dates: Sequence[str]) -> list[str]:
    """Les jours à étiqueter sur l'axe des dates, parmi ``dates`` (ordre chronologique).

    Les dates répétées (une ligne par service et par jour) ne comptent qu'une
    fois. Un jour sur ``date_label_step`` en partant du premier ; le dernier
    jour est toujours ajouté. S'il tombe à moins d'un demi-écart de l'étiquette
    précédente, celle-ci est retirée pour que les deux ne se chevauchent pas.
    Toutes les barres restent tracées : seules les étiquettes sont espacées.
    """
    dates = list(dict.fromkeys(dates))
    if not dates:
        return []
    step = date_label_step(len(dates))
    indexes = list(range(0, len(dates), step))
    last = len(dates) - 1
    if indexes[-1] != last:
        if len(indexes) > 1 and last - indexes[-1] < step / 2:
            indexes.pop()
        indexes.append(last)
    return [dates[index] for index in indexes]


def service_kpis(summary: MonitoringSummaryResponse) -> dict[str, list[Kpi]]:
    """Cinq indicateurs par service, dans l'ordre de ``SERVICES``.

    Latences calculées sur les requêtes réussies : la médiane donne la durée
    habituelle, le maximum la pire latence observée sur la période. Un service
    absent du résumé ou sans aucun appel affiche « — » là où il n'y a pas de
    mesure.
    """
    entries = {entry.service: entry for entry in summary.services}
    kpis = empty_service_kpis()
    for service in SERVICES:
        entry = entries.get(service)
        if entry is None:
            continue
        values = (
            format_count(entry.total_requests),
            format_rate(entry.success_rate),
            format_count(entry.error_count),
            format_latency(entry.latency_ms.median),
            format_latency(entry.latency_ms.max),
        )
        kpis[service] = [Kpi(label, value) for label, value in zip(SERVICE_KPI_LABELS, values)]
    return kpis


def empty_service_kpis() -> dict[str, list[Kpi]]:
    """Indicateurs par service sans valeur (« — »), quand aucun résumé n'est disponible."""
    return {service: [Kpi(label, MISSING) for label in SERVICE_KPI_LABELS] for service in SERVICES}


def recent_errors_frame(response: MonitoringRequestsResponse) -> pd.DataFrame:
    """Les derniers appels en erreur, dans l'ordre reçu (du plus récent au plus ancien).

    Date à l'heure de Paris, à la seconde, statut HTTP entier, code
    d'erreur court, message public de l'API et corps JSON compact envoyé.
    """
    rows = [
        [
            _table_date(item.timestamp),
            item.service,
            item.status_code,
            item.error_type or MISSING,
            item.error_message or MISSING,
            format_ms(item.duration_ms),
            compact_json(item.request_payload),
        ]
        for item in response.items
    ]
    return pd.DataFrame(rows, columns=RECENT_ERRORS_COLUMNS)


def recent_successes_frame(response: MonitoringRequestsResponse) -> pd.DataFrame:
    """Les derniers appels réussis, dans l'ordre reçu (du plus récent au plus ancien).

    Pas de colonne de statut : ces appels ont tous réussi.
    """
    rows = [
        [
            _table_date(item.timestamp),
            item.service,
            format_ms(item.duration_ms),
            item.model_version or MISSING,
            compact_json(item.request_payload),
        ]
        for item in response.items
    ]
    return pd.DataFrame(rows, columns=RECENT_SUCCESSES_COLUMNS)


def _table_date(moment: datetime) -> str:
    """Date d'un tableau : heure de Paris à la seconde."""
    return format_datetime(moment, with_seconds=True)


# --- Formatage des valeurs ---------------------------------------------------


def format_count(value: int) -> str:
    """Entier à la française : ``1234`` → ``1 234`` (espace fine insécable)."""
    return fr_number(value)


def format_rate(rate: float | None) -> str:
    """Taux entre 0 et 1 en pourcentage à une décimale : ``0.976`` → ``97,6 %``."""
    if rate is None:
        return MISSING
    return f"{fr_number(rate * 100, 1)} %"


def format_ms(value: int) -> str:
    """Durée d'un appel en millisecondes : ``14`` → ``14 ms``."""
    return f"{fr_number(value)} ms"


def format_latency(value: float | None) -> str:
    """Latence à une décimale, sans « ,0 » inutile : ``5.5`` → ``5,5 ms``, ``12.0`` → ``12 ms``."""
    if value is None:
        return MISSING
    text = fr_number(value, 1)
    if text.endswith(",0"):
        text = text[:-2]
    return f"{text}\u202fms"


def to_display_time(moment: datetime) -> datetime:
    """Date UTC de l'API ramenée à l'heure de Paris ; une date sans fuseau est lue comme UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(DISPLAY_TIMEZONE)


def format_datetime(moment: datetime | None, *, with_seconds: bool = False) -> str:
    """Date et heure de Paris : ``2026-10-01 15:45`` ; ``None`` → ``—``."""
    if moment is None:
        return MISSING
    pattern = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    return to_display_time(moment).strftime(pattern)


def format_readable_datetime(moment: datetime | None) -> str:
    """Date lisible en français, heure de Paris : ``05 oct. 2026 · 10:51`` ; ``None`` → ``—``.

    Le mois vient de ``FRENCH_MONTHS`` : le résultat ne dépend pas de la
    langue configurée sur la machine.
    """
    if moment is None:
        return MISSING
    moment = to_display_time(moment)
    month = FRENCH_MONTHS[moment.month - 1]
    return f"{moment.day:02d} {month} {moment.year} · {moment:%H:%M}"


def compact_json(payload: Any) -> str:
    """Corps JSON sur une seule ligne, sans espaces, accents conservés.

    L'ordre des clés est celui de la requête archivée (déjà stable) : on ne
    le trie pas, pour garder l'ordre naturel des champs envoyés.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
