"""Mise en forme des données de monitoring pour l'interface de suivi.

Fonctions pures : elles reçoivent les réponses déjà validées par
``api_client`` (``MonitoringSummaryResponse``, ``MonitoringRequestsResponse``)
et renvoient des valeurs prêtes à afficher. Aucun appel HTTP, aucune
dépendance à l'interface graphique : les widgets sont construits ailleurs.

- ``summary_kpis`` : les quatre indicateurs du bandeau (libellé + valeur) ;
- ``daily_volume_frame`` : volume quotidien au format « long » pour un
  graphique en barres empilées (une ligne par jour et par service) ;
- ``services_frame`` : une ligne par service, latences comprises ;
- ``errors_frame`` : nombre d'erreurs par type ;
- ``recent_requests_frame`` : les derniers appels, une ligne par appel.

Les tableaux sont des ``pandas.DataFrame`` aux colonnes fixes, même vides.
Les dates restent en UTC, sans conversion vers un fuseau local. Une mesure
absente s'affiche « — », jamais « 0 ».
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from agritech.api.schemas.monitoring import (
    LatencySummary,
    MonitoringRequestsResponse,
    MonitoringSummaryResponse,
)
from agritech.ui.components import fr_number


# Valeur affichée quand une mesure n'existe pas (aucun appel, aucune latence...).
MISSING = "—"

# Ordre d'affichage des services, identique à celui de l'API.
SERVICES = ("predict", "recommend")

# Colonnes du graphique quotidien : noms techniques, utilisés comme x / y / couleur.
DAILY_COLUMNS = ["date", "service", "requests"]

# Colonnes des tableaux affichés tels quels.
SERVICES_COLUMNS = [
    "Service",
    "Requêtes",
    "Erreurs",
    "Taux de succès",
    "Latence moyenne",
    "Latence médiane",
    "Latence max",
]
ERRORS_COLUMNS = ["Type d'erreur", "Nombre"]
REQUESTS_COLUMNS = ["Date (UTC)", "Service", "Statut", "Durée", "Modèle", "Erreur", "Entrées"]


@dataclass(frozen=True)
class Kpi:
    """Un indicateur du bandeau : son libellé et sa valeur déjà formatée."""

    label: str
    value: str


def summary_kpis(summary: MonitoringSummaryResponse) -> list[Kpi]:
    """Les quatre indicateurs de la période, dans l'ordre du bandeau."""
    return [
        Kpi("Requêtes", format_count(summary.total_requests)),
        Kpi("Taux de succès", format_rate(summary.success_rate)),
        Kpi("Erreurs", format_count(summary.error_count)),
        Kpi("Dernière requête", format_utc(summary.last_request_at)),
    ]


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


def services_frame(summary: MonitoringSummaryResponse) -> pd.DataFrame:
    """Une ligne par service : volumes, taux de succès et latences des appels réussis."""
    rows = [
        [
            entry.service,
            format_count(entry.total_requests),
            format_count(entry.error_count),
            format_rate(entry.success_rate),
            *_latency_cells(entry.latency_ms),
        ]
        for entry in summary.services
    ]
    return pd.DataFrame(rows, columns=SERVICES_COLUMNS)


def errors_frame(summary: MonitoringSummaryResponse) -> pd.DataFrame:
    """Erreurs par type, de la plus fréquente à la moins fréquente.

    À nombre égal, les types sont classés par ordre alphabétique. Sans
    erreur, le tableau est vide (aucune ligne inventée).
    """
    ordered = sorted(summary.errors_by_type.items(), key=lambda item: (-item[1], item[0]))
    return pd.DataFrame([list(item) for item in ordered], columns=ERRORS_COLUMNS)


def recent_requests_frame(response: MonitoringRequestsResponse) -> pd.DataFrame:
    """Les derniers appels, dans l'ordre reçu (du plus récent au plus ancien).

    La date est en UTC (indiqué dans l'en-tête de colonne), à la seconde ;
    le statut HTTP reste un entier ; l'erreur affichée est le code court
    (``validation_error``...) ; les entrées sont le corps JSON compact.
    """
    rows = [
        [
            format_utc(item.timestamp, with_seconds=True, with_suffix=False),
            item.service,
            item.status_code,
            format_ms(item.duration_ms),
            item.model_version or MISSING,
            item.error_type or item.error_message or MISSING,
            compact_json(item.request_payload),
        ]
        for item in response.items
    ]
    return pd.DataFrame(rows, columns=REQUESTS_COLUMNS)


# --- Formatage des valeurs ---------------------------------------------------


def format_count(value: int) -> str:
    """Entier à la française : ``1234`` → ``1 234`` (espace fine insécable)."""
    return fr_number(value)


def format_rate(rate: float | None) -> str:
    """Taux entre 0 et 1 en pourcentage à une décimale : ``0.976`` → ``97,6 %``."""
    if rate is None:
        return MISSING
    return f"{fr_number(rate * 100, 1)} %"


def format_ms(value: float | None, decimals: int = 0) -> str:
    """Durée en millisecondes : ``14`` → ``14 ms`` ; ``None`` → ``—``."""
    if value is None:
        return MISSING
    return f"{fr_number(value, decimals)} ms"


def format_utc(
    moment: datetime | None, *, with_seconds: bool = False, with_suffix: bool = True
) -> str:
    """Date et heure en UTC : ``2026-10-01 13:45 UTC`` ; ``None`` → ``—``.

    ``with_seconds`` ajoute les secondes ; ``with_suffix=False`` retire le
    « UTC » final quand l'en-tête de colonne l'indique déjà. Aucune
    conversion vers un fuseau local : la date est ramenée en UTC.
    """
    if moment is None:
        return MISSING
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    pattern = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    text = moment.astimezone(timezone.utc).strftime(pattern)
    return f"{text} UTC" if with_suffix else text


def compact_json(payload: Any) -> str:
    """Corps JSON sur une seule ligne, sans espaces, accents conservés.

    L'ordre des clés est celui de la requête archivée (déjà stable) : on ne
    le trie pas, pour garder l'ordre naturel des champs envoyés.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _latency_cells(latency: LatencySummary) -> list[str]:
    """Moyenne, médiane et max, tous à une décimale pour un affichage homogène."""
    return [
        format_ms(latency.mean, decimals=1),
        format_ms(latency.median, decimals=1),
        format_ms(latency.max, decimals=1),
    ]
